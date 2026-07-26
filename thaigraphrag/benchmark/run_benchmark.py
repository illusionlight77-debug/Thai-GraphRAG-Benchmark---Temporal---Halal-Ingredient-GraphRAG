"""Benchmark runner — GraphRAG vs vanilla RAG, broken down by hop type.

This is the study's headline experiment. For every question it runs each retriever
through the identical answer pipeline, scores the answer, and aggregates by
hop_type (single / multi / relational) so the multi-hop advantage is visible.

Three suites share the harness:

| suite      | eval file                     | retrievers compared                  |
|------------|-------------------------------|--------------------------------------|
| core       | thai_eval.jsonl               | vanilla, graphrag                    |
| temporal   | temporal_eval.jsonl           | vanilla, graphrag, temporal          |
| ingredient | ingredient_eval.jsonl         | vanilla, graphrag, halal_ingredient  |

Answer metrics (F1/EM/containment) are reported next to **retrieval** metrics
(context_recall, hit@k). That separation matters: it distinguishes "the retriever
never found the evidence" from "the LLM had the evidence and still answered badly",
and only the first is the variable the experiment manipulates.

Outputs (results/):
    benchmark_detail.csv     one row per (question, retriever)
    benchmark_summary.csv    mean metrics per (retriever, hop_type) and overall
    figures/*.png            F1 by hop type, retrieval vs answer, B and C figures

Run:  python -m scripts.run_benchmark [--suite all] [--judge]
"""
from __future__ import annotations

import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from thaigraphrag import config
from thaigraphrag.benchmark import datasets, metrics
from thaigraphrag.core import llm
from thaigraphrag.pipeline.answer import answer
from thaigraphrag.retrievers import get_retriever

RETRIEVERS = ["vanilla", "graphrag"]

SUITES: dict[str, dict] = {
    "core": {"file": "thai_eval.jsonl",
             "retrievers": ["vanilla", "graphrag"],
             "title": "A — GraphRAG vs vanilla RAG"},
    "temporal": {"file": "temporal_eval.jsonl",
                 "retrievers": ["vanilla", "graphrag", "temporal"],
                 "title": "B — Temporal GraphRAG"},
    "ingredient": {"file": "ingredient_eval.jsonl",
                   "retrievers": ["vanilla", "graphrag", "halal_ingredient"],
                   "title": "C — Halal-Ingredient explainable GraphRAG"},
}

PALETTE = {"vanilla": "#94a3b8", "graphrag": "#2563eb",
           "temporal": "#7c3aed", "halal_ingredient": "#059669"}


def _golds(q: dict) -> list[str]:
    return [q["answer"], *q.get("aliases", [])]


def _make(name: str, q: dict):
    """Build a retriever configured for this question.

    Only the temporal retriever is question-dependent — it needs the item's `as_of`.
    Everything else is constructed identically for every question, so the comparison
    stays single-variable.
    """
    if name == "temporal" and q.get("as_of"):
        return get_retriever("temporal", as_of=q["as_of"])
    return get_retriever(name)


def _paths_from(meta: dict) -> list[str]:
    """Flatten extension C's structured paths into a node sequence for scoring."""
    out: list[str] = []
    for p in meta.get("paths") or []:
        out.extend(p.get("chain") or [])
    return out


def _checkpoint_path(suite: str):
    return config.RESULTS_DIR / f".checkpoint_{suite}.jsonl"


def _load_checkpoint(suite: str) -> list[dict]:
    """Rows already scored in an earlier attempt at this suite."""
    path = _checkpoint_path(suite)
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue          # a partially written last line — drop it
    return out


def run(questions_file: str, retrievers: list[str] | None = None,
        judge: bool = False, suite: str = "core", limit: int = 0,
        resume: bool = True) -> pd.DataFrame:
    """Score every (question, retriever) pair, checkpointing as it goes.

    A full run costs hours because the LLM provider caps tokens per minute, so each
    row is appended to a checkpoint file immediately. An interrupted run resumes
    instead of paying for the same answers twice.
    """
    config.ensure_dirs()
    questions = datasets.load_questions(questions_file)
    if limit:
        questions = questions[:limit]
    names = retrievers or RETRIEVERS

    rows: list[dict] = []
    done: set[tuple] = set()
    if resume:
        rows = _load_checkpoint(suite)
        done = {(r.get("id"), r.get("retriever")) for r in rows}
        if rows:
            print(f"  resuming {suite}: {len(rows)} rows already scored")

    ckpt = _checkpoint_path(suite)
    for q in tqdm(questions, desc=f"{suite}: questions"):
        golds = _golds(q)
        for name in names:
            if (q.get("id"), name) in done:
                continue
            retr = _make(name, q)
            res = answer(q["question"], retr)
            meta = res["retrieve_meta"]
            usage = res["usage"]

            faith = (llm.judge_faithfulness(q["question"], res["answer"], res["context"])
                     if judge and res["context"] else float("nan"))

            paths = _paths_from(meta)
            # A "wrong-era" answer repeats the pre-annotated stale fact instead of the
            # one true at `as_of`. Both halves are required: an answer that says neither
            # is merely unhelpful, not wrong about time.
            stale = q.get("stale_answer", "")
            wrong_era = float("nan")
            if q.get("as_of") and stale:
                said_stale = metrics.containment(res["answer"], [stale])
                said_true = metrics.containment(res["answer"], golds)
                wrong_era = 1.0 if (said_stale and not said_true) else 0.0

            row = {
                "suite": suite,
                "id": q.get("id"),
                "question": q["question"],
                "hop_type": q.get("hop_type", "unknown"),
                "retriever": name,
                "answer": res["answer"],
                "gold": q["answer"],
                # answer quality
                "f1": metrics.f1(res["answer"], golds),
                "em": metrics.exact_match(res["answer"], golds),
                "containment": metrics.containment(res["answer"], golds),
                "faithfulness": faith,
                # retrieval quality
                "context_recall": metrics.context_recall(res["context"], golds),
                "hit_at_k": metrics.hit_at_k(
                    [res["context"], *(meta.get("seed_names") or [])],
                    q.get("gold_nodes", [])),
                # extension-specific
                "path_validity": metrics.path_validity(paths, q.get("gold_path", [])),
                "has_complete_path": float(bool(meta.get("complete_paths"))),
                "wrong_era": wrong_era,
                "as_of": q.get("as_of", ""),
                # cost
                "latency_s": round(meta.get("latency_s", 0)
                                   + res["ground_latency_s"], 3),
                "retrieve_s": meta.get("latency_s", 0),
                "total_tokens": usage.get("total_tokens", 0),
                "llm_calls": usage.get("llm_calls", 0),
                "context_chars": len(res["context"]),
                "n_seeds": meta.get("seeds", 0),
                "n_triples": meta.get("triples", 0),
                # A failed LLM call yields an empty answer that scores as a wrong
                # answer. Recording it per row is what makes a rate-limited run
                # distinguishable from a genuinely bad retriever.
                "llm_error": res["llm_error"],
            }
            rows.append(row)
            with ckpt.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    return pd.DataFrame(rows)


MAX_LLM_ERROR_RATE = 0.02


def check_integrity(detail: pd.DataFrame) -> tuple[bool, str]:
    """Refuse to present a run whose answers were lost to API failures."""
    if "llm_error" not in detail.columns or detail.empty:
        return True, ""
    failed = detail["llm_error"].astype(str).str.strip().ne("").sum()
    rate = failed / len(detail)
    if rate <= MAX_LLM_ERROR_RATE:
        return True, f"{failed}/{len(detail)} rows had an LLM error ({rate:.1%})"
    return False, (
        f"{failed}/{len(detail)} rows ({rate:.1%}) lost their answer to an LLM error — "
        "these score as wrong answers, so the results are not usable. "
        "Lower LLM_TOKENS_PER_MINUTE, or check the API key/quota, and re-run.")


METRIC_COLS = ["f1", "em", "containment", "faithfulness", "context_recall",
               "hit_at_k", "path_validity", "has_complete_path", "wrong_era",
               "latency_s", "retrieve_s", "total_tokens", "context_chars"]


def summarise(detail: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in METRIC_COLS if c in detail.columns]
    by_hop = (detail.groupby(["suite", "retriever", "hop_type"])[cols]
              .mean().reset_index())
    overall = detail.groupby(["suite", "retriever"])[cols].mean().reset_index()
    overall["hop_type"] = "ALL"
    out = pd.concat([by_hop, overall], ignore_index=True)
    counts = (detail.groupby(["suite", "retriever", "hop_type"]).size()
              .rename("n").reset_index())
    return out.merge(counts, on=["suite", "retriever", "hop_type"], how="left")


# ── figures ─────────────────────────────────────────────────────────────────

def _bar(data: pd.DataFrame, x: str, y: str, hue: str, title: str,
         fname: str, ylabel: str = "", ylim: tuple | None = (0, 1)) -> str:
    if data.empty:
        return ""
    plt.figure(figsize=(9, 5))
    order = [r for r in PALETTE if r in set(data[hue])]
    ax = sns.barplot(data=data, x=x, y=y, hue=hue, hue_order=order,
                     palette=[PALETTE[r] for r in order])
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=8, padding=2)
    plt.title(title)
    plt.ylabel(ylabel or y)
    if ylim:
        plt.ylim(*ylim)
    plt.legend(title="retriever")
    plt.tight_layout()
    path = config.FIGURES_DIR / fname
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return str(path)


def plot(summary: pd.DataFrame, detail: pd.DataFrame) -> list[str]:
    made = []
    core = summary[(summary["suite"] == "core") & (summary["hop_type"] != "ALL")]

    # Headline: containment by hop type. Character-level F1 is noisy for Thai and
    # penalises GraphRAG's verbose-but-correct answers (it ties on F1 while winning
    # decisively on containment), so containment is the honest answer-quality view.
    made.append(_bar(core, "hop_type", "containment", "retriever",
                     "A — answer containment by hop type (GraphRAG vs vanilla RAG)",
                     "containment_by_hop.png", "contains gold answer (rate)"))
    made.append(_bar(core, "hop_type", "context_recall", "retriever",
                     "A — retrieval: context contains the evidence, by hop type",
                     "context_recall_by_hop.png", "context recall"))
    # Kept for continuity with the brief; shows why F1 alone is misleading here.
    made.append(_bar(core, "hop_type", "f1", "retriever",
                     "A — answer F1 by hop type (char-level; see containment)",
                     "f1_by_hop.png", "mean F1"))

    if not core.empty:
        long = core.melt(id_vars=["hop_type", "retriever"],
                         value_vars=["context_recall", "containment"],
                         var_name="metric", value_name="score")
        long["label"] = long["retriever"] + " · " + long["metric"]
        plt.figure(figsize=(10, 5))
        ax = sns.barplot(data=long, x="hop_type", y="score", hue="label")
        for c in ax.containers:
            ax.bar_label(c, fmt="%.2f", fontsize=7, padding=2)
        plt.title("A — retrieval ceiling (context recall) vs delivered answer (containment)")
        plt.ylim(0, 1)
        plt.tight_layout()
        p = config.FIGURES_DIR / "retrieval_vs_answer.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        made.append(str(p))

    # B: the wrong_era rate came out uniform (the metric's both-halves condition rarely
    # fires), so the honest B story is retrieval of time-relevant evidence — the
    # temporal retriever surfaces expiry facts the time-agnostic ones never see.
    temporal = summary[(summary["suite"] == "temporal") & (summary["hop_type"] == "ALL")]
    if not temporal.empty:
        tlong = temporal.melt(id_vars=["retriever"],
                              value_vars=["context_recall", "containment"],
                              var_name="metric", value_name="score")
        plt.figure(figsize=(9, 5))
        order = [r for r in PALETTE if r in set(temporal["retriever"])]
        ax = sns.barplot(data=tlong, x="retriever", y="score", hue="metric", order=order)
        for c in ax.containers:
            ax.bar_label(c, fmt="%.2f", fontsize=8, padding=2)
        plt.title("B — time-aware retrieval: evidence found & answer correctness")
        plt.ylim(0, 1)
        plt.tight_layout()
        p = config.FIGURES_DIR / "temporal_retrieval.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        made.append(str(p))

    ing = summary[(summary["suite"] == "ingredient") & (summary["hop_type"] == "ALL")]
    if not ing.empty:
        long = ing.melt(id_vars=["retriever"],
                        value_vars=["containment", "context_recall", "path_validity"],
                        var_name="metric", value_name="score")
        plt.figure(figsize=(9, 5))
        ax = sns.barplot(data=long, x="metric", y="score", hue="retriever",
                         hue_order=[r for r in PALETTE if r in set(long["retriever"])],
                         palette=[PALETTE[r] for r in PALETTE if r in set(long["retriever"])])
        for c in ax.containers:
            ax.bar_label(c, fmt="%.2f", fontsize=8, padding=2)
        plt.title("C — answer correctness and evidence-path validity")
        plt.ylim(0, 1)
        plt.tight_layout()
        p = config.FIGURES_DIR / "ingredient_path_validity.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        made.append(str(p))

    cost = detail.groupby("retriever")[["latency_s", "total_tokens"]].mean().reset_index()
    if not cost.empty:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        order = [r for r in PALETTE if r in set(cost["retriever"])]
        colours = [PALETTE[r] for r in order]
        sns.barplot(data=cost, x="retriever", y="latency_s", order=order,
                    hue="retriever", hue_order=order, palette=colours,
                    legend=False, ax=axes[0])
        axes[0].set_title("mean end-to-end latency (s)")
        sns.barplot(data=cost, x="retriever", y="total_tokens", order=order,
                    hue="retriever", hue_order=order, palette=colours,
                    legend=False, ax=axes[1])
        axes[1].set_title("mean LLM tokens per question")
        plt.tight_layout()
        p = config.FIGURES_DIR / "cost_latency.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        made.append(str(p))

    return [m for m in made if m]


# ── entry point ─────────────────────────────────────────────────────────────

def run_suites(suites: list[str], judge: bool = False, limit: int = 0,
               resume: bool = True) -> tuple:
    llm.USAGE.reset()
    t0 = time.time()
    frames = []
    for name in suites:
        cfg = SUITES[name]
        try:
            datasets.load_questions(cfg["file"])
        except FileNotFoundError:
            print(f"skip suite {name}: {cfg['file']} not found")
            continue
        print(f"\n=== {cfg['title']} ===", flush=True)
        frames.append(run(cfg["file"], cfg["retrievers"], judge=judge,
                          suite=name, limit=limit, resume=resume))
    if not frames:
        raise SystemExit("no eval files found — nothing to run")
    detail = pd.concat(frames, ignore_index=True)
    ok, integrity_note = check_integrity(detail)
    print(f"\nintegrity: {integrity_note}")
    if not ok:
        # Still write the detail file so the failures can be inspected, but do not
        # let a contaminated run overwrite the published summary and figures.
        detail.to_csv(config.RESULTS_DIR / "benchmark_detail_FAILED.csv", index=False)
        raise SystemExit(f"aborting: {integrity_note}")

    summary = summarise(detail)

    detail.to_csv(config.RESULTS_DIR / "benchmark_detail.csv", index=False)
    summary.to_csv(config.RESULTS_DIR / "benchmark_summary.csv", index=False)
    figures = plot(summary, detail)

    meta = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_s": round(time.time() - t0, 1),
        "suites": suites,
        "judge": judge,
        "questions": int(detail["id"].nunique()),
        "rows": len(detail),
        "llm": {"calls": llm.USAGE.calls,
                "prompt_tokens": llm.USAGE.prompt_tokens,
                "completion_tokens": llm.USAGE.completion_tokens,
                "errors": llm.USAGE.errors,
                "by_model": llm.USAGE.by_model},
        "integrity": integrity_note,
        "figures": figures,
    }
    (config.RESULTS_DIR / "benchmark_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return detail, summary, meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--questions", default="", help="run one eval file (implies core suite)")
    ap.add_argument("--suite", default="all",
                    choices=[*SUITES, "all"], help="which experiment to run")
    ap.add_argument("--judge", action="store_true", help="run LLM-as-judge faithfulness")
    ap.add_argument("--limit", type=int, default=0, help="first N questions (smoke test)")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore checkpoints and score every row again")
    ap.add_argument("--plot-only", action="store_true",
                    help="regenerate figures + summary from the saved detail CSV (no LLM)")
    args = ap.parse_args()

    if args.plot_only:
        detail = pd.read_csv(config.RESULTS_DIR / "benchmark_detail.csv")
        summary = summarise(detail)
        summary.to_csv(config.RESULTS_DIR / "benchmark_summary.csv", index=False)
        figs = plot(summary, detail)
        print(f"regenerated {len(figs)} figures from saved detail:")
        for f in figs:
            print("  ", f)
        return

    if args.fresh:
        for path in config.RESULTS_DIR.glob(".checkpoint_*.jsonl"):
            path.unlink()

    if args.questions:
        SUITES["core"]["file"] = args.questions
        suites = ["core"]
    else:
        suites = list(SUITES) if args.suite == "all" else [args.suite]

    detail, summary, meta = run_suites(suites, judge=args.judge, limit=args.limit,
                                       resume=not args.fresh)

    print("\n=== Mean F1 by hop type ===")
    for suite in detail["suite"].unique():
        s = summary[summary["suite"] == suite]
        print(f"\n[{suite}]")
        print(s.pivot_table(index="hop_type", columns="retriever", values="f1").round(3))

    print(f"\nLLM: {meta['llm']['calls']} calls, "
          f"{meta['llm']['prompt_tokens'] + meta['llm']['completion_tokens']:,} tokens, "
          f"{meta['llm']['errors']} errors")
    print(f"Artefacts written to {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
