"""FastAPI backend + static SPA for the demo UI.

Every endpoint talks to the real Neo4j / Qdrant / TEI / LLM stack — there is no mock
data anywhere in this app. Swagger lives at /docs.

Pages served by the SPA (all backed by the endpoints below):
    Overview · Ask/Compare · KG Explorer · Benchmark · Eval Set · Temporal · Ingredient
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from thaigraphrag import config
from thaigraphrag.benchmark import datasets
from thaigraphrag.core import embeddings, entity_linking, llm, neo4j_client
from thaigraphrag.core import qdrant_client as qc
from thaigraphrag.kg.schema import LABEL_TH, REL_TH
from thaigraphrag.pipeline.answer import answer as answer_pipeline
from thaigraphrag.retrievers import available_retrievers, get_retriever

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="Thai GraphRAG + Benchmark",
    description=(
        "GraphRAG vs vanilla RAG on a Thai knowledge graph, with Temporal (B) and "
        "Halal-Ingredient (C) extensions. Every endpoint runs against the live stack."
    ),
    version="1.0.0",
)


# ── models ──────────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str = Field(..., examples=["มัสยิดกลางปัตตานีตั้งอยู่จังหวัดใด"])
    retrievers: list[str] = Field(default_factory=lambda: ["vanilla", "graphrag"])


class TemporalAskRequest(BaseModel):
    question: str = Field(..., examples=["ณ ปี 2570 สินค้านี้ยังได้รับรองฮาลาลหรือไม่"])
    as_of: int | None = Field(None, examples=[2570])
    compare_baseline: bool = True


class ExplainRequest(BaseModel):
    query: str = Field(..., examples=["เจลาตินฮาลาลหรือไม่"])
    ground: bool = True


class BenchmarkRequest(BaseModel):
    suite: str = "core"
    judge: bool = False
    limit: int = 0


class EvalSaveRequest(BaseModel):
    items: list[dict]


# ── health / stats ──────────────────────────────────────────────────────────

def _service_status() -> dict:
    status = {}
    try:
        neo4j_client.run("RETURN 1 AS ok")
        status["neo4j"] = "up"
    except Exception as e:
        status["neo4j"] = f"down: {type(e).__name__}"
    try:
        qc.get_qdrant().get_collections()
        status["qdrant"] = "up"
    except Exception as e:
        status["qdrant"] = f"down: {type(e).__name__}"
    status["embeddings"] = "up" if embeddings.health() else "down"
    status["llm"] = "configured" if llm.available() else "no API key"
    return status


@app.get("/health", tags=["system"])
def health() -> dict:
    """Liveness + dependency status. Used by the compose healthcheck."""
    svc = _service_status()
    ok = svc["neo4j"] == "up" and svc["qdrant"] == "up"
    return JSONResponse(status_code=200 if ok else 503,
                        content={"ok": ok, "services": svc})


@app.get("/api/stats", tags=["system"])
def stats() -> dict:
    """Live counts for the Overview page — nodes, relations, vectors, eval sets."""
    from thaigraphrag.kg.build_kg import graph_counts

    try:
        counts = graph_counts()
    except Exception:
        counts = {"nodes": {}, "relationships": {}, "total_nodes": 0, "total_rels": 0}
    try:
        eval_files = datasets.list_files()
    except Exception:
        eval_files = []
    return {
        "services": _service_status(),
        "graph": counts,
        "vectors": qc.count(),
        "eval_files": eval_files,
        "questions_total": sum(f["count"] for f in eval_files),
        "retrievers": available_retrievers(),
        "label_th": LABEL_TH,
        "settings": {
            "top_k": config.get_settings().top_k,
            "graph_hops": config.get_settings().graph_hops,
            "llm_model": config.get_settings().llm_model,
            "embedding_dim": config.get_settings().embedding_dim,
            "max_product_rows": config.get_settings().max_product_rows,
        },
    }


# ── ask / compare ───────────────────────────────────────────────────────────

@app.post("/api/ask", tags=["ask"])
def ask(req: AskRequest) -> dict:
    """Answer one Thai question with several retrievers for side-by-side comparison."""
    if not req.question.strip():
        raise HTTPException(400, "question is empty")
    results = []
    for name in req.retrievers:
        try:
            retr = get_retriever(name)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        res = answer_pipeline(req.question, retr)
        results.append({
            "retriever": name,
            "answer": res["answer"],
            "context": res["context"],
            "provenance": res["provenance"][:20],
            "meta": res["retrieve_meta"],
            "usage": res["usage"],
            "ground_latency_s": res["ground_latency_s"],
            "error": res["llm_error"],
        })
    return {"question": req.question, "results": results}


@app.get("/api/link", tags=["ask"])
def link(q: str = Query(..., description="Thai query to entity-link"),
         limit: int = 5) -> dict:
    """Entity-linking trace — which KG nodes a question resolves to, and why."""
    return {"query": q, "seeds": entity_linking.link_entities(q, limit=limit)}


# ── KG explorer ─────────────────────────────────────────────────────────────

@app.get("/api/kg/search", tags=["kg"])
def kg_search(q: str = Query("", description="name substring"),
              label: str = "", limit: int = 25) -> dict:
    """Search nodes by name, optionally filtered by label."""
    where = ["n.name IS NOT NULL"]
    params: dict = {"limit": limit}
    if q:
        where.append("(toLower(n.name) CONTAINS toLower($q) "
                     "OR toLower(coalesce(n.name_en,'')) CONTAINS toLower($q))")
        params["q"] = q
    if label:
        where.append("$label IN labels(n)")
        params["label"] = label
    rows = neo4j_client.run(
        f"""
        MATCH (n) WHERE {' AND '.join(where)}
        RETURN elementId(n) AS id, labels(n) AS labels, n.name AS name,
               coalesce(n.name_en,'') AS name_en, coalesce(n.text,'') AS text,
               coalesce(n.province,'') AS province,
               size([(n)--() | 1]) AS degree
        ORDER BY degree DESC LIMIT $limit
        """, **params)
    return {"query": q, "label": label, "count": len(rows), "nodes": rows}


@app.get("/api/kg/labels", tags=["kg"])
def kg_labels() -> dict:
    rows = neo4j_client.run(
        "MATCH (n) UNWIND labels(n) AS label RETURN label, count(*) AS count "
        "ORDER BY count DESC")
    return {"labels": rows, "label_th": LABEL_TH}


@app.get("/api/kg/node/{node_id:path}", tags=["kg"])
def kg_node(node_id: str, limit: int = 30) -> dict:
    """One node with its properties and its immediate neighbourhood (graph view)."""
    rows = neo4j_client.run(
        """
        MATCH (n) WHERE elementId(n) = $id
        RETURN elementId(n) AS id, labels(n) AS labels, properties(n) AS props
        """, id=node_id)
    if not rows:
        raise HTTPException(404, "node not found")
    neighbours = neo4j_client.run(
        """
        MATCH (n) WHERE elementId(n) = $id
        MATCH (n)-[r]-(m)
        RETURN elementId(m) AS id, labels(m) AS labels,
               coalesce(m.name, labels(m)[0]) AS name,
               type(r) AS rel, startNode(r) = n AS outgoing,
               properties(r) AS rel_props
        LIMIT $limit
        """, id=node_id, limit=limit)
    return {"node": rows[0], "neighbours": neighbours, "rel_th": REL_TH}


@app.get("/api/kg/graph", tags=["kg"])
def kg_graph(q: str = Query(..., description="Thai query"), hops: int = 1,
             limit: int = 40) -> dict:
    """A small subgraph around a query's seed nodes, shaped for the UI's graph view."""
    seeds = entity_linking.link_entities(q, limit=3)
    if not seeds:
        return {"query": q, "nodes": [], "edges": [], "seeds": []}
    ids = [s["node_id"] for s in seeds]
    rows = neo4j_client.run(
        f"""
        MATCH (s) WHERE elementId(s) IN $ids
        MATCH (s)-[r*1..{max(1, min(hops, 2))}]-(m)
        WITH s, r, m LIMIT $limit
        UNWIND r AS rel
        RETURN DISTINCT elementId(startNode(rel)) AS source_id,
               coalesce(startNode(rel).name, labels(startNode(rel))[0]) AS source,
               labels(startNode(rel))[0] AS source_label,
               elementId(endNode(rel)) AS target_id,
               coalesce(endNode(rel).name, labels(endNode(rel))[0]) AS target,
               labels(endNode(rel))[0] AS target_label,
               type(rel) AS rel
        """, ids=ids, limit=limit)
    nodes: dict[str, dict] = {}
    edges = []
    for r in rows:
        for side in ("source", "target"):
            nodes.setdefault(r[f"{side}_id"], {
                "id": r[f"{side}_id"], "name": r[side], "label": r[f"{side}_label"],
                "is_seed": r[f"{side}_id"] in ids})
        edges.append({"source": r["source_id"], "target": r["target_id"],
                      "rel": r["rel"], "rel_th": REL_TH.get(r["rel"], r["rel"])})
    return {"query": q, "seeds": seeds, "nodes": list(nodes.values()), "edges": edges}


# ── benchmark ───────────────────────────────────────────────────────────────

_bench_state: dict = {"running": False, "started_at": None, "finished_at": None,
                      "suite": None, "error": None, "log": []}
_bench_lock = threading.Lock()


def _run_benchmark(suite: str, judge: bool, limit: int) -> None:
    from thaigraphrag.benchmark.run_benchmark import SUITES, run_suites
    try:
        suites = list(SUITES) if suite == "all" else [suite]
        run_suites(suites, judge=judge, limit=limit)
        _bench_state["error"] = None
    except Exception as e:                              # surfaced in the UI
        _bench_state["error"] = f"{type(e).__name__}: {e}"
    finally:
        _bench_state["running"] = False
        _bench_state["finished_at"] = time.time()


@app.post("/api/benchmark/run", tags=["benchmark"])
def benchmark_run(req: BenchmarkRequest, background: BackgroundTasks) -> dict:
    """Kick off a benchmark run in the background; poll /api/benchmark/status."""
    with _bench_lock:
        if _bench_state["running"]:
            raise HTTPException(409, "a benchmark run is already in progress")
        _bench_state.update({"running": True, "started_at": time.time(),
                             "finished_at": None, "suite": req.suite, "error": None})
    background.add_task(_run_benchmark, req.suite, req.judge, req.limit)
    return {"started": True, "suite": req.suite, "judge": req.judge}


@app.get("/api/benchmark/status", tags=["benchmark"])
def benchmark_status() -> dict:
    return {**_bench_state,
            "elapsed_s": round(time.time() - _bench_state["started_at"], 1)
            if _bench_state["running"] and _bench_state["started_at"] else None}


@app.get("/api/benchmark/results", tags=["benchmark"])
def benchmark_results() -> dict:
    """Latest summary + per-question detail + generated figure URLs."""
    summary_path = config.RESULTS_DIR / "benchmark_summary.csv"
    detail_path = config.RESULTS_DIR / "benchmark_detail.csv"
    meta_path = config.RESULTS_DIR / "benchmark_meta.json"
    if not summary_path.exists():
        return {"available": False,
                "message": "no results yet — run the benchmark first"}
    summary = pd.read_csv(summary_path)
    detail = pd.read_csv(detail_path) if detail_path.exists() else pd.DataFrame()
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    figures = [f"/figures/{p.name}" for p in sorted(config.FIGURES_DIR.glob("*.png"))]
    return {
        "available": True,
        "summary": json.loads(summary.to_json(orient="records")),
        "detail": json.loads(detail.head(600).to_json(orient="records")),
        "meta": meta,
        "figures": figures,
    }


# ── eval sets ───────────────────────────────────────────────────────────────

@app.get("/api/eval/files", tags=["eval"])
def eval_files() -> dict:
    return {"files": datasets.list_files()}


@app.get("/api/eval/{name}", tags=["eval"])
def eval_get(name: str) -> dict:
    try:
        items = datasets.load_questions(name)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e
    return {"name": name, "count": len(items), "items": items,
            "problems": datasets.validate(items)}


@app.put("/api/eval/{name}", tags=["eval"])
def eval_put(name: str, req: EvalSaveRequest) -> dict:
    problems = datasets.validate(req.items)
    if problems:
        raise HTTPException(400, {"problems": problems})
    n = datasets.save_questions(name, req.items)
    return {"saved": n, "name": name}


# ── extension B: temporal ───────────────────────────────────────────────────

@app.post("/api/temporal/ask", tags=["temporal"])
def temporal_ask(req: TemporalAskRequest) -> dict:
    """Answer a question as of a Buddhist-Era year, next to the time-agnostic answer."""
    results = []
    retr = get_retriever("temporal", as_of=req.as_of)
    results.append({"retriever": "temporal", **_summarise_answer(
        answer_pipeline(req.question, retr))})
    if req.compare_baseline:
        results.append({"retriever": "graphrag", **_summarise_answer(
            answer_pipeline(req.question, get_retriever("graphrag")))})
    return {"question": req.question, "as_of": req.as_of, "results": results}


@app.get("/api/temporal/coverage", tags=["temporal"])
def temporal_coverage() -> dict:
    """Which relations carry validity, and how certifications spread across years."""
    from thaigraphrag.extensions.temporal.temporal_kg import coverage
    try:
        return coverage()
    except Exception as e:
        raise HTTPException(503, f"temporal layer not built: {e}") from e


@app.get("/api/temporal/timeline", tags=["temporal"])
def temporal_timeline(as_of: int | None = None) -> dict:
    """Regulations in force, optionally restricted to a year."""
    rows = neo4j_client.run(
        """
        MATCH (r:Regulation)
        WHERE $as_of IS NULL OR
              ((r.valid_from IS NULL OR r.valid_from <= $as_of) AND
               (r.valid_to   IS NULL OR r.valid_to   >= $as_of))
        OPTIONAL MATCH (r)-[:ISSUED_BY]->(o:Organisation)
        RETURN r.reg_id AS id, r.name AS name, r.kind AS kind,
               r.valid_from AS valid_from, r.valid_to AS valid_to,
               r.topic AS topic, o.name AS issuer
        ORDER BY r.valid_from
        """, as_of=as_of)
    return {"as_of": as_of, "regulations": rows}


# ── extension C: halal ingredient ───────────────────────────────────────────

@app.post("/api/ingredient/explain", tags=["ingredient"])
def ingredient_explain(req: ExplainRequest) -> dict:
    """Ruling paths for an ingredient or product, with the grounded Thai answer."""
    retr = get_retriever("halal_ingredient")
    if req.ground:
        res = answer_pipeline(req.query, retr)
        meta = res["retrieve_meta"]
        return {"query": req.query, "answer": res["answer"], "context": res["context"],
                "paths": meta.get("paths", []), "products": meta.get("products", []),
                "meta": meta, "usage": res["usage"]}
    ctx = retr.retrieve(req.query)
    return {"query": req.query, "answer": None, "context": ctx.text,
            "paths": ctx.meta.get("paths", []), "products": ctx.meta.get("products", []),
            "meta": ctx.meta}


@app.get("/api/ingredient/list", tags=["ingredient"])
def ingredient_list() -> dict:
    """All ingredients with their per-source rulings — powers the C page's browser."""
    rows = neo4j_client.run(
        """
        MATCH (i:Ingredient)-[h:HAS_RULING]->(r:Ruling)
        MATCH (i)-[:DERIVED_FROM]->(s:Source {name: h.via_source})
        OPTIONAL MATCH (p:Product)-[:CONTAINS]->(i)
        RETURN i.name AS ingredient, i.name_en AS ingredient_en,
               coalesce(i.e_number,'') AS e_number,
               s.name AS source, s.source_type AS source_type,
               r.status AS ruling, h.basis AS basis,
               count(DISTINCT p) AS products
        ORDER BY ingredient, ruling
        """)
    return {"count": len(rows), "rulings": rows}


@app.get("/api/ingredient/summary", tags=["ingredient"])
def ingredient_summary() -> dict:
    from thaigraphrag.extensions.halal_ingredient.ingredient_kg import summary
    try:
        return summary()
    except Exception as e:
        raise HTTPException(503, f"ingredient layer not built: {e}") from e


def _summarise_answer(res: dict) -> dict:
    return {"answer": res["answer"], "context": res["context"],
            "meta": res["retrieve_meta"], "usage": res["usage"]}


# ── static assets ───────────────────────────────────────────────────────────

config.ensure_dirs()
app.mount("/figures", StaticFiles(directory=str(config.FIGURES_DIR)), name="figures")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
