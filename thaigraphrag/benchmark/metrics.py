"""Answer-quality and retrieval metrics.

Thai text is whitespace-sparse, so answer scoring falls back to character-level
comparison, which is robust for short factual answers. Faithfulness (LLM-judge)
lives in core/llm.py and is called by the runner.

Retrieval metrics are separated from answer metrics on purpose. A low F1 can mean
either "the retriever never found the evidence" or "the LLM had the evidence and
still answered badly". `context_recall` and `hit_at_k` measure only the first,
which is the variable the experiment actually manipulates.
"""
from __future__ import annotations

import re

_PUNCT = re.compile(r"[\s\.,!?;:()\[\]\"'`‘’“”\-–—/\\|]+")


def _normalize(s: str) -> str:
    return _PUNCT.sub("", (s or "").lower().strip())


def _char_tokens(s: str) -> list[str]:
    return list(_normalize(s))


def exact_match(pred: str, golds: list[str]) -> float:
    p = _normalize(pred)
    return 1.0 if any(p == _normalize(g) for g in golds) else 0.0


def containment(pred: str, golds: list[str]) -> float:
    """1.0 if any gold answer appears inside the prediction (or vice-versa)."""
    p = _normalize(pred)
    if not p:
        return 0.0
    for g in golds:
        gg = _normalize(g)
        if gg and (gg in p or p in gg):
            return 1.0
    return 0.0


def f1(pred: str, golds: list[str]) -> float:
    """Best token-level F1 against any accepted gold answer (char tokens)."""
    best = 0.0
    ptoks = _char_tokens(pred)
    for g in golds:
        gtoks = _char_tokens(g)
        if not ptoks or not gtoks:
            continue
        common = 0
        gcopy = list(gtoks)
        for t in ptoks:
            if t in gcopy:
                gcopy.remove(t)
                common += 1
        if common == 0:
            continue
        prec = common / len(ptoks)
        rec = common / len(gtoks)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


# ── retrieval metrics ───────────────────────────────────────────────────────

def context_recall(context: str, golds: list[str]) -> float:
    """1.0 if the retrieved context contains a gold answer string.

    The ceiling on answer quality: the grounding prompt forbids outside knowledge,
    so a retriever that scores 0 here cannot be answered correctly by any LLM.
    Needs no extra annotation, which is why it is reported for every question.
    """
    c = _normalize(context)
    if not c:
        return 0.0
    return 1.0 if any(_normalize(g) and _normalize(g) in c for g in golds) else 0.0


def hit_at_k(retrieved: list[str], gold_nodes: list[str]) -> float:
    """Fraction of gold provenance entities present in the retrieved node set.

    `gold_nodes` is the optional `gold_nodes` field of an eval item — the entity
    names a correct answer must have passed through. Returns NaN when an item has
    no provenance annotation so it is excluded from the mean rather than counted 0.
    """
    if not gold_nodes:
        return float("nan")
    hay = _normalize(" ".join(retrieved))
    if not hay:
        return 0.0
    hits = sum(1 for g in gold_nodes if _normalize(g) and _normalize(g) in hay)
    return hits / len(gold_nodes)


def path_validity(path: list[str], gold_path: list[str]) -> float:
    """Extension C: does the returned evidence path cover the gold reasoning chain?

    Order-sensitive — a path that visits the right nodes in the wrong order is not
    a valid explanation. Returns the fraction of gold hops matched in sequence.
    """
    if not gold_path:
        return float("nan")
    if not path:
        return 0.0
    norm_path = [_normalize(p) for p in path]
    matched, cursor = 0, 0
    for g in gold_path:
        gg = _normalize(g)
        if not gg:
            continue
        for i in range(cursor, len(norm_path)):
            if gg in norm_path[i] or norm_path[i] in gg:
                matched += 1
                cursor = i + 1
                break
    return matched / len(gold_path)
