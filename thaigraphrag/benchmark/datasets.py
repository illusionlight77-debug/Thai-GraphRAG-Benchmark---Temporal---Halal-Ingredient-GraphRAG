"""Evaluation-set loader.

A question record (one JSON object per line, .jsonl):

    {
      "id": "q001",
      "question": "ร้านอาหารฮาลาลในจังหวัดเดียวกับมัสยิดกลางปัตตานีมีอะไรบ้าง",
      "answer": "…",                # gold answer (short text or entity)
      "hop_type": "multi",          # single | multi | relational
      "aliases": ["…"],             # optional accepted answer variants
      "gold_nodes": ["…"],          # optional provenance entities → retrieval hit@k
      "as_of": 2570,                # optional — Temporal (B); Buddhist-Era year
      "stale_answer": "…",          # optional — what a time-agnostic system says (B)
      "gold_path": ["…", "…"]       # optional — reasoning chain (C) → path_validity
    }

This is where the released Thai multi-hop eval set lives. The `*.example.jsonl`
files are the small starter samples that shipped with the seed; the `*_eval.jsonl`
files are the full released sets the benchmark actually runs.
"""
from __future__ import annotations

import json
from pathlib import Path

from thaigraphrag.config import QUESTIONS_DIR

HOP_TYPES = ("single", "multi", "relational")


def resolve(name: str) -> Path:
    """Locate an eval file by bare name, relative path, or absolute path."""
    path = Path(name)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return QUESTIONS_DIR / name


def load_questions(name: str = "thai_eval.jsonl") -> list[dict]:
    path = resolve(name)
    if not path.exists():
        raise FileNotFoundError(f"eval file not found: {name} (looked in {QUESTIONS_DIR})")
    items = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ValueError(f"{path.name} line {lineno}: {e}") from e
    return items


def save_questions(name: str, items: list[dict]) -> int:
    """Overwrite an eval file — backs the UI's Eval Set editor."""
    path = resolve(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(it, ensure_ascii=False) for it in items) + "\n"
    path.write_text(text, encoding="utf-8")
    return len(items)


def list_files() -> list[dict]:
    """Every eval file with its item count and hop-type breakdown."""
    out = []
    for path in sorted(QUESTIONS_DIR.glob("*.jsonl")):
        try:
            items = load_questions(path.name)
        except (ValueError, FileNotFoundError):
            continue
        by_hop: dict[str, int] = {}
        for it in items:
            key = it.get("hop_type", "unknown")
            by_hop[key] = by_hop.get(key, 0) + 1
        out.append({"name": path.name, "count": len(items), "by_hop": by_hop,
                    "is_example": ".example." in path.name})
    return out


def validate(items: list[dict]) -> list[str]:
    """Structural problems in an eval set — surfaced by the UI before saving."""
    problems, seen = [], set()
    for i, it in enumerate(items, 1):
        qid = it.get("id") or f"<line {i}>"
        if not it.get("question"):
            problems.append(f"{qid}: missing 'question'")
        if not it.get("answer"):
            problems.append(f"{qid}: missing 'answer'")
        hop = it.get("hop_type")
        if hop not in HOP_TYPES:
            problems.append(f"{qid}: hop_type {hop!r} not in {HOP_TYPES}")
        if it.get("id") in seen:
            problems.append(f"{qid}: duplicate id")
        seen.add(it.get("id"))
    return problems
