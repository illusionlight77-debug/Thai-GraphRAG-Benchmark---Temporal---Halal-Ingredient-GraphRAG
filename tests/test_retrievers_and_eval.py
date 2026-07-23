"""Retriever registry, extension separability, and eval-set structure.

The registry tests encode the CLAUDE.md §2 invariant in executable form: core A must
construct without either extension, and every retriever must expose the same shape.
"""
import json

import pytest

from thaigraphrag.benchmark import datasets
from thaigraphrag.retrievers import (
    CORE_RETRIEVERS, RetrievedContext, Retriever, available_retrievers, get_retriever,
)


# ── registry ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", ["vanilla", "graphrag", "temporal", "halal_ingredient"])
def test_every_retriever_constructs_and_names_itself(name):
    r = get_retriever(name)
    assert isinstance(r, Retriever)
    assert r.name == name
    # `retrieve` is the only contract the pipeline relies on.
    assert callable(r.retrieve)


def test_core_retrievers_need_no_extensions():
    for name in CORE_RETRIEVERS:
        assert get_retriever(name).name == name


def test_unknown_retriever_is_rejected_with_a_useful_message():
    with pytest.raises(ValueError) as e:
        get_retriever("nope")
    assert "available" in str(e.value)


def test_available_retrievers_includes_core():
    assert set(CORE_RETRIEVERS) <= set(available_retrievers())


def test_temporal_accepts_as_of_and_parses_it_from_text():
    from thaigraphrag.extensions.temporal.temporal_retriever import parse_as_of
    assert get_retriever("temporal", as_of=2570).as_of == 2570
    assert parse_as_of("ณ ปี พ.ศ. 2571 ยังรับรองอยู่ไหม") == 2571
    assert parse_as_of("ไม่มีปีในคำถามนี้") is None


def test_retrieved_context_defaults():
    ctx = RetrievedContext(text="x")
    assert ctx.provenance == [] and ctx.meta == {}


# ── eval sets ───────────────────────────────────────────────────────────────

def test_every_shipped_eval_file_is_structurally_valid():
    files = datasets.list_files()
    assert files, "no eval files found"
    for f in files:
        items = datasets.load_questions(f["name"])
        problems = datasets.validate(items)
        assert not problems, f"{f['name']}: {problems[:5]}"


def test_eval_items_round_trip_as_jsonl():
    items = datasets.load_questions(datasets.list_files()[0]["name"])
    for it in items:
        assert json.loads(json.dumps(it, ensure_ascii=False)) == it


def test_validate_catches_missing_fields_and_dupes():
    bad = [{"id": "a", "question": "q", "hop_type": "single"},          # no answer
           {"id": "a", "question": "q2", "answer": "x", "hop_type": "nope"}]
    problems = datasets.validate(bad)
    assert any("answer" in p for p in problems)
    assert any("hop_type" in p for p in problems)
    assert any("duplicate" in p for p in problems)


def test_hop_types_are_balanced_in_the_core_set():
    """The headline result is F1 *by hop type*, so no bucket may be near-empty."""
    names = [f["name"] for f in datasets.list_files()]
    core = "thai_eval.jsonl" if "thai_eval.jsonl" in names else "thai_eval.example.jsonl"
    items = datasets.load_questions(core)
    counts = {h: sum(1 for i in items if i.get("hop_type") == h)
              for h in datasets.HOP_TYPES}
    assert all(counts.values()), f"empty hop bucket: {counts}"
    assert min(counts.values()) / max(counts.values()) >= 0.3, counts
