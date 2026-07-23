# Extension B — Temporal GraphRAG

**Goal:** answer time-sensitive questions correctly — *"ณ ปี 2571 สินค้านี้ยังได้รับ
การรับรองฮาลาลอยู่หรือไม่"* — by making the graph and the retriever **time-aware**.

## How it works

Relations carry a validity interval (`valid_from` / `valid_to`, Buddhist Era). The
retriever subclasses the core `GraphRAGRetriever` and applies the standard interval
predicate to every expansion, with NULL meaning unbounded:

```cypher
(r.valid_from IS NULL OR r.valid_from <= $as_of) AND
(r.valid_to   IS NULL OR r.valid_to   >= $as_of)
```

Relations with no validity at all (`LOCATED_IN`, `BELONGS_TO`, …) are timeless and
always pass, so turning the extension on never *loses* untimed facts — it only
removes statements that were not true in the requested year.

`as_of` comes from the eval item's `as_of` field, or is parsed out of the question
text (`ณ ปี 2571 …`) when the retriever is used interactively.

The retriever also reports **what expired**, not just what is missing. A
time-agnostic system cannot tell "no such fact" apart from "that lapsed in 2569", and
that confusion is precisely what produces wrong-era answers.

## Two temporal layers, different evidential weight

| layer | source | rows | weight |
|-------|--------|-----:|--------|
| **Certification validity** | `product_processed.csv :: expire_date` — 100% populated, real | 222k | headline result |
| **Regulation timeline** | `data/halal/regulation_timeline.csv` — curated | 5 | qualitative only |

`valid_to` is the real expiry year. **`valid_from` is deliberately NULL**: the
registry records no issue date, and the trailing digits of `halal_code` do not encode
one reliably (measured implied term: −29 to +4 years, so it was rejected rather than
guessed at).

This makes the experiment real: at `as_of = 2570` roughly half the registry has
already expired, so a time-agnostic retriever confidently reports lapsed products as
currently certified.

## Build and run

```bash
python -m scripts.build_kg              # Product nodes must exist first
python -m scripts.build_temporal_kg     # adds CERTIFIED_HALAL + the timeline
python -m scripts.generate_eval --suite temporal
python -m scripts.run_benchmark --suite temporal --judge
```

## Metric

`wrong_era` — the answer repeats the item's pre-annotated `stale_answer` **and**
fails to state the answer true at `as_of`. Both halves are required: an answer that
says neither is merely unhelpful, not wrong about time. Reported as a rate per
retriever, so the headline is the **% reduction** versus time-agnostic GraphRAG.

## Files

| file | role |
|------|------|
| `temporal_kg.py` | `annotate_certifications()`, `build_timeline()`, `coverage()` |
| `temporal_retriever.py` | `TemporalGraphRAGRetriever(as_of=…)`, `parse_as_of()` |

Registered as `"temporal"` in `retrievers.get_retriever`; no core file is modified.

## Limitation

Because `valid_from` is NULL, "was X certified in 2565?" is answered as
"unbounded start". Eval items stay in the year range where `valid_to` discriminates.

See `docs/REFERENCES.md` (TG-RAG, ECT-QA, ATOM) for the related work.
