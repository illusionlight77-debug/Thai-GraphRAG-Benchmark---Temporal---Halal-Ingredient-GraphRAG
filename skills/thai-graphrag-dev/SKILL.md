---
name: thai-graphrag-dev
description: >
  Development guide for the Thai GraphRAG + Benchmark research repo (Neo4j + Qdrant +
  local bge-m3/TEI + OpenAI-compatible LLM + FastAPI demo UI). Use whenever working
  inside this repo — building/loading the knowledge graph, adding or tuning retrievers
  (vanilla vs graphrag), regenerating the Thai multi-hop eval set, running or extending
  the benchmark, entity linking, the demo UI/API, Docker boot problems, or the Temporal
  (B) and Halal-Ingredient (C) extensions. Trigger for any "how do I run/add/measure X
  here", any stack trace in this project, and code review of changes here.
---

# Thai GraphRAG — Dev Skill

Read `CLAUDE.md` first (architecture, invariants) and `docs/METHODOLOGY.md`
(the experiment). This skill is the actionable how-to.

## Golden rules

- **Keep the comparison fair.** Every retriever returns `RetrievedContext`; the answer
  pipeline and LLM/prompt are shared. `pipeline/answer.py` must never branch on
  `retriever.name`.
- **GraphRAG must not touch Qdrant.** Entity linking is lexical over Neo4j only. Adding
  a vector fallback there makes GraphRAG a superset of vanilla and destroys the result.
- **Core (A) must run without the extensions.** B/C plug in via `get_retriever` + the
  benchmark's suite table; don't edit core so an extension works.
- **Both retrievers must see the same node set.** `build_kg` writes Neo4j and Qdrant in
  one pass for exactly this reason — don't split them.
- **Embeddings = bge-m3 via TEI**, **LLM = OpenAI-compatible**. No hardcoded secrets.

## Run

```bash
docker compose up --build -d          # everything, including the UI on :8000
# or, host-side development:
docker compose up -d neo4j qdrant embeddings
python -m scripts.build_kg --reset
python -m scripts.build_temporal_kg
python -m scripts.build_ingredient_kg
python -m scripts.generate_eval --suite all
python -m scripts.run_benchmark --suite all --judge
pytest
uvicorn thaigraphrag.app.main:app --reload --port 8000
```

## Recipes

**Add a retriever.** Subclass `Retriever` (set `name`, implement
`retrieve(query) -> RetrievedContext`), register it in
`retrievers/__init__.py :: get_retriever`, and add it to a suite in
`benchmark/run_benchmark.py :: SUITES`. It is now in the comparison, the API, and the
UI's retriever checkboxes automatically.

**Add a data source.** Append a `SourceSpec` to `kg/schema.py :: SOURCES` — filename,
label, key, name columns, province/address/lat-lon columns, and `EdgeSpec`s for the
attribute columns. `build_kg` is a generic engine over those specs; no loader code.
Give every new relation type a Thai label in `REL_TH` (a test enforces this).

**Regenerate the eval set.** `python -m scripts.generate_eval --suite all`. Gold
answers come from Cypher queries that already know the answer, so they cannot drift
from the graph. Only graph-unique entity names become questions. Re-run after any
rebuild. Hand-written knowledge items live in `KNOWLEDGE` in that script.

**Improve entity linking.** `core/entity_linking.py` — three passes (gazetteer →
Neo4j full-text → n-gram CONTAINS), scored by Dice. To plug in WangchanBERTa NER, add
a pass that returns the same `[{node_id, label, name, score, span, matched_by}]` shape
and merge it in `link_entities`. Watch out for generic Thai words: `_REQUIRED_PREFIX`
exists because Region "กลาง" was matching "มัสยิด**กลาง**ปัตตานี".

**Tune GraphRAG.** `retrievers/graphrag.py` — `GRAPH_HOPS`, `MAX_TRIPLES`, and the
per-node caps at the top of the file. Keep expansions **aggregated** (`count` +
`collect(...)[0..n]`); raw path enumeration explodes on hub nodes like a province.

**Extension B.** Time filter in `extensions/temporal/temporal_retriever.py`; validity
annotation in `temporal_kg.py`. Eval items need `as_of` and `stale_answer`.

**Extension C.** Rulings in `data/halal/ingredient_rulings.csv` (one row per
ingredient+source pair); graph build in `ingredient_kg.py`; paths in
`explain_retriever.py :: meta["paths"]`. Eval items need `gold_path`.

## Gotchas

- `build_kg` needs the CSVs in `DATA_DIR` and a running Neo4j + TEI.
- **bge-m3 on CPU embeds ~13 texts/s.** A full build is ~60k nodes ≈ 75 minutes. Use
  `--limit` or `MAX_PRODUCT_ROWS` while iterating, and `--no-embed` to test Cypher only.
- **TEI answers 413** when a batch exceeds `--max-client-batch-size` (64 in compose).
  `embed_many` splits and retries automatically; keep `EMBED_BATCH` ≤ that number.
- **The province columns are nearly empty** in the raw data — that is expected, and
  `kg/provinces.py` recovers them. Check `province_source` on a node to see which tier
  produced it.
- Thai is whitespace-sparse → metrics use **character** tokens (`benchmark/metrics.py`).
- Metrics return **NaN** for unannotated items (`hit_at_k`, `path_validity`) so they
  drop out of means. Don't "fix" that to 0.
- Without `LLM_API_KEY`, `ground()` returns raw context and the judge returns NaN —
  the pipeline still runs, but answer metrics are not meaningful.
- Neo4j `District` is keyed by `district|province`, not name: "เมือง" exists 77 times.
