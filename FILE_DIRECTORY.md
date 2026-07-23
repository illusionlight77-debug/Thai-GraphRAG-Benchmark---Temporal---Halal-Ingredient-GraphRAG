# 🗂️ File directory

Annotated tree of the whole repo. ⭐ marks the files that carry the research argument.

```
thai-graphrag-benchmark/
│
├── 🐳 Docker + config
│   ├── docker-compose.yml          4 services: neo4j · qdrant · embeddings(TEI) · app
│   ├── Dockerfile                  app image (thin HTTP client; CSVs bind-mounted, not copied)
│   ├── .dockerignore               keeps data/ out of the image (product CSV alone is 133MB)
│   ├── .env.example                every setting, documented
│   ├── .gitignore                  excludes .env + source CSVs; keeps curated data + figures
│   ├── requirements.txt            pinned deps
│   └── pyproject.toml              package metadata
│
├── 📖 Docs
│   ├── README.md                   the research report (results, findings, links map)
│   ├── CLAUDE.md                   architecture + design invariants + how to run
│   ├── FILE_DIRECTORY.md           this file
│   ├── LICENSE                     view-only
│   ├── KICKOFF_CLAUDE_CODE.md      the original build brief
│   └── docs/
│       ├── METHODOLOGY.md          ⭐ the controlled experiment, metrics, known limitations
│       ├── REFERENCES.md           papers grounding every claim
│       └── img/                    screenshots + tour GIF used by the README
│
├── 🧠 thaigraphrag/                the package
│   │
│   ├── config.py                   Settings from .env (pydantic) + results paths
│   │
│   ├── core/                       shared infrastructure — identical for every retriever
│   │   ├── embeddings.py           bge-m3 via TEI; splits on 413, retries, wait_ready()
│   │   ├── llm.py                  ⭐ ground_detailed() + judge_faithfulness() + Usage
│   │   │                             (real token/call accounting; ONE shared prompt)
│   │   ├── neo4j_client.py         driver + run(cypher) → list[dict]
│   │   ├── qdrant_client.py        kg_nodes collection; ensure/reset/count
│   │   └── entity_linking.py       ⭐ query → seed nodes. gazetteer → Neo4j fulltext →
│   │                                 n-gram fallback, scored by Dice. NO vector search
│   │                                 here — that would confound the comparison.
│   │
│   ├── kg/                         graph construction
│   │   ├── provinces.py            ⭐ 77-province gazetteer + point-in-polygon geocoder.
│   │   │                             Recovers LOCATED_IN from data where addr_province
│   │   │                             is ~4% populated. Without it there is no hierarchy.
│   │   ├── schema.py               ⭐ declarative SourceSpec/EdgeSpec per CSV + node_text()
│   │   └── build_kg.py             ⭐ the engine: CSV → Neo4j + Qdrant in one pass, so
│   │                                 both retrievers see an identical node set
│   │
│   ├── retrievers/                 the only thing that differs between conditions
│   │   ├── base.py                 ⭐ Retriever ABC + RetrievedContext (the seam)
│   │   ├── vanilla.py              baseline — Qdrant top-k over node text
│   │   ├── graphrag.py             ⭐ seed cards + aggregated neighbourhoods + bridge
│   │   │                             paths. Aggregation replaces path enumeration, which
│   │   │                             exploded on hub nodes.
│   │   └── __init__.py             ⭐ get_retriever() — extensions register here (lazy)
│   │
│   ├── pipeline/
│   │   └── answer.py               ⭐ retrieve → ground. Must never branch on retriever
│   │                                 name; that is the fair-comparison guarantee in code.
│   │
│   ├── benchmark/
│   │   ├── datasets.py             load/save/validate eval .jsonl
│   │   ├── metrics.py              ⭐ F1(char)/EM/containment + context_recall / hit@k /
│   │   │                             path_validity (NaN = unannotated, excluded from means)
│   │   └── run_benchmark.py        ⭐ 3 suites → results/ CSVs + figures + run metadata
│   │
│   ├── extensions/                 plug in via get_retriever; never modify core
│   │   ├── temporal/               B — answer "as of <year>"
│   │   │   ├── temporal_kg.py      valid_to from the registry's real expire_date (100%
│   │   │   │                         populated); valid_from deliberately NULL
│   │   │   ├── temporal_retriever.py  interval filter on every expansion + expiry report
│   │   │   └── README.md           design, metric, limitation
│   │   └── halal_ingredient/       C — ingredient → source → ruling PATH
│   │       ├── ingredient_kg.py    curated rulings + CONTAINS from real product names
│   │       ├── explain_retriever.py   returns structured meta["paths"]
│   │       └── README.md           why the ruling belongs to the (ingredient, source) pair
│   │
│   └── app/                        the demo
│       ├── main.py                 ⭐ FastAPI — every endpoint hits the live stack
│       └── static/                 index.html · app.js · style.css
│                                   7 pages, zero external assets (works offline)
│
├── 🔧 scripts/                     all runnable as `python -m scripts.<name>`
│   ├── build_kg.py                 core KG + vector index
│   ├── build_temporal_kg.py        extension B layer
│   ├── build_ingredient_kg.py      extension C layer
│   ├── generate_eval.py            ⭐ generates eval sets FROM the graph, so gold
│   │                                 answers are correct by construction
│   ├── run_benchmark.py            the experiment
│   └── bootstrap.py                first-boot sequence inside the app container
│
├── 📊 data/
│   ├── README.md                   where each file comes from + the sampling policy
│   ├── *.csv                       source CSVs (NOT committed)
│   ├── thailand_provinces.json     77 province boundaries (geocoding)
│   ├── halal/
│   │   ├── ingredient_rulings.csv  ⭐ committed research artefact — 90 facts,
│   │   │                             52 ingredients, 34 sources, 22 source-dependent
│   │   └── regulation_timeline.csv small curated Thai halal governance timeline
│   └── questions/
│       ├── thai_eval.jsonl         ⭐ the released Thai multi-hop set (suite A)
│       ├── temporal_eval.jsonl     suite B — items carry as_of + stale_answer
│       ├── ingredient_eval.jsonl   suite C — items carry gold_path
│       └── *.example.jsonl         the small starter samples from the seed
│
├── ✅ tests/                       55 tests, none require a running service
│   ├── test_smoke.py               metrics, linking helpers, factory
│   ├── test_provinces.py           gazetteer completeness + geocoder accuracy
│   ├── test_metrics.py             including NaN semantics and path order-sensitivity
│   ├── test_schema_and_data.py     schema wiring + curated ruling-table invariants
│   └── test_retrievers_and_eval.py registry, extension separability, eval structure
│
├── 📈 results/                     regenerated by run_benchmark
│   ├── benchmark_detail.csv        one row per (question, retriever) — gitignored
│   ├── benchmark_summary.csv       means per (suite, retriever, hop_type) — committed
│   ├── benchmark_meta.json         run metadata + real LLM spend — committed
│   └── figures/*.png               committed so the README renders on GitHub
│
└── 🎓 skills/thai-graphrag-dev/SKILL.md   dev recipes for this repo
```

## Where to look first

| I want to… | Read |
|------------|------|
| understand the experiment | `docs/METHODOLOGY.md` |
| check the comparison is fair | `pipeline/answer.py`, `retrievers/base.py`, `core/llm.py` |
| see how GraphRAG actually retrieves | `retrievers/graphrag.py` |
| know where the eval answers came from | `scripts/generate_eval.py` |
| audit a halal ruling | `data/halal/ingredient_rulings.csv` (`basis_th` column) |
| add a retriever | `skills/thai-graphrag-dev/SKILL.md` |
