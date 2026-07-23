# Kickoff prompt — paste into Claude Code

> Copy everything inside the code block into Claude Code as your first message in this
> repo. It is the full build brief: implement all 3 topics, make it run in one Docker
> command, build a real demo UI for every page, run the experiments, and write the
> results + screenshots + GIF + system map + file directory into the README.

```text
You are the lead engineer for the "Thai GraphRAG + Benchmark" research monorepo.
This is ONE big project with three topics that must all work and be separable:
  A (core)  = GraphRAG vs vanilla RAG on a Thai knowledge graph + a benchmark by hop type
  B (ext)   = Temporal GraphRAG (answer "as of <year>")
  C (ext)   = Halal-Ingredient explainable GraphRAG (ingredient → source → ruling PATH)

The repo is a working SEED: core code, benchmark harness, extension scaffolds, docs,
example eval sets, and docker-compose already exist. Your job is to finish it into a
runnable, portfolio-grade research repo with a real demo UI and written-up results.

═══════════════════════════════════════════════════════════════════════════
READ FIRST (do not skip)
═══════════════════════════════════════════════════════════════════════════
- CLAUDE.md — architecture, design invariants (§2), Qdrant/Neo4j roles, TODO (§6).
- docs/METHODOLOGY.md — the controlled experiment (only the retriever changes).
- docs/REFERENCES.md — the papers grounding every claim (cite these in the README).
- skills/thai-graphrag-dev/SKILL.md — recipes (add a retriever, grow eval set, etc.).
- thaigraphrag/extensions/{temporal,halal_ingredient}/README.md — B and C designs.

DESIGN INVARIANTS (must not break):
1. Fair comparison: every retriever returns RetrievedContext; the answer pipeline,
   embeddings (bge-m3 via TEI) and LLM/prompt are SHARED. Only retrieval differs.
2. Core (A) runs without B/C. Extensions plug into retrievers.get_retriever and the
   benchmark's RETRIEVERS list — never edit core to make an extension work.
3. Neo4j = the graph; Qdrant = the vanilla baseline only. Secrets only via .env/config.
4. License stays VIEW-ONLY (see LICENSE) — do not relicense.

═══════════════════════════════════════════════════════════════════════════
TASK 1 — Finish core A
═══════════════════════════════════════════════════════════════════════════
- Implement/verify KG build from DATA_DIR CSVs (thaigraphrag/kg/build_kg.py): nodes,
  Province + LOCATED_IN, and (from product_processed.csv) the Product/Category links.
- Strengthen entity_linking.py (char n-gram + CONTAINS baseline → add fuzzy/embedding
  or WangchanBERTa NER; keep the return shape).
- Improve graphrag.py subgraph selection + linearisation (concise triples; consider
  NodeRAG/LightRAG ideas from REFERENCES).
- Add metrics: retrieval hit@k (needs gold provenance) and real token/call cost.
- GROW the eval set: data/questions/thai_eval.example.jsonl from ~30 → 150–300 items,
  balanced single/multi/relational. After building the KG, VERIFY data-dependent gold
  answers (province/"มี-ไม่มี") against the real graph. This released Thai multi-hop
  KGQA set is the project's novel contribution.

═══════════════════════════════════════════════════════════════════════════
TASK 2 — Implement extensions B and C fully
═══════════════════════════════════════════════════════════════════════════
B (temporal): add valid_from/valid_to to relations (temporal_kg.py), implement the
  time filter in temporal_retriever.py, register "temporal" in get_retriever and add it
  to the benchmark. Use data/questions/temporal_eval.example.jsonl (as_of). Report the
  % reduction in wrong-era answers vs the time-agnostic GraphRAG.
C (halal-ingredient): implement ingredient_kg.build() from product_processed.csv
  ((:Product)-[:CONTAINS]->(:Ingredient)-[:DERIVED_FROM]->(:Source),
   (:Ingredient)-[:HAS_RULING]->(:Ruling {status})), return real ruling PATHS in
  explain_retriever.py, register "halal_ingredient". Use ingredient_eval.example.jsonl
  (gold_path). Report answer correctness AND % of answers with a valid, complete path.

═══════════════════════════════════════════════════════════════════════════
TASK 3 — One-command Docker (whole system boots in a single command)
═══════════════════════════════════════════════════════════════════════════
- Extend docker-compose.yml so `docker compose up --build -d` brings up EVERYTHING:
  neo4j + qdrant + embeddings(TEI) + a new `app` service (the demo web UI + API).
- Add a Dockerfile for the app. On first boot the app must: wait for neo4j/qdrant
  readiness → build the KG (if empty) → run the benchmark once → then serve.
- Provide an .env-driven flag to skip the (slow) auto-build on later boots.
- Document the exact one command in the README and in CLAUDE.md §5.
- Handle cold-start races (retry/wait), TEI model download (~2GB once), and memory
  caps — mirror the fixes documented in the sibling Chatbot-CoreEngine repo.

═══════════════════════════════════════════════════════════════════════════
TASK 4 — Real demo UI (every page must actually work end-to-end)
═══════════════════════════════════════════════════════════════════════════
Build a web UI (FastAPI + a static SPA, or Streamlit — your call) served by the app
container. Every page calls the real backend, no mock data:
  1. Overview        — what the project is + live counts (KG nodes, collections, #questions).
  2. Ask / Compare   — type a Thai question → show Vanilla vs GraphRAG answers SIDE BY SIDE,
                       with retrieved context, the GraphRAG subgraph triples, scores, latency.
  3. KG Explorer     — browse/search nodes and relations (Neo4j), show a small graph view.
  4. Benchmark       — button to run the benchmark; render the summary table + the
                       "F1 by hop type" chart; let the user pick the eval file.
  5. Eval Set        — view (and ideally edit) the question sets, per hop_type.
  6. Temporal (B)    — an `as_of` year selector; show how the answer changes across years.
  7. Halal-Ingredient (C) — enter an ingredient/product → show the ruling PATH visually
                       (ingredient → source → ruling) so the reasoning is explainable.
Expose a REST API for each (e.g. /api/ask, /api/benchmark/run, /api/kg/search,
/api/temporal/ask, /api/ingredient/explain) and Swagger at /docs.

═══════════════════════════════════════════════════════════════════════════
TASK 5 — Run experiments + capture screenshots + build the GIF
═══════════════════════════════════════════════════════════════════════════
- Run the full benchmark (A, plus B and C) with --judge; save artefacts to results/
  (benchmark_detail.csv, benchmark_summary.csv, figures/f1_by_hop.png, plus B/C figures).
- Screenshot EVERY UI page into docs/img/ (01-overview.png, 02-ask-compare.png,
  03-kg-explorer.png, 04-benchmark.png, 05-eval-set.png, 06-temporal.png,
  07-ingredient.png). Assemble a docs/img/00-tour.gif that walks through all pages.

═══════════════════════════════════════════════════════════════════════════
TASK 6 — Write the README (research report + map + directory)
═══════════════════════════════════════════════════════════════════════════
The README must contain, in Thai+English where natural:
- Problem + the research question, and the hypothesis (GraphRAG's edge grows with hops).
- System architecture diagram + the "fair comparison" invariant.
- 🎬 System tour: embed docs/img/00-tour.gif, then each page screenshot with a caption.
- 📊 RESEARCH RESULTS: the benchmark tables (F1/EM/faithfulness/latency) and the
  F1-by-hop chart, with the written FINDINGS for A, B, and C — state clearly what won,
  by how much, and where (single vs multi vs relational). Cite docs/REFERENCES.md.
- 🔗 SYSTEM LINKS MAP: a copy-paste list of every local URL the running stack exposes
  (UI pages, /docs, /health, Neo4j browser :7474, Qdrant dashboard :6333, TEI :8080).
- 🗂️ FILE DIRECTORY: the full annotated file tree (what each folder/file does).
- 🐳 One-command run + everyday docker commands + a "bugs fixed to make it boot" table.
- 🔬 Research system structure: how A/B/C compose, and how to run each alone.
Keep a matching FILE_DIRECTORY.md if the README gets long.

═══════════════════════════════════════════════════════════════════════════
TASK 7 — Hygiene
═══════════════════════════════════════════════════════════════════════════
- LICENSE stays view-only. .gitignore already excludes .env and raw CSVs — keep it so.
- No secrets or raw source CSVs committed. results/ figures + summary CSVs MAY be
  committed so the README renders on GitHub (adjust .gitignore accordingly).

═══════════════════════════════════════════════════════════════════════════
ACCEPTANCE CRITERIA (definition of done)
═══════════════════════════════════════════════════════════════════════════
□ `docker compose up --build -d` on a clean machine brings the whole stack up; the UI
  is reachable and every one of the 7 pages works against the real backend.
□ `pytest` is green.
□ `python -m scripts.run_benchmark --judge` reproduces results/ and the F1-by-hop chart,
  and the chart shows GraphRAG's margin over vanilla widening from single → multi/relational.
□ README shows the tour GIF, every page screenshot, the results tables/chart with written
  findings for A/B/C, the system-links map, and the full file directory.
□ B and C are runnable and separable (A still runs alone).

WORK METHOD: after each task, run pytest + the benchmark and briefly report what changed
and the numbers. Verify data-dependent gold answers against the built KG before quoting
final results. Ask me for the TOR .docx wording if you need exact halal-regulation text.
```
