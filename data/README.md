# Data

`DATA_DIR` (default `./data`) is where every source file lives. Docker bind-mounts
this directory read-only into the `app` container.

## KG source CSVs (not committed — see `.gitignore`)

| File | Rows | Becomes | Where to get it |
|------|-----:|---------|-----------------|
| `restuarant.csv` *(sic)* | 411 | `:Restaurant` | Halal Tourism `data_scraped` |
| `hotel.csv` | 26,224 | `:Hotel` | Halal Tourism `data_scraped` |
| `store_processed.csv` | 7,217 | `:Store` | Halal Tourism `data_scraped` |
| `attractions.csv` | 5,112 | `:Attraction` | Halal Tourism `data_scraped` |
| `mosque.csv` | 2,022 | `:Mosque` | Halal Tourism `data_scraped` |
| `product_processed.csv` | 222,752 | `:Product` | CICOT halal product registry |
| `thailand_provinces.json` | 77 features | province boundaries | Halal Tourism repo |

Copy them in, then build:

```bash
python -m scripts.build_kg --reset
```

### ⚠️ The province columns are mostly empty — this is expected

`addr_province` is populated for only ~4% of hotels, ~1.7% of mosques, and **0%** of
attractions. Province is recovered by a three-tier cascade
(`thaigraphrag/kg/provinces.py`): explicit column → province name inside the Thai
address text → point-in-polygon against `thailand_provinces.json`. Every node stores
`province_source` recording which tier produced it, so any `LOCATED_IN` edge can be
audited. Without this recovery there would be almost no location hierarchy, and
therefore no multi-hop questions.

### Product sampling

`product_processed.csv` has 222k rows; embedding all of them on CPU TEI takes hours.
`MAX_PRODUCT_ROWS` (default 20,000) takes a deterministic, category-stratified sample
with a per-category floor. The **same** sample is written to Neo4j and Qdrant, which
keeps the vanilla baseline and GraphRAG looking at an identical node set. Set it to
`0` to ingest every row.

## Curated research data (committed)

These are artefacts of this project, small and hand-checked, so they live in git.

### `halal/ingredient_rulings.csv` — 90 rows, 52 ingredients, 34 sources

One row per **(ingredient, source)** pair, because that pair — not the ingredient
alone — determines the ruling:

| column | meaning |
|--------|---------|
| `ingredient_th` / `ingredient_en` / `aliases` | names + `\|`-separated search terms |
| `e_number` | E-number where one applies (E441, E120, …) |
| `source_th` / `source_en` / `source_type` | provenance and its class |
| `ruling` | `halal` \| `haram` \| `mashbooh` |
| `basis_th` | why — the reasoning behind the verdict |
| `note_th` | practical caveat |

22 of the 52 ingredients carry **more than one ruling** depending on source
(เจลาตินจากหมู = หะรอม, เจลาตินจากปลา = ฮาลาล). That is the whole point of
extension C: without naming the source, the question cannot be answered.

> **Not a fatwa.** This is a research dataset for a retrieval experiment. Genuinely
> disputed items (คาร์ไมน์, เชลแล็ก, ethanol as a flavour carrier) are marked
> `mashbooh` with a note rather than resolved, and `basis_th` states the reasoning
> for every row. Consult a qualified certifying body for real decisions.

### `halal/regulation_timeline.csv` — 5 rows

A deliberately small timeline of well-established Thai halal governance facts
(พ.ร.บ. 2540, CICOT, HSC Chula 2546, HSIT 2546, มกษ. 8400-2550) with `valid_from` /
`valid_to` in Buddhist Era. It backs the qualitative "ณ ปีใด" questions. The headline
temporal result comes from the certification layer instead, which is grounded in the
registry's real 100%-populated `expire_date`.

## Evaluation sets (`questions/`)

| File | Suite | Notes |
|------|-------|-------|
| `thai_eval.jsonl` | A (core) | the released Thai multi-hop set, generated from the graph |
| `temporal_eval.jsonl` | B | items carry `as_of` and `stale_answer` |
| `ingredient_eval.jsonl` | C | items carry `gold_path` |
| `*.example.jsonl` | — | the small starter samples that shipped with the seed |

One JSON object per line:

```json
{"id": "m001", "question": "…", "answer": "…", "hop_type": "multi",
 "aliases": ["…"], "gold_nodes": ["…"], "as_of": 2570,
 "stale_answer": "…", "gold_path": ["…"], "provenance": "…"}
```

`hop_type` ∈ `single | multi | relational`.

### Gold answers are generated from the built graph

Rather than hand-writing answers that drift out of sync with the data, every
data-dependent item is derived from a Cypher query that already knows the answer:

```bash
python -m scripts.generate_eval --suite all
```

Only entities with a **graph-unique name** become questions, so no item is ambiguous.
Each carries `gold_nodes` (for retrieval hit@k) and `provenance` naming the Cypher
pattern and the province tier it rests on. Re-run the generator after any rebuild.
Knowledge-based items (halal definitions and rulings) are hand-written and stable.
