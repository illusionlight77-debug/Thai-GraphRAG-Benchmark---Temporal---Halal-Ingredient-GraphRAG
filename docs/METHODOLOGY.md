# Methodology

## Question

Does graph-structured retrieval (**GraphRAG**) beat dense retrieval (**vanilla RAG**)
for Thai question answering, and **where** — i.e. does the advantage grow with the
number of reasoning hops, as reported for English (arXiv:2502.11371)?

## Controlled comparison

Everything is held constant except the **retriever**:

| Component | Setting |
|-----------|---------|
| Knowledge graph | Neo4j, built from the Halal Tourism CSVs + the CICOT product registry |
| Embeddings | bge-m3 (TEI), 1024-d, cosine |
| LLM grounding | one OpenAI-compatible model (Groq default), temperature 0 |
| Answer pipeline | `pipeline/answer.py` — identical for every retriever |
| **Retriever** | **vanilla** (Qdrant top-k node text) vs **graphrag** (k-hop subgraph) |

Both retrievers return the same `RetrievedContext`, so any metric difference is
attributable to retrieval strategy, not prompt/model differences.

Three things are load-bearing for that claim and are enforced in code:

1. **`pipeline/answer.py` never branches on `retriever.name`.** The system prompt,
   model, and temperature live in `core/llm.py` and are shared.
2. **GraphRAG never touches Qdrant.** Entity linking (`core/entity_linking.py`) is
   lexical over Neo4j only. Letting it fall back to vector search would make
   GraphRAG a strict superset of vanilla, and any win would then be trivially
   explained by "it also has the baseline".
3. **Both retrievers see the same node set.** The KG build embeds exactly the nodes
   it writes to Neo4j, in the same pass, so the baseline is never handicapped by a
   smaller corpus.

The context length is bounded (`MAX_TRIPLES`, default 60 facts) for the same reason:
an unbounded graph dump would win on prompt budget rather than on structure.

## Data

### Knowledge graph

Built from six CSVs into a location hierarchy plus typed attribute nodes:

```
(Place)-[:LOCATED_IN]->(Province)-[:IN_REGION]->(Region)
(Place)-[:IN_DISTRICT]->(District)-[:IN_PROVINCE]->(Province)
(Restaurant)-[:SERVES_CUISINE]->(Cuisine)
(Store)-[:IS_TYPE]->(StoreType) , (Store)-[:HAS_BRAND]->(Brand)
(Product)-[:BELONGS_TO]->(Category) , (Product)-[:MADE_BY]->(Company)
```

The hierarchy is what makes multi-hop questions possible at all: two places in
different CSVs that share a province are exactly 2 hops apart, and a dense retriever
over independent node texts has no way to traverse that.

**Province recovery.** The raw data is nearly unusable for location as shipped —
`addr_province` is populated for ~4% of hotels, ~1.7% of mosques, and *0%* of
attractions. Province is therefore resolved by a three-tier cascade, and every node
records which tier produced it (`province_source`) so any derived edge is auditable:

| tier | source | notes |
|------|--------|-------|
| `explicit` | a province column in the CSV | normalised EN → canonical Thai |
| `address` | province name found in the Thai address text | longest-name-first match |
| `geo` | point-in-polygon of (lat, lon) | 77 province boundaries, 107 rings |

**Product sampling.** `product_processed.csv` holds 222,752 certified products.
Embedding all of them through CPU TEI takes hours and adds little to a tourism
multi-hop benchmark, so the build takes a **deterministic, category-stratified
sample** (`MAX_PRODUCT_ROWS`, default 20,000, `random_state=42`) with a per-category
floor so small categories survive. The sample goes into **both** Neo4j and Qdrant, so
the fairness invariant holds. Set `MAX_PRODUCT_ROWS=0` to ingest everything.

### Evaluation set

Questions are labelled by `hop_type`:

- **single** — answerable from one node.
- **multi** — needs ≥2 nodes joined by relations.
- **relational** — the answer is a relation/attribute reached by traversal.

The released Thai multi-hop eval set is the project's novel artefact — Thai KGQA
multi-hop sets are effectively absent publicly.

**Gold answers are derived from the graph, not written by hand.**
`scripts/generate_eval.py` builds each data-dependent item from a Cypher query that
already knows the answer, so the gold is correct by construction and re-running the
script after a rebuild keeps the set honest. Two guards apply:

- only entities whose **name is unique across the graph** become questions, because
  an ambiguous name has no single gold answer;
- every item records `gold_nodes` (entities a correct answer must pass through) and a
  `provenance` string naming the Cypher pattern and the province tier it rests on.

Knowledge-based items (halal definitions, rulings) are hand-written, stable, and
independent of the graph.

## Metrics

**Answer quality**

- token-level **F1** over character tokens (Thai is whitespace-sparse), **Exact
  Match**, **Containment**.
- **Faithfulness** — LLM-as-judge (0/1): is the answer supported by the retrieved
  context?

**Retrieval quality** — reported separately, because a low F1 is ambiguous between
"the retriever never found the evidence" and "the LLM had the evidence and still
answered badly", and only the first is the manipulated variable.

- **context_recall** — does the retrieved context contain a gold answer string? This
  is the *ceiling* on answer quality: the grounding prompt forbids outside knowledge,
  so a retriever scoring 0 here cannot be rescued by any LLM. Needs no annotation, so
  it is reported for every question.
- **hit@k** — fraction of an item's `gold_nodes` present in the retrieved set. `NaN`
  for unannotated items, so they are excluded from the mean rather than scored 0.

**Cost** — end-to-end latency, retrieval latency, real prompt/completion token counts
and call counts read from the API response (`core/llm.py :: Usage`).

Aggregate **by hop_type** and overall. The headline figure is **F1 vs hop type**
(`results/figures/f1_by_hop.png`): the expected, testable pattern is that GraphRAG's
margin over vanilla widens from single → multi/relational.

## Extensions reuse the same harness

Both plug in through `retrievers.get_retriever` and add themselves to the benchmark's
suite table. Neither modifies a core file.

### B — Temporal

Relations carry `valid_from` / `valid_to`; the retriever applies the standard
interval predicate, with NULL meaning unbounded:

```
(r.valid_from IS NULL OR r.valid_from <= as_of) AND
(r.valid_to   IS NULL OR r.valid_to   >= as_of)
```

Untimed relations always pass, so switching the extension on never *loses* facts —
it only removes statements that were untrue in the requested year.

Two temporal layers with **different evidential weight**, reported separately:

1. **Certification validity — real data.** `expire_date` is populated for 100% of the
   222k-row registry in Thai Buddhist-Era `DD/MM/YYYY`. `valid_to` is that expiry
   year. `valid_from` is deliberately left NULL: the registry has no issue date, and
   the trailing digits of `halal_code` do not encode one reliably (measured implied
   term: −29 to +4 years). This layer carries the headline number.
2. **Regulation timeline — small curated set.** Five well-established Thai halal
   governance facts. Deliberately limited to undisputed items; it supports the
   qualitative "ณ ปีใด" questions, not the headline.

**Metric:** `wrong_era` — the answer repeats the item's pre-annotated `stale_answer`
*and* fails to state the answer true at `as_of`. Both halves are required: an answer
that says neither is unhelpful, not wrong about time.

### C — Halal-Ingredient

```
(:Product)-[:CONTAINS]->(:Ingredient)-[:DERIVED_FROM]->(:Source)
(:Ingredient)-[:HAS_RULING {via_source, basis}]->(:Ruling {status})
(:Source)-[:TYPICAL_RULING]->(:Ruling)
```

The ruling is a property of the **(ingredient, source) pair**, not of the ingredient:
เจลาตินจากหมู is หะรอม while เจลาตินจากปลา is ฮาลาล. In the curated table, 22 of 52
ingredients carry more than one ruling. That is what makes the reasoning path
*necessary* rather than decorative — a system that cannot name the source cannot
answer correctly, and the honest answer to an unqualified "เจลาตินฮาลาลไหม" is
"ขึ้นอยู่กับแหล่งที่มา".

`HAS_RULING.via_source` joins a verdict back to the provenance edge that produced it,
which is how a complete ingredient → source → ruling chain is assembled.

Product links are extracted from **real Thai product names** in the registry, with
the matched substring stored on the edge (`CONTAINS.matched_term`) so every link is
auditable.

**Metrics:** answer correctness **and** `path_validity` — the fraction of the item's
`gold_path` matched *in order*, since a path visiting the right nodes in the wrong
order is not a valid explanation — plus the share of answers carrying a complete
three-hop path.

## Known limitations

- **Gold answers inherit the data's errors.** A mosque whose OSM coordinates are
  wrong yields a wrong province; the `province_source` field makes such items
  traceable but does not fix them.
- **`valid_from` is unknown for certifications**, so "was X certified in 2565?"
  is answered as "unbounded start". Eval items stay in the range where `valid_to`
  discriminates.
- **The ruling table is a research artefact, not a fatwa.** Disputed items
  (คาร์ไมน์, เชลแล็ก, ethanol as a carrier) are marked `mashbooh` with a note rather
  than resolved, and the basis column cites the reasoning for every row.
- **Character-level F1 rewards partial string overlap**, which inflates scores for
  verbose answers; EM and containment are reported alongside for that reason.
