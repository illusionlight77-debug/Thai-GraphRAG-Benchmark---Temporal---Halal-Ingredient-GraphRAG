# Extension C — Halal-Ingredient explainable GraphRAG

**Goal:** answer *"ส่วนผสมนี้ฮาลาลไหม"* with a **traceable reasoning path** —
`เจลาติน → สุกร → ไม่ฮาลาล (หะรอม)` — instead of an unexplained yes/no.

## The core idea: the ruling belongs to the *pair*, not the ingredient

An ingredient has no ruling on its own. In the curated table, **22 of 52 ingredients
carry more than one ruling** depending on where they came from:

| ส่วนผสม | แหล่งที่มา | คำวินิจฉัย |
|---------|-----------|-----------|
| เจลาติน | สุกร | ไม่ฮาลาล |
| เจลาติน | ปลา | ฮาลาล |
| เจลาติน | วัวไม่ระบุวิธีการเชือด | คลุมเครือ |

So the path is *necessary*, not decorative: a system that cannot name the source
cannot answer correctly, and the honest answer to an unqualified "เจลาตินฮาลาลไหม" is
**"ขึ้นอยู่กับแหล่งที่มา"**. That item type is exactly where a flat retriever fails —
it picks one ruling and states it as fact.

## Graph shape

```
(:Product)-[:CONTAINS {matched_term}]->(:Ingredient)-[:DERIVED_FROM]->(:Source)
(:Ingredient)-[:HAS_RULING {via_source, basis}]->(:Ruling {status})
(:Source)-[:TYPICAL_RULING]->(:Ruling)
```

`HAS_RULING.via_source` joins a verdict back to the `DERIVED_FROM` edge that produced
it — that join is how a complete three-hop chain is assembled.

`TYPICAL_RULING` is created only for sources that are unambiguous. Two are
deliberately excluded: เอทานอล from petrochemistry is `mashbooh` while วานิลลิน from
the same route is `halal`, because ethanol is an intoxicant regardless of synthesis
route. Encoding that exception rather than flattening it keeps the table honest.

## Two data layers

| layer | source | scale |
|-------|--------|-------|
| **Ruling layer** | `data/halal/ingredient_rulings.csv` — curated, committed | 90 facts · 52 ingredients · 34 sources |
| **Product layer** | `product_processed.csv` — the real CICOT registry | 20k sampled products |

`CONTAINS` edges are extracted from **real Thai product names** (เจลาติน appears in
281 of them, เวย์ in 361, คอลลาเจน in 854), and the matched substring is stored on
the edge so every link is auditable.

A nice property of the registry: it lists products that are **already certified
halal**, so `หมู` appears in only 2 of 222k names. It works as a positive control.

## Build and run

```bash
python -m scripts.build_kg               # Product nodes must exist first
python -m scripts.build_ingredient_kg    # ruling layer + CONTAINS links
python -m scripts.generate_eval --suite ingredient
python -m scripts.run_benchmark --suite ingredient --judge
```

## Metrics

- **answer correctness** — F1 / EM / containment, as everywhere else;
- **`path_validity`** — fraction of the item's `gold_path` matched **in order**, since
  a path visiting the right nodes in the wrong order is not a valid explanation;
- **share of answers carrying a complete three-hop path** — the explainability claim.

## Files

| file | role |
|------|------|
| `ingredient_kg.py` | `build_rulings()`, `link_products()`, `summary()` |
| `explain_retriever.py` | `IngredientExplainRetriever` — returns structured `meta["paths"]` |

Registered as `"halal_ingredient"` in `retrievers.get_retriever`; no core file is
modified.

## ⚠️ Not a fatwa

This is a research dataset for a retrieval experiment. Genuinely disputed items
(คาร์ไมน์/E120, เชลแล็ก/E904, ethanol as a flavour carrier) are marked `mashbooh`
with a note rather than resolved, and `basis_th` states the reasoning for every row.
Consult a qualified certifying body for real decisions.
