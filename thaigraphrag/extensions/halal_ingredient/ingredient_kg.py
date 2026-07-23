"""C — build the ingredient graph.

Two layers are joined here:

* **Ruling layer** — a curated `data/halal/ingredient_rulings.csv` of
  (ingredient, source) → ruling facts. One row per *pair*, because the whole point
  of extension C is that an ingredient has no ruling on its own: เจลาตินจากหมู is
  หะรอม while เจลาตินจากปลา is ฮาลาล. 22 of the 52 ingredients carry more than one
  ruling, so a system that cannot name the source cannot answer correctly.

* **Product layer** — the real `product_processed.csv` registry already loaded as
  `:Product` nodes. Ingredient mentions are extracted from the actual Thai product
  names, so `(:Product)-[:CONTAINS]->(:Ingredient)` edges are evidence found in the
  data, not invented.

Resulting shape:

    (:Product)-[:CONTAINS]->(:Ingredient)-[:DERIVED_FROM]->(:Source)
    (:Ingredient)-[:HAS_RULING {via_source, basis}]->(:Ruling {status})
    (:Source)-[:TYPICAL_RULING]->(:Ruling)      # only where the source is unambiguous

`HAS_RULING.via_source` is what lets the retriever join a verdict back to the source
that produced it, which is how a complete ingredient → source → ruling path is built.

Run:  python -m scripts.build_ingredient_kg
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from thaigraphrag.config import get_settings
from thaigraphrag.core import entity_linking, neo4j_client

RULINGS_CSV = "halal/ingredient_rulings.csv"

# Thai display names for the three verdicts.
RULING_TH = {"halal": "ฮาลาล", "haram": "ไม่ฮาลาล (หะรอม)", "mashbooh": "คลุมเครือ (มัชบูฮ์)"}
RULING_ORDER = {"haram": 0, "mashbooh": 1, "halal": 2}   # worst-first for aggregation

# Aliases shorter than this are too generic to match product names safely
# ('นม' inside 'นมข้น' is fine, but 2-char latin fragments are not).
_MIN_ALIAS_LEN = 3


def _load(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path).fillna("")
    required = {"ingredient_th", "source_th", "ruling"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path.name} missing columns: {sorted(missing)}")
    bad = set(df["ruling"]) - set(RULING_TH)
    if bad:
        raise ValueError(f"unknown ruling values: {sorted(bad)}")
    return df


def _aliases(row: pd.Series) -> list[str]:
    """Search terms for one ingredient: Thai name, English name, and `aliases`."""
    out = [str(row["ingredient_th"]).strip(), str(row.get("ingredient_en", "")).strip()]
    out += [a.strip() for a in str(row.get("aliases", "")).split("|")]
    if row.get("e_number"):
        out.append(str(row["e_number"]).strip())
    seen, terms = set(), []
    for a in out:
        if len(a) >= _MIN_ALIAS_LEN and a.lower() not in seen:
            seen.add(a.lower())
            terms.append(a)
    return terms


def build_rulings(df: pd.DataFrame) -> dict:
    """Create Ingredient / Source / Ruling nodes and the edges between them."""
    # Ruling nodes (exactly three).
    neo4j_client.run(
        """
        UNWIND $rows AS row
        MERGE (r:Ruling {status: row.status})
          SET r.name = row.name_th, r.text = 'คำวินิจฉัย: ' + row.name_th
        """,
        rows=[{"status": k, "name_th": v} for k, v in RULING_TH.items()],
    )

    ingredients = (df.groupby("ingredient_th")
                     .agg(ingredient_en=("ingredient_en", "first"),
                          e_number=("e_number", "first"))
                     .reset_index())
    alias_map = {r["ingredient_th"]: _aliases(r) for _, r in df.iterrows()}
    neo4j_client.run(
        """
        UNWIND $rows AS row
        MERGE (i:Ingredient {name: row.name})
          SET i.name_en = row.name_en, i.e_number = row.e_number,
              i.aliases = row.aliases,
              i.text = 'ส่วนผสม: ' + row.name
                     + CASE WHEN row.name_en <> '' THEN ' (' + row.name_en + ')' ELSE '' END
                     + CASE WHEN row.e_number <> '' THEN ' รหัส ' + row.e_number ELSE '' END
        """,
        rows=[{"name": r["ingredient_th"], "name_en": r["ingredient_en"],
               "e_number": r["e_number"], "aliases": alias_map.get(r["ingredient_th"], [])}
              for _, r in ingredients.iterrows()],
    )

    sources = (df.groupby("source_th")
                 .agg(source_en=("source_en", "first"), source_type=("source_type", "first"))
                 .reset_index())
    neo4j_client.run(
        """
        UNWIND $rows AS row
        MERGE (s:Source {name: row.name})
          SET s.name_en = row.name_en, s.source_type = row.source_type,
              s.text = 'แหล่งที่มา: ' + row.name + ' (ประเภท ' + row.source_type + ')'
        """,
        rows=[{"name": r["source_th"], "name_en": r["source_en"],
               "source_type": r["source_type"]} for _, r in sources.iterrows()],
    )

    facts = [{"ing": r["ingredient_th"], "src": r["source_th"], "ruling": r["ruling"],
              "basis": r["basis_th"], "note": r["note_th"]} for _, r in df.iterrows()]
    neo4j_client.run(
        """
        UNWIND $rows AS row
        MATCH (i:Ingredient {name: row.ing})
        MATCH (s:Source {name: row.src})
        MATCH (r:Ruling {status: row.ruling})
        MERGE (i)-[d:DERIVED_FROM]->(s)
          SET d.ruling = row.ruling
        MERGE (i)-[h:HAS_RULING {via_source: row.src}]->(r)
          SET h.basis = row.basis, h.note = row.note
        """,
        rows=facts,
    )

    # Source-level default, created only for sources that always imply one ruling.
    # เอทานอล from petrochemistry is mashbooh while วานิลลิน from the same route is
    # halal, so those two sources deliberately get no TYPICAL_RULING edge.
    unambiguous = (df.groupby("source_th")["ruling"].nunique() == 1)
    typical = [{"src": s, "ruling": df[df["source_th"] == s]["ruling"].iloc[0]}
               for s, ok in unambiguous.items() if ok]
    neo4j_client.run(
        """
        UNWIND $rows AS row
        MATCH (s:Source {name: row.src})
        MATCH (r:Ruling {status: row.ruling})
        MERGE (s)-[:TYPICAL_RULING]->(r)
        """,
        rows=typical,
    )
    return {"ingredients": len(ingredients), "sources": len(sources),
            "facts": len(facts), "typical": len(typical)}


def link_products(df: pd.DataFrame, batch: int = 2000) -> dict:
    """Attach `(:Product)-[:CONTAINS]->(:Ingredient)` by scanning real product names.

    Matching is literal substring over the product name that is already in the graph,
    so the edges reflect what the registry actually says. `matched_term` is stored on
    the edge so any link can be audited back to the string that produced it.
    """
    terms: list[tuple[str, str]] = []      # (search term, ingredient name)
    for _, r in df.drop_duplicates("ingredient_th").iterrows():
        for t in _aliases(r):
            terms.append((t.lower(), r["ingredient_th"]))
    # Longest first so 'น้ำมันมะพร้าว' wins over 'น้ำมัน'.
    terms.sort(key=lambda x: -len(x[0]))

    products = neo4j_client.run(
        "MATCH (p:Product) RETURN p.halal_code AS code, coalesce(p.name,'') AS name")
    edges: list[dict] = []
    for p in products:
        name = (p["name"] or "").lower()
        if not name:
            continue
        hit: dict[str, str] = {}
        for term, ing in terms:
            if ing in hit:
                continue
            if term in name:
                hit[ing] = term
        for ing, term in hit.items():
            edges.append({"code": p["code"], "ing": ing, "term": term})

    for i in range(0, len(edges), batch):
        neo4j_client.run(
            """
            UNWIND $rows AS row
            MATCH (p:Product {halal_code: row.code})
            MATCH (i:Ingredient {name: row.ing})
            MERGE (p)-[c:CONTAINS]->(i)
              SET c.matched_term = row.term
            """,
            rows=edges[i:i + batch],
        )
    return {"products_scanned": len(products), "contains_edges": len(edges),
            "products_linked": len({e["code"] for e in edges})}


def _constraints() -> None:
    for label in ("Ingredient", "Source"):
        neo4j_client.run(
            f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.name IS UNIQUE")
    neo4j_client.run(
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Ruling) REQUIRE n.status IS UNIQUE")


def build(csv_name: str = RULINGS_CSV, link: bool = True) -> dict:
    """Build the ingredient layer. Safe to re-run; everything is MERGEd."""
    path = Path(get_settings().data_dir) / csv_name
    if not path.exists():
        raise FileNotFoundError(
            f"{csv_name} not found under DATA_DIR ({get_settings().data_dir}). "
            "This curated file ships with the repo — restore it from git.")
    df = _load(path)
    _constraints()

    stats = build_rulings(df)
    print(f"rulings: {stats['ingredients']} ingredients, {stats['sources']} sources, "
          f"{stats['facts']} ruling facts, {stats['typical']} source defaults")

    if link:
        linked = link_products(df)
        stats.update(linked)
        print(f"products: scanned {linked['products_scanned']:,}, "
              f"linked {linked['products_linked']:,} "
              f"via {linked['contains_edges']:,} CONTAINS edges")

    # The vanilla baseline must be able to retrieve these nodes too, or extension C
    # would win on corpus size instead of on reasoning structure.
    from thaigraphrag.kg.build_kg import index_labels
    stats["indexed"] = index_labels(["Ingredient", "Source", "Ruling"])
    print(f"indexed {stats['indexed']} ingredient-layer nodes into Qdrant")
    entity_linking.clear_cache()
    return stats


def summary() -> dict:
    """Counts used by the UI's ingredient page and /api/stats."""
    rows = neo4j_client.run(
        """
        MATCH (i:Ingredient)-[h:HAS_RULING]->(r:Ruling)
        RETURN r.status AS status, count(DISTINCT i) AS ingredients, count(h) AS facts
        ORDER BY facts DESC
        """)
    totals = neo4j_client.run(
        """
        OPTIONAL MATCH (i:Ingredient) WITH count(i) AS ingredients
        OPTIONAL MATCH (s:Source) WITH ingredients, count(s) AS sources
        OPTIONAL MATCH ()-[c:CONTAINS]->() RETURN ingredients, sources, count(c) AS contains
        """)
    return {"by_ruling": rows, **(totals[0] if totals else {})}


if __name__ == "__main__":
    build()
