"""C — Halal-Ingredient explainable retriever.

Returns the **ingredient → source → ruling** path so every verdict can be traced
hop by hop, instead of an unexplained yes/no.

The ruling is a property of the (ingredient, source) *pair*, so the retriever never
reports a bare verdict: it enumerates every source an ingredient can come from,
each with its own ruling and the basis for it. When the sources disagree —
เจลาตินจากหมู vs เจลาตินจากปลา — that disagreement is the answer, and the context
says so explicitly rather than picking one.

A query naming a product resolves through `(:Product)-[:CONTAINS]->(:Ingredient)`
first, so "ผลิตภัณฑ์นี้มีส่วนผสมที่คลุมเครือไหม" is answerable end to end.

`meta["paths"]` carries the structured chains that the UI draws and that
`metrics.path_validity` scores against each question's `gold_path`.
"""
from __future__ import annotations

import time

from thaigraphrag.core import entity_linking, neo4j_client
from thaigraphrag.extensions.halal_ingredient.ingredient_kg import RULING_ORDER, RULING_TH
from thaigraphrag.retrievers.base import RetrievedContext, Retriever

_MAX_INGREDIENTS = 8
_MAX_PRODUCTS = 5


def _ruling_paths(node_ids: list[str]) -> list[dict]:
    """Every (ingredient, source, ruling) chain reachable from the seed nodes.

    Seeds may be Ingredient nodes directly, or Product nodes whose CONTAINS edges
    lead to them. `via_source` joins the verdict edge back to the provenance edge.
    """
    if not node_ids:
        return []
    return neo4j_client.run(
        """
        MATCH (seed) WHERE elementId(seed) IN $ids
        OPTIONAL MATCH (seed)-[:CONTAINS]->(pi:Ingredient)
        WITH collect(DISTINCT seed) + collect(DISTINCT pi) AS cands
        UNWIND cands AS n
        WITH n WHERE n:Ingredient
        MATCH (n)-[h:HAS_RULING]->(r:Ruling)
        MATCH (n)-[:DERIVED_FROM]->(s:Source {name: h.via_source})
        RETURN DISTINCT n.name AS ingredient, n.name_en AS ingredient_en,
               n.e_number AS e_number,
               s.name AS source, s.source_type AS source_type,
               r.status AS ruling, h.basis AS basis, h.note AS note
        ORDER BY ingredient, ruling
        LIMIT 60
        """,
        ids=node_ids,
    )


def _products_for(node_ids: list[str]) -> list[dict]:
    """Real certified products that contain the seed ingredients."""
    if not node_ids:
        return []
    return neo4j_client.run(
        """
        MATCH (n) WHERE elementId(n) IN $ids AND n:Ingredient
        MATCH (p:Product)-[c:CONTAINS]->(n)
        RETURN n.name AS ingredient, p.name AS product, p.halal_code AS halal_code,
               c.matched_term AS matched_term
        LIMIT $limit
        """,
        ids=node_ids, limit=_MAX_PRODUCTS * 2,
    )


def _verdict(rulings: list[str]) -> str:
    """Worst-case aggregation: one haram source makes the whole question haram-risky."""
    if not rulings:
        return ""
    return sorted(rulings, key=lambda r: RULING_ORDER.get(r, 9))[0]


class IngredientExplainRetriever(Retriever):
    name = "halal_ingredient"

    def retrieve(self, query: str) -> RetrievedContext:
        t0 = time.time()
        seeds = entity_linking.link_entities(query, limit=6)
        seed_ids = [x["node_id"] for x in seeds]

        rows = _ruling_paths(seed_ids)
        if not rows:
            return RetrievedContext(
                text="", provenance=seed_ids,
                meta={"strategy": self.name, "paths": [], "n_paths": 0,
                      "seeds": len(seeds), "complete_paths": 0,
                      "latency_s": round(time.time() - t0, 3), "llm_calls": 0})

        # Group by ingredient so conflicting sources are presented together.
        by_ing: dict[str, list[dict]] = {}
        for r in rows:
            by_ing.setdefault(r["ingredient"], []).append(r)

        blocks, paths = [], []
        for ing, items in list(by_ing.items())[:_MAX_INGREDIENTS]:
            lines = []
            for it in items:
                th = RULING_TH.get(it["ruling"], it["ruling"])
                chain = f"{ing} → {it['source']} → {th}"
                lines.append(f"  - {chain}\n    เหตุผล: {it['basis']}"
                             + (f"\n    หมายเหตุ: {it['note']}" if it.get("note") else ""))
                paths.append({
                    "ingredient": ing, "source": it["source"],
                    "source_type": it["source_type"], "ruling": it["ruling"],
                    "ruling_th": th, "basis": it["basis"], "note": it.get("note") or "",
                    "chain": [ing, it["source"], th],
                })
            verdict = _verdict([i["ruling"] for i in items])
            head = f"ส่วนผสม: {ing}"
            if items[0].get("e_number"):
                head += f" (รหัส {items[0]['e_number']})"
            if len({i["ruling"] for i in items}) > 1:
                head += ("  ⚠ สถานะขึ้นกับแหล่งที่มา — ต้องระบุแหล่งที่มาจึงจะตัดสินได้ "
                         f"(กรณีแย่ที่สุด: {RULING_TH.get(verdict, verdict)})")
            else:
                head += f"  → {RULING_TH.get(verdict, verdict)}"
            blocks.append(head + "\n" + "\n".join(lines))

        text = "เส้นทางคำวินิจฉัยส่วนผสม (ส่วนผสม → แหล่งที่มา → คำวินิจฉัย):\n\n" + "\n\n".join(blocks)

        products = _products_for(seed_ids)
        if products:
            plines = [f"- {p['product']} (รหัสฮาลาล {p['halal_code']}) — พบคำว่า '{p['matched_term']}'"
                      for p in products[:_MAX_PRODUCTS]]
            text += "\n\nสินค้าที่ได้รับรองฮาลาลซึ่งมีส่วนผสมนี้:\n" + "\n".join(plines)

        return RetrievedContext(
            text=text,
            provenance=seed_ids,
            meta={"strategy": self.name,
                  "seeds": len(seeds),
                  "seed_names": [x["name"] for x in seeds],
                  "paths": paths,
                  "n_paths": len(paths),
                  # A path counts as complete only when all three hops are present.
                  "complete_paths": sum(1 for p in paths if all(p["chain"])),
                  "ingredients": list(by_ing)[:_MAX_INGREDIENTS],
                  "products": products[:_MAX_PRODUCTS],
                  "latency_s": round(time.time() - t0, 3), "llm_calls": 0},
        )
