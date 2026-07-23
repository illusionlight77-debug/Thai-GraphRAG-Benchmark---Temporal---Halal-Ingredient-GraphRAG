"""GraphRAG retriever — the core contribution of the study.

Strategy
--------
  1. entity-link the query to seed nodes in Neo4j,
  2. build a **seed card** per seed: the node's own text plus an *aggregated*
     summary of its typed relations,
  3. expand one more level through attribute nodes (Province, Category, Cuisine …),
  4. find **bridge paths** between seed pairs — the explicit multi-hop evidence,
  5. linearise everything into concise Thai facts for the LLM.

Why aggregation instead of raw path enumeration
-----------------------------------------------
The seed version ran `MATCH path = (s)-[*1..k]-(m) … LIMIT 200`. When a seed is a
hub — จังหวัดกรุงเทพมหานคร has tens of thousands of neighbours — that enumerates
millions of paths before the LIMIT applies, and the 200 facts that survive are an
arbitrary slice. Here every expansion is aggregated at the database
(`count` + a small `collect` sample), so a hub costs one row per relation type and
the context stays both bounded and representative. This follows the node/community
summary idea from NodeRAG and LightRAG (see docs/REFERENCES.md).

Note: no vector search anywhere in this path. Retrieval here is graph-only, which
is what makes the contrast with the vanilla baseline a clean single-variable
comparison (CLAUDE.md §2).
"""
from __future__ import annotations

import time
from itertools import combinations

from thaigraphrag.config import get_settings
from thaigraphrag.core import entity_linking, neo4j_client
from thaigraphrag.kg.schema import LABEL_TH, REL_TH
from thaigraphrag.retrievers.base import RetrievedContext, Retriever

# Per-seed / per-level caps that keep the LLM context bounded.
_REL_TYPES_PER_NODE = 10     # distinct (rel, target-label) groups per node
_SAMPLES_PER_GROUP = 5       # example target names shown per group
_EXPAND_ATTRS = 4            # attribute neighbours expanded to a second level
_BRIDGE_PAIRS = 6            # seed pairs searched for a connecting path


def th_label(label: str) -> str:
    return LABEL_TH.get(label, label)


def th_rel(rel: str) -> str:
    return REL_TH.get(rel, rel)


def _neighbour_summary(node_ids: list[str]) -> list[dict]:
    """Aggregated 1-hop neighbourhood for each node.

    One row per (node, relation type, neighbour label) with a count and a few
    example names — never one row per neighbour.
    """
    if not node_ids:
        return []
    return neo4j_client.run(
        """
        MATCH (s) WHERE elementId(s) IN $ids
        MATCH (s)-[r]-(m)
        WITH s, type(r) AS rel, labels(m)[0] AS mlabel,
             startNode(r) = s AS outgoing,
             count(DISTINCT m) AS n,
             collect(DISTINCT coalesce(m.name, m.text))[0..$samples] AS samples,
             collect(DISTINCT elementId(m))[0..$samples] AS sample_ids
        RETURN elementId(s) AS node_id, coalesce(s.name, labels(s)[0]) AS node_name,
               labels(s)[0] AS node_label,
               rel, mlabel, outgoing, n, samples, sample_ids
        ORDER BY node_id, n DESC
        """,
        ids=node_ids, samples=_SAMPLES_PER_GROUP,
    )


def _node_texts(node_ids: list[str]) -> list[dict]:
    if not node_ids:
        return []
    return neo4j_client.run(
        """
        MATCH (n) WHERE elementId(n) IN $ids
        RETURN elementId(n) AS node_id, labels(n)[0] AS label,
               coalesce(n.name, '') AS name, coalesce(n.text, '') AS text,
               coalesce(n.province, '') AS province
        """,
        ids=node_ids,
    )


def _bridges(seed_ids: list[str], max_hops: int) -> list[dict]:
    """Shortest connecting path between each seed pair — the multi-hop evidence.

    `shortestPath` is index-backed and bidirectional, so this stays cheap even
    when both endpoints are hubs.
    """
    pairs = list(combinations(seed_ids, 2))[:_BRIDGE_PAIRS]
    if not pairs:
        return []
    return neo4j_client.run(
        f"""
        UNWIND $pairs AS pair
        MATCH (a) WHERE elementId(a) = pair[0]
        MATCH (b) WHERE elementId(b) = pair[1]
        MATCH p = shortestPath((a)-[*1..{max_hops}]-(b))
        RETURN [n IN nodes(p) | coalesce(n.name, labels(n)[0])] AS names,
               [n IN nodes(p) | labels(n)[0]] AS labels,
               [r IN relationships(p) | type(r)] AS rels,
               length(p) AS hops
        """,
        pairs=[list(p) for p in pairs],
    )


def _fmt_group(row: dict) -> str:
    """One aggregated relation group → one Thai fact line."""
    rel, mlabel, n = th_rel(row["rel"]), th_label(row["mlabel"]), row["n"]
    samples = [s for s in (row.get("samples") or []) if s]
    arrow = "→" if row.get("outgoing") else "←"
    head = f"({row['node_name']}) {arrow}[{rel}]→ {mlabel}"
    if n == 1 and samples:
        return f"{head}: {samples[0]}"
    shown = ", ".join(str(s)[:60] for s in samples[:_SAMPLES_PER_GROUP])
    more = f" (จาก {n:,} รายการ)" if n > len(samples) else ""
    return f"{head} จำนวน {n:,}: {shown}{more}"


class GraphRAGRetriever(Retriever):
    name = "graphrag"

    def __init__(self, hops: int | None = None, max_triples: int | None = None):
        s = get_settings()
        self.hops = hops if hops is not None else s.graph_hops
        self.max_triples = max_triples if max_triples is not None else s.max_triples

    # -- hooks the Temporal extension overrides ----------------------------
    def _link(self, query: str, limit: int) -> list[dict]:
        return entity_linking.link_entities(query, limit=limit)

    def _empty_meta(self, t0: float) -> dict:
        return {"strategy": self.name, "hops": self.hops, "seeds": 0, "triples": 0,
                "latency_s": round(time.time() - t0, 3), "llm_calls": 0}

    def retrieve(self, query: str) -> RetrievedContext:
        s = get_settings()
        t0 = time.time()

        seeds = self._link(query, limit=s.top_k)
        if not seeds:
            return RetrievedContext(text="", provenance=[], meta=self._empty_meta(t0))

        seed_ids = [x["node_id"] for x in seeds]
        facts: list[str] = []
        provenance: list[str] = list(seed_ids)

        # 1) seed descriptions — carries single-hop questions.
        lines = []
        for row in _node_texts(seed_ids):
            desc = row["text"] or row["name"]
            if desc:
                lines.append(f"- [{th_label(row['label'])}] {desc}")
        if lines:
            facts.append("ข้อมูลเอนทิตีที่ตรงกับคำถาม:\n" + "\n".join(lines))

        # 2) aggregated 1-hop neighbourhood of every seed.
        groups = _neighbour_summary(seed_ids)
        per_node: dict[str, list[dict]] = {}
        for g in groups:
            per_node.setdefault(g["node_id"], []).append(g)

        hop1 = []
        expand_ids: list[str] = []
        for nid in seed_ids:
            rows = per_node.get(nid, [])[:_REL_TYPES_PER_NODE]
            for r in rows:
                hop1.append(_fmt_group(r))
                # Queue attribute neighbours (จังหวัด, หมวดหมู่ …) for level 2: they are
                # the join points that turn two unrelated places into a 2-hop path.
                if r["n"] <= 3 and r["mlabel"] not in ("Hotel", "Restaurant", "Store",
                                                       "Attraction", "Mosque", "Product"):
                    expand_ids.extend(r.get("sample_ids") or [])
        if hop1:
            facts.append("ความสัมพันธ์รอบเอนทิตี (1 hop):\n" + "\n".join(f"- {x}" for x in hop1))

        # 3) second level through the attribute join points.
        if self.hops >= 2 and expand_ids:
            expand_ids = [i for i in dict.fromkeys(expand_ids) if i not in seed_ids][:_EXPAND_ATTRS]
            provenance.extend(expand_ids)
            hop2 = []
            per_node2: dict[str, list[dict]] = {}
            for g in _neighbour_summary(expand_ids):
                per_node2.setdefault(g["node_id"], []).append(g)
            for nid in expand_ids:
                for r in per_node2.get(nid, [])[:_REL_TYPES_PER_NODE]:
                    hop2.append(_fmt_group(r))
            if hop2:
                facts.append("ความสัมพันธ์ขยายอีกขั้น (2 hops):\n"
                             + "\n".join(f"- {x}" for x in hop2))

        # 4) explicit bridge paths between seeds.
        bridges = _bridges(seed_ids, max_hops=max(2, self.hops * 2))
        if bridges:
            blines = []
            for b in bridges:
                names, rels = b["names"], b["rels"]
                chain = names[0]
                for rel, nxt in zip(rels, names[1:]):
                    chain += f" -[{th_rel(rel)}]-> {nxt}"
                blines.append(f"- ({b['hops']} hop) {chain}")
            facts.append("เส้นทางเชื่อมระหว่างเอนทิตี:\n" + "\n".join(blines))

        seed_txt = ", ".join(f"{x['name']} [{th_label(x['label'])}]" for x in seeds)
        text = f"เอนทิตีตั้งต้น: {seed_txt}\n\n" + "\n\n".join(facts)

        # Bound the context: the fair-comparison rule fixes the prompt, not its length,
        # but an unbounded graph dump would beat vanilla on budget rather than on structure.
        n_facts = sum(f.count("\n- ") for f in facts)
        if n_facts > self.max_triples:
            kept, total = [], 0
            for block in facts:
                head, *items = block.split("\n- ")
                room = max(0, self.max_triples - total)
                if room == 0:
                    break
                kept.append(head + ("\n- " + "\n- ".join(items[:room]) if items else ""))
                total += min(len(items), room)
            text = f"เอนทิตีตั้งต้น: {seed_txt}\n\n" + "\n\n".join(kept)
            n_facts = total

        return RetrievedContext(
            text=text,
            provenance=provenance,
            meta={"strategy": self.name, "hops": self.hops,
                  "seeds": len(seeds), "triples": n_facts,
                  "bridges": len(bridges),
                  "seed_names": [x["name"] for x in seeds],
                  "latency_s": round(time.time() - t0, 3), "llm_calls": 0},
        )
