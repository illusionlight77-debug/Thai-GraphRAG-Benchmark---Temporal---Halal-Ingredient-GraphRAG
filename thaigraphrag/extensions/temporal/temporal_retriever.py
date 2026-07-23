"""B — Temporal GraphRAG retriever.

Subclasses the core `GraphRAGRetriever` and restricts every expansion to relations
valid at `as_of` (a Thai Buddhist-Era year, e.g. 2570).

The filter is the standard interval predicate, with NULL meaning unbounded:

    (r.valid_from IS NULL OR r.valid_from <= as_of) AND
    (r.valid_to   IS NULL OR r.valid_to   >= as_of)

Relations that carry no validity at all — LOCATED_IN, BELONGS_TO … — are timeless and
always pass. Only the annotated ones (CERTIFIED_HALAL, ISSUED_BY, SUPERSEDES) can be
excluded, so switching this retriever on never *loses* untimed facts; it only removes
statements that were not true in the requested year.

`as_of` is taken from the question's `as_of` field by the benchmark, or parsed out of
the question text ("ณ ปี 2570 …") when the retriever is used interactively.

Nothing in core is modified: this plugs in through `retrievers.get_retriever`
(CLAUDE.md §2).
"""
from __future__ import annotations

import re
import time

from thaigraphrag.core import neo4j_client
from thaigraphrag.kg.schema import LABEL_TH
from thaigraphrag.retrievers.base import RetrievedContext
from thaigraphrag.retrievers.graphrag import (
    _SAMPLES_PER_GROUP, GraphRAGRetriever, _fmt_group, th_label,
)

# Buddhist-Era years in the range the data actually covers.
_YEAR = re.compile(r"\b(25\d{2})\b")

_TIME_FILTER = """
    ((r.valid_from IS NULL OR r.valid_from <= $as_of) AND
     (r.valid_to   IS NULL OR r.valid_to   >= $as_of))
"""


def parse_as_of(text: str) -> int | None:
    """Pull a Buddhist-Era year out of a Thai question, if it names one."""
    m = _YEAR.search(text or "")
    return int(m.group(1)) if m else None


def _neighbour_summary_at(node_ids: list[str], as_of: int) -> list[dict]:
    """Aggregated 1-hop neighbourhood, restricted to relations valid at `as_of`."""
    if not node_ids:
        return []
    return neo4j_client.run(
        f"""
        MATCH (s) WHERE elementId(s) IN $ids
        MATCH (s)-[r]-(m)
        WHERE {_TIME_FILTER}
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
        ids=node_ids, as_of=as_of, samples=_SAMPLES_PER_GROUP,
    )


def _expired_at(node_ids: list[str], as_of: int) -> list[dict]:
    """Certifications that exist in the graph but were NOT valid at `as_of`.

    Stating the expiry explicitly is what turns "no fact found" into "this lapsed in
    2569" — a time-agnostic retriever cannot distinguish the two, and that confusion
    is exactly what produces wrong-era answers.
    """
    if not node_ids:
        return []
    return neo4j_client.run(
        """
        MATCH (p) WHERE elementId(p) IN $ids
        MATCH (p)-[e:CERTIFIED_HALAL]->(c)
        WHERE e.valid_to IS NOT NULL AND e.valid_to < $as_of
        RETURN coalesce(p.name, p.halal_code) AS name, p.halal_code AS code,
               e.valid_to AS valid_to, e.expire_date AS expire_date, c.name AS certifier
        LIMIT 20
        """,
        ids=node_ids, as_of=as_of,
    )


def _timed_facts(as_of: int) -> list[dict]:
    """Regulations/organisations in force at `as_of` — the timeline layer."""
    return neo4j_client.run(
        """
        MATCH (r:Regulation)
        WHERE (r.valid_from IS NULL OR r.valid_from <= $as_of)
          AND (r.valid_to   IS NULL OR r.valid_to   >= $as_of)
        OPTIONAL MATCH (r)-[:ISSUED_BY]->(o:Organisation)
        RETURN r.name AS name, r.kind AS kind, r.valid_from AS valid_from,
               r.valid_to AS valid_to, r.topic AS topic, o.name AS issuer
        ORDER BY r.valid_from
        LIMIT 20
        """,
        as_of=as_of,
    )


class TemporalGraphRAGRetriever(GraphRAGRetriever):
    name = "temporal"

    def __init__(self, as_of: int | str | None = None, **kwargs):
        super().__init__(**kwargs)
        self.as_of = int(as_of) if as_of not in (None, "") else None

    def _as_of_for(self, query: str) -> int | None:
        return self.as_of if self.as_of is not None else parse_as_of(query)

    def retrieve(self, query: str) -> RetrievedContext:
        as_of = self._as_of_for(query)
        if as_of is None:
            # No year anywhere → behave exactly like core GraphRAG, and say so in meta
            # so the benchmark can separate "no temporal signal" from "filtered".
            ctx = super().retrieve(query)
            ctx.meta.update({"strategy": self.name, "as_of": None, "time_filtered": False})
            return ctx

        t0 = time.time()
        seeds = self._link(query, limit=5)
        if not seeds:
            return RetrievedContext(
                text="", provenance=[],
                meta={"strategy": self.name, "as_of": as_of, "time_filtered": True,
                      "seeds": 0, "triples": 0, "expired": 0,
                      "latency_s": round(time.time() - t0, 3), "llm_calls": 0})

        seed_ids = [x["node_id"] for x in seeds]
        facts: list[str] = []

        rows = neo4j_client.run(
            """
            MATCH (n) WHERE elementId(n) IN $ids
            RETURN labels(n)[0] AS label, coalesce(n.text, n.name, '') AS text
            """, ids=seed_ids)
        lines = [f"- [{th_label(r['label'])}] {r['text']}" for r in rows if r["text"]]
        if lines:
            facts.append("ข้อมูลเอนทิตีที่ตรงกับคำถาม:\n" + "\n".join(lines))

        groups = _neighbour_summary_at(seed_ids, as_of)
        per_node: dict[str, list[dict]] = {}
        for g in groups:
            per_node.setdefault(g["node_id"], []).append(g)
        hop1 = [_fmt_group(r) for nid in seed_ids for r in per_node.get(nid, [])[:10]]
        if hop1:
            facts.append(f"ข้อเท็จจริงที่ยังมีผล ณ ปี พ.ศ. {as_of}:\n"
                         + "\n".join(f"- {x}" for x in hop1))

        expired = _expired_at(seed_ids, as_of)
        if expired:
            elines = [f"- {e['name']} (รหัส {e['code']}) การรับรองสิ้นสุด "
                      f"{e['expire_date']} คือ พ.ศ. {e['valid_to']} "
                      f"ซึ่ง**หมดอายุแล้ว** ณ ปี {as_of}" for e in expired]
            facts.append("การรับรองที่หมดอายุก่อนปีที่ถาม:\n" + "\n".join(elines))

        timeline = _timed_facts(as_of)
        if timeline:
            tlines = [f"- {t['name']} (บังคับใช้ตั้งแต่ พ.ศ. {t['valid_from']}"
                      + (f" ถึง {t['valid_to']}" if t["valid_to"] else " ถึงปัจจุบัน")
                      + (f", ออกโดย {t['issuer']}" if t.get("issuer") else "") + ")"
                      for t in timeline]
            facts.append(f"ระเบียบ/กฎหมายที่บังคับใช้ ณ ปี พ.ศ. {as_of}:\n" + "\n".join(tlines))

        seed_txt = ", ".join(f"{x['name']} [{LABEL_TH.get(x['label'], x['label'])}]"
                             for x in seeds)
        text = (f"คำถามนี้ถามถึงสถานะ ณ ปี พ.ศ. {as_of} "
                f"— ใช้เฉพาะข้อเท็จจริงที่มีผลบังคับในปีนั้น\n"
                f"เอนทิตีตั้งต้น: {seed_txt}\n\n" + "\n\n".join(facts))

        return RetrievedContext(
            text=text,
            provenance=seed_ids,
            meta={"strategy": self.name, "as_of": as_of, "time_filtered": True,
                  "hops": self.hops, "seeds": len(seeds),
                  "seed_names": [x["name"] for x in seeds],
                  "triples": sum(f.count("\n- ") for f in facts),
                  "expired": len(expired),
                  "in_force": len(timeline),
                  "latency_s": round(time.time() - t0, 3), "llm_calls": 0},
        )
