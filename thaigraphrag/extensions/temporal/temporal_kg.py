"""B — add validity intervals to KG relations.

Two temporal layers are built, and they carry very different evidential weight —
the README reports them separately for that reason.

1. **Certification validity (real data, ~222k rows).**
   `product_processed.csv` has a 100%-populated `expire_date` in Thai Buddhist-Era
   `DD/MM/YYYY`. Every certified product gets

       (:Product)-[:CERTIFIED_HALAL {valid_from, valid_to}]->(:Certifier)

   `valid_to` is the real expiry year. `valid_from` is left **NULL on purpose**: the
   registry does not record an issue date, and the trailing digits of `halal_code`
   do not encode one reliably (checked: implied term ranges from -29 to +4 years).
   A NULL `valid_from` means "unbounded start", which the time filter handles.

   This is what makes the experiment real: at as_of 2570 roughly half the registry
   has already expired, so a time-agnostic retriever confidently reports products as
   currently certified when they are not.

2. **Regulation timeline (small curated set).**
   `data/halal/regulation_timeline.csv` — five well-established Thai halal
   governance facts with SUPERSEDES/ISSUED_BY edges. Deliberately small and limited
   to facts that are not in dispute; it supports the qualitative "ณ ปีใด" questions
   rather than the headline number.

Run:  python -m scripts.build_temporal_kg
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from thaigraphrag.config import get_settings
from thaigraphrag.core import entity_linking, neo4j_client

TIMELINE_CSV = "halal/regulation_timeline.csv"
CERTIFIER = "คณะกรรมการกลางอิสลามแห่งประเทศไทย"

_DATE = re.compile(r"^\s*(\d{1,2})/(\d{1,2})/(\d{4})\s*$")


def parse_be_year(value: str) -> int | None:
    """Year from a Thai Buddhist-Era `DD/MM/YYYY` string, or None.

    Values outside 2500–2600 are treated as data errors (the registry contains one
    row with expiry 2655) rather than propagated into the graph.
    """
    m = _DATE.match(str(value or ""))
    if not m:
        return None
    year = int(m.group(3))
    return year if 2500 <= year <= 2600 else None


def annotate_certifications(batch: int = 2000) -> dict:
    """Attach CERTIFIED_HALAL edges with validity to every Product already in the graph.

    Reads `expire_date` from the CSV and joins on `halal_code`, so only the products
    sampled into the KG are annotated — the graph and the vector index stay in sync.
    """
    s = get_settings()
    path = Path(s.data_dir) / "product_processed.csv"
    if not path.exists():
        raise FileNotFoundError(f"product_processed.csv not found in {s.data_dir}")

    codes = {r["code"] for r in neo4j_client.run(
        "MATCH (p:Product) RETURN p.halal_code AS code") if r["code"]}
    if not codes:
        return {"products": 0, "annotated": 0,
                "note": "no Product nodes — run scripts.build_kg first"}

    df = pd.read_csv(path, low_memory=False, dtype=str,
                     usecols=["halal_code", "expire_date"]).fillna("")
    df["halal_code"] = df["halal_code"].str.strip()
    df = df[df["halal_code"].isin(codes)].drop_duplicates("halal_code")

    neo4j_client.run(
        """
        MERGE (c:Certifier {name: $name})
          SET c.text = 'หน่วยงานผู้ให้การรับรองฮาลาล: ' + $name
        """, name=CERTIFIER)

    rows, unparsed = [], 0
    for _, r in df.iterrows():
        year = parse_be_year(r["expire_date"])
        if year is None:
            unparsed += 1
            continue
        rows.append({"code": r["halal_code"], "valid_to": year,
                     "expire_date": r["expire_date"].strip()})

    for i in range(0, len(rows), batch):
        neo4j_client.run(
            """
            UNWIND $rows AS row
            MATCH (p:Product {halal_code: row.code})
            MATCH (c:Certifier {name: $name})
            MERGE (p)-[e:CERTIFIED_HALAL]->(c)
              SET e.valid_to = row.valid_to,
                  e.valid_from = NULL,
                  e.expire_date = row.expire_date
            SET p.cert_valid_to = row.valid_to
            """,
            rows=rows[i:i + batch], name=CERTIFIER,
        )

    by_year = {r["y"]: r["c"] for r in neo4j_client.run(
        """
        MATCH ()-[e:CERTIFIED_HALAL]->()
        RETURN e.valid_to AS y, count(*) AS c ORDER BY y
        """)}
    return {"products": len(codes), "annotated": len(rows),
            "unparsed_dates": unparsed, "by_expiry_year": by_year}


def build_timeline(csv_name: str = TIMELINE_CSV) -> dict:
    """Load the curated regulation timeline as time-bounded nodes and edges."""
    path = Path(get_settings().data_dir) / csv_name
    if not path.exists():
        raise FileNotFoundError(f"{csv_name} not found under DATA_DIR")
    df = pd.read_csv(path).fillna("")

    neo4j_client.run(
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Regulation) REQUIRE n.reg_id IS UNIQUE")
    neo4j_client.run(
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Organisation) REQUIRE n.name IS UNIQUE")
    neo4j_client.run(
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Certifier) REQUIRE n.name IS UNIQUE")

    regs = df[df["kind"] != "organisation"]
    orgs = df[df["kind"] == "organisation"]

    neo4j_client.run(
        """
        UNWIND $rows AS row
        MERGE (r:Regulation {reg_id: row.reg_id})
          SET r.name = row.name_th, r.name_en = row.name_en, r.kind = row.kind,
              r.valid_from = row.valid_from, r.valid_to = row.valid_to,
              r.topic = row.topic, r.note = row.note,
              r.text = row.name_th + ' — บังคับใช้ตั้งแต่ พ.ศ. ' + toString(row.valid_from)
                     + CASE WHEN row.valid_to IS NULL THEN ' ถึงปัจจุบัน'
                            ELSE ' ถึง พ.ศ. ' + toString(row.valid_to) END
                     + ' | ' + row.topic
        """,
        rows=[{"reg_id": r["reg_id"], "name_th": r["name_th"], "name_en": r["name_en"],
               "kind": r["kind"], "valid_from": int(r["valid_from"]),
               "valid_to": int(r["valid_to"]) if str(r["valid_to"]).strip() else None,
               "topic": r["topic_th"], "note": r["note_th"]}
              for _, r in regs.iterrows()],
    )
    neo4j_client.run(
        """
        UNWIND $rows AS row
        MERGE (o:Organisation {name: row.name_th})
          SET o.name_en = row.name_en, o.valid_from = row.valid_from,
              o.text = row.name_th + ' — ก่อตั้ง พ.ศ. ' + toString(row.valid_from)
                     + ' | ' + row.topic
        """,
        rows=[{"name_th": r["name_th"], "name_en": r["name_en"],
               "valid_from": int(r["valid_from"]), "topic": r["topic_th"]}
              for _, r in orgs.iterrows()],
    )

    # Issuer + supersession edges carry the same validity as their regulation, so a
    # time filter on relations alone is enough to answer "ณ ปี X ใครออกระเบียบนี้".
    issued = [{"reg_id": r["reg_id"], "issuer": r["issuer_th"],
               "valid_from": int(r["valid_from"]),
               "valid_to": int(r["valid_to"]) if str(r["valid_to"]).strip() else None}
              for _, r in regs.iterrows() if str(r["issuer_th"]).strip()]
    neo4j_client.run(
        """
        UNWIND $rows AS row
        MATCH (r:Regulation {reg_id: row.reg_id})
        MERGE (o:Organisation {name: row.issuer})
          SET o.text = coalesce(o.text, 'หน่วยงาน: ' + row.issuer)
        MERGE (r)-[e:ISSUED_BY]->(o)
          SET e.valid_from = row.valid_from, e.valid_to = row.valid_to
        """,
        rows=issued,
    )
    sup = [{"reg_id": r["reg_id"], "old": r["supersedes"],
            "valid_from": int(r["valid_from"])}
           for _, r in regs.iterrows() if str(r["supersedes"]).strip()]
    if sup:
        neo4j_client.run(
            """
            UNWIND $rows AS row
            MATCH (new:Regulation {reg_id: row.reg_id})
            MATCH (old:Regulation {reg_id: row.old})
            MERGE (new)-[e:SUPERSEDES]->(old)
              SET e.valid_from = row.valid_from
            """,
            rows=sup,
        )
    return {"regulations": len(regs), "organisations": len(orgs),
            "issued_by": len(issued), "supersedes": len(sup)}


def annotate_relation(rel_type: str, key: str, valid_from: str, valid_to: str = "") -> None:
    """Set validity on relations of a given type (kept from the seed API)."""
    neo4j_client.run(
        f"""
        MATCH ()-[r:{rel_type} {{key: $key}}]-()
        SET r.valid_from = $vf, r.valid_to = $vt
        """,
        key=key, vf=valid_from, vt=valid_to or None,
    )


def coverage() -> dict:
    """How much of the graph is time-annotated — shown on the UI's Temporal page."""
    rows = neo4j_client.run(
        """
        MATCH ()-[r]->()
        WITH type(r) AS rel, count(*) AS total,
             sum(CASE WHEN r.valid_from IS NOT NULL OR r.valid_to IS NOT NULL
                      THEN 1 ELSE 0 END) AS timed
        WHERE timed > 0
        RETURN rel, total, timed ORDER BY timed DESC
        """)
    years = neo4j_client.run(
        """
        MATCH ()-[e:CERTIFIED_HALAL]->()
        RETURN e.valid_to AS year, count(*) AS n ORDER BY year
        """)
    return {"by_relation": rows, "cert_expiry_years": years}


def build() -> dict:
    stats = {"timeline": build_timeline(), "certifications": annotate_certifications()}
    print(f"timeline: {stats['timeline']}")
    print(f"certifications: annotated {stats['certifications']['annotated']:,} products "
          f"| expiry years {stats['certifications'].get('by_expiry_year')}")

    # Same fairness rule as extension C: the vanilla baseline must be able to retrieve
    # the timeline nodes this layer introduces.
    from thaigraphrag.kg.build_kg import index_labels
    stats["indexed"] = index_labels(["Regulation", "Organisation", "Certifier"])
    print(f"indexed {stats['indexed']} temporal-layer nodes into Qdrant")
    entity_linking.clear_cache()
    return stats


if __name__ == "__main__":
    build()
