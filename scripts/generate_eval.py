"""Generate the Thai multi-hop eval set **from the built graph**.

Why generate rather than hand-write
-----------------------------------
The brief requires data-dependent gold answers to be verified against the real KG.
Deriving each question from a Cypher query that already knows the answer makes the
gold correct *by construction*, and re-running this script after a rebuild keeps the
set honest instead of silently stale.

Every generated item records `gold_nodes` (the entities a correct answer must pass
through → retrieval hit@k) and a `provenance` note naming the Cypher pattern and the
province tier the fact rests on, so any item can be audited.

Hop types follow docs/METHODOLOGY.md:
    single      answerable from one node's own properties
    multi       needs ≥2 nodes joined by relations
    relational  the answer *is* a relation/attribute reached by traversal

Knowledge-based items (halal definitions, rulings) are hand-written and stable —
they do not depend on the graph and are appended verbatim.

Run:  python -m scripts.generate_eval [--out thai_eval.jsonl] [--target 200]
"""
from __future__ import annotations

import argparse
import json
import random

from thaigraphrag.config import QUESTIONS_DIR
from thaigraphrag.core import neo4j_client
from thaigraphrag.kg import provinces

SEED = 20260723


# ── helpers ─────────────────────────────────────────────────────────────────

def _unique_named(label: str, limit: int, require_province: bool = True) -> list[dict]:
    """Nodes whose name is unique across the whole graph.

    Ambiguous names make a question unanswerable in principle — 'โรงแรมสุขสันต์อยู่
    จังหวัดใด' has no single gold answer if three of them exist — so they are excluded
    rather than resolved arbitrarily.
    """
    return neo4j_client.run(
        f"""
        MATCH (n:{label})
        WHERE n.name IS NOT NULL AND size(n.name) >= 5
          {"AND n.province <> ''" if require_province else ""}
        WITH n.name AS name, collect(n) AS ns
        WHERE size(ns) = 1
        WITH ns[0] AS n
        RETURN n.name AS name, n.province AS province,
               coalesce(n.province_source,'') AS province_source,
               labels(n)[0] AS label
        ORDER BY n.name
        LIMIT $limit
        """,
        limit=limit,
    )


def _mosque_phrase(name: str) -> str:
    """'มัสยิดกรือเซะ' but 'มัสยิด Pakistan Mosque' — Thai runs together, latin needs a space."""
    stripped = name.replace("มัสยิด", "").strip()
    if not stripped:
        return name.strip()
    return f"มัสยิด{stripped}" if "฀" <= stripped[0] <= "๿" else f"มัสยิด {stripped}"


def _item(qid: str, question: str, answer: str, hop: str, *,
          aliases: list[str] | None = None, gold_nodes: list[str] | None = None,
          provenance: str = "") -> dict:
    item = {"id": qid, "question": question, "answer": answer, "hop_type": hop}
    if aliases:
        item["aliases"] = sorted(set(a for a in aliases if a and a != answer))
    if gold_nodes:
        item["gold_nodes"] = gold_nodes
    if provenance:
        item["provenance"] = provenance
    return item


# ── single-hop: one node's own properties ───────────────────────────────────

def gen_single(rng: random.Random, n: int) -> list[dict]:
    out: list[dict] = []
    templates = [
        ("Mosque", "{name} ตั้งอยู่จังหวัดใด"),
        ("Hotel", "ที่พักชื่อ {name} อยู่จังหวัดอะไร"),
        ("Attraction", "สถานที่ท่องเที่ยว {name} อยู่ในจังหวัดใด"),
        ("Restaurant", "ร้านอาหารชื่อ {name} อยู่จังหวัดอะไร"),
        ("Store", "ร้าน {name} ตั้งอยู่ในจังหวัดใด"),
    ]
    per = max(1, n // len(templates))
    for label, tmpl in templates:
        rows = _unique_named(label, limit=per * 8)
        rng.shuffle(rows)
        for r in rows[:per]:
            name = r["name"].strip()
            # Do not repeat the label word when the name already carries it.
            q = tmpl.format(name=_mosque_phrase(name) if label == "Mosque" else name)
            out.append(_item(
                f"s{len(out) + 1:03d}", q, r["province"], "single",
                aliases=[provinces.TH_TO_EN.get(r["province"], "")],
                gold_nodes=[name, r["province"]],
                provenance=f"({label} {{name}})-[:LOCATED_IN]->(Province) "
                           f"| province_source={r['province_source']}"))
    return out


def gen_single_attr(rng: random.Random, n: int) -> list[dict]:
    """Single-node attribute questions (cuisine, store type, tourism type)."""
    out: list[dict] = []
    specs = [
        ("Restaurant", "SERVES_CUISINE", "Cuisine",
         "ร้านอาหาร {name} เสิร์ฟอาหารประเภทใด"),
        ("Store", "IS_TYPE", "StoreType", "ร้าน {name} เป็นร้านประเภทใด"),
        ("Hotel", "IS_TYPE", "TourismType", "ที่พัก {name} เป็นที่พักประเภทใด"),
    ]
    per = max(1, n // len(specs))
    for label, rel, target, tmpl in specs:
        rows = neo4j_client.run(
            f"""
            MATCH (n:{label})-[:{rel}]->(t:{target})
            WHERE n.name IS NOT NULL AND size(n.name) >= 5
            WITH n.name AS name, collect(DISTINCT t.name) AS vals
            WHERE size(vals) = 1
            WITH name, vals[0] AS value, count(*) AS dup
            WHERE dup = 1
            RETURN name, value ORDER BY name LIMIT $limit
            """, limit=per * 8)
        rng.shuffle(rows)
        for r in rows[:per]:
            out.append(_item(
                f"sa{len(out) + 1:03d}", tmpl.format(name=r["name"]), r["value"], "single",
                gold_nodes=[r["name"], r["value"]],
                provenance=f"({label})-[:{rel}]->({target})"))
    return out


# ── multi-hop: ≥2 nodes joined by relations ─────────────────────────────────

def gen_multi(rng: random.Random, n: int) -> list[dict]:
    out: list[dict] = []

    # Place → Province → Region (2 hops)
    rows = neo4j_client.run(
        """
        MATCH (m:Mosque)-[:LOCATED_IN]->(p:Province)-[:IN_REGION]->(r:Region)
        WHERE m.name IS NOT NULL AND size(m.name) >= 6
        WITH m.name AS name, collect(DISTINCT p.name) AS ps, collect(DISTINCT r.name) AS rs
        WHERE size(ps) = 1 AND size(rs) = 1
        RETURN name, ps[0] AS province, rs[0] AS region ORDER BY name LIMIT 400
        """)
    rng.shuffle(rows)
    for r in rows[:n // 3]:
        out.append(_item(
            f"m{len(out) + 1:03d}",
            f"{_mosque_phrase(r['name'])} อยู่ในภาคใดของประเทศไทย",
            f"ภาค{r['region']}", "multi",
            aliases=[r["region"]],
            gold_nodes=[r["name"], r["province"], r["region"]],
            provenance="(Mosque)-[:LOCATED_IN]->(Province)-[:IN_REGION]->(Region)"))

    # Two places joined through a shared province — the classic 2-hop join.
    rows = neo4j_client.run(
        """
        MATCH (m:Mosque)-[:LOCATED_IN]->(p:Province)<-[:LOCATED_IN]-(h:Hotel)
        WHERE m.name IS NOT NULL AND size(m.name) >= 6
        WITH p, m.name AS mosque, count(DISTINCT h) AS hotels
        WHERE hotels >= 3
        RETURN mosque, p.name AS province, hotels ORDER BY mosque LIMIT 400
        """)
    rng.shuffle(rows)
    for r in rows[:n // 3]:
        out.append(_item(
            f"m{len(out) + 1:03d}",
            f"ในจังหวัดเดียวกับ{_mosque_phrase(r['mosque'])} มีที่พักอยู่กี่แห่ง",
            str(r["hotels"]), "multi",
            aliases=[f"{r['hotels']} แห่ง"],
            gold_nodes=[r["mosque"], r["province"]],
            provenance="(Mosque)-[:LOCATED_IN]->(Province)<-[:LOCATED_IN]-(Hotel), "
                       "count(Hotel)"))

    # Same-province yes/no between two named places.
    rows = neo4j_client.run(
        """
        MATCH (m:Mosque)-[:LOCATED_IN]->(p:Province)<-[:LOCATED_IN]-(a:Attraction)
        WHERE m.name IS NOT NULL AND a.name IS NOT NULL
          AND size(m.name) >= 6 AND size(a.name) >= 6
        RETURN m.name AS mosque, a.name AS attraction, p.name AS province
        ORDER BY m.name LIMIT 400
        """)
    rng.shuffle(rows)
    for r in rows[:n - len(out)]:
        out.append(_item(
            f"m{len(out) + 1:03d}",
            f"{_mosque_phrase(r['mosque'])} กับ {r['attraction']} "
            f"อยู่จังหวัดเดียวกันหรือไม่",
            "อยู่จังหวัดเดียวกัน", "multi",
            aliases=["ใช่", "จังหวัดเดียวกัน", r["province"]],
            gold_nodes=[r["mosque"], r["attraction"], r["province"]],
            provenance="(Mosque)-[:LOCATED_IN]->(Province)<-[:LOCATED_IN]-(Attraction)"))
    return out


# ── relational: the answer *is* a relation reached by traversal ─────────────

def gen_relational(rng: random.Random, n: int) -> list[dict]:
    out: list[dict] = []

    # Province → Region: pure relation lookup, no node properties involved.
    names = list(provinces.TH_NAMES)
    rng.shuffle(names)
    for th in names[:n // 3]:
        out.append(_item(
            f"r{len(out) + 1:03d}", f"จังหวัด{th}อยู่ในภาคใด",
            f"ภาค{provinces.TH_TO_REGION[th]}", "relational",
            aliases=[provinces.TH_TO_REGION[th]],
            gold_nodes=[th, provinces.TH_TO_REGION[th]],
            provenance="(Province)-[:IN_REGION]->(Region)"))

    # Region → member provinces (sampled, count is the gold).
    rows = neo4j_client.run(
        """
        MATCH (p:Province)-[:IN_REGION]->(r:Region)
        RETURN r.name AS region, count(p) AS n ORDER BY r.name
        """)
    for r in rows:
        out.append(_item(
            f"r{len(out) + 1:03d}", f"ภาค{r['region']}มีกี่จังหวัด", str(r["n"]),
            "relational", aliases=[f"{r['n']} จังหวัด"],
            gold_nodes=[r["region"]],
            provenance="(Province)-[:IN_REGION]->(Region), count(Province)"))

    # Which cuisine/brand a place relates to, phrased as a relation question.
    rows = neo4j_client.run(
        """
        MATCH (s:Store)-[:HAS_BRAND]->(b:Brand)
        MATCH (s)-[:IS_TYPE]->(t:StoreType)
        WITH b.name AS brand, collect(DISTINCT t.name) AS types, count(s) AS n
        WHERE size(types) = 1 AND n >= 5
        RETURN brand, types[0] AS store_type, n ORDER BY n DESC LIMIT 40
        """)
    rng.shuffle(rows)
    for r in rows[:n // 4]:
        out.append(_item(
            f"r{len(out) + 1:03d}",
            f"ร้านแบรนด์ {r['brand']} จัดอยู่ในประเภทร้านใด", r["store_type"],
            "relational", gold_nodes=[r["brand"], r["store_type"]],
            provenance="(Store)-[:HAS_BRAND]->(Brand), (Store)-[:IS_TYPE]->(StoreType)"))

    # Product → category / company.
    rows = neo4j_client.run(
        """
        MATCH (p:Product)-[:BELONGS_TO]->(c:Category)
        WHERE p.name IS NOT NULL AND size(p.name) >= 8
        WITH p.name AS name, collect(DISTINCT c.name) AS cats
        WHERE size(cats) = 1
        RETURN name, cats[0] AS category ORDER BY name LIMIT 400
        """)
    rng.shuffle(rows)
    for r in rows[:n - len(out)]:
        out.append(_item(
            f"r{len(out) + 1:03d}",
            f"สินค้า \"{r['name'][:60]}\" จัดอยู่ในหมวดหมู่ใด", r["category"],
            "relational", gold_nodes=[r["category"]],
            provenance="(Product)-[:BELONGS_TO]->(Category)"))
    return out


# ── stable, knowledge-based items (no graph dependency) ─────────────────────

KNOWLEDGE = [
    ("ฮาลาลแปลว่าอะไร", "สิ่งที่อนุมัติหรืออนุญาตตามหลักศาสนาอิสลาม", "single",
     ["อนุมัติ", "อนุญาต"]),
    ("หะรอมแปลว่าอะไร", "สิ่งที่ต้องห้ามตามหลักศาสนาอิสลาม", "single", ["ต้องห้าม", "haram"]),
    ("มัชบูฮ์หมายถึงอะไร", "สิ่งที่คลุมเครือ ไม่ชัดเจนว่าฮาลาลหรือหะรอม", "single",
     ["คลุมเครือ", "น่าสงสัย", "mashbooh"]),
    ("หน่วยงานใดมีอำนาจออกเครื่องหมายรับรองฮาลาลของประเทศไทย",
     "คณะกรรมการกลางอิสลามแห่งประเทศไทย", "single", ["สกอท.", "CICOT"]),
    ("เนื้อสุกรมีสถานะฮาลาลอย่างไร", "ไม่ฮาลาล", "single", ["หะรอม", "haram", "ต้องห้าม"]),
    ("สัตว์น้ำเช่นปลามีสถานะฮาลาลอย่างไรตามมัซฮับชาฟิอีย์", "ฮาลาล", "single",
     ["halal", "อนุมัติ"]),
    ("เจลาตินที่ทำจากหมูมีสถานะฮาลาลหรือไม่ เพราะเหตุใด", "ไม่ฮาลาล เพราะมาจากสุกร",
     "multi", ["ไม่ฮาลาล", "หะรอม", "haram"]),
    ("เจลาตินที่ทำจากปลามีสถานะฮาลาลหรือไม่", "ฮาลาล", "multi", ["halal", "อนุมัติ"]),
    ("เอนไซม์เรนเนตที่ผลิตจากจุลินทรีย์มีสถานะฮาลาลหรือไม่", "ฮาลาล", "multi",
     ["halal", "อนุมัติ"]),
    ("เหตุใดส่วนผสมชนิดเดียวกันจึงมีคำวินิจฉัยฮาลาลต่างกันได้",
     "เพราะคำวินิจฉัยขึ้นกับแหล่งที่มาของส่วนผสมนั้น", "relational",
     ["ขึ้นกับแหล่งที่มา", "แหล่งที่มาต่างกัน"]),
    ("แอลกอฮอล์ที่ได้จากการหมักเพื่อผลิตสุรามีสถานะอย่างไร", "ไม่ฮาลาล", "relational",
     ["หะรอม", "haram", "ต้องห้าม"]),
    ("การเชือดสัตว์ตามหลักศาสนาอิสลามมีผลต่อสถานะฮาลาลของเนื้อสัตว์อย่างไร",
     "เนื้อสัตว์จะฮาลาลก็ต่อเมื่อเชือดตามหลักศาสนา", "relational",
     ["ต้องเชือดตามหลักศาสนา", "ถ้าไม่เชือดตามหลักศาสนาจะไม่ฮาลาล"]),
]


def gen_knowledge() -> list[dict]:
    return [_item(f"k{i:03d}", q, a, hop, aliases=al, provenance="knowledge-based (stable)")
            for i, (q, a, hop, al) in enumerate(KNOWLEDGE, 1)]


# ── entry point ─────────────────────────────────────────────────────────────

def build(target: int = 200) -> list[dict]:
    rng = random.Random(SEED)
    knowledge = gen_knowledge()
    budget = target - len(knowledge)
    per = budget // 3

    items = (gen_single(rng, per - per // 3)
             + gen_single_attr(rng, per // 3)
             + gen_multi(rng, per)
             + gen_relational(rng, per)
             + knowledge)

    # Renumber so ids are contiguous and hop-prefixed.
    seen, out = set(), []
    counters = {"single": 0, "multi": 0, "relational": 0}
    for it in items:
        if it["question"] in seen:
            continue
        seen.add(it["question"])
        hop = it["hop_type"]
        counters[hop] += 1
        it["id"] = f"{hop[0]}{counters[hop]:03d}"
        out.append(it)
    return out


# ── B: temporal eval, grounded in real certification expiry dates ───────────

def build_temporal(target: int = 60) -> list[dict]:
    """Time-sensitive questions whose gold answer flips with `as_of`.

    Each certification item carries a `stale_answer` — what a time-agnostic system
    says because the graph does contain a CERTIFIED_HALAL edge — so the benchmark can
    measure the wrong-era rate precisely instead of inferring it from a low F1.
    """
    rng = random.Random(SEED + 1)
    rows = neo4j_client.run(
        """
        MATCH (p:Product)-[e:CERTIFIED_HALAL]->()
        WHERE p.name IS NOT NULL AND size(p.name) >= 8 AND e.valid_to IS NOT NULL
        RETURN p.name AS name, p.halal_code AS code, e.valid_to AS valid_to,
               e.expire_date AS expire_date
        ORDER BY p.halal_code LIMIT 800
        """)
    if not rows:
        print("  ! no CERTIFIED_HALAL edges — run scripts.build_temporal_kg first")
    rng.shuffle(rows)

    out: list[dict] = []
    # Ask each product about a year on both sides of its expiry so the set is balanced
    # between "still valid" and "lapsed" rather than skewed by the registry's shape.
    for r in rows[: target // 2]:
        vt = int(r["valid_to"])
        as_of = vt + 1 if len(out) % 2 == 0 else vt - 1
        expired = as_of > vt
        short = r["name"][:55]
        out.append(_item(
            f"t{len(out) + 1:03d}",
            f"ณ ปี พ.ศ. {as_of} สินค้า \"{short}\" (รหัสฮาลาล {r['code']}) "
            f"ยังได้รับการรับรองฮาลาลอยู่หรือไม่",
            "หมดอายุแล้ว" if expired else "ยังได้รับการรับรอง",
            "multi",
            aliases=["ไม่ได้รับการรับรองแล้ว", "สิ้นสุดแล้ว"] if expired
            else ["ยังรับรองอยู่", "ยังมีผล"],
            gold_nodes=[r["code"]],
            provenance="(Product)-[:CERTIFIED_HALAL {valid_to}]->(Certifier) "
                       f"| expire_date={r['expire_date']}"))
        out[-1]["as_of"] = as_of
        out[-1]["stale_answer"] = "ยังได้รับการรับรอง" if expired else "หมดอายุแล้ว"

    for r in rows[target // 2: target // 2 + target // 4]:
        out.append(_item(
            f"t{len(out) + 1:03d}",
            f"การรับรองฮาลาลของสินค้ารหัส {r['code']} สิ้นสุดในปี พ.ศ. ใด",
            str(r["valid_to"]), "single",
            aliases=[r["expire_date"]], gold_nodes=[r["code"]],
            provenance="(Product)-[:CERTIFIED_HALAL {valid_to}]->(Certifier)"))
        out[-1]["as_of"] = int(r["valid_to"])

    # Regulation timeline — before/after the year each body or standard came into force.
    # Issuer nodes created as a side effect of ISSUED_BY carry no validity of their
    # own, so they are excluded rather than defaulted to some invented year.
    regs = neo4j_client.run(
        """
        MATCH (r) WHERE (r:Regulation OR r:Organisation) AND r.valid_from IS NOT NULL
        RETURN labels(r)[0] AS label, r.name AS name, r.valid_from AS valid_from
        ORDER BY r.valid_from
        """)
    for r in regs:
        vf = int(r["valid_from"])
        for as_of, ans, stale in ((vf + 2, "มีผลบังคับใช้แล้ว", "ยังไม่มีผลบังคับใช้"),
                                  (vf - 2, "ยังไม่มีผลบังคับใช้", "มีผลบังคับใช้แล้ว")):
            out.append(_item(
                f"t{len(out) + 1:03d}",
                f"ณ ปี พ.ศ. {as_of} \"{r['name']}\" มีผลบังคับใช้แล้วหรือยัง",
                ans, "relational",
                aliases=["แล้ว", "ใช่"] if "แล้ว" in ans else ["ยัง", "ไม่"],
                gold_nodes=[r["name"]],
                provenance=f"({r['label']} {{valid_from}}) | valid_from={vf}"))
            out[-1]["as_of"] = as_of
            out[-1]["stale_answer"] = stale
    return out


# ── C: ingredient eval, generated from the curated ruling table ─────────────

def build_ingredient(target: int = 80) -> list[dict]:
    """Ruling questions with a `gold_path` — the reasoning chain a correct answer needs."""
    rng = random.Random(SEED + 2)
    rows = neo4j_client.run(
        """
        MATCH (i:Ingredient)-[h:HAS_RULING]->(r:Ruling)
        MATCH (i)-[:DERIVED_FROM]->(s:Source {name: h.via_source})
        RETURN i.name AS ingredient, s.name AS source, r.status AS ruling,
               r.name AS ruling_th, h.basis AS basis
        ORDER BY i.name, s.name
        """)
    if not rows:
        print("  ! no ingredient rulings — run scripts.build_ingredient_kg first")
        return []

    by_ing: dict[str, list[dict]] = {}
    for r in rows:
        by_ing.setdefault(r["ingredient"], []).append(r)

    out: list[dict] = []

    # 1) source-specified → a single determinate ruling (the 2-hop path).
    pool = list(rows)
    rng.shuffle(pool)
    for r in pool[: target // 2]:
        out.append(_item(
            f"c{len(out) + 1:03d}",
            f"ส่วนผสม \"{r['ingredient']}\" ที่ได้มาจาก{r['source']} "
            f"มีสถานะฮาลาลอย่างไร",
            r["ruling_th"], "multi",
            aliases=[r["ruling"], {"halal": "ฮาลาล", "haram": "ไม่ฮาลาล",
                                   "mashbooh": "คลุมเครือ"}[r["ruling"]]],
            gold_nodes=[r["ingredient"], r["source"]],
            provenance="(Ingredient)-[:DERIVED_FROM]->(Source), "
                       "(Ingredient)-[:HAS_RULING]->(Ruling)"))
        out[-1]["gold_path"] = [r["ingredient"], r["source"], r["ruling_th"]]

    # 2) source unspecified → the correct answer is "it depends on the source".
    #    This is the item type a flat retriever cannot get right: it will pick one
    #    ruling and state it as fact.
    conflicted = [(ing, items) for ing, items in by_ing.items()
                  if len({i["ruling"] for i in items}) > 1]
    for ing, items in conflicted:
        sources = ", ".join(i["source"] for i in items[:3])
        out.append(_item(
            f"c{len(out) + 1:03d}",
            f"\"{ing}\" ฮาลาลหรือไม่",
            "ขึ้นอยู่กับแหล่งที่มา", "relational",
            aliases=["ขึ้นกับแหล่งที่มา", "ต้องดูแหล่งที่มา", "ไม่สามารถตอบได้หากไม่ทราบแหล่งที่มา"],
            gold_nodes=[ing, *[i["source"] for i in items[:3]]],
            provenance=f"(Ingredient)-[:DERIVED_FROM]->(Source) — {len(items)} "
                       f"sources with differing rulings: {sources}"))
        out[-1]["gold_path"] = [ing, items[0]["source"], items[0]["ruling_th"]]

    # 3) single-hop provenance: which source does this ruling rest on.
    determinate = [(ing, items) for ing, items in by_ing.items()
                   if len({i["ruling"] for i in items}) == 1]
    rng.shuffle(determinate)
    for ing, items in determinate[: target // 4]:
        out.append(_item(
            f"c{len(out) + 1:03d}",
            f"\"{ing}\" ได้มาจากแหล่งใด", items[0]["source"], "single",
            gold_nodes=[ing, items[0]["source"]],
            provenance="(Ingredient)-[:DERIVED_FROM]->(Source)"))
        out[-1]["gold_path"] = [ing, items[0]["source"]]

    for i, it in enumerate(out, 1):
        it["id"] = f"c{i:03d}"
    return out


def _write(items: list[dict], name: str) -> None:
    path = QUESTIONS_DIR / name
    path.write_text(
        "\n".join(json.dumps(i, ensure_ascii=False) for i in items) + "\n",
        encoding="utf-8")
    by_hop: dict[str, int] = {}
    for i in items:
        by_hop[i["hop_type"]] = by_hop.get(i["hop_type"], 0) + 1
    print(f"wrote {len(items):>4} questions → {path.name}   by hop: {by_hop}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", default="all",
                    choices=["core", "temporal", "ingredient", "all"])
    ap.add_argument("--target", type=int, default=200, help="core suite size")
    args = ap.parse_args()

    if args.suite in ("core", "all"):
        _write(build(args.target), "thai_eval.jsonl")
    if args.suite in ("temporal", "all"):
        _write(build_temporal(), "temporal_eval.jsonl")
    if args.suite in ("ingredient", "all"):
        _write(build_ingredient(), "ingredient_eval.jsonl")


if __name__ == "__main__":
    main()
