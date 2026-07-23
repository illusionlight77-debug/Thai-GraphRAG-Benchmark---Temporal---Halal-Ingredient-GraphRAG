"""Build the knowledge graph in Neo4j and index node text into Qdrant.

A generic engine over `schema.SOURCES`: for every CSV it creates the place/product
nodes, resolves each row's province through the three-tier cascade, and creates the
typed attribute edges (cuisine, brand, category, …) that give the graph its
multi-hop structure. The same node text is then embedded into Qdrant so the
vanilla-RAG baseline searches **exactly the same node set** as GraphRAG traverses —
the fairness invariant from CLAUDE.md §2.

Run:  python -m scripts.build_kg [--reset] [--only restuarant.csv] [--no-embed]
"""
from __future__ import annotations

import argparse
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from qdrant_client.models import PointStruct
from tqdm import tqdm

from thaigraphrag.config import get_settings
from thaigraphrag.core import embeddings, entity_linking, neo4j_client
from thaigraphrag.core.qdrant_client import NODES, ensure_collection, get_qdrant, reset_collection
from thaigraphrag.kg import provinces, schema

_BATCH = 500
_EMBED_WORKERS = 6      # keep below TEI's --max-concurrent-requests


# ── graph setup ─────────────────────────────────────────────────────────────

def _constraints() -> None:
    """Uniqueness constraints (which also create the lookup indexes)."""
    for spec in schema.SOURCES:
        neo4j_client.run(
            f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{spec.label}) "
            f"REQUIRE n.{spec.key} IS UNIQUE")
    for label in schema.ATTRIBUTE_LABELS:
        # District is keyed by district+province ('เมือง' is not unique nationally).
        prop = "key" if label == "District" else "name"
        neo4j_client.run(
            f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.{prop} IS UNIQUE")
    neo4j_client.run("CREATE INDEX IF NOT EXISTS FOR (n:District) ON (n.name)")
    # Entity linking scans `name` across every place label.
    for label in schema.PLACE_LABELS:
        neo4j_client.run(
            f"CREATE INDEX IF NOT EXISTS FOR (n:{label}) ON (n.name)")

    # Full-text index backing entity_linking pass 2. The `thai` analyzer does
    # dictionary-based word segmentation, which is what makes a whitespace-free Thai
    # question match a short entity name at all. Lucene supplies recall; the linker
    # re-scores every hit itself, because Lucene's ranking is term-frequency based
    # and not a measure of how well a name answers the question.
    labels = "|".join(schema.PLACE_LABELS + schema.ATTRIBUTE_LABELS)
    neo4j_client.run(
        f"""
        CREATE FULLTEXT INDEX {entity_linking.FULLTEXT_INDEX} IF NOT EXISTS
        FOR (n:{labels}) ON EACH [n.name, n.name_en]
        OPTIONS {{indexConfig: {{ `fulltext.analyzer`: 'thai' }}}}
        """)


def reset_graph() -> None:
    """Drop all nodes/relations (keeps constraints)."""
    while neo4j_client.run(
            "MATCH (n) WITH n LIMIT 50000 DETACH DELETE n RETURN count(n) AS c"
    )[0]["c"]:
        pass


def build_province_hierarchy() -> int:
    """Create all 77 Province nodes and their Region edges up front.

    Doing this before the place loaders means every province exists even if no
    place resolved to it, so 'which provinces are in ภาคใต้' is answerable from the
    graph rather than from whatever happened to be in the CSVs.
    """
    rows = [{"name": th, "name_en": provinces.TH_TO_EN[th],
             "region": provinces.TH_TO_REGION[th]} for th in provinces.TH_NAMES]
    neo4j_client.run(
        """
        UNWIND $rows AS row
        MERGE (p:Province {name: row.name})
          SET p.name_en = row.name_en, p.region = row.region,
              p.text = 'จังหวัด' + row.name + ' อยู่ภาค' + row.region
        MERGE (r:Region {name: row.region})
          SET r.text = 'ภาค' + row.region
        MERGE (p)-[:IN_REGION]->(r)
        """,
        rows=rows,
    )
    return len(rows)


# ── CSV loading ─────────────────────────────────────────────────────────────

def _sample_products(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    """Deterministic, category-stratified sample of the product registry.

    Proportional allocation per category with a floor, so small categories
    (ยาและเวชภัณฑ์, ความงาม…) survive instead of being crowded out by 'อื่นๆ'.
    A fixed seed keeps the KG reproducible across rebuilds.
    """
    if limit <= 0 or len(df) <= limit:
        return df
    cats = df["category"].fillna("อื่นๆ").replace("", "อื่นๆ")
    out = []
    counts = cats.value_counts()
    floor = min(200, limit // max(len(counts), 1))
    for cat, n in counts.items():
        take = max(floor, int(round(limit * n / len(df))))
        sub = df[cats == cat]
        out.append(sub.sample(n=min(take, len(sub)), random_state=42))
    sampled = pd.concat(out)
    if len(sampled) > limit:
        sampled = sampled.sample(n=limit, random_state=42)
    return sampled.sort_index()


def load_rows(spec: schema.SourceSpec, data_dir: Path, limit: int = 0) -> list[dict]:
    """Read one CSV into de-duplicated row dicts keyed by `spec.key`."""
    path = data_dir / spec.filename
    if not path.exists():
        return []
    df = pd.read_csv(path, low_memory=False, dtype=str)
    df.columns = [c.lstrip("﻿") for c in df.columns]   # strip BOM on col 1
    if spec.key not in df.columns:
        print(f"  ! {spec.filename}: missing key column {spec.key!r} — skipped")
        return []
    df = df[df[spec.key].notna() & (df[spec.key].astype(str).str.strip() != "")]
    df = df.drop_duplicates(subset=[spec.key], keep="first")
    if spec.label == "Product" and limit:
        before = len(df)
        df = _sample_products(df, limit)
        print(f"  sampled {len(df):,} of {before:,} products "
              f"(category-stratified, seed=42)")
    return df.fillna("").to_dict("records")


# ── the engine ──────────────────────────────────────────────────────────────

def _point_id(label: str, key: str) -> str:
    """Stable Qdrant point id so re-running the build upserts instead of duplicating."""
    return hashlib.md5(f"{label}:{key}".encode()).hexdigest()


def ingest_source(spec: schema.SourceSpec, data_dir: Path, locator,
                  limit: int, embed: bool = True) -> dict:
    rows = load_rows(spec, data_dir, limit=limit)
    if not rows:
        print(f"skip {spec.filename} (not found in {data_dir})")
        return {"label": spec.label, "rows": 0}

    stats = {"label": spec.label, "rows": len(rows), "province": 0,
             "tiers": {"explicit": 0, "address": 0, "geo": 0}, "edges": 0}

    # 1) nodes + province/district edges ------------------------------------
    payloads, texts = [], []
    for r in rows:
        province, tier = schema.resolve_province(r, spec, locator)
        if province:
            stats["province"] += 1
            stats["tiers"][tier] += 1
        text = schema.node_text(spec.label, r, spec, province)
        props = {k: schema.clean(r.get(k)) for k in spec.keep_props}
        props = {k: v for k, v in props.items() if v}
        payloads.append({
            "key": schema.clean(r[spec.key]),
            "name": schema.first(r, spec.name_cols),
            "name_en": schema.clean(r.get("name_en")),
            "province": province,
            "province_source": tier,
            "district": schema.clean(r.get(spec.district_col)) if spec.district_col else "",
            "lat": schema.first(r, spec.lat_cols),
            "lon": schema.first(r, spec.lon_cols),
            "text": text,
            "props": props,
        })
        texts.append(text)

    # Three passes rather than one long WITH-chain: each is independently correct,
    # and a row missing a province still gets its node and its district.
    for i in tqdm(range(0, len(payloads), _BATCH), desc=f"  nodes {spec.label}", leave=False):
        chunk = payloads[i:i + _BATCH]
        neo4j_client.run(
            f"""
            UNWIND $rows AS row
            MERGE (n:{spec.label} {{{spec.key}: row.key}})
              SET n.name = row.name, n.text = row.text,
                  n.name_en = row.name_en,
                  n.province = row.province,
                  n.province_source = row.province_source,
                  n.lat = row.lat, n.lon = row.lon,
                  n += row.props
            """,
            rows=chunk,
        )
        located = [c for c in chunk if c["province"]]
        if located:
            neo4j_client.run(
                f"""
                UNWIND $rows AS row
                MATCH (n:{spec.label} {{{spec.key}: row.key}})
                MATCH (p:Province {{name: row.province}})
                MERGE (n)-[l:LOCATED_IN]->(p)
                  SET l.source = row.province_source
                """,
                rows=located,
            )
        # District names repeat across provinces ('เมือง' exists 77 times), so the
        # node is keyed by district+province, not by name alone.
        districts = [{**c, "dkey": f"{c['district']}|{c['province']}"}
                     for c in chunk if c["district"] and c["province"]]
        if districts:
            neo4j_client.run(
                f"""
                UNWIND $rows AS row
                MATCH (n:{spec.label} {{{spec.key}: row.key}})
                MATCH (p:Province {{name: row.province}})
                MERGE (d:District {{key: row.dkey}})
                  SET d.name = row.district, d.province = row.province,
                      d.text = 'อำเภอ/เขต ' + row.district + ' จังหวัด' + row.province
                MERGE (n)-[:IN_DISTRICT]->(d)
                MERGE (d)-[:IN_PROVINCE]->(p)
                """,
                rows=districts,
            )

    # 2) typed attribute edges ---------------------------------------------
    for espec in spec.edges:
        pairs = []
        for r in rows:
            key = schema.clean(r[spec.key])
            for val in schema.edge_values(r, espec):
                pairs.append({"key": key, "val": val})
        if not pairs:
            continue
        stats["edges"] += len(pairs)
        for i in range(0, len(pairs), _BATCH):
            neo4j_client.run(
                f"""
                UNWIND $rows AS row
                MATCH (n:{spec.label} {{{spec.key}: row.key}})
                MERGE (t:{espec.label} {{name: row.val}})
                  SET t.text = coalesce(t.text, row.val)
                MERGE (n)-[:{espec.rel}]->(t)
                """,
                rows=pairs[i:i + _BATCH],
            )

    # 3) embed node text into Qdrant (vanilla baseline) ---------------------
    # TEI serves ~32 texts/s per connection on CPU; the KG has ~60k nodes, so the
    # batches are embedded over a small thread pool (TEI is started with
    # --max-concurrent-requests 32). Upserts stay on the main thread.
    if embed:
        s = get_settings()
        qdrant = get_qdrant()
        bs = s.embed_batch
        batches = [payloads[i:i + bs] for i in range(0, len(payloads), bs)]

        def _embed(batch: list[dict]) -> tuple[list[dict], list[list[float]]]:
            return batch, embeddings.embed_many([c["text"] for c in batch])

        with ThreadPoolExecutor(max_workers=_EMBED_WORKERS) as pool:
            for batch, vectors in tqdm(
                pool.map(_embed, batches), total=len(batches),
                desc=f"  embed {spec.label}", leave=False,
            ):
                qdrant.upsert(NODES, [
                    PointStruct(
                        id=_point_id(spec.label, c["key"]),
                        vector=v,
                        payload={"node_key": c["key"], "label": spec.label,
                                 "name": c["name"], "province": c["province"],
                                 "text": c["text"]},
                    )
                    for c, v in zip(batch, vectors)
                ])
    return stats


def index_labels(labels: list[str], batch: int | None = None) -> int:
    """Embed every node of the given labels into Qdrant.

    Extensions create their own nodes (Ingredient, Source, Ruling, Regulation …).
    If those only existed in Neo4j, the vanilla baseline could never retrieve them and
    the extension would win by having a bigger corpus rather than a better strategy.
    This keeps the two views of the graph identical (CLAUDE.md §2).
    """
    s = get_settings()
    rows = neo4j_client.run(
        """
        MATCH (n) WHERE any(l IN labels(n) WHERE l IN $labels)
        RETURN labels(n)[0] AS label,
               coalesce(n.name, n.status, '') AS name,
               coalesce(n.text, n.name, '') AS text
        """,
        labels=labels)
    rows = [r for r in rows if r["text"]]
    if not rows:
        return 0

    ensure_collection()
    qdrant = get_qdrant()
    bs = batch or s.embed_batch
    for i in range(0, len(rows), bs):
        chunk = rows[i:i + bs]
        vectors = embeddings.embed_many([c["text"] for c in chunk])
        qdrant.upsert(NODES, [
            PointStruct(
                id=_point_id(c["label"], c["name"]),
                vector=v,
                payload={"node_key": c["name"], "label": c["label"],
                         "name": c["name"], "province": "", "text": c["text"]},
            )
            for c, v in zip(chunk, vectors)
        ])
    return len(rows)


def graph_counts() -> dict:
    """Node/relation census used by the build report and the /api/stats endpoint."""
    nodes = {r["label"]: r["c"] for r in neo4j_client.run(
        "MATCH (n) UNWIND labels(n) AS label RETURN label, count(*) AS c ORDER BY c DESC")}
    rels = {r["rel"]: r["c"] for r in neo4j_client.run(
        "MATCH ()-[r]->() RETURN type(r) AS rel, count(*) AS c ORDER BY c DESC")}
    return {"nodes": nodes, "relationships": rels,
            "total_nodes": sum(nodes.values()), "total_rels": sum(rels.values())}


def build(only: str = "", reset: bool = False, embed: bool = True,
          limit: int | None = None) -> dict:
    s = get_settings()
    data_dir = Path(s.data_dir)
    t0 = time.time()

    print(f"DATA_DIR = {data_dir.resolve()}")
    ensure_collection()
    if reset:
        print("resetting graph + vector collection …")
        reset_graph()
        reset_collection()
    _constraints()

    n = build_province_hierarchy()
    print(f"Province hierarchy: {n} provinces + regions")

    locator = provinces.get_locator(str(data_dir / s.province_geojson))
    if locator is None:
        print(f"  ! {s.province_geojson} not found — geo province tier disabled")

    product_limit = s.max_product_rows if limit is None else limit
    all_stats = []
    for spec in schema.SOURCES:
        if only and spec.filename != only:
            continue
        print(f"\n── {spec.label}  ({spec.filename})")
        st = ingest_source(spec, data_dir, locator, product_limit, embed=embed)
        all_stats.append(st)
        if st["rows"]:
            cov = 100 * st["province"] / st["rows"]
            print(f"  {st['rows']:,} nodes | province {st['province']:,} ({cov:.1f}%) "
                  f"{st['tiers']} | attr-edges {st['edges']:,}")

    counts = graph_counts()
    print(f"\nKG build complete in {time.time() - t0:.0f}s")
    print(f"  nodes: {counts['total_nodes']:,}   relationships: {counts['total_rels']:,}")
    return {"sources": all_stats, "counts": counts,
            "elapsed_s": round(time.time() - t0, 1)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reset", action="store_true", help="wipe the graph first")
    ap.add_argument("--only", default="", help="build a single CSV, e.g. mosque.csv")
    ap.add_argument("--no-embed", action="store_true", help="skip Qdrant indexing")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap product rows (0 = all); default MAX_PRODUCT_ROWS")
    args = ap.parse_args()
    build(only=args.only, reset=args.reset, embed=not args.no_embed, limit=args.limit)


if __name__ == "__main__":
    main()
