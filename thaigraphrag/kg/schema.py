"""Knowledge-graph schema — declarative CSV → graph mapping.

The graph is deliberately built around a **location hierarchy** plus a set of
typed attribute nodes, because that is what creates genuine multi-hop paths:

    (Place)-[:LOCATED_IN]->(Province)-[:IN_REGION]->(Region)
    (Restaurant)-[:SERVES_CUISINE]->(Cuisine)
    (Store)-[:IS_TYPE]->(StoreType) , (Store)-[:HAS_BRAND]->(Brand)
    (Product)-[:BELONGS_TO]->(Category) , (Product)-[:MADE_BY]->(Company)

Two places in different CSVs that share a province are now 2 hops apart, which a
dense retriever over independent node texts cannot traverse — that asymmetry is
exactly what the benchmark measures.

Each CSV is described by a `SourceSpec`; `build_kg.py` is a generic engine over
these specs, so adding a source means adding a spec, not writing loader code.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from thaigraphrag.kg import provinces


@dataclass(frozen=True)
class EdgeSpec:
    """One attribute column → one related node + relation."""
    column: str            # source column on the row
    label: str             # label of the target node
    rel: str               # relation type from the place/product to the target
    split: str = ""        # if set, split the cell on this char (OSM uses ';')
    lower: bool = False    # normalise the value to lower case
    max_values: int = 4    # guard against pathological multi-valued cells


@dataclass(frozen=True)
class SourceSpec:
    """How one CSV becomes nodes + edges."""
    filename: str
    label: str
    key: str                                  # unique-key column
    name_cols: tuple[str, ...]
    province_cols: tuple[str, ...] = ()
    address_cols: tuple[str, ...] = ()        # scanned for a province name (tier 2)
    lat_cols: tuple[str, ...] = ()
    lon_cols: tuple[str, ...] = ()
    district_col: str = ""
    edges: tuple[EdgeSpec, ...] = ()
    keep_props: tuple[str, ...] = ()          # extra columns copied onto the node
    text_props: tuple[str, ...] = ()          # extra columns folded into node_text
    geo_locate: bool = True                   # allow the point-in-polygon tier


SOURCES: tuple[SourceSpec, ...] = (
    SourceSpec(
        filename="restuarant.csv",            # (sic) matches the source filename
        label="Restaurant",
        key="osm_id",
        name_cols=("name", "name_th", "name_en"),
        province_cols=("addr_province",),
        address_cols=("addr_full", "addr_street"),
        lat_cols=("lat",), lon_cols=("lon",),
        district_col="addr_district",
        edges=(
            EdgeSpec("cuisine", "Cuisine", "SERVES_CUISINE", split=";", lower=True),
            EdgeSpec("amenity", "Amenity", "HAS_AMENITY", split=";", lower=True),
        ),
        keep_props=("diet_halal", "cuisine", "amenity", "phone", "website"),
        text_props=("cuisine", "amenity", "diet_halal"),
    ),
    SourceSpec(
        filename="hotel.csv",
        label="Hotel",
        key="osm_id",
        name_cols=("name", "name_th", "name_en"),
        province_cols=("addr_province",),
        address_cols=("addr_full", "addr_street"),
        lat_cols=("lat",), lon_cols=("lon",),
        district_col="addr_district",
        edges=(EdgeSpec("tourism_type", "TourismType", "IS_TYPE", lower=True),),
        keep_props=("tourism_type", "stars", "halal", "phone", "website"),
        text_props=("tourism_type", "stars", "halal"),
    ),
    SourceSpec(
        filename="store_processed.csv",
        label="Store",
        key="osm_id",
        name_cols=("name", "name_th", "name_en", "store_brand"),
        province_cols=("addr_province",),
        address_cols=("addr_full", "addr_street"),
        lat_cols=("lat",), lon_cols=("lon",),
        district_col="addr_district",
        edges=(
            EdgeSpec("store_type", "StoreType", "IS_TYPE", lower=True),
            EdgeSpec("store_brand", "Brand", "HAS_BRAND"),
        ),
        keep_props=("store_type", "store_brand", "group", "halal"),
        text_props=("store_type", "store_brand", "halal"),
    ),
    SourceSpec(
        filename="attractions.csv",
        label="Attraction",
        key="ttd_id",
        name_cols=("name_th", "name_en"),
        province_cols=("province_th", "province_en"),
        address_cols=("address_th", "address_en"),
        lat_cols=("lat",), lon_cols=("lon",),
        edges=(EdgeSpec("category_th", "Category", "BELONGS_TO", split=","),),
        keep_props=("category_th", "phone", "website"),
        text_props=("category_th",),
    ),
    SourceSpec(
        filename="mosque.csv",
        label="Mosque",
        key="osm_id",
        name_cols=("name", "name_th", "name_en", "alt_name"),
        province_cols=("addr_province",),
        address_cols=("addr_full", "addr_street"),
        lat_cols=("lat",), lon_cols=("lon",),
        district_col="addr_district",
        edges=(EdgeSpec("denomination", "Denomination", "HAS_DENOMINATION", lower=True),),
        keep_props=("denomination", "religion", "phone"),
        text_props=("denomination", "religion"),
    ),
    SourceSpec(
        filename="product_processed.csv",
        label="Product",
        key="halal_code",
        name_cols=("product_name", "trademark_name"),
        edges=(
            EdgeSpec("category", "Category", "BELONGS_TO"),
            EdgeSpec("company_name", "Company", "MADE_BY"),
            EdgeSpec("trademark_name", "Brand", "HAS_BRAND"),
        ),
        keep_props=("trademark_name", "category", "company_name", "fda_number", "barcode"),
        text_props=("trademark_name", "category", "company_name"),
        geo_locate=False,
    ),
)

# Back-compat with the seed API (tests / older scripts import these).
CSV_TO_LABEL = {s.filename: (s.label, s.key) for s in SOURCES}
NAME_COLS = ("name", "name_th", "name_en")
PROVINCE_COLS = ("addr_province", "province", "province_th")
LAT_COLS = ("lat", "latitude")
LON_COLS = ("lon", "lng", "longitude")

# Labels that carry a searchable `name` and take part in entity linking.
PLACE_LABELS = ("Restaurant", "Hotel", "Store", "Attraction", "Mosque", "Product")
ATTRIBUTE_LABELS = (
    "Province", "Region", "District", "Cuisine", "Amenity", "TourismType",
    "StoreType", "Brand", "Category", "Company", "Denomination",
)

# Human-readable Thai labels used when linearising a subgraph for the LLM.
LABEL_TH = {
    "Restaurant": "ร้านอาหาร", "Hotel": "ที่พัก", "Store": "ร้านค้า",
    "Attraction": "สถานที่ท่องเที่ยว", "Mosque": "มัสยิด", "Product": "สินค้า",
    "Province": "จังหวัด", "Region": "ภาค", "District": "อำเภอ/เขต",
    "Cuisine": "ประเภทอาหาร", "Amenity": "ประเภทสถานที่", "TourismType": "ประเภทที่พัก",
    "StoreType": "ประเภทร้าน", "Brand": "แบรนด์", "Category": "หมวดหมู่",
    "Company": "บริษัท", "Denomination": "นิกาย",
    "Ingredient": "ส่วนผสม", "Source": "แหล่งที่มา", "Ruling": "คำวินิจฉัย",
    "Regulation": "ระเบียบ/กฎหมาย", "Organisation": "หน่วยงาน",
}

REL_TH = {
    "LOCATED_IN": "ตั้งอยู่ในจังหวัด", "IN_REGION": "อยู่ในภาค",
    "IN_DISTRICT": "อยู่ในอำเภอ/เขต", "IN_PROVINCE": "อยู่ในจังหวัด",
    "SERVES_CUISINE": "เสิร์ฟอาหารประเภท", "HAS_AMENITY": "เป็นสถานที่ประเภท",
    "IS_TYPE": "เป็นประเภท", "HAS_BRAND": "แบรนด์", "BELONGS_TO": "อยู่ในหมวด",
    "MADE_BY": "ผลิตโดย", "HAS_DENOMINATION": "นิกาย",
    "CONTAINS": "มีส่วนผสม", "DERIVED_FROM": "ได้มาจาก", "HAS_RULING": "คำวินิจฉัยคือ",
    "ISSUED_BY": "ออกโดย", "SUPERSEDES": "ใช้แทนฉบับ", "GOVERNS": "กำกับดูแล",
}


def first(row: dict, cols) -> str:
    """First non-empty value among `cols`."""
    for c in cols:
        v = row.get(c)
        if v is not None:
            sv = str(v).strip()
            if sv and sv.lower() not in ("nan", "none", "null"):
                return sv
    return ""


def clean(value) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    return "" if s.lower() in ("nan", "none", "null") else s


def edge_values(row: dict, spec: EdgeSpec) -> list[str]:
    """Values for one EdgeSpec — handles OSM's ';'-separated multi-values."""
    raw = clean(row.get(spec.column))
    if not raw:
        return []
    parts = [p.strip() for p in raw.split(spec.split)] if spec.split else [raw]
    out: list[str] = []
    for p in parts:
        if not p:
            continue
        p = p.lower() if spec.lower else p
        if p not in out:
            out.append(p)
    return out[: spec.max_values]


def resolve_province(row: dict, spec: SourceSpec, locator=None) -> tuple[str, str]:
    """Three-tier province resolution → (canonical Thai name, source tier).

    Tier order is explicit column → address text → point-in-polygon. Returns
    ('', '') when no tier fires so the caller can skip the LOCATED_IN edge.
    """
    raw = first(row, spec.province_cols)
    if raw:
        canon = provinces.normalise(raw)
        if canon:
            return canon, "explicit"

    for col in spec.address_cols:
        canon = provinces.from_address(clean(row.get(col)))
        if canon:
            return canon, "address"

    if spec.geo_locate and locator is not None:
        lat, lon = first(row, spec.lat_cols), first(row, spec.lon_cols)
        if lat and lon:
            canon = locator.locate(lat, lon)
            if canon:
                return canon, "geo"
    return "", ""


def node_text(label: str, row: dict, spec: SourceSpec | None = None,
              province: str = "") -> str:
    """The text that gets embedded (vanilla RAG) and shown in the UI.

    Kept identical for both retrievers' view of a node so the comparison stays fair:
    vanilla embeds this string, GraphRAG shows it as the seed node's description.
    """
    name_cols = spec.name_cols if spec else NAME_COLS
    text_props = spec.text_props if spec else ()
    name = first(row, name_cols)
    th = LABEL_TH.get(label, label)
    parts = [f"{th}: {name}" if name else th]
    if province:
        parts.append(f"จังหวัด {province}")
        region = provinces.TH_TO_REGION.get(province)
        if region:
            parts.append(f"ภาค{region}")
    for k in text_props:
        v = clean(row.get(k))
        if v:
            parts.append(f"{k}={v}")
    return " | ".join(parts)
