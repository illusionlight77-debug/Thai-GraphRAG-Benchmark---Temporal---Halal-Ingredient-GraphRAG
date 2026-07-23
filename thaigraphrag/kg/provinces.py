"""Thai province gazetteer + resolver — the backbone of the location hierarchy.

Why this module exists
----------------------
The raw OSM/TTD CSVs are almost useless for location on their own:
`addr_province` is populated for only ~4% of hotels, ~1.7% of mosques, and *0%*
of attractions. Without provinces there are no `LOCATED_IN` edges, and without
those there are no multi-hop paths — which is the entire point of the study.

So province is resolved with a deterministic three-tier cascade, best source first:

    1. `explicit`  — a province column in the CSV (normalised EN → canonical TH)
    2. `address`   — the province name found inside the Thai address text
    3. `geo`       — point-in-polygon of (lat, lon) against province boundaries

Every node records which tier produced its province (`province_source`) so the
provenance of a derived edge is always auditable in the graph.

Canonical form is the **Thai** province name; English names match those in
`data/thailand_provinces.json` exactly so the geo tier can join on them.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

# (thai, english-as-in-geojson, region-th)
PROVINCES: list[tuple[str, str, str]] = [
    # ── ภาคเหนือ (9) ─────────────────────────────────────────────
    ("เชียงใหม่", "Chiang Mai", "เหนือ"),
    ("เชียงราย", "Chiang Rai", "เหนือ"),
    ("ลำปาง", "Lampang", "เหนือ"),
    ("ลำพูน", "Lamphun", "เหนือ"),
    ("แม่ฮ่องสอน", "Mae Hong Son", "เหนือ"),
    ("น่าน", "Nan", "เหนือ"),
    ("พะเยา", "Phayao", "เหนือ"),
    ("แพร่", "Phrae", "เหนือ"),
    ("อุตรดิตถ์", "Uttaradit", "เหนือ"),
    # ── ภาคตะวันออกเฉียงเหนือ (20) ────────────────────────────────
    ("กาฬสินธุ์", "Kalasin", "ตะวันออกเฉียงเหนือ"),
    ("ขอนแก่น", "Khon Kaen", "ตะวันออกเฉียงเหนือ"),
    ("ชัยภูมิ", "Chaiyaphum", "ตะวันออกเฉียงเหนือ"),
    ("นครพนม", "Nakhon Phanom", "ตะวันออกเฉียงเหนือ"),
    ("นครราชสีมา", "Nakhon Ratchasima", "ตะวันออกเฉียงเหนือ"),
    ("บึงกาฬ", "Bueng Kan", "ตะวันออกเฉียงเหนือ"),
    ("บุรีรัมย์", "Buri Ram", "ตะวันออกเฉียงเหนือ"),
    ("มหาสารคาม", "Maha Sarakham", "ตะวันออกเฉียงเหนือ"),
    ("มุกดาหาร", "Mukdahan", "ตะวันออกเฉียงเหนือ"),
    ("ยโสธร", "Yasothon", "ตะวันออกเฉียงเหนือ"),
    ("ร้อยเอ็ด", "Roi Et", "ตะวันออกเฉียงเหนือ"),
    ("เลย", "Loei", "ตะวันออกเฉียงเหนือ"),
    ("ศรีสะเกษ", "Si Sa Ket", "ตะวันออกเฉียงเหนือ"),
    ("สกลนคร", "Sakon Nakhon", "ตะวันออกเฉียงเหนือ"),
    ("สุรินทร์", "Surin", "ตะวันออกเฉียงเหนือ"),
    ("หนองคาย", "Nong Khai", "ตะวันออกเฉียงเหนือ"),
    ("หนองบัวลำภู", "Nong Bua Lam Phu", "ตะวันออกเฉียงเหนือ"),
    ("อำนาจเจริญ", "Amnat Charoen", "ตะวันออกเฉียงเหนือ"),
    ("อุดรธานี", "Udon Thani", "ตะวันออกเฉียงเหนือ"),
    ("อุบลราชธานี", "Ubon Ratchathani", "ตะวันออกเฉียงเหนือ"),
    # ── ภาคกลาง (22) ─────────────────────────────────────────────
    ("กรุงเทพมหานคร", "Bangkok Metropolis", "กลาง"),
    ("กำแพงเพชร", "Kamphaeng Phet", "กลาง"),
    ("ชัยนาท", "Chai Nat", "กลาง"),
    ("นครนายก", "Nakhon Nayok", "กลาง"),
    ("นครปฐม", "Nakhon Pathom", "กลาง"),
    ("นครสวรรค์", "Nakhon Sawan", "กลาง"),
    ("นนทบุรี", "Nonthaburi", "กลาง"),
    ("ปทุมธานี", "Pathum Thani", "กลาง"),
    ("พระนครศรีอยุธยา", "Phra Nakhon Si Ayutthaya", "กลาง"),
    ("พิจิตร", "Phichit", "กลาง"),
    ("พิษณุโลก", "Phitsanulok", "กลาง"),
    ("เพชรบูรณ์", "Phetchabun", "กลาง"),
    ("ลพบุรี", "Lop Buri", "กลาง"),
    ("สมุทรปราการ", "Samut Prakan", "กลาง"),
    ("สมุทรสงคราม", "Samut Songkhram", "กลาง"),
    ("สมุทรสาคร", "Samut Sakhon", "กลาง"),
    ("สิงห์บุรี", "Sing Buri", "กลาง"),
    ("สุโขทัย", "Sukhothai", "กลาง"),
    ("สุพรรณบุรี", "Suphan Buri", "กลาง"),
    ("สระบุรี", "Saraburi", "กลาง"),
    ("อ่างทอง", "Ang Thong", "กลาง"),
    ("อุทัยธานี", "Uthai Thani", "กลาง"),
    # ── ภาคตะวันออก (7) ──────────────────────────────────────────
    ("จันทบุรี", "Chanthaburi", "ตะวันออก"),
    ("ฉะเชิงเทรา", "Chachoengsao", "ตะวันออก"),
    ("ชลบุรี", "Chon Buri", "ตะวันออก"),
    ("ตราด", "Trat", "ตะวันออก"),
    ("ปราจีนบุรี", "Prachin Buri", "ตะวันออก"),
    ("ระยอง", "Rayong", "ตะวันออก"),
    ("สระแก้ว", "Sa Kaeo", "ตะวันออก"),
    # ── ภาคตะวันตก (5) ───────────────────────────────────────────
    ("กาญจนบุรี", "Kanchanaburi", "ตะวันตก"),
    ("ตาก", "Tak", "ตะวันตก"),
    ("ประจวบคีรีขันธ์", "Prachuap Khiri Khan", "ตะวันตก"),
    ("เพชรบุรี", "Phetchaburi", "ตะวันตก"),
    ("ราชบุรี", "Ratchaburi", "ตะวันตก"),
    # ── ภาคใต้ (14) ──────────────────────────────────────────────
    ("กระบี่", "Krabi", "ใต้"),
    ("ชุมพร", "Chumphon", "ใต้"),
    ("ตรัง", "Trang", "ใต้"),
    ("นครศรีธรรมราช", "Nakhon Si Thammarat", "ใต้"),
    ("นราธิวาส", "Narathiwat", "ใต้"),
    ("ปัตตานี", "Pattani", "ใต้"),
    ("พังงา", "Phangnga", "ใต้"),
    ("พัทลุง", "Phatthalung", "ใต้"),
    ("ภูเก็ต", "Phuket", "ใต้"),
    ("ยะลา", "Yala", "ใต้"),
    ("ระนอง", "Ranong", "ใต้"),
    ("สงขลา", "Songkhla", "ใต้"),
    ("สตูล", "Satun", "ใต้"),
    ("สุราษฎร์ธานี", "Surat Thani", "ใต้"),
]

TH_NAMES: list[str] = [p[0] for p in PROVINCES]
TH_TO_EN: dict[str, str] = {p[0]: p[1] for p in PROVINCES}
TH_TO_REGION: dict[str, str] = {p[0]: p[2] for p in PROVINCES}

# Extra spellings seen in OSM / TTD data that are not the geojson name.
_EXTRA_ALIASES: dict[str, str] = {
    "bangkok": "กรุงเทพมหานคร",
    "bangkok metropolitan": "กรุงเทพมหานคร",
    "krung thep maha nakhon": "กรุงเทพมหานคร",
    "กทม": "กรุงเทพมหานคร",
    "กทม.": "กรุงเทพมหานคร",
    "กรุงเทพ": "กรุงเทพมหานคร",
    "กรุงเทพฯ": "กรุงเทพมหานคร",
    "ayutthaya": "พระนครศรีอยุธยา",
    "อยุธยา": "พระนครศรีอยุธยา",
    "buriram": "บุรีรัมย์",
    "chonburi": "ชลบุรี",
    "lopburi": "ลพบุรี",
    "sisaket": "ศรีสะเกษ",
    "singburi": "สิงห์บุรี",
    "suphanburi": "สุพรรณบุรี",
    "chainat": "ชัยนาท",
    "nongbua lamphu": "หนองบัวลำภู",
    "phang nga": "พังงา",
    "phang-nga": "พังงา",
    "pattani province": "ปัตตานี",
    "prachuap khirikhan": "ประจวบคีรีขันธ์",
    "ประจวบ": "ประจวบคีรีขันธ์",
    "nakhon si thammarat province": "นครศรีธรรมราช",
    "korat": "นครราชสีมา",
    "โคราช": "นครราชสีมา",
    "ubon": "อุบลราชธานี",
    "udon": "อุดรธานี",
    "hat yai": "สงขลา",
    "หาดใหญ่": "สงขลา",
}


@lru_cache
def _lookup() -> dict[str, str]:
    """Normalised alias → canonical Thai name."""
    table: dict[str, str] = {}
    for th, en, _ in PROVINCES:
        table[th.lower()] = th
        table[en.lower()] = th
        # "Chon Buri" is also written "Chonburi", "Buri Ram" as "Buriram", ...
        table[en.lower().replace(" ", "")] = th
        table[en.lower().replace(" ", "-")] = th
    table.update(_EXTRA_ALIASES)
    return table


def normalise(name: str) -> str:
    """Map any spelling of a province to its canonical Thai name ('' if unknown)."""
    if not name:
        return ""
    key = str(name).strip().lower()
    for prefix in ("จังหวัด", "จ.", "province of ", "changwat "):
        if key.startswith(prefix):
            key = key[len(prefix):].strip()
    if key.endswith(" province"):
        key = key[: -len(" province")].strip()
    return _lookup().get(key, "")


def from_address(text: str) -> str:
    """Find a province name inside free Thai/English address text.

    Longest name first so 'นครศรีธรรมราช' is not shadowed by a shorter substring.
    """
    if not text:
        return ""
    t = str(text)
    tl = t.lower()
    for th in sorted(TH_NAMES, key=len, reverse=True):
        if th in t:
            return th
    for en, th in sorted(
        ((p[1], p[0]) for p in PROVINCES), key=lambda x: len(x[0]), reverse=True
    ):
        if en.lower() in tl:
            return th
    for alias, th in sorted(_EXTRA_ALIASES.items(), key=lambda x: len(x[0]), reverse=True):
        if alias in tl:
            return th
    return ""


class ProvinceLocator:
    """Point-in-polygon province lookup from `thailand_provinces.json` (GeoJSON).

    Uses a bounding-box prefilter then `matplotlib.path.Path.contains_point`, which
    keeps ~40k lookups well under a second without pulling in a geometry library.
    """

    def __init__(self, geojson_path: str | Path):
        from matplotlib.path import Path as MplPath  # local: keeps import cost off the CLI

        self._entries: list[tuple[str, tuple[float, float, float, float], object]] = []
        data = json.loads(Path(geojson_path).read_text(encoding="utf-8"))
        for feat in data.get("features", []):
            th = normalise(feat.get("properties", {}).get("name", ""))
            if not th:
                continue
            geom = feat.get("geometry", {})
            polys = (
                geom.get("coordinates", [])
                if geom.get("type") == "Polygon"
                else [ring for part in geom.get("coordinates", []) for ring in part]
            )
            for ring in polys:
                if not ring or len(ring) < 4:
                    continue
                xs = [c[0] for c in ring]
                ys = [c[1] for c in ring]
                bbox = (min(xs), min(ys), max(xs), max(ys))
                self._entries.append((th, bbox, MplPath([(c[0], c[1]) for c in ring])))

    def __len__(self) -> int:
        return len(self._entries)

    def locate(self, lat: float, lon: float) -> str:
        """Canonical Thai province containing (lat, lon), or '' if outside Thailand."""
        try:
            lat, lon = float(lat), float(lon)
        except (TypeError, ValueError):
            return ""
        # Thailand's rough envelope — rejects the Malaysian/Lao rows in the OSM dumps.
        if not (5.0 <= lat <= 21.0 and 96.0 <= lon <= 106.0):
            return ""
        for th, (x0, y0, x1, y1), path in self._entries:
            if x0 <= lon <= x1 and y0 <= lat <= y1 and path.contains_point((lon, lat)):
                return th
        return ""


@lru_cache
def get_locator(geojson_path: str) -> ProvinceLocator | None:
    """Cached locator; returns None if the boundary file is absent (geo tier off)."""
    p = Path(geojson_path)
    if not p.exists():
        return None
    return ProvinceLocator(p)
