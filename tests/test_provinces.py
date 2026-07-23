"""The province gazetteer and geocoder — the backbone of every LOCATED_IN edge.

If this module is wrong, the location hierarchy is wrong, and with it every
multi-hop question in the benchmark. These tests need no services.
"""
import json
from pathlib import Path

import pytest

from thaigraphrag.kg import provinces

GEOJSON = Path(__file__).resolve().parents[1] / "data" / "thailand_provinces.json"


def test_gazetteer_is_complete_and_unique():
    assert len(provinces.PROVINCES) == 77
    assert len(set(provinces.TH_NAMES)) == 77
    assert len(set(provinces.TH_TO_EN.values())) == 77


def test_region_membership_totals():
    counts: dict[str, int] = {}
    for _, _, region in provinces.PROVINCES:
        counts[region] = counts.get(region, 0) + 1
    # Thailand's official six-region grouping.
    assert counts == {"เหนือ": 9, "ตะวันออกเฉียงเหนือ": 20, "กลาง": 22,
                      "ตะวันออก": 7, "ตะวันตก": 5, "ใต้": 14}


@pytest.mark.parametrize("raw,expected", [
    ("Surat Thani", "สุราษฎร์ธานี"),
    ("จังหวัดสงขลา", "สงขลา"),
    ("Bangkok", "กรุงเทพมหานคร"),
    ("Bangkok Metropolis", "กรุงเทพมหานคร"),
    ("Chonburi", "ชลบุรี"),
    ("Chon Buri", "ชลบุรี"),
    ("ประจวบ", "ประจวบคีรีขันธ์"),
    ("Pattani Province", "ปัตตานี"),
    ("", ""),
    ("Kuala Lumpur", ""),
])
def test_normalise(raw, expected):
    assert provinces.normalise(raw) == expected


def test_from_address_prefers_longest_name():
    assert provinces.from_address("123 ถ.สุขุมวิท อ.หาดใหญ่ จ.สงขลา 90110") == "สงขลา"
    # 'นครศรีธรรมราช' must not be shadowed by a shorter province substring.
    assert provinces.from_address("ต.ในเมือง อ.เมือง จ.นครศรีธรรมราช") == "นครศรีธรรมราช"
    assert provinces.from_address("no province here") == ""


@pytest.mark.skipif(not GEOJSON.exists(), reason="boundary file not present")
def test_every_geojson_name_resolves():
    names = [f["properties"]["name"]
             for f in json.loads(GEOJSON.read_text(encoding="utf-8"))["features"]]
    assert len(names) == 77
    unresolved = [n for n in names if not provinces.normalise(n)]
    assert unresolved == []


@pytest.mark.skipif(not GEOJSON.exists(), reason="boundary file not present")
@pytest.mark.parametrize("lat,lon,expected", [
    (13.7563, 100.5018, "กรุงเทพมหานคร"),   # Bangkok
    (6.8694, 101.2502, "ปัตตานี"),          # Pattani
    (7.8804, 98.3923, "ภูเก็ต"),            # Phuket
    (18.7883, 98.9853, "เชียงใหม่"),        # Chiang Mai
    (7.0083, 100.4767, "สงขลา"),            # Hat Yai
])
def test_point_in_polygon(lat, lon, expected):
    assert provinces.get_locator(str(GEOJSON)).locate(lat, lon) == expected


@pytest.mark.skipif(not GEOJSON.exists(), reason="boundary file not present")
def test_locator_rejects_outside_thailand():
    loc = provinces.get_locator(str(GEOJSON))
    assert loc.locate(3.139, 101.687) == ""      # Kuala Lumpur
    assert loc.locate(None, None) == ""
    assert loc.locate("nope", "nope") == ""
