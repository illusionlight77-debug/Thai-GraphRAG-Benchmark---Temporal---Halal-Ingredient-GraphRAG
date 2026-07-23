"""Schema mapping and the curated data files that ship with the repo.

The ruling table is a research artefact, so its invariants are asserted here rather
than trusted: a typo'd ruling value or a missing basis would silently corrupt every
extension-C result.
"""
from pathlib import Path

import pandas as pd
import pytest

from thaigraphrag.kg import schema

DATA = Path(__file__).resolve().parents[1] / "data"
RULINGS = DATA / "halal" / "ingredient_rulings.csv"
TIMELINE = DATA / "halal" / "regulation_timeline.csv"


# ── schema helpers ──────────────────────────────────────────────────────────

def test_first_skips_empty_and_nan():
    row = {"name": "", "name_th": "nan", "name_en": "Central Mosque"}
    assert schema.first(row, ("name", "name_th", "name_en")) == "Central Mosque"
    assert schema.first({}, ("name",)) == ""


def test_clean_treats_nan_strings_as_empty():
    assert schema.clean("nan") == ""
    assert schema.clean(None) == ""
    assert schema.clean("  ok  ") == "ok"


def test_edge_values_splits_osm_multivalues():
    spec = schema.EdgeSpec("cuisine", "Cuisine", "SERVES_CUISINE", split=";", lower=True)
    assert schema.edge_values({"cuisine": "Malaysian;Thai;malaysian"}, spec) == \
        ["malaysian", "thai"]
    assert schema.edge_values({"cuisine": ""}, spec) == []


def test_edge_values_respects_max():
    spec = schema.EdgeSpec("c", "X", "R", split=";", max_values=2)
    assert len(schema.edge_values({"c": "a;b;c;d"}, spec)) == 2


def test_province_resolution_prefers_explicit_then_address():
    spec = next(s for s in schema.SOURCES if s.label == "Mosque")
    assert schema.resolve_province({"addr_province": "Songkhla"}, spec) == ("สงขลา", "explicit")
    assert schema.resolve_province({"addr_full": "อ.เมือง จ.ยะลา"}, spec) == ("ยะลา", "address")
    assert schema.resolve_province({}, spec) == ("", "")


def test_every_source_spec_is_wired():
    assert {s.label for s in schema.SOURCES} == {
        "Restaurant", "Hotel", "Store", "Attraction", "Mosque", "Product"}
    for spec in schema.SOURCES:
        assert spec.key and spec.name_cols
        for edge in spec.edges:
            assert edge.label in schema.ATTRIBUTE_LABELS
            assert edge.rel in schema.REL_TH, f"{edge.rel} missing a Thai label"


def test_node_text_includes_province_and_region():
    spec = next(s for s in schema.SOURCES if s.label == "Mosque")
    text = schema.node_text("Mosque", {"name": "มัสยิดกลางปัตตานี"}, spec, "ปัตตานี")
    assert "มัสยิด" in text and "ปัตตานี" in text and "ภาคใต้" in text


# ── curated ruling table ────────────────────────────────────────────────────

@pytest.mark.skipif(not RULINGS.exists(), reason="ruling table not present")
def test_ruling_table_invariants():
    df = pd.read_csv(RULINGS).fillna("")
    assert len(df) >= 80
    assert set(df["ruling"]) <= {"halal", "haram", "mashbooh"}
    for col in ("ingredient_th", "source_th", "ruling", "basis_th", "source_type"):
        assert (df[col].astype(str).str.strip() != "").all(), f"{col} has blanks"
    # One row per (ingredient, source) pair — a duplicate would create two
    # contradictory HAS_RULING edges for the same provenance.
    assert not df.duplicated(subset=["ingredient_th", "source_th"]).any()


@pytest.mark.skipif(not RULINGS.exists(), reason="ruling table not present")
def test_ruling_table_has_source_dependent_ingredients():
    """Extension C only demonstrates anything if some ingredients genuinely disagree."""
    df = pd.read_csv(RULINGS)
    conflicted = df.groupby("ingredient_th")["ruling"].nunique()
    assert (conflicted > 1).sum() >= 15


@pytest.mark.skipif(not RULINGS.exists(), reason="ruling table not present")
def test_pork_derived_is_always_haram():
    df = pd.read_csv(RULINGS)
    pork = df[df["source_type"] == "animal_prohibited"]
    assert len(pork) >= 10
    assert set(pork["ruling"]) == {"haram"}


@pytest.mark.skipif(not TIMELINE.exists(), reason="timeline not present")
def test_regulation_timeline_years_are_buddhist_era():
    df = pd.read_csv(TIMELINE).fillna("")
    assert len(df) >= 5
    assert df["valid_from"].astype(int).between(2500, 2600).all()
