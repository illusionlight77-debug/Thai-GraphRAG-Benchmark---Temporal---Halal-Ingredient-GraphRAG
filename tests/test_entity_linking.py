"""Entity-linking logic that needs no database.

Two regressions are pinned here because both were found by running the real system
and both silently degrade retrieval rather than raising:

1. Region "กลาง" matched "มัสยิด**กลาง**ปัตตานี" and became the top seed.
2. Capping the candidate list to the longest n-grams meant a short latin name like
   "Mooz" was never tried, so a question about an entity that *is* in the graph
   returned no seeds at all.
"""
from thaigraphrag.core import entity_linking as el


def test_dice_rewards_exact_name_match():
    assert el._dice("ปัตตานี", "ปัตตานี") == 1.0
    # A longer node name containing the span scores lower — the Province node should
    # beat "มัสยิดกลางปัตตานี" when the query is just the province.
    assert el._dice("ปัตตานี", "มัสยิดกลางปัตตานี") < 1.0
    assert el._dice("", "x") == 0.0


def test_candidates_put_whitespace_tokens_first():
    cands = el._candidates("ร้านอาหารชื่อ Mooz อยู่จังหวัดอะไร")
    assert "Mooz" in cands[:4], cands[:6]


def test_candidates_drop_stopwords():
    cands = el._candidates("ร้านอาหารฮาลาลในจังหวัดสงขลา")
    assert "ฮาลาล" not in cands
    assert any("สงขลา" in c for c in cands)


def test_region_names_require_their_prefix():
    """'กลาง' and 'ใต้' are ordinary Thai words; they only count with 'ภาค'."""
    q = "มัสยิดกลางปัตตานีตั้งอยู่จังหวัดใด"
    assert not el._accepts("กลาง", "Region", q, q.lower())
    q2 = "มัสยิดในภาคใต้"
    assert el._accepts("ใต้", "Region", q2, q2.lower())


def test_district_names_require_their_prefix():
    q = "อำเภอหาดใหญ่อยู่จังหวัดใด"
    assert el._accepts("หาดใหญ่", "District", q, q.lower())
    assert not el._accepts("หาดใหญ่", "District", "ร้านหาดใหญ่โภชนา", "ร้านหาดใหญ่โภชนา")


def test_long_names_match_without_a_prefix():
    q = "ร้านอาหารในจังหวัดนครศรีธรรมราช"
    assert el._accepts("นครศรีธรรมราช", "Province", q, q.lower())


def test_thai_len_ignores_combining_marks():
    """Python's len() overstates Thai names: tone marks and vowels are code points."""
    assert len("น่าน") == 4 and el.thai_len("น่าน") == 3
    assert len("เลย") == 3 and el.thai_len("เลย") == 3
    assert el.thai_len("นครศรีธรรมราช") == 12
    assert el.thai_len("Songkhla") == 8


def test_short_names_need_a_qualifier():
    # 'ตรัง' is four letters — distinctive enough to stand alone.
    q = "โรงแรมในจังหวัดตรัง"
    assert el._accepts("ตรัง", "Province", q, q.lower())
    # 'น่าน' is only three letters, and 'เลย'/'ตาก' are everyday Thai words, so all
    # of them must be qualified or they match mid-word.
    assert not el._accepts("น่าน", "Province", "ร้านอาหารน่านฟ้า", "ร้านอาหารน่านฟ้า")
    assert not el._accepts("เลย", "Province", "ไม่มีเลยสักร้าน", "ไม่มีเลยสักร้าน")
    assert el._accepts("น่าน", "Province", "จังหวัดน่านอยู่ภาคใด", "จังหวัดน่านอยู่ภาคใด")


def test_lcs_len_measures_shared_substring():
    """Full-text hits are re-ranked by this, not by Lucene's term-frequency score."""
    assert el._lcs_len("น้ำตกโตนงาช้างอยู่จังหวัดอะไร", "น้ำตกโตนงาช้าง") == len("น้ำตกโตนงาช้าง")
    # A near-miss shares only the common prefix, so it must score strictly lower.
    exact = el._lcs_len("น้ำตกโตนงาช้างอยู่ที่ไหน", "น้ำตกโตนงาช้าง")
    near = el._lcs_len("น้ำตกโตนงาช้างอยู่ที่ไหน", "น้ำตกโตนลาด")
    assert near < exact
    assert el._lcs_len("", "x") == 0 and el._lcs_len("x", "") == 0


def test_lucene_escaping_survives_special_characters():
    assert "\\-" in el._escape_lucene("7-Eleven")
    assert "\\:" in el._escape_lucene("a:b")


def test_clean_query_strips_punctuation():
    assert el._clean_query('  "ปัตตานี",  ') == "ปัตตานี"


def test_empty_query_links_nothing():
    assert el.link_entities("") == []
    assert el.link_entities("   ") == []
