"""Answer + retrieval metrics. NaN semantics matter: an unannotated item must be
excluded from a mean, not counted as a zero."""
import math

from thaigraphrag.benchmark import metrics


def test_containment_both_directions():
    assert metrics.containment("จังหวัดสงขลา", ["สงขลา"]) == 1.0
    assert metrics.containment("สงขลา", ["จังหวัดสงขลา"]) == 1.0
    assert metrics.containment("ปัตตานี", ["สงขลา"]) == 0.0
    assert metrics.containment("", ["สงขลา"]) == 0.0


def test_f1_partial_overlap_is_between_zero_and_one():
    score = metrics.f1("จังหวัดสงขลาและปัตตานี", ["สงขลา"])
    assert 0.0 < score < 1.0


def test_f1_takes_best_gold():
    assert metrics.f1("halal", ["ฮาลาล", "halal"]) == 1.0


def test_normalisation_ignores_punctuation_and_case():
    assert metrics.exact_match(" Songkhla. ", ["songkhla"]) == 1.0


def test_context_recall_is_the_answer_ceiling():
    ctx = "มัสยิดกลางปัตตานี -[ตั้งอยู่ในจังหวัด]-> ปัตตานี"
    assert metrics.context_recall(ctx, ["ปัตตานี"]) == 1.0
    assert metrics.context_recall(ctx, ["สงขลา"]) == 0.0
    assert metrics.context_recall("", ["ปัตตานี"]) == 0.0


def test_hit_at_k_is_nan_without_annotation():
    assert math.isnan(metrics.hit_at_k(["anything"], []))
    assert metrics.hit_at_k(["ปัตตานี ยะลา"], ["ปัตตานี", "ยะลา"]) == 1.0
    assert metrics.hit_at_k(["ปัตตานี"], ["ปัตตานี", "ยะลา"]) == 0.5
    assert metrics.hit_at_k([], ["ปัตตานี"]) == 0.0


def test_path_validity_is_order_sensitive():
    gold = ["เจลาติน", "สุกร", "ไม่ฮาลาล"]
    assert metrics.path_validity(["เจลาติน", "สุกร", "ไม่ฮาลาล"], gold) == 1.0
    # Right nodes, wrong order → not a valid explanation.
    assert metrics.path_validity(["ไม่ฮาลาล", "สุกร", "เจลาติน"], gold) < 1.0
    assert metrics.path_validity([], gold) == 0.0
    assert math.isnan(metrics.path_validity(["เจลาติน"], []))


def test_path_validity_partial_credit():
    gold = ["เจลาติน", "สุกร", "ไม่ฮาลาล"]
    assert metrics.path_validity(["เจลาติน", "สุกร"], gold) == 2 / 3
