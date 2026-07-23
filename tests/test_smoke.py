"""Smoke tests — pure logic that needs no Neo4j/Qdrant/network."""
from thaigraphrag.benchmark import metrics
from thaigraphrag.core import entity_linking


def test_metrics_exact_and_f1():
    assert metrics.exact_match("สงขลา", ["สงขลา"]) == 1.0
    assert metrics.containment("อยู่จังหวัดสงขลา", ["สงขลา"]) == 1.0
    assert metrics.f1("สงขลา", ["สงขลา"]) == 1.0
    assert metrics.f1("", ["สงขลา"]) == 0.0


def test_entity_candidates_drop_stopwords():
    cands = entity_linking._candidates("ร้านอาหารฮาลาลในจังหวัดสงขลา")
    assert "ฮาลาล" not in cands  # stopword removed
    assert any("สงขลา" in c for c in cands)


def test_retriever_factory():
    from thaigraphrag.retrievers import get_retriever
    assert get_retriever("vanilla").name == "vanilla"
    assert get_retriever("graphrag").name == "graphrag"
