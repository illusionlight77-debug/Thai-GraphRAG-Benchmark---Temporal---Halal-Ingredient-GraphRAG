"""Retriever registry — the single seam every retrieval strategy plugs into.

Core (A) is `vanilla` and `graphrag`. The extensions register themselves here so the
benchmark can compare them without any core file knowing they exist; extension
imports are lazy so that A still runs if an extension is removed (CLAUDE.md §2).
"""
from thaigraphrag.retrievers.base import RetrievedContext, Retriever
from thaigraphrag.retrievers.graphrag import GraphRAGRetriever
from thaigraphrag.retrievers.vanilla import VanillaRetriever

__all__ = [
    "RetrievedContext", "Retriever", "VanillaRetriever", "GraphRAGRetriever",
    "get_retriever", "available_retrievers",
]

CORE_RETRIEVERS = ("vanilla", "graphrag")
EXTENSION_RETRIEVERS = ("temporal", "halal_ingredient")


def get_retriever(name: str, **kwargs) -> Retriever:
    """Factory used by the benchmark runner and the API.

    'vanilla' | 'graphrag' | 'temporal' | 'halal_ingredient'.
    `kwargs` are passed to the retriever (e.g. `as_of=2570` for temporal).
    """
    name = (name or "").lower().strip()
    if name == "vanilla":
        return VanillaRetriever()
    if name == "graphrag":
        return GraphRAGRetriever(**kwargs)
    if name == "temporal":
        from thaigraphrag.extensions.temporal.temporal_retriever import (
            TemporalGraphRAGRetriever,
        )
        return TemporalGraphRAGRetriever(**kwargs)
    if name in ("halal_ingredient", "ingredient"):
        from thaigraphrag.extensions.halal_ingredient.explain_retriever import (
            IngredientExplainRetriever,
        )
        return IngredientExplainRetriever()
    raise ValueError(
        f"unknown retriever: {name!r} "
        f"(available: {', '.join(CORE_RETRIEVERS + EXTENSION_RETRIEVERS)})")


def available_retrievers() -> list[str]:
    """Names that `get_retriever` can actually construct in this install."""
    out = list(CORE_RETRIEVERS)
    for name in EXTENSION_RETRIEVERS:
        try:
            get_retriever(name)
            out.append(name)
        except Exception:      # extension removed or broken → core still works
            pass
    return out
