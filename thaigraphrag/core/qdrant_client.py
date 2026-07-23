"""Qdrant vector store for the vanilla-RAG baseline.

Holds one point per KG node: vector = bge-m3(node text), payload = {node_id, label,
name, text}. The vanilla retriever searches this collection; GraphRAG does not use it.
"""
from __future__ import annotations

from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from thaigraphrag.config import get_settings

NODES = "kg_nodes"


@lru_cache
def get_qdrant() -> QdrantClient:
    s = get_settings()
    return QdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key or None)


def ensure_collection() -> None:
    client = get_qdrant()
    dim = get_settings().embedding_dim
    existing = {c.name for c in client.get_collections().collections}
    if NODES not in existing:
        client.create_collection(
            NODES, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))


def reset_collection() -> None:
    """Drop and recreate the collection — used by `build_kg --reset`."""
    client = get_qdrant()
    if NODES in {c.name for c in client.get_collections().collections}:
        client.delete_collection(NODES)
    ensure_collection()


def count() -> int:
    """Number of indexed node vectors (0 if the collection does not exist yet)."""
    try:
        return get_qdrant().count(NODES, exact=True).count
    except Exception:
        return 0
