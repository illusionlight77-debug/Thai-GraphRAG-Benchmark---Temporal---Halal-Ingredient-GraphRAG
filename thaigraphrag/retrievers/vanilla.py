"""Vanilla RAG baseline — dense retrieval over KG node text (no graph structure).

embed(query) → Qdrant top-k node texts → concatenate. This is the control condition
the GraphRAG retriever is compared against.
"""
from __future__ import annotations

import time

from thaigraphrag.config import get_settings
from thaigraphrag.core import embeddings
from thaigraphrag.core.qdrant_client import NODES, get_qdrant
from thaigraphrag.retrievers.base import RetrievedContext, Retriever


class VanillaRetriever(Retriever):
    name = "vanilla"

    def retrieve(self, query: str) -> RetrievedContext:
        s = get_settings()
        t0 = time.time()
        hits = get_qdrant().query_points(
            collection_name=NODES,
            query=embeddings.embed(query),
            limit=s.top_k,
            with_payload=True,
        ).points
        texts, prov = [], []
        for h in hits:
            pl = h.payload or {}
            texts.append(pl.get("text", ""))
            prov.append(str(pl.get("node_key", h.id)))
        return RetrievedContext(
            text="\n".join(texts),
            provenance=prov,
            meta={"strategy": "vanilla", "k": s.top_k,
                  "latency_s": round(time.time() - t0, 3), "llm_calls": 0},
        )
