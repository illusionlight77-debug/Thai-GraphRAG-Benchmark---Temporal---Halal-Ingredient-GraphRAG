"""Retriever interface — the seam that makes GraphRAG vs vanilla a fair comparison.

Both retrievers return the same `RetrievedContext`, so the answer pipeline and the
benchmark treat them identically; only the retrieval strategy differs.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class RetrievedContext:
    text: str                        # linearised context handed to the LLM
    provenance: list[str] = field(default_factory=list)  # node keys / ids used
    meta: dict = field(default_factory=dict)             # latency, hops, #nodes, ...


class Retriever(ABC):
    name: str = "base"

    @abstractmethod
    def retrieve(self, query: str) -> RetrievedContext:
        ...
