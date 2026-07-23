"""Answer pipeline — retrieve → ground. Identical for every retriever.

This function is the fair-comparison guarantee in code: it takes a `Retriever`,
calls `retrieve`, and hands the resulting context to the *same* prompt and model
regardless of which retriever produced it. Nothing here may branch on
`retriever.name`.
"""
from __future__ import annotations

import time

from thaigraphrag.core import llm
from thaigraphrag.retrievers.base import Retriever

NO_CONTEXT = "ไม่พบข้อมูลที่เกี่ยวข้อง"


def answer(query: str, retriever: Retriever) -> dict:
    ctx = retriever.retrieve(query)
    t0 = time.time()
    if ctx.text:
        res = llm.ground_detailed(query, ctx.text)
        text = res.text or NO_CONTEXT
    else:
        # No retrieval → no LLM call. Recorded as zero cost rather than skipped,
        # so a retriever that often finds nothing is penalised on quality, not hidden.
        res = llm.LLMResult(text=NO_CONTEXT)
        text = NO_CONTEXT
    ground_latency = round(time.time() - t0, 3)

    return {
        "retriever": retriever.name,
        "answer": text,
        "context": ctx.text,
        "provenance": ctx.provenance,
        "retrieve_meta": ctx.meta,
        "ground_latency_s": ground_latency,
        "usage": res.as_meta(),
        "llm_error": res.error,
    }
