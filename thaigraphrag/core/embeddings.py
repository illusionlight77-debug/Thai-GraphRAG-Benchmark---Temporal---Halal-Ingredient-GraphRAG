"""bge-m3 embeddings via a local TEI container (HTTP). Thai-aware, 1024-d.

Matches the chatbot project's embedding setup so both systems stay consistent.

TEI enforces both a max **client batch size** and a max **batch token** budget and
answers 413 when either is exceeded. Rather than making callers guess a safe batch,
`embed_many` splits on 413 and retries transient failures — the KG build pushes
~70k texts through here, so one bad batch must not lose the whole run.
"""
from __future__ import annotations

import time

import httpx

from thaigraphrag.config import get_settings

_MAX_RETRIES = 4
_TIMEOUT = 120.0


def _post(texts: list[str]) -> list[list[float]]:
    url = get_settings().embedding_url.rstrip("/") + "/embed"
    r = httpx.post(url, json={"inputs": texts, "truncate": True}, timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()


def embed(text: str) -> list[float]:
    return embed_many([text or ""])[0]


def embed_many(texts: list[str]) -> list[list[float]]:
    """Call TEI /embed. Returns one 1024-d vector per input, order preserved."""
    if not texts:
        return []
    texts = [t if (t and t.strip()) else "-" for t in texts]

    last_error: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            return _post(texts)
        except httpx.HTTPStatusError as e:
            last_error = e
            # 413 = batch too large for TEI; halve it and recurse.
            if e.response.status_code == 413 and len(texts) > 1:
                mid = len(texts) // 2
                return embed_many(texts[:mid]) + embed_many(texts[mid:])
            if e.response.status_code < 500:
                raise
            time.sleep(2 ** attempt)
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError) as e:
            last_error = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"TEI /embed failed after {_MAX_RETRIES} attempts") from last_error


def health() -> bool:
    """True when the TEI container is up and has finished loading the model."""
    try:
        url = get_settings().embedding_url.rstrip("/") + "/health"
        return httpx.get(url, timeout=5).status_code == 200
    except Exception:
        return False


def wait_ready(timeout_s: int = 900, interval_s: int = 5) -> bool:
    """Block until TEI is serving. bge-m3 downloads ~2GB on the very first boot."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if health():
            return True
        time.sleep(interval_s)
    return False
