"""LLM grounding + LLM-as-judge over any OpenAI-compatible endpoint (Groq default).

`ground()` turns retrieved context into a Thai answer.
`judge_faithfulness()` scores whether an answer is supported by the context (0/1).

Both return real **token/call usage** alongside the text so the benchmark can report
cost per retriever. The system prompt and decoding parameters are defined here once
and are identical for every retriever — the fair-comparison invariant means the only
thing that may differ between conditions is the CONTEXT string itself.
"""
from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from functools import lru_cache

import httpx

from thaigraphrag.config import get_settings

_SYSTEM = (
    "You are a helpful assistant. Answer the user's question in Thai using ONLY the "
    "CONTEXT provided. If the context does not contain the answer, say you don't know. "
    "Be concise and factual."
)

_JUDGE_SYSTEM = (
    "You are a strict grader. Reply with a single digit: 1 if the ANSWER is "
    "fully supported by the CONTEXT, 0 otherwise."
)

_TIMEOUT = 90.0
_MAX_RETRIES = 6
_RETRY_STATUS = (429, 500, 502, 503, 504)


class RateLimiter:
    """Sliding-window token pacer for the provider's tokens-per-minute cap.

    Groq's free tier allows 6,000 tokens/minute. A benchmark run pushes hundreds of
    thousands of tokens through, so without pacing almost every call hits 429 — and a
    429 that outlives its retries returns an empty answer, which the pipeline scores
    as a wrong answer. Silent contamination of the results is far worse than a slow
    run, so requests are paced *before* they are sent rather than only retried after.

    Budget is charged optimistically on an estimate, then corrected with the real
    usage from the response.
    """

    def __init__(self, tokens_per_minute: int):
        self.tpm = max(tokens_per_minute, 0)
        self._window: list[tuple[float, int]] = []   # (timestamp, tokens)
        self._lock = threading.Lock()

    def _prune(self, now: float) -> None:
        cutoff = now - 60.0
        while self._window and self._window[0][0] < cutoff:
            self._window.pop(0)

    def used(self) -> int:
        with self._lock:
            self._prune(time.time())
            return sum(t for _, t in self._window)

    def acquire(self, estimated_tokens: int) -> float:
        """Block until `estimated_tokens` fit in the window. Returns seconds waited."""
        if not self.tpm:
            return 0.0
        waited = 0.0
        while True:
            with self._lock:
                now = time.time()
                self._prune(now)
                used = sum(t for _, t in self._window)
                if used + estimated_tokens <= self.tpm or not self._window:
                    self._window.append((now, estimated_tokens))
                    return waited
                # Sleep until the oldest entry leaves the window.
                sleep_for = max(0.5, 60.0 - (now - self._window[0][0]) + 0.25)
            time.sleep(sleep_for)
            waited += sleep_for

    def correct(self, estimated: int, actual: int) -> None:
        """Replace the estimate with the real token count once it is known."""
        if not self.tpm or actual <= 0:
            return
        with self._lock:
            for i in range(len(self._window) - 1, -1, -1):
                if self._window[i][1] == estimated:
                    self._window[i] = (self._window[i][0], actual)
                    return


# Reserving the full `max_tokens` for every call wastes most of the budget: the
# answers here are short factual sentences (measured: 15-80 completion tokens) while
# max_tokens is 512, so a run reserves ~3x the tokens it actually spends and crawls at
# a third of the allowed rate. Reserve a realistic completion instead — `correct()`
# trues the window up with the real usage as soon as the response lands, and the 429
# path still covers the occasional underestimate safely.
_COMPLETION_RESERVE = 96


def _estimate_tokens(system: str, user: str, max_tokens: int) -> int:
    """Rough prompt cost. Thai is ~2-3 chars/token for llama-family tokenisers."""
    return int((len(system) + len(user)) / 2.5) + min(max_tokens, _COMPLETION_RESERVE) + 16


@lru_cache
def _limiter() -> RateLimiter:
    return RateLimiter(get_settings().llm_tokens_per_minute)


def _retry_after(response) -> float | None:
    """Seconds the provider asked us to wait, from whichever header it used."""
    h = response.headers
    for key in ("retry-after", "x-ratelimit-reset-tokens", "x-ratelimit-reset-requests"):
        raw = h.get(key)
        if not raw:
            continue
        m = re.match(r"^\s*(?:(\d+(?:\.\d+)?)m)?(\d+(?:\.\d+)?)s?\s*$", str(raw))
        if m:
            minutes = float(m.group(1) or 0)
            return minutes * 60 + float(m.group(2))
    return None


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    latency_s: float = 0.0
    error: str = ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def as_meta(self) -> dict:
        return {"prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
                "llm_calls": self.calls,
                "llm_latency_s": round(self.latency_s, 3)}


@dataclass
class Usage:
    """Process-wide accounting, so a benchmark run can report total spend."""
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    errors: int = 0
    by_model: dict = field(default_factory=dict)

    def add(self, model: str, r: LLMResult) -> None:
        self.calls += r.calls
        self.prompt_tokens += r.prompt_tokens
        self.completion_tokens += r.completion_tokens
        if r.error:
            self.errors += 1
        m = self.by_model.setdefault(model, {"calls": 0, "tokens": 0})
        m["calls"] += r.calls
        m["tokens"] += r.total_tokens

    def reset(self) -> None:
        self.calls = self.prompt_tokens = self.completion_tokens = self.errors = 0
        self.by_model = {}


USAGE = Usage()


def _chat(model: str, system: str, user: str, temperature: float = 0.0,
          max_tokens: int = 512) -> LLMResult:
    s = get_settings()
    t0 = time.time()
    estimate = _estimate_tokens(system, user, max_tokens)
    last = ""

    for attempt in range(_MAX_RETRIES):
        _limiter().acquire(estimate)
        try:
            r = httpx.post(
                s.llm_base_url.rstrip("/") + "/chat/completions",
                headers={"Authorization": f"Bearer {s.llm_api_key}"},
                json={"model": model, "temperature": temperature,
                      "max_tokens": max_tokens,
                      "messages": [{"role": "system", "content": system},
                                   {"role": "user", "content": user}]},
                timeout=_TIMEOUT,
            )
            if r.status_code in _RETRY_STATUS:
                last = f"HTTP {r.status_code}"
                # Honour the provider's own reset hint; guessing short backoffs is
                # what let rate-limited calls fall through as empty answers before.
                wait = _retry_after(r)
                time.sleep(min(wait + 0.5, 70.0) if wait else min(2 ** attempt + 1, 30))
                continue
            r.raise_for_status()
            data = r.json()
            usage = data.get("usage") or {}
            total = int(usage.get("prompt_tokens", 0)) + int(usage.get("completion_tokens", 0))
            _limiter().correct(estimate, total)
            res = LLMResult(
                text=data["choices"][0]["message"]["content"].strip(),
                prompt_tokens=int(usage.get("prompt_tokens", 0)),
                completion_tokens=int(usage.get("completion_tokens", 0)),
                calls=1, latency_s=time.time() - t0,
            )
            USAGE.add(model, res)
            return res
        except Exception as e:                       # noqa: BLE001 — reported, not raised
            last = f"{type(e).__name__}: {e}"
            time.sleep(min(2 ** attempt, 30))

    res = LLMResult(text="", calls=0, latency_s=time.time() - t0, error=last)
    USAGE.add(model, res)
    return res


def ground_detailed(question: str, context: str) -> LLMResult:
    """Answer `question` from `context`; usage included."""
    s = get_settings()
    if not s.llm_api_key:
        # No key → return the raw context so the pipeline still runs offline.
        return LLMResult(text=context[:500], error="no LLM_API_KEY")
    user = f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER (ภาษาไทย):"
    return _chat(s.llm_model, _SYSTEM, user)


def ground(question: str, context: str) -> str:
    """Back-compatible string API."""
    return ground_detailed(question, context).text


def judge_faithfulness(question: str, answer: str, context: str) -> float:
    """Return 1.0 if the answer is supported by the context, else 0.0 (NaN if no key)."""
    s = get_settings()
    if not s.llm_api_key:
        return float("nan")
    user = (f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\n"
            f"ANSWER: {answer}\n\nGrade (0/1):")
    res = _chat(s.judge_model, _JUDGE_SYSTEM, user, max_tokens=8)
    if res.error and not res.text:
        return float("nan")
    m = re.search(r"[01]", res.text)
    return float(m.group()) if m else float("nan")


def available() -> bool:
    return bool(get_settings().llm_api_key)
