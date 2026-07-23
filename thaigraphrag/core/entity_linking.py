"""Thai entity linking — map a natural-language query to seed KG nodes.

Thai has no word delimiters, so the query is matched against the graph's own
vocabulary rather than against tokens. Three passes run in order of precision and
their candidates are merged:

  1. **gazetteer**  — longest-match of attribute-node names (จังหวัด, ภาค, ประเภทอาหาร,
     แบรนด์, หมวดหมู่ …) that are read straight out of the graph and cached. Exact and
     cheap, and it is what makes "ในภาคใต้" or "ร้านสะดวกซื้อ" resolve to a real node.
  2. **full-text**  — Neo4j full-text index over place/product names, built with the
     `thai` analyzer so Lucene segments Thai words rather than treating a whole
     question as one token. This covers the long tail of proper nouns the gazetteer
     cannot hold ("น้ำตกโตนงาช้าง", "Mooz").
  3. **char n-gram + CONTAINS** — the original fallback, used when the two passes above
     come back empty (e.g. before the full-text index exists).

Scoring is a **Dice coefficient** between the matched span and the node's name:
`2·|span| / (|name| + |span|)`. An exact name match scores 1.0, while a node whose
name merely contains the span scores lower the longer the rest of its name is — so
querying "ปัตตานี" prefers the Province node over "มัสยิดกลางปัตตานี". Lucene's own
score is *not* used for ranking: it reflects term frequency across the index, not how
well a name answers the question.

Precision then depends on three guards, each added after watching the real system
return a confidently wrong seed:

  * names that are ordinary Thai words need their qualifier (`ภาคกลาง`, not `กลาง`);
  * a shared fragment that is only a function word ("จังหวัด") does not count as a match;
  * seeds scoring far below the best seed are dropped, so a single-entity question is
    not padded out with near-misses whose facts are noise.

Deliberately **no vector search here.** Entity linking is inside the GraphRAG path,
and reaching into Qdrant would turn GraphRAG into "vanilla + graph", confounding the
controlled comparison this whole repo is built around (CLAUDE.md §2).

Return shape is stable: `[{node_id, label, name, score, ...}]`, best first.
"""
from __future__ import annotations

import re
import time

from thaigraphrag.core import neo4j_client

# Function words and domain words that must never become entity candidates on
# their own — they match half the graph and drown out the real entity.
_STOPWORDS = {
    "ฮาลาล", "ร้าน", "ร้านอาหาร", "ใกล้", "ใน", "ที่", "มี", "ไหน", "อะไร", "จังหวัด",
    "อยู่", "คือ", "ของ", "และ", "หรือ", "กับ", "ให้", "ได้", "เป็น", "ไม่", "จาก",
    "แห่ง", "บ้าง", "ใด", "กี่", "ชื่อ", "ประเภท", "แบบ", "ทั้งหมด", "เดียวกัน",
    "อะไรบ้าง", "ที่ไหน", "เท่าไหร่", "อย่างไร", "ทำไม", "ภาค", "อำเภอ", "เขต",
}

# Attribute labels whose names are small enough to hold in memory as a gazetteer.
_GAZETTEER_LABELS = (
    "Province", "Region", "Cuisine", "StoreType", "TourismType", "Amenity",
    "Category", "Brand", "Denomination", "District", "Source", "Ruling", "Ingredient",
)

# Some attribute names are ordinary Thai words that appear in almost any question.
# Region names are the worst offenders: 'กลาง' matches "มัสยิด**กลาง**ปัตตานี" and
# 'ใต้' matches "ข้าง**ใต้**". These labels only match when the query uses the name
# with its qualifying prefix, which is how Thai actually writes them.
_REQUIRED_PREFIX = {
    "Region": ("ภาค",),
    "District": ("อำเภอ", "เขต", "อ."),
}
# Names with fewer than this many Thai *letters* need the prefix even for labels not
# listed above — 'เลย', 'ตาก' and 'น่าน' are provinces and also everyday Thai words.
_SHORT_NAME_LEN = 4

# Generic words that are node names in the graph but never a useful seed on their own.
_GENERIC_NAMES = {
    "อื่นๆ", "อื่น ๆ", "other", "others", "ทั่วไป", "regional", "yes", "no", "only",
    "เมือง", "ทารก", "อาหาร", "นม", "ปลา", "ไข่", "แป้ง",
}

FULLTEXT_INDEX = "entity_names"

_gazetteer_cache: tuple[float, list[dict]] | None = None
_GAZETTEER_TTL_S = 300


# ── helpers ─────────────────────────────────────────────────────────────────

# Thai combining marks: above/below vowels and tone marks. They occupy a code point
# but not a letter position, so `len()` overstates how distinctive a Thai name is —
# "น่าน" is len 4 but only three letters, and "เลย" is len 3 and an everyday word.
_COMBINING = set(
    "ั" + "".join(chr(c) for c in range(0x0E34, 0x0E3B))
    + "".join(chr(c) for c in range(0x0E47, 0x0E4F))
)


def thai_len(s: str) -> int:
    """Letter count ignoring Thai combining vowels and tone marks."""
    return sum(1 for ch in s if ch not in _COMBINING)


def _dice(span: str, name: str) -> float:
    """Dice similarity between the matched span and the full node name."""
    if not span or not name:
        return 0.0
    return 2 * len(span) / (len(name) + len(span))


def _candidates(query: str, min_len: int = 3, max_len: int = 12) -> list[str]:
    """Candidate spans of the query, most promising first, stopwords dropped.

    Whitespace-delimited tokens come first. Thai runs together, but the names that
    actually need this fallback — latin brand names, transliterations — are written
    with spaces around them, so a token is usually the exact entity string.

    Ordering matters because the caller only issues a bounded number of queries: a
    pure longest-first n-gram list spends its whole budget on 12-character Thai
    substrings and never reaches a 4-character name like "Mooz".
    """
    q = query.strip()
    tokens = [t.strip() for t in re.split(r"\s+", q)]
    ordered = [t for t in tokens
               if len(t) >= min_len and t not in _STOPWORDS]

    spans = set()
    for n in range(max_len, min_len - 1, -1):
        for i in range(0, len(q) - n + 1):
            span = q[i:i + n].strip()
            if span and span not in _STOPWORDS:
                spans.add(span)
    for span in sorted(spans, key=len, reverse=True):
        if span not in ordered:
            ordered.append(span)
    return ordered


def _clean_query(query: str) -> str:
    """Strip punctuation so spans do not straddle it."""
    return re.sub(r"[\s\.,!?;:()\[\]\"'`‘’“”]+", " ", query or "").strip()


# ── pass 1: gazetteer ───────────────────────────────────────────────────────

def _load_gazetteer() -> list[dict]:
    """All attribute-node names from the graph, cached for a few minutes."""
    global _gazetteer_cache
    now = time.time()
    if _gazetteer_cache and now - _gazetteer_cache[0] < _GAZETTEER_TTL_S:
        return _gazetteer_cache[1]
    labels = "|".join(_GAZETTEER_LABELS)
    try:
        rows = neo4j_client.run(
            f"""
            MATCH (n)
            WHERE any(l IN labels(n) WHERE l IN $labels)
              AND n.name IS NOT NULL AND size(n.name) >= 2
            RETURN elementId(n) AS node_id, labels(n)[0] AS label, n.name AS name
            LIMIT 40000
            """,
            labels=list(_GAZETTEER_LABELS),
        )
    except Exception:
        rows = []
    _ = labels
    _gazetteer_cache = (now, rows)
    return rows


def _accepts(name: str, label: str, q: str, ql: str) -> bool:
    """Is this gazetteer name a genuine mention in the query, not an accidental substring?"""
    prefixes = _REQUIRED_PREFIX.get(label)
    if prefixes is None and thai_len(name) < _SHORT_NAME_LEN:
        # Short names outside the prefix-controlled labels still need a qualifier,
        # otherwise 2-3 character node names match nearly every question.
        prefixes = ("จังหวัด", "ภาค", "อำเภอ", "เขต", "ประเภท", "แบรนด์", "ยี่ห้อ")
    if prefixes:
        return any((p + name) in q for p in prefixes)
    return name in q or name.lower() in ql


def _gazetteer_matches(query: str) -> list[dict]:
    q = _clean_query(query)
    ql = q.lower()
    out = []
    for row in _load_gazetteer():
        name = row["name"]
        if len(name) < 2 or name in _STOPWORDS or name.strip().lower() in _GENERIC_NAMES:
            continue
        if _accepts(name, row["label"], q, ql):
            out.append({**row, "score": _dice(name, name), "span": name,
                        "matched_by": "gazetteer"})
    # Drop names fully contained in a longer match ('ยะลา' inside 'ปัตตานี ยะลา' is fine,
    # but 'ตานี' inside 'ปัตตานี' is not a separate entity).
    spans = sorted({o["span"] for o in out}, key=len, reverse=True)
    shadowed = {s for s in spans if any(s != t and s in t for t in spans)}
    return [o for o in out if o["span"] not in shadowed]


# ── pass 2: Neo4j full-text ─────────────────────────────────────────────────

def _escape_lucene(text: str) -> str:
    return re.sub(r'([+\-!(){}\[\]^"~*?:\\/]|&&|\|\|)', r"\\\1", text)


def _lcs(a: str, b: str) -> str:
    """Longest common substring — the part of `b` that the query actually contains."""
    if not a or not b:
        return ""
    prev = [0] * (len(b) + 1)
    best, end = 0, 0
    for ch in a:
        cur = [0] * (len(b) + 1)
        for j, ch2 in enumerate(b, 1):
            if ch == ch2:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best, end = cur[j], j
        prev = cur
    return b[end - best:end]


def _lcs_len(a: str, b: str) -> int:
    return len(_lcs(a, b))


# A shared fragment that is only a function word proves nothing. Every Thai question
# about a place contains "จังหวัด", which would otherwise make every node whose name
# contains "จังหวัด" a candidate seed.
_WEAK_OVERLAPS = _STOPWORDS | {
    "จังหวัด", "อำเภอ", "ตำบล", "ที่พัก", "ร้านอาหาร", "สถานที่", "ประเทศไทย",
    "แห่งประเทศไทย", "กลางจังหวัด", "ในจังหวัด",
    # Bare category words. Some nodes are literally named just "มัสยิด" or "ห้องพัก";
    # they match perfectly and tell you nothing, so they are never useful seeds.
    "มัสยิด", "โรงแรม", "ห้องพัก", "ร้านค้า", "ร้าน", "ตลาด", "วัด", "โรงเรียน",
    "น้ำตก", "สินค้า", "ผลิตภัณฑ์", "อาหาร",
}
# Minimum re-scored similarity for a full-text hit to become a seed.
_FULLTEXT_MIN_SCORE = 0.45
# A seed must score at least this fraction of the best seed's score to be kept.
_SEED_SCORE_RATIO = 0.75


def _fulltext_matches(query: str, limit: int) -> list[dict]:
    """Query the `entity_names` full-text index; silent no-op if it is absent.

    The **raw** query is handed to Lucene rather than a whitespace-split term list.
    The index is built with Neo4j's `thai` analyzer, which segments Thai properly —
    pre-splitting on whitespace turned a whole Thai question into one unmatched
    60-character term and made this pass useless for exactly the language it exists for.

    Lucene's score reflects term frequency across the index, not how well a name
    matches the question, so each hit is re-scored with Dice over the longest common
    substring: "น้ำตกโตนงาช้าง" scores 1.0 while the near-miss "น้ำตกโตนลาด" does not.
    """
    q = _clean_query(query)
    if not q:
        return []
    try:
        rows = neo4j_client.run(
            """
            CALL db.index.fulltext.queryNodes($index, $q, {limit: $limit})
            YIELD node, score
            RETURN elementId(node) AS node_id, labels(node)[0] AS label,
                   node.name AS name, score AS lucene
            """,
            index=FULLTEXT_INDEX, q=_escape_lucene(q), limit=limit * 4,
        )
    except Exception:
        return []

    ql = q.lower()
    out = []
    for r in rows:
        name = r.get("name") or ""
        bare = name.strip()
        if not name or bare.lower() in _GENERIC_NAMES or bare in _WEAK_OVERLAPS:
            continue
        # The prefix guard has to apply here too, not only to the gazetteer: Lucene
        # happily segments "มัสยิด|กลาง|ปัตตานี" and hands back the Region "กลาง".
        if r["label"] in _REQUIRED_PREFIX and not _accepts(name, r["label"], q, ql):
            continue
        span = _lcs(q, name)
        if len(span) < 2 or span.strip() in _WEAK_OVERLAPS:
            continue
        score = 2 * len(span) / (len(name) + len(span))
        if score < _FULLTEXT_MIN_SCORE:
            continue
        out.append({"node_id": r["node_id"], "label": r["label"], "name": name,
                    "score": round(score, 4), "span": span,
                    "matched_by": "fulltext"})
    return out


# ── pass 3: char n-gram + CONTAINS (fallback) ───────────────────────────────

def _contains_matches(query: str, limit: int, max_spans: int = 12) -> list[dict]:
    """Last-resort scan. Capped at `max_spans` Neo4j round-trips.

    Uncapped this issued one CONTAINS query per n-gram — measured at 8.2s for a short
    query, which is slower than the retrieval it is meant to support. The longest
    spans are the informative ones, and `_candidates` already returns them first.
    """
    seen, results = set(), []
    for span in _candidates(_clean_query(query))[:max_spans]:
        rows = neo4j_client.run(
            """
            MATCH (n)
            WHERE n.name CONTAINS $span OR coalesce(n.name_en,'') CONTAINS $span
            RETURN elementId(n) AS node_id, labels(n)[0] AS label,
                   coalesce(n.name, n.name_en) AS name
            LIMIT 10
            """,
            span=span,
        )
        for r in rows:
            if r["node_id"] in seen:
                continue
            seen.add(r["node_id"])
            results.append({**r, "score": round(_dice(span, r["name"] or ""), 4),
                            "span": span, "matched_by": "contains"})
        if len(results) >= limit * 3:
            break
    return results


# ── public API ──────────────────────────────────────────────────────────────

def link_entities(query: str, limit: int = 5) -> list[dict]:
    """Return seed nodes `[{node_id, label, name, score, span, matched_by}]`, best first.

    Requires a populated Neo4j KG; returns an empty list when nothing matches.
    """
    if not query or not query.strip():
        return []

    candidates = _gazetteer_matches(query) + _fulltext_matches(query, limit)
    if not candidates:
        candidates = _contains_matches(query, limit)

    # Merge duplicates, keeping the best-scoring evidence for each node.
    best: dict[str, dict] = {}
    for c in candidates:
        if not c.get("name"):
            continue
        prev = best.get(c["node_id"])
        if prev is None or c["score"] > prev["score"]:
            best[c["node_id"]] = c

    ranked = sorted(best.values(), key=lambda x: (-x["score"], len(x["name"])))
    if not ranked:
        return []

    # Relative cutoff. A question naming two entities scores both near 1.0, so they
    # both survive; a single-entity question is otherwise padded out to `limit` with
    # near-misses whose facts are pure noise in the grounding context.
    floor = ranked[0]["score"] * _SEED_SCORE_RATIO

    # Diversity: one seed per distinct entity name, so a query mentioning two
    # entities gets both instead of five spellings of the same one.
    out, used_names = [], set()
    for c in ranked:
        if c["score"] < floor:
            break
        key = c["name"].strip().lower()
        if key in used_names:
            continue
        used_names.add(key)
        out.append(c)
        if len(out) >= limit:
            break
    return out


def clear_cache() -> None:
    """Drop the gazetteer cache — call after rebuilding the graph."""
    global _gazetteer_cache
    _gazetteer_cache = None
