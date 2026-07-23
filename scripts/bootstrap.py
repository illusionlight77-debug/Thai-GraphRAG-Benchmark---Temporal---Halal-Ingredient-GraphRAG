"""First-boot bootstrap for the `app` container.

Runs before uvicorn in the compose command:

    wait for neo4j + qdrant + TEI  →  build KG (if empty)  →  build B and C layers
    →  run the benchmark once  →  hand over to the server

Every step is idempotent and individually skippable, because a cold `docker compose
up` has to survive three slow, racy things at once: Neo4j taking ~20s to accept bolt,
TEI downloading ~2GB of bge-m3 weights, and the KG build itself taking tens of
minutes on CPU. Nothing here aborts the boot — if a step fails the API still comes
up and reports the failure on /health and the Overview page, which is far easier to
debug than a container that exits.

Env flags (see .env.example):
    SKIP_BOOTSTRAP=1     skip everything below — normal for later boots
    SKIP_KG_BUILD=1      keep the graph as-is
    SKIP_BENCHMARK=1     do not run the benchmark on boot
    BOOTSTRAP_LIMIT=n    build only the first n products (fast smoke boot)
"""
from __future__ import annotations

import os
import sys
import time

from thaigraphrag.config import get_settings
from thaigraphrag.core import embeddings, neo4j_client
from thaigraphrag.core import qdrant_client as qc


def _flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


def log(msg: str) -> None:
    print(f"[bootstrap] {msg}", flush=True)


def wait_neo4j(timeout_s: int = 180) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            neo4j_client.run("RETURN 1 AS ok")
            return True
        except Exception:
            time.sleep(3)
    return False


def wait_qdrant(timeout_s: int = 120) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            qc.get_qdrant().get_collections()
            return True
        except Exception:
            time.sleep(3)
    return False


def graph_is_empty() -> bool:
    try:
        rows = neo4j_client.run("MATCH (n) RETURN count(n) AS c")
        # The 77 provinces + 6 regions are created by every build, so a graph that
        # only holds the hierarchy still counts as empty.
        return rows[0]["c"] <= 100
    except Exception:
        return True


def main() -> int:
    if _flag("SKIP_BOOTSTRAP"):
        log("SKIP_BOOTSTRAP=1 — serving immediately")
        return 0

    s = get_settings()
    log(f"neo4j={s.neo4j_uri} qdrant={s.qdrant_url} tei={s.embedding_url}")

    if not wait_neo4j():
        log("!! Neo4j never became reachable — starting the API anyway")
        return 0
    log("Neo4j ready")

    if not wait_qdrant():
        log("!! Qdrant never became reachable — starting the API anyway")
        return 0
    log("Qdrant ready")

    # bge-m3 downloads ~2GB on the very first boot; only the KG build needs it.
    log("waiting for TEI (first boot downloads ~2GB of bge-m3 weights)…")
    if embeddings.wait_ready(timeout_s=1800):
        log("TEI ready")
    else:
        log("!! TEI not ready — skipping embedding-dependent steps")
        return 0

    limit = int(os.getenv("BOOTSTRAP_LIMIT", "0") or 0)

    if _flag("SKIP_KG_BUILD"):
        log("SKIP_KG_BUILD=1 — leaving the graph as-is")
    elif not graph_is_empty():
        log("graph already populated — skipping KG build "
            "(set SKIP_KG_BUILD=0 and wipe the volume to rebuild)")
    else:
        log("building the knowledge graph — this takes a while on CPU…")
        try:
            from thaigraphrag.kg.build_kg import build
            build(limit=limit or None)
            log("KG build finished")
        except Exception as e:
            log(f"!! KG build failed: {type(e).__name__}: {e}")
            return 0

    for name, fn in (("temporal (B)", _build_temporal), ("ingredient (C)", _build_ingredient)):
        try:
            fn()
            log(f"{name} layer ready")
        except Exception as e:
            log(f"!! {name} layer failed: {type(e).__name__}: {e}")

    if _flag("SKIP_BENCHMARK"):
        log("SKIP_BENCHMARK=1 — not running the benchmark")
    else:
        try:
            from thaigraphrag.benchmark.run_benchmark import SUITES, run_suites
            log("running the benchmark once…")
            run_suites(list(SUITES), judge=_flag("BOOTSTRAP_JUDGE"))
            log("benchmark finished — results/ populated")
        except Exception as e:
            log(f"!! benchmark failed: {type(e).__name__}: {e}")

    log("bootstrap complete")
    return 0


def _build_temporal() -> None:
    from thaigraphrag.extensions.temporal.temporal_kg import build
    build()


def _build_ingredient() -> None:
    from thaigraphrag.extensions.halal_ingredient.ingredient_kg import build
    build()


if __name__ == "__main__":
    sys.exit(main())
