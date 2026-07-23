"""Neo4j connection helper (knowledge graph store)."""
from __future__ import annotations

from functools import lru_cache

from neo4j import Driver, GraphDatabase

from thaigraphrag.config import get_settings


@lru_cache
def get_driver() -> Driver:
    s = get_settings()
    kwargs: dict = {"auth": (s.neo4j_user, s.neo4j_password)}
    # Queries legitimately mention labels/relations that do not exist yet — the
    # extensions are optional layers, and `build_kg` runs before anything is created.
    # Neo4j warns per query about each one, which buries real errors in the logs.
    try:
        from neo4j import NotificationDisabledClassification

        kwargs["notifications_disabled_classifications"] = [
            NotificationDisabledClassification.UNRECOGNIZED,
        ]
    except ImportError:      # older driver — the warnings are cosmetic, carry on
        pass
    return GraphDatabase.driver(s.neo4j_uri, **kwargs)


def run(cypher: str, **params) -> list[dict]:
    """Execute a Cypher query and return rows as dicts."""
    with get_driver().session() as session:
        return [r.data() for r in session.run(cypher, **params)]


def close() -> None:
    """Close the pooled driver (used by tests and shutdown hooks)."""
    if get_driver.cache_info().currsize:
        get_driver().close()
        get_driver.cache_clear()
