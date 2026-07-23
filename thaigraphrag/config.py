"""Configuration loaded from environment (.env)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
QUESTIONS_DIR = ROOT_DIR / "data" / "questions"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Neo4j (knowledge graph)
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "change-me"

    # Qdrant (vanilla-RAG vector store)
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""

    # Embeddings (TEI / bge-m3)
    embedding_url: str = "http://localhost:8080"
    embedding_dim: int = 1024

    # LLM grounding + judge (OpenAI-compatible)
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_api_key: str = ""
    llm_model: str = "llama-3.1-8b-instant"
    judge_model: str = "llama-3.1-8b-instant"
    # Provider tokens-per-minute cap. Requests are paced to stay under it, because a
    # 429 that outlives its retries returns an empty answer that the benchmark would
    # score as a wrong answer. Groq's free tier is 6,000 TPM. Set 0 to disable pacing.
    llm_tokens_per_minute: int = 6000

    # Retrieval / GraphRAG
    top_k: int = 5
    graph_hops: int = 2
    similarity_threshold: float = 0.5
    max_triples: int = 60          # subgraph facts handed to the LLM per question

    # Data source (Halal KG CSVs)
    data_dir: str = "./data"
    province_geojson: str = "thailand_provinces.json"   # relative to data_dir

    # KG build
    # product_processed.csv holds 222k certified products; embedding all of them on
    # a CPU TEI takes hours and adds little to a tourism-multi-hop benchmark. The
    # build takes a deterministic, category-stratified sample of this size. Both
    # Neo4j *and* Qdrant receive exactly the same sample, so the comparison stays fair.
    # Set to 0 to ingest every row.
    max_product_rows: int = 20000
    embed_batch: int = 64

    # App / bootstrap
    skip_bootstrap: bool = False
    app_port: int = 8000


@lru_cache
def get_settings() -> Settings:
    return Settings()


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
