"""Central configuration, driven entirely by environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    # --- LLM providers -------------------------------------------------
    ollama_base_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    ollama_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b"))
    openai_api_key: str | None = field(default_factory=lambda: os.getenv("OPENAI_API_KEY") or None)
    openai_base_url: str | None = field(
        default_factory=lambda: os.getenv("OPENAI_BASE_URL") or None
    )
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    force_provider: str | None = field(default_factory=lambda: os.getenv("FORCE_PROVIDER") or None)

    # --- Embeddings / vector store -------------------------------------
    embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )
    chroma_dir: str = field(default_factory=lambda: os.getenv("CHROMA_DIR", "./data/chroma"))
    collection: str = field(default_factory=lambda: os.getenv("CHROMA_COLLECTION", "knowledge"))

    # --- Web search fallback -------------------------------------------
    enable_web_search: bool = field(
        default_factory=lambda: os.getenv("ENABLE_WEB_SEARCH", "false").lower()
        in {"1", "true", "yes"}
    )
    tavily_api_key: str | None = field(
        default_factory=lambda: os.getenv("TAVILY_API_KEY") or None
    )
    web_search_results: int = field(default_factory=lambda: _int("WEB_SEARCH_RESULTS", 3))

    # --- Chunking / retrieval ------------------------------------------
    chunk_size: int = field(default_factory=lambda: _int("CHUNK_SIZE", 800))
    chunk_overlap: int = field(default_factory=lambda: _int("CHUNK_OVERLAP", 120))
    retrieval_k: int = field(default_factory=lambda: _int("RETRIEVAL_K", 4))
    max_transform_attempts: int = field(default_factory=lambda: _int("MAX_TRANSFORM_ATTEMPTS", 1))

    def __post_init__(self) -> None:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.retrieval_k < 1:
            raise ValueError("retrieval_k must be >= 1")


def get_settings() -> Settings:
    """Build settings fresh so tests can monkeypatch env vars."""
    return Settings()
