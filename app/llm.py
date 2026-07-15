"""Provider selection.

Order of preference:
  1. FORCE_PROVIDER env var, if set ("ollama" | "openai") -- used by CI and tests.
  2. Ollama, if a local daemon answers /api/tags. Zero cost, runs offline.
  3. OpenAI-compatible API, if OPENAI_API_KEY is present.

Any OpenAI-compatible endpoint works via OPENAI_BASE_URL (Groq, Together,
OpenRouter, a local vLLM, etc.), so the fallback is not vendor-locked.
"""
from __future__ import annotations

import logging

import httpx

from app.config import get_settings

log = logging.getLogger(__name__)


class ProviderUnavailableError(RuntimeError):
    """Raised when no LLM backend can be reached."""


def ollama_available(base_url: str, timeout: float = 2.0) -> bool:
    """Cheap health probe. Never raises -- an unreachable daemon is a normal state."""
    try:
        resp = httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=timeout)
        return resp.status_code == 200
    except Exception as exc:  # noqa: BLE001 - any transport error means "not available"
        log.debug("Ollama probe failed at %s: %s", base_url, exc)
        return False


def resolve_provider() -> str:
    """Return the provider name we would use right now: 'ollama' or 'openai'."""
    settings = get_settings()

    if settings.force_provider:
        forced = settings.force_provider.lower()
        if forced not in {"ollama", "openai"}:
            raise ValueError(f"FORCE_PROVIDER must be 'ollama' or 'openai', got {forced!r}")
        return forced

    if ollama_available(settings.ollama_base_url):
        return "ollama"
    if settings.openai_api_key:
        return "openai"

    raise ProviderUnavailableError(
        "No LLM backend available. Either start Ollama "
        f"(expected at {settings.ollama_base_url}) or set OPENAI_API_KEY."
    )


def get_llm(temperature: float = 0.0):
    """Return a LangChain chat model for the active provider."""
    settings = get_settings()
    provider = resolve_provider()

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        log.info("Using Ollama model=%s", settings.ollama_model)
        return ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=temperature,
        )

    from langchain_openai import ChatOpenAI

    log.info("Using OpenAI-compatible model=%s", settings.openai_model)
    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        temperature=temperature,
    )
