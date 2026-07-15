"""Web search fallback for the corrective-RAG loop.

Used only when the local corpus cannot answer. Results are returned as
Documents with the origin URL in metadata, so the generate node cites a real
source and the grounding check still has something to check against.

Deliberately raw httpx rather than a LangChain integration: one endpoint, one
POST, and a client we can fake in tests without patching a vendor package.
"""
from __future__ import annotations

import logging

import httpx
from langchain_core.documents import Document

from app.config import get_settings

log = logging.getLogger(__name__)

TAVILY_ENDPOINT = "https://api.tavily.com/search"


class WebSearchUnavailable(RuntimeError):
    """No search backend is configured."""


class TavilyClient:
    """Thin Tavily wrapper. Returns [] on any failure -- search is a fallback,
    not a hard dependency, so a search outage must not take down /ask."""

    def __init__(self, api_key: str, timeout: float = 10.0) -> None:
        if not api_key:
            raise WebSearchUnavailable("TAVILY_API_KEY is not set")
        self.api_key = api_key
        self.timeout = timeout

    def search(self, query: str, max_results: int = 3) -> list[dict]:
        payload = {
            "api_key": self.api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",
        }
        try:
            resp = httpx.post(TAVILY_ENDPOINT, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json().get("results", [])
        except Exception as exc:  # noqa: BLE001 - degrade, never crash the request
            log.warning("Web search failed for %r: %s", query, exc)
            return []


def results_to_documents(results: list[dict]) -> list[Document]:
    """Convert raw search hits into Documents tagged as web-origin."""
    docs: list[Document] = []
    for item in results:
        content = (item.get("content") or "").strip()
        if not content:
            continue
        url = item.get("url", "unknown")
        docs.append(
            Document(
                page_content=content,
                metadata={
                    "source": url,
                    "title": item.get("title", ""),
                    "origin": "web",
                },
            )
        )
    return docs


def get_web_search_client():
    """Return a client, or None when web search is disabled/unconfigured.

    None is a normal state: the agent falls back to an honest "I don't know".
    """
    settings = get_settings()
    if not settings.enable_web_search:
        return None
    if not settings.tavily_api_key:
        log.info("Web search enabled but TAVILY_API_KEY is unset; disabling fallback")
        return None
    return TavilyClient(settings.tavily_api_key)
