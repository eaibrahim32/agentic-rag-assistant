from __future__ import annotations

import httpx
import pytest

from app import websearch as ws
from app.websearch import (
    TavilyClient,
    WebSearchUnavailable,
    get_web_search_client,
    results_to_documents,
)


class _Resp:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("boom", request=None, response=None)

    def json(self) -> dict:
        return self._payload


SAMPLE = {
    "results": [
        {
            "title": "Paris",
            "url": "https://en.wikipedia.org/wiki/Paris",
            "content": "Paris is the capital of France.",
        },
        {"title": "France", "url": "https://example.com/france", "content": "France is in Europe."},
    ]
}


# --- client -------------------------------------------------------------

def test_client_requires_api_key():
    with pytest.raises(WebSearchUnavailable):
        TavilyClient("")


def test_search_returns_results(monkeypatch):
    monkeypatch.setattr(ws.httpx, "post", lambda *a, **k: _Resp(SAMPLE))
    results = TavilyClient("tvly-test").search("capital of France")
    assert len(results) == 2
    assert results[0]["url"].endswith("Paris")


def test_search_sends_query_and_key(monkeypatch):
    captured = {}

    def capture(url, json=None, timeout=None):
        captured.update(json)
        return _Resp(SAMPLE)

    monkeypatch.setattr(ws.httpx, "post", capture)
    TavilyClient("tvly-secret").search("my query", max_results=5)
    assert captured["query"] == "my query"
    assert captured["api_key"] == "tvly-secret"
    assert captured["max_results"] == 5


def test_search_returns_empty_on_network_error(monkeypatch):
    def boom(*a, **k):
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(ws.httpx, "post", boom)
    # Search is a fallback, not a hard dependency: an outage must not 500 /ask.
    assert TavilyClient("tvly-test").search("anything") == []


def test_search_returns_empty_on_http_error(monkeypatch):
    monkeypatch.setattr(ws.httpx, "post", lambda *a, **k: _Resp({}, status_code=429))
    assert TavilyClient("tvly-test").search("rate limited") == []


def test_search_handles_missing_results_key(monkeypatch):
    monkeypatch.setattr(ws.httpx, "post", lambda *a, **k: _Resp({"unexpected": "shape"}))
    assert TavilyClient("tvly-test").search("q") == []


# --- document conversion ------------------------------------------------

def test_results_to_documents_maps_url_to_source():
    docs = results_to_documents(SAMPLE["results"])
    assert docs[0].metadata["source"] == "https://en.wikipedia.org/wiki/Paris"
    assert "capital of France" in docs[0].page_content


def test_results_to_documents_tags_web_origin():
    docs = results_to_documents(SAMPLE["results"])
    assert all(d.metadata["origin"] == "web" for d in docs)


def test_results_to_documents_skips_empty_content():
    docs = results_to_documents([{"url": "https://x.com", "content": "   "}])
    assert docs == []


def test_results_to_documents_handles_missing_fields():
    docs = results_to_documents([{"content": "orphan text"}])
    assert docs[0].metadata["source"] == "unknown"


def test_results_to_documents_of_empty_list():
    assert results_to_documents([]) == []


# --- factory ------------------------------------------------------------

def test_factory_returns_none_when_disabled(monkeypatch):
    monkeypatch.setenv("ENABLE_WEB_SEARCH", "false")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    assert get_web_search_client() is None


def test_factory_returns_none_when_key_missing(monkeypatch):
    monkeypatch.setenv("ENABLE_WEB_SEARCH", "true")
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    # Enabled but unconfigured degrades to "no fallback", not a crash.
    assert get_web_search_client() is None


def test_factory_returns_client_when_configured(monkeypatch):
    monkeypatch.setenv("ENABLE_WEB_SEARCH", "true")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    assert isinstance(get_web_search_client(), TavilyClient)


@pytest.mark.parametrize("value", ["true", "TRUE", "1", "yes", "Yes"])
def test_factory_accepts_truthy_spellings(monkeypatch, value):
    monkeypatch.setenv("ENABLE_WEB_SEARCH", value)
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    assert get_web_search_client() is not None
