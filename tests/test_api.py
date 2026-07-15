from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from app import api as api_module


class FakeGraph:
    def __init__(self, result: dict) -> None:
        self.result = result
        self.invocations: list[dict] = []

    def invoke(self, state: dict) -> dict:
        self.invocations.append(state)
        return self.result


@pytest.fixture
def client(monkeypatch):
    """Bypass lifespan startup -- we inject a fake graph instead of loading models."""
    fake = FakeGraph(
        {
            "generation": "HPA scales on CPU. [source: kubernetes.md]",
            "grounded": True,
            "attempts": 0,
            "used_web": False,
            "documents": [
                Document(
                    page_content="HPA scales replicas.",
                    metadata={"source": "docs/kubernetes.md"},
                )
            ],
        }
    )
    api_module._state["graph"] = fake
    api_module._state["provider"] = "openai"
    # No `with` block: that would run the lifespan handler, which loads the
    # embedding model and Chroma. We inject a fake graph instead.
    c = TestClient(api_module.app)
    c.fake_graph = fake
    yield c
    api_module._state.clear()


def test_health_reports_active_provider(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["provider"] == "openai"


def test_health_reports_web_search_state(client):
    assert client.get("/health").json()["web_search"] is False


def test_root_points_at_the_docs(client):
    body = client.get("/").json()
    assert body["docs"] == "/docs"
    assert "Agentic RAG" in body["service"]


def test_ask_returns_answer_and_sources(client):
    resp = client.post("/ask", json={"question": "How does HPA work?"})
    assert resp.status_code == 200
    body = resp.json()
    assert "HPA scales on CPU" in body["answer"]
    assert body["grounded"] is True
    assert body["sources"][0]["source"] == "docs/kubernetes.md"


def test_ask_passes_original_question_into_state(client):
    client.post("/ask", json={"question": "How does HPA work?"})
    state = client.fake_graph.invocations[0]
    assert state["question"] == "How does HPA work?"
    assert state["original_question"] == "How does HPA work?"
    assert state["attempts"] == 0
    assert state["web_searched"] is False


def test_ask_reports_used_web_flag(client):
    client.fake_graph.result["used_web"] = True
    body = client.post("/ask", json={"question": "capital of France?"}).json()
    assert body["used_web"] is True


def test_ask_rejects_too_short_question(client):
    assert client.post("/ask", json={"question": "hi"}).status_code == 422


def test_ask_rejects_missing_question(client):
    assert client.post("/ask", json={}).status_code == 422


def test_ask_rejects_oversized_question(client):
    assert client.post("/ask", json={"question": "x" * 1001}).status_code == 422


def test_ask_returns_503_when_agent_not_initialised():
    api_module._state.clear()
    resp = TestClient(api_module.app).post("/ask", json={"question": "anything at all"})
    assert resp.status_code == 503


def test_health_is_degraded_without_any_provider(monkeypatch):
    api_module._state.clear()
    monkeypatch.setattr(
        api_module,
        "resolve_provider",
        lambda: (_ for _ in ()).throw(api_module.ProviderUnavailableError("none")),
    )
    body = TestClient(api_module.app).get("/health").json()
    assert body["status"] == "degraded"
    assert body["provider"] is None


def test_source_snippets_are_truncated(client):
    client.fake_graph.result["documents"] = [
        Document(page_content="y" * 900, metadata={"source": "big.md"})
    ]
    body = client.post("/ask", json={"question": "long doc question"}).json()
    assert len(body["sources"][0]["snippet"]) == 200
