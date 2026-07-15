"""Shared fixtures.

The whole suite runs with fakes: no Ollama daemon, no API key, no model
download, no network. That is deliberate -- CI must stay fast and free.
"""
from __future__ import annotations

import pytest
from langchain_core.documents import Document


class FakeMessage:
    """Mimics a LangChain AIMessage (only .content is used downstream)."""

    def __init__(self, content: str) -> None:
        self.content = content


class FakeLLM:
    """Routes prompts to canned replies based on which prompt template fired.

    Keeps a call log so tests can assert on how the graph actually behaved.
    """

    def __init__(
        self,
        *,
        grade: str = "yes",
        grounded: str = "yes",
        answer: str = "Kubernetes HPA scales pods on CPU. [source: k8s.md]",
        rewrite: str = "kubernetes horizontal pod autoscaler CPU scaling",
        no_context: str = "I don't have information on that topic.",
    ) -> None:
        self.grade = grade
        self.grounded = grounded
        self.answer = answer
        self.rewrite = rewrite
        self.no_context = no_context
        self.calls: list[str] = []

    def invoke(self, prompt: str):
        self.calls.append(prompt)
        if "Rewrite the question" in prompt:
            return FakeMessage(self.rewrite)
        if "no relevant information" in prompt:
            return FakeMessage(self.no_context)
        if "grading whether a retrieved document" in prompt:
            grade = self.grade
            if isinstance(grade, list):
                grade = grade.pop(0) if grade else "no"
            return FakeMessage(grade)
        if "checking whether an answer is supported" in prompt:
            return FakeMessage(self.grounded)
        return FakeMessage(self.answer)


class FakeRetriever:
    """Returns a fixed document set and records every query it saw."""

    def __init__(self, docs: list[Document] | None = None) -> None:
        self.docs = docs if docs is not None else []
        self.queries: list[str] = []

    def invoke(self, query: str) -> list[Document]:
        self.queries.append(query)
        return list(self.docs)


class FakeWebClient:
    """Returns canned search hits and records the queries it was given."""

    def __init__(self, results: list[dict] | None = None) -> None:
        self.results = results if results is not None else [
            {
                "title": "Paris",
                "url": "https://en.wikipedia.org/wiki/Paris",
                "content": "Paris is the capital and most populous city of France.",
            }
        ]
        self.queries: list[str] = []

    def search(self, query: str, max_results: int = 3) -> list[dict]:
        self.queries.append(query)
        return list(self.results)


@pytest.fixture
def fake_web_client() -> FakeWebClient:
    return FakeWebClient()


@pytest.fixture
def sample_docs() -> list[Document]:
    return [
        Document(
            page_content="The Horizontal Pod Autoscaler scales replicas based on CPU utilization.",
            metadata={"source": "docs/k8s.md"},
        ),
        Document(
            page_content="Terraform provisions AWS EC2 instances declaratively from code.",
            metadata={"source": "docs/terraform.md"},
        ),
    ]


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def fake_retriever(sample_docs) -> FakeRetriever:
    return FakeRetriever(sample_docs)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Stop the developer's real .env from leaking into tests."""
    for var in (
        "FORCE_PROVIDER",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OLLAMA_BASE_URL",
        "CHUNK_SIZE",
        "CHUNK_OVERLAP",
        "RETRIEVAL_K",
        "MAX_TRANSFORM_ATTEMPTS",
        "ENABLE_WEB_SEARCH",
        "TAVILY_API_KEY",
        "WEB_SEARCH_RESULTS",
    ):
        monkeypatch.delenv(var, raising=False)
