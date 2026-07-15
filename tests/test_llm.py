from __future__ import annotations

import httpx
import pytest

from app import llm as llm_module
from app.llm import ProviderUnavailableError, ollama_available, resolve_provider


class _Resp:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


def test_ollama_available_when_daemon_responds(monkeypatch):
    monkeypatch.setattr(llm_module.httpx, "get", lambda *a, **k: _Resp(200))
    assert ollama_available("http://localhost:11434") is True


def test_ollama_unavailable_on_bad_status(monkeypatch):
    monkeypatch.setattr(llm_module.httpx, "get", lambda *a, **k: _Resp(500))
    assert ollama_available("http://localhost:11434") is False


def test_ollama_probe_swallows_connection_errors(monkeypatch):
    def boom(*a, **k):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(llm_module.httpx, "get", boom)
    # An unreachable daemon is a normal state, not an exception.
    assert ollama_available("http://localhost:11434") is False


def test_ollama_probe_strips_trailing_slash(monkeypatch):
    seen = {}

    def capture(url, **kwargs):
        seen["url"] = url
        return _Resp(200)

    monkeypatch.setattr(llm_module.httpx, "get", capture)
    ollama_available("http://localhost:11434/")
    assert seen["url"] == "http://localhost:11434/api/tags"


def test_prefers_ollama_when_available(monkeypatch):
    monkeypatch.setattr(llm_module, "ollama_available", lambda *a, **k: True)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert resolve_provider() == "ollama"


def test_falls_back_to_openai_when_ollama_down(monkeypatch):
    monkeypatch.setattr(llm_module, "ollama_available", lambda *a, **k: False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert resolve_provider() == "openai"


def test_raises_when_no_provider_available(monkeypatch):
    monkeypatch.setattr(llm_module, "ollama_available", lambda *a, **k: False)
    with pytest.raises(ProviderUnavailableError, match="No LLM backend"):
        resolve_provider()


def test_force_provider_overrides_probe(monkeypatch):
    monkeypatch.setattr(llm_module, "ollama_available", lambda *a, **k: True)
    monkeypatch.setenv("FORCE_PROVIDER", "openai")
    assert resolve_provider() == "openai"


def test_force_provider_rejects_unknown_value(monkeypatch):
    monkeypatch.setenv("FORCE_PROVIDER", "banana")
    with pytest.raises(ValueError, match="FORCE_PROVIDER"):
        resolve_provider()
