from __future__ import annotations

import pytest

from app.config import Settings, get_settings
from app.grader import grade_document, grade_grounded, parse_binary

# --- config ------------------------------------------------------------

def test_settings_have_sane_defaults():
    s = get_settings()
    assert s.chunk_size > s.chunk_overlap
    assert s.retrieval_k >= 1
    assert s.ollama_base_url.startswith("http")


def test_settings_read_from_env(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "1234")
    assert get_settings().chunk_size == 1234


def test_settings_ignore_non_numeric_env(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_K", "not-a-number")
    assert get_settings().retrieval_k == 4  # falls back to default


def test_settings_reject_overlap_larger_than_chunk():
    with pytest.raises(ValueError, match="chunk_overlap"):
        Settings(chunk_size=100, chunk_overlap=100)


def test_settings_reject_zero_k():
    with pytest.raises(ValueError, match="retrieval_k"):
        Settings(retrieval_k=0)


def test_empty_api_key_becomes_none(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    assert get_settings().openai_api_key is None


# --- grader parsing ----------------------------------------------------

@pytest.mark.parametrize(
    "text",
    ["yes", "Yes", "YES", "yes.", "Yes, this document is relevant.", "  yes  "],
)
def test_parse_binary_accepts_positives(text):
    assert parse_binary(text) is True


@pytest.mark.parametrize("text", ["no", "No.", "NO", "no, unrelated"])
def test_parse_binary_accepts_negatives(text):
    assert parse_binary(text) is False


def test_parse_binary_handles_not_relevant_phrase():
    # "not relevant" contains "relevant" -- naive matching would return True.
    assert parse_binary("This document is not relevant to the question.") is False


def test_parse_binary_uses_default_when_ambiguous():
    assert parse_binary("perhaps, hard to say", default=True) is True
    assert parse_binary("perhaps, hard to say", default=False) is False


def test_parse_binary_handles_empty_string():
    assert parse_binary("", default=True) is True


def test_parse_binary_only_inspects_the_opening():
    # A verbose model may ramble; we only trust the first 200 chars.
    text = "yes" + " padding" * 100 + " no"
    assert parse_binary(text) is True


# --- grader integration with a fake LLM --------------------------------

def test_grade_document_relevant(fake_llm):
    assert grade_document(fake_llm, "what is HPA?", "HPA scales pods") is True


def test_grade_document_irrelevant(fake_llm):
    fake_llm.grade = "no"
    assert grade_document(fake_llm, "what is HPA?", "unrelated text") is False


def test_grade_document_truncates_long_input(fake_llm):
    grade_document(fake_llm, "q", "x" * 9999)
    assert len(fake_llm.calls[0]) < 3000


def test_grade_grounded_defaults_true_on_ambiguous(fake_llm):
    fake_llm.grounded = "unclear"
    # Hallucination check fails open: we don't discard an answer on a vague grade.
    assert grade_grounded(fake_llm, "docs", "answer") is True


def test_grade_grounded_detects_unsupported(fake_llm):
    fake_llm.grounded = "no"
    assert grade_grounded(fake_llm, "docs", "invented answer") is False
