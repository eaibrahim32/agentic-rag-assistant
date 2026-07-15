from __future__ import annotations

import pytest
from langchain_core.documents import Document

from app.ingest import chunk_documents, load_documents


def test_load_documents_reads_supported_files(tmp_path):
    (tmp_path / "a.md").write_text("# Alpha\ncontent here")
    (tmp_path / "b.txt").write_text("bravo content")
    docs = load_documents(tmp_path)
    assert len(docs) == 2


def test_load_documents_skips_unsupported_suffixes(tmp_path):
    (tmp_path / "keep.md").write_text("keep me")
    (tmp_path / "skip.pdf").write_text("binary-ish")
    (tmp_path / "skip.png").write_text("image")
    docs = load_documents(tmp_path)
    assert len(docs) == 1
    assert "keep" in docs[0].page_content


def test_load_documents_skips_empty_files(tmp_path):
    (tmp_path / "empty.md").write_text("   \n  ")
    (tmp_path / "real.md").write_text("real content")
    assert len(load_documents(tmp_path)) == 1


def test_load_documents_recurses_subdirectories(tmp_path):
    nested = tmp_path / "sub" / "deeper"
    nested.mkdir(parents=True)
    (nested / "deep.md").write_text("deep content")
    assert len(load_documents(tmp_path)) == 1


def test_load_documents_records_source_metadata(tmp_path):
    (tmp_path / "x.md").write_text("some content")
    docs = load_documents(tmp_path)
    assert docs[0].metadata["source"].endswith("x.md")


def test_load_documents_raises_on_missing_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path / "nope")


def test_chunk_documents_splits_long_text(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "100")
    monkeypatch.setenv("CHUNK_OVERLAP", "10")
    doc = Document(page_content="word " * 500, metadata={"source": "big.md"})
    chunks = chunk_documents([doc])
    assert len(chunks) > 1
    assert all(len(c.page_content) <= 120 for c in chunks)


def test_chunk_documents_preserves_metadata(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "100")
    monkeypatch.setenv("CHUNK_OVERLAP", "10")
    doc = Document(page_content="text " * 200, metadata={"source": "keep.md"})
    chunks = chunk_documents([doc])
    assert all(c.metadata["source"] == "keep.md" for c in chunks)


def test_chunk_documents_handles_empty_input():
    assert chunk_documents([]) == []


def test_chunk_documents_leaves_short_docs_intact():
    doc = Document(page_content="short", metadata={"source": "s.md"})
    chunks = chunk_documents([doc])
    assert len(chunks) == 1
    assert chunks[0].page_content == "short"
