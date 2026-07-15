"""Document ingestion: load -> chunk -> embed -> persist to Chroma."""
from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings

log = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".md", ".txt", ".rst"}


def load_documents(source_dir: str | Path) -> list[Document]:
    """Read every supported file under `source_dir` into a Document."""
    root = Path(source_dir)
    if not root.exists():
        raise FileNotFoundError(f"Source directory not found: {root}")

    docs: list[Document] = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in SUPPORTED_SUFFIXES or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            log.warning("Skipping empty file: %s", path)
            continue
        docs.append(Document(page_content=text, metadata={"source": str(path)}))

    log.info("Loaded %d documents from %s", len(docs), root)
    return docs


def chunk_documents(docs: list[Document]) -> list[Document]:
    """Split documents into overlapping chunks sized for the embedding model."""
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    log.info("Split %d documents into %d chunks", len(docs), len(chunks))
    return chunks


def get_embeddings():
    """Local CPU embeddings -- ~90MB model, no API key, no per-call cost."""
    from langchain_huggingface import HuggingFaceEmbeddings

    settings = get_settings()
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vectorstore(chunks: list[Document] | None = None):
    """Open the Chroma collection, optionally adding chunks first."""
    from langchain_chroma import Chroma

    settings = get_settings()
    Path(settings.chroma_dir).mkdir(parents=True, exist_ok=True)

    store = Chroma(
        collection_name=settings.collection,
        embedding_function=get_embeddings(),
        persist_directory=settings.chroma_dir,
    )
    if chunks:
        store.add_documents(chunks)
        log.info("Indexed %d chunks into collection '%s'", len(chunks), settings.collection)
    return store


def ingest(source_dir: str | Path = "docs") -> int:
    """Full pipeline. Returns the number of chunks indexed."""
    chunks = chunk_documents(load_documents(source_dir))
    if not chunks:
        log.warning("Nothing to ingest from %s", source_dir)
        return 0
    build_vectorstore(chunks)
    return len(chunks)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    count = ingest()
    print(f"Indexed {count} chunks.")
