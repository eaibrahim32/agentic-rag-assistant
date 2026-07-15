"""FastAPI wrapper around the CRAG agent.

The vector store and LLM are built once at startup, not per request -- loading
the embedding model on every call would dominate latency.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.config import get_settings
from app.graph import build_graph
from app.llm import ProviderUnavailableError, resolve_provider

log = logging.getLogger(__name__)

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.ingest import build_vectorstore
    from app.llm import get_llm
    from app.websearch import get_web_search_client

    settings = get_settings()
    log.info("Loading vector store from %s", settings.chroma_dir)
    store = build_vectorstore()
    retriever = store.as_retriever(search_kwargs={"k": settings.retrieval_k})
    llm = get_llm()
    web_client = get_web_search_client()

    _state["graph"] = build_graph(retriever, llm, web_client)
    _state["provider"] = resolve_provider()
    _state["web_search"] = web_client is not None
    log.info("Ready. Provider=%s web_search=%s", _state["provider"], _state["web_search"])
    yield
    _state.clear()


app = FastAPI(title="Agentic RAG Knowledge Assistant", version="1.0.0", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)


class Source(BaseModel):
    source: str
    snippet: str


class AskResponse(BaseModel):
    answer: str
    grounded: bool
    rewrites: int
    used_web: bool
    sources: list[Source]
    provider: str


@app.get("/")
def root() -> dict:
    return {
        "service": "Agentic RAG Knowledge Assistant",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health() -> dict:
    try:
        provider = _state.get("provider") or resolve_provider()
    except ProviderUnavailableError:
        provider = None
    return {
        "status": "ok" if provider else "degraded",
        "provider": provider,
        "web_search": _state.get("web_search", False),
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    graph = _state.get("graph")
    if graph is None:
        raise HTTPException(status_code=503, detail="Agent not initialised")

    result = graph.invoke(
        {
            "question": req.question,
            "original_question": req.question,
            "attempts": 0,
            "web_searched": False,
            "used_web": False,
        }
    )
    return AskResponse(
        answer=result.get("generation", ""),
        grounded=result.get("grounded", False),
        rewrites=result.get("attempts", 0),
        used_web=result.get("used_web", False),
        sources=[
            Source(
                source=d.metadata.get("source", "unknown"),
                snippet=d.page_content[:200],
            )
            for d in result.get("documents", [])
        ],
        provider=_state.get("provider", "unknown"),
    )
