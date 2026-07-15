"""Corrective RAG (CRAG) agent built on LangGraph.

Flow:

    retrieve -> grade_documents -> [relevant?]
                                     |-- yes --> generate -> check_grounded -> END
                                     |-- no  --> transform_query -> retrieve  (bounded retries)
                                                 |-- out of retries --> web_search -> generate
                                                     |-- no web backend --> generate (with a
                                                         "no supporting context" instruction)

The agent never answers from the model's parametric memory. The corpus is the
first hop; the web is the fallback; if neither yields anything it says so.

The "corrective" part: instead of blindly stuffing whatever the retriever
returned into the prompt, we grade each document, drop the irrelevant ones, and
rewrite the question if nothing survived. Bounded retries keep it from looping.
"""
from __future__ import annotations

import logging
from typing import TypedDict
from urllib.parse import urlparse

from langchain_core.documents import Document
from langgraph.graph import END, StateGraph

from app.config import get_settings
from app.grader import grade_document, grade_grounded

log = logging.getLogger(__name__)


class GraphState(TypedDict, total=False):
    question: str
    original_question: str
    documents: list[Document]
    generation: str
    attempts: int
    grounded: bool
    provider: str
    web_searched: bool
    used_web: bool


ANSWER_PROMPT = """You are a precise assistant. Answer the question using ONLY the context below.
If the context does not contain the answer, say you don't know. Do not invent facts.
Cite sources inline as [source: <filename>].

Context:
{context}

Question: {question}

Answer:"""

NO_CONTEXT_PROMPT = """The knowledge base contains no relevant information for this question.

Question: {question}

Reply in one sentence stating that you don't have information on this topic.
Do not attempt to answer from memory."""

REWRITE_PROMPT = """Rewrite the question below to be a better vector-search query.
Expand abbreviations, add likely synonyms and domain keywords. Keep the original intent.
Return ONLY the rewritten question, nothing else.

Original question: {question}

Rewritten question:"""


def source_label(doc: Document) -> str:
    """Human-meaningful citation label.

    Corpus docs are file paths -> use the filename.
    Web docs are URLs -> use the domain. Taking the last path segment of a URL
    yields things like "122227403774048080", which is worse than useless in a
    citation: it looks like a reference but identifies nothing.
    """
    source = doc.metadata.get("source", "unknown")
    if doc.metadata.get("origin") == "web" or source.startswith(("http://", "https://")):
        domain = urlparse(source).netloc
        return domain.removeprefix("www.") or "web"
    return source.split("/")[-1]


def format_context(docs: list[Document]) -> str:
    """Render documents into a prompt-ready block with source labels."""
    parts = []
    for doc in docs:
        parts.append(f"[source: {source_label(doc)}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def _text(response) -> str:
    if isinstance(response, str):
        return response
    return str(getattr(response, "content", "") or "")


def build_graph(retriever, llm, web_client=None):
    """Compile the CRAG graph.

    All three dependencies are injected so tests can pass fakes. `web_client`
    is optional -- pass None to disable the web fallback, in which case an
    unanswerable question returns an honest "I don't know".
    """
    settings = get_settings()

    # ---- nodes --------------------------------------------------------

    def retrieve(state: GraphState) -> GraphState:
        question = state["question"]
        docs = retriever.invoke(question)
        log.info("retrieve: %d docs for %r", len(docs), question)
        return {"documents": docs}

    def grade_documents(state: GraphState) -> GraphState:
        question = state["question"]
        kept: list[Document] = []
        for doc in state.get("documents", []):
            if grade_document(llm, question, doc.page_content):
                kept.append(doc)
            else:
                log.info("grade: dropped %s", doc.metadata.get("source", "?"))
        log.info("grade: kept %d/%d docs", len(kept), len(state.get("documents", [])))
        return {"documents": kept}

    def transform_query(state: GraphState) -> GraphState:
        attempts = state.get("attempts", 0) + 1
        rewritten = _text(llm.invoke(REWRITE_PROMPT.format(question=state["question"]))).strip()
        rewritten = rewritten.split("\n")[0].strip().strip('"').strip() or state["question"]
        log.info("transform (attempt %d): %r -> %r", attempts, state["question"], rewritten)
        return {"question": rewritten, "attempts": attempts}

    def web_search(state: GraphState) -> GraphState:
        """Last resort: search the web using the (possibly rewritten) query."""
        from app.websearch import results_to_documents

        query = state["question"]
        results = web_client.search(query, max_results=settings.web_search_results)
        docs = results_to_documents(results)
        log.info("web_search: %d results for %r", len(docs), query)
        return {"documents": docs, "web_searched": True, "used_web": bool(docs)}

    def generate(state: GraphState) -> GraphState:
        docs = state.get("documents", [])
        question = state.get("original_question") or state["question"]

        if not docs:
            answer = _text(llm.invoke(NO_CONTEXT_PROMPT.format(question=question))).strip()
            return {"generation": answer, "grounded": True}

        answer = _text(
            llm.invoke(ANSWER_PROMPT.format(context=format_context(docs), question=question))
        ).strip()
        return {"generation": answer}

    def check_grounded(state: GraphState) -> GraphState:
        docs = state.get("documents", [])
        if not docs:
            return {"grounded": True}
        grounded = grade_grounded(llm, format_context(docs), state.get("generation", ""))
        if not grounded:
            log.warning("check_grounded: answer not supported by context")
        return {"grounded": grounded}

    # ---- conditional edges --------------------------------------------

    def decide_after_grading(state: GraphState) -> str:
        if state.get("documents"):
            return "generate"
        if state.get("attempts", 0) < settings.max_transform_attempts:
            return "transform_query"
        if web_client is not None and not state.get("web_searched"):
            log.info("decide: corpus exhausted, falling back to web search")
            return "web_search"
        log.info("decide: no context available, answering honestly")
        return "generate"

    # ---- wiring -------------------------------------------------------

    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("web_search", web_search)
    workflow.add_node("generate", generate)
    workflow.add_node("check_grounded", check_grounded)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_after_grading,
        {
            "generate": "generate",
            "transform_query": "transform_query",
            "web_search": "web_search",
        },
    )
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", "check_grounded")
    workflow.add_edge("check_grounded", END)

    return workflow.compile()


def answer_question(question: str, retriever, llm, web_client=None) -> GraphState:
    """Convenience wrapper: run the graph once for a single question."""
    graph = build_graph(retriever, llm, web_client)
    return graph.invoke(
        {
            "question": question,
            "original_question": question,
            "attempts": 0,
            "web_searched": False,
            "used_web": False,
        }
    )
