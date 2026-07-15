from __future__ import annotations

from langchain_core.documents import Document

from app.graph import answer_question, build_graph, format_context, source_label
from tests.conftest import FakeLLM, FakeRetriever

# --- context formatting -------------------------------------------------

def test_format_context_labels_each_source(sample_docs):
    out = format_context(sample_docs)
    assert "[source: k8s.md]" in out
    assert "[source: terraform.md]" in out


def test_format_context_handles_missing_metadata():
    out = format_context([Document(page_content="orphan", metadata={})])
    assert "unknown" in out


def test_format_context_of_empty_list_is_empty():
    assert format_context([]) == ""


# --- happy path ---------------------------------------------------------

def test_relevant_docs_produce_grounded_answer(fake_retriever, fake_llm):
    result = answer_question("How does HPA work?", fake_retriever, fake_llm)
    assert result["generation"]
    assert result["grounded"] is True
    assert len(result["documents"]) == 2


def test_happy_path_does_not_rewrite_the_query(fake_retriever, fake_llm):
    answer_question("How does HPA work?", fake_retriever, fake_llm)
    assert fake_retriever.queries == ["How does HPA work?"]


def test_original_question_is_used_for_generation(fake_retriever, fake_llm):
    answer_question("How does HPA work?", fake_retriever, fake_llm)
    answer_prompts = [c for c in fake_llm.calls if "precise assistant" in c]
    assert "How does HPA work?" in answer_prompts[0]


# --- grading drops noise -------------------------------------------------

def test_irrelevant_docs_are_filtered_out(fake_retriever):
    # First doc relevant, second not.
    llm = FakeLLM(grade=["yes", "no"])
    result = answer_question("How does HPA work?", fake_retriever, llm)
    assert len(result["documents"]) == 1
    assert "Horizontal Pod Autoscaler" in result["documents"][0].page_content


# --- corrective loop ------------------------------------------------------

def test_all_docs_irrelevant_triggers_query_rewrite(fake_retriever):
    llm = FakeLLM(grade="no")
    answer_question("vague question", fake_retriever, llm)
    # Retrieved once with the original, once with the rewritten query.
    assert len(fake_retriever.queries) == 2
    assert fake_retriever.queries[1] == "kubernetes horizontal pod autoscaler CPU scaling"


def test_rewrite_is_bounded_by_retry_budget(fake_retriever, monkeypatch):
    monkeypatch.setenv("MAX_TRANSFORM_ATTEMPTS", "1")
    llm = FakeLLM(grade="no")
    result = answer_question("vague question", fake_retriever, llm)
    # Must terminate rather than loop forever.
    assert result["attempts"] == 1
    assert len(fake_retriever.queries) == 2


def test_higher_retry_budget_allows_more_rewrites(fake_retriever, monkeypatch):
    monkeypatch.setenv("MAX_TRANSFORM_ATTEMPTS", "2")
    llm = FakeLLM(grade="no")
    result = answer_question("vague question", fake_retriever, llm)
    assert result["attempts"] == 2
    assert len(fake_retriever.queries) == 3


def test_exhausted_retries_answers_without_context(fake_retriever):
    llm = FakeLLM(grade="no")
    result = answer_question("off-topic question", fake_retriever, llm)
    assert result["documents"] == []
    assert "don't have information" in result["generation"]


def test_no_context_path_never_uses_the_answer_prompt(fake_retriever):
    llm = FakeLLM(grade="no")
    answer_question("off-topic question", fake_retriever, llm)
    # The model must not be invited to answer from parametric memory.
    assert not any("precise assistant" in c for c in llm.calls)


def test_empty_retriever_still_terminates():
    llm = FakeLLM(grade="no")
    result = answer_question("anything", FakeRetriever([]), llm)
    assert result["generation"]
    assert result["grounded"] is True


def test_rewrite_falls_back_to_original_when_model_returns_blank(fake_retriever):
    llm = FakeLLM(grade="no", rewrite="")
    answer_question("keep me", fake_retriever, llm)
    assert fake_retriever.queries[1] == "keep me"


def test_rewrite_strips_quotes_and_extra_lines(fake_retriever):
    llm = FakeLLM(grade="no", rewrite='"clean query"\nsome trailing junk')
    answer_question("q", fake_retriever, llm)
    assert fake_retriever.queries[1] == "clean query"


# --- hallucination check --------------------------------------------------

def test_unsupported_answer_is_flagged(fake_retriever):
    llm = FakeLLM(grounded="no")
    result = answer_question("How does HPA work?", fake_retriever, llm)
    assert result["grounded"] is False


def test_grounded_check_skipped_when_no_documents(fake_retriever):
    llm = FakeLLM(grade="no", grounded="no")
    result = answer_question("off-topic", fake_retriever, llm)
    # Nothing to be grounded against, so the no-context reply is not penalised.
    assert result["grounded"] is True


# --- graph structure ------------------------------------------------------

def test_graph_compiles(fake_retriever, fake_llm):
    graph = build_graph(fake_retriever, fake_llm)
    assert graph is not None


def test_graph_exposes_expected_nodes(fake_retriever, fake_llm):
    graph = build_graph(fake_retriever, fake_llm)
    nodes = set(graph.get_graph().nodes)
    for expected in ("retrieve", "grade_documents", "transform_query", "generate"):
        assert expected in nodes


# --- web search fallback --------------------------------------------------

def test_web_search_fires_when_corpus_cannot_answer(fake_retriever, fake_web_client):
    llm = FakeLLM(grade="no")
    result = answer_question(
        "What is the capital of France?", fake_retriever, llm, fake_web_client
    )
    assert result["used_web"] is True
    assert result["documents"]
    assert "Paris" in result["documents"][0].page_content


def test_web_search_only_runs_after_the_rewrite_budget(fake_retriever, fake_web_client):
    llm = FakeLLM(grade="no")
    answer_question("q", fake_retriever, llm, fake_web_client)
    # Corpus gets the original query and one rewrite before the web is touched.
    assert len(fake_retriever.queries) == 2
    assert len(fake_web_client.queries) == 1


def test_web_search_uses_the_rewritten_query(fake_retriever, fake_web_client):
    llm = FakeLLM(grade="no", rewrite="france capital city")
    answer_question("q", fake_retriever, llm, fake_web_client)
    assert fake_web_client.queries[0] == "france capital city"


def test_web_results_are_cited_as_sources(fake_retriever, fake_web_client):
    llm = FakeLLM(grade="no")
    result = answer_question("q", fake_retriever, llm, fake_web_client)
    assert result["documents"][0].metadata["source"].startswith("https://")
    assert result["documents"][0].metadata["origin"] == "web"


def test_web_search_does_not_run_when_corpus_answers(fake_retriever, fake_web_client):
    llm = FakeLLM(grade="yes")
    result = answer_question("How does HPA work?", fake_retriever, llm, fake_web_client)
    assert fake_web_client.queries == []
    assert result["used_web"] is False


def test_web_search_runs_at_most_once(fake_retriever, fake_web_client):
    # Web returns nothing useful; the agent must not loop back for more.
    fake_web_client.results = []
    llm = FakeLLM(grade="no")
    result = answer_question("q", fake_retriever, llm, fake_web_client)
    assert len(fake_web_client.queries) == 1
    assert result["documents"] == []
    assert "don't have information" in result["generation"]


def test_empty_web_results_fall_back_to_honest_refusal(fake_retriever, fake_web_client):
    fake_web_client.results = []
    llm = FakeLLM(grade="no")
    result = answer_question("q", fake_retriever, llm, fake_web_client)
    assert result["used_web"] is False
    assert not any("precise assistant" in c for c in llm.calls)


def test_no_web_client_preserves_honest_refusal(fake_retriever):
    llm = FakeLLM(grade="no")
    result = answer_question("q", fake_retriever, llm, None)
    assert result["documents"] == []
    assert "don't have information" in result["generation"]


def test_web_answer_still_passes_the_grounding_check(fake_retriever, fake_web_client):
    llm = FakeLLM(grade="no", grounded="no")
    result = answer_question("q", fake_retriever, llm, fake_web_client)
    # Grounding is enforced on web context exactly as on corpus context.
    assert result["grounded"] is False


def test_graph_exposes_web_search_node(fake_retriever, fake_llm, fake_web_client):
    graph = build_graph(fake_retriever, fake_llm, fake_web_client)
    assert "web_search" in set(graph.get_graph().nodes)


# --- citation labels ------------------------------------------------------

def test_source_label_uses_filename_for_corpus_docs():
    doc = Document(page_content="x", metadata={"source": "docs/kubernetes.md"})
    assert source_label(doc) == "kubernetes.md"


def test_source_label_uses_domain_for_web_docs():
    doc = Document(
        page_content="x",
        metadata={"source": "https://www.britannica.com/place/France", "origin": "web"},
    )
    # Not "France" -- the last path segment of a URL is not a citation.
    assert source_label(doc) == "britannica.com"


def test_source_label_never_returns_an_opaque_id():
    doc = Document(
        page_content="x",
        metadata={
            "source": "https://www.facebook.com/groups/376100836095453/posts/2780652808973565",
            "origin": "web",
        },
    )
    assert source_label(doc) == "facebook.com"


def test_source_label_detects_urls_without_origin_metadata():
    doc = Document(page_content="x", metadata={"source": "https://example.com/a/b"})
    assert source_label(doc) == "example.com"


def test_source_label_handles_missing_source():
    assert source_label(Document(page_content="x", metadata={})) == "unknown"


def test_format_context_labels_web_docs_by_domain():
    docs = [
        Document(
            page_content="Paris is the capital.",
            metadata={"source": "https://en.wikipedia.org/wiki/Paris", "origin": "web"},
        )
    ]
    assert "[source: en.wikipedia.org]" in format_context(docs)
