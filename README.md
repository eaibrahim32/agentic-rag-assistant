# Agentic RAG Knowledge Assistant

A **corrective RAG (CRAG)** agent built on LangGraph. Instead of stuffing whatever
the retriever returned into a prompt and hoping, the agent grades each retrieved
document, discards the irrelevant ones, rewrites the query if nothing survived,
and checks the final answer against its own sources before returning it.

Runs fully offline on a local Ollama model, or against any OpenAI-compatible API.
Built and tested entirely in GitHub Codespaces — no local setup required.

---

## Why "corrective"

Naive RAG has one failure mode that matters: the retriever returns something
plausible-but-wrong, and the model confidently answers from it. This graph adds
three checkpoints:

| Stage | Question it answers | On failure |
|---|---|---|
| `grade_documents` | Is each retrieved chunk actually relevant? | Drop it |
| `decide_after_grading` | Did anything survive? | Rewrite the query and retry |
| `web_search` | Corpus exhausted -- can the web answer? | Fall through to an honest refusal |
| `check_grounded` | Is the answer supported by the sources? | Flag `grounded: false` |

**The agent never answers from the model's parametric memory.** The corpus is the
first hop, the web is the fallback, and if neither yields supporting context it
says so. Ask it "what is the capital of France?" with web search off and it will
refuse -- even though the underlying model obviously knows. That refusal is the
point: a naive RAG pipeline answers "Paris" and looks fine, right up until it
does the same confident thing with a fact that actually matters.

```
        ┌──────────┐
        │ retrieve │◀────────────────┐
        └────┬─────┘                 │
             ▼                       │
     ┌─────────────────┐             │
     │ grade_documents │             │
     └────────┬────────┘             │
              ▼                      │
        ╱ any relevant? ╲            │
       ╱                 ╲           │
     no                   yes        │
      │                    │         │
      ▼                    │   ┌─────────────────┐
 retry budget left? ───yes─┼──▶│ transform_query │
      │                    │   └─────────────────┘
      no                   │
      │                    │
      └────────┬───────────┘
               ▼
          ┌──────────┐
          │ generate │
          └────┬─────┘
               ▼
       ┌────────────────┐
       │ check_grounded │
       └────────┬───────┘
                ▼
               END
```

The retry loop is **bounded** by `MAX_TRANSFORM_ATTEMPTS`. An unbounded corrective
loop is an infinite loop waiting for a question your corpus can't answer.

---

## Quick start (GitHub Codespaces)

```bash
# 1. Open in Codespaces, then:
cp .env.example .env
pip install -r requirements-dev.txt

# 2. Run the tests — no API key, no model download, no network needed
pytest -q

# 3. Point at a provider. Fastest path: any OpenAI-compatible endpoint.
#    Edit .env:
#      OPENAI_API_KEY=sk-...
#      FORCE_PROVIDER=openai

# 4. Index the sample corpus
python -m app.ingest

# 5. Serve
uvicorn app.api:app --reload --port 8000
```

```bash
curl -X POST localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "How does HPA decide when to scale?"}'
```

### Running fully offline with Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull qwen2.5:1.5b     # ~1GB, fits a 2-core/8GB Codespace
unset FORCE_PROVIDER          # the app auto-detects the daemon
```

**Expect it to be slow.** On a CPU-only 2-core Codespace a 1.5B model generates
roughly 5–10 tokens/sec, and the CRAG loop makes several LLM calls per question
(one grade per document, plus generation, plus the grounding check). A single
answer can take 30–60 seconds. That is a fine demo of "runs with zero API cost
and no data leaving the box" — it is not a pleasant dev loop. Develop against an
API, demo the Ollama path.

---

## Configuration

Everything is env-driven — see `.env.example`.

| Variable | Default | Notes |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Probed at startup; unreachable is fine |
| `OLLAMA_MODEL` | `qwen2.5:1.5b` | Small enough for 8GB CPU-only |
| `OPENAI_API_KEY` | — | Fallback provider |
| `OPENAI_BASE_URL` | — | Any OpenAI-compatible host (Groq, Together, vLLM) |
| `FORCE_PROVIDER` | — | `ollama` \| `openai`. Skips the probe |
| `RETRIEVAL_K` | `4` | Chunks fetched before grading |
| `MAX_TRANSFORM_ATTEMPTS` | `1` | Query-rewrite retry budget |
| `ENABLE_WEB_SEARCH` | `false` | Web fallback when the corpus can't answer |
| `TAVILY_API_KEY` | — | Free tier: 1000 searches/month, no card |
| `WEB_SEARCH_RESULTS` | `3` | Hits fetched from the web |

Provider resolution order: `FORCE_PROVIDER` → Ollama (if the daemon answers) →
OpenAI-compatible (if a key is set) → raise.

---

## Testing

```bash
pytest -q                          # 104 tests, ~2.5s (87% coverage)
pytest --cov=app --cov-report=term-missing
```

The suite runs on injected fakes (`tests/conftest.py`) — no daemon, no API key,
no embedding model download, no network. That's what makes it viable in CI on a
free runner. Both `build_graph` and `answer_question` take the retriever and LLM
as arguments precisely so they can be faked.

Coverage includes the corrective paths that are easy to get wrong: bounded
retries, the no-context escape hatch, lenient grade parsing (`"not relevant"`
must not match as `"relevant"`), and the rule that an unanswerable question
never reaches the answer prompt.

---

## CI

`.github/workflows/ci.yml` runs ruff, then pytest with coverage, then builds the
Docker image on a separate job gated behind the tests.

---

## Known limitations

- **Grading costs one LLM call per document.** At `RETRIEVAL_K=4` that's 4 calls
  before generation. Fine at this scale, expensive at production volume — batch
  grading into a single call would be the first optimisation.
- **No reranker.** A cross-encoder rerank before grading would cut the grading
  calls and improve precision.
- **No conversation memory.** Single-turn only.

## Stack

LangGraph · LangChain · ChromaDB · sentence-transformers · Ollama · FastAPI ·
Docker · GitHub Actions · pytest
