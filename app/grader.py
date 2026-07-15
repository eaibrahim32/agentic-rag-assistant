"""Binary graders used by the corrective-RAG loop.

Design note: small local models (1.5B class) are unreliable at strict JSON /
function-calling output. Rather than depend on `with_structured_output`, we ask
for a bare yes/no and parse leniently. This keeps the same code path working on
both a 1.5B Ollama model and a frontier API model.
"""
from __future__ import annotations

import re
from typing import Any

_YES = re.compile(r"\b(yes|relevant|grounded|useful|true)\b", re.IGNORECASE)
_NO = re.compile(r"\b(no|irrelevant|not relevant|ungrounded|false)\b", re.IGNORECASE)


def parse_binary(text: str, *, default: bool = False) -> bool:
    """Parse a model's free-text answer into a boolean.

    Checks negatives first: "not relevant" contains "relevant", so a naive
    positive-first check would misread it.
    """
    if not text:
        return default
    snippet = text.strip()[:200]
    if _NO.search(snippet):
        return False
    if _YES.search(snippet):
        return True
    return default


def _content(response: Any) -> str:
    """Extract text from a LangChain message or a plain string."""
    if isinstance(response, str):
        return response
    return str(getattr(response, "content", "") or "")


GRADE_DOCUMENT_PROMPT = """You are grading whether a retrieved document is relevant
to a user question. Answer with a single word: yes or no.

Question: {question}

Document:
{document}

Relevant (yes/no):"""

GRADE_GROUNDED_PROMPT = """You are checking whether an answer is supported by the
given source documents. Answer with a single word: yes or no.

Documents:
{documents}

Answer:
{generation}

Is the answer supported by the documents (yes/no):"""


def grade_document(llm, question: str, document: str) -> bool:
    """True if `document` is relevant to `question`."""
    prompt = GRADE_DOCUMENT_PROMPT.format(question=question, document=document[:2000])
    return parse_binary(_content(llm.invoke(prompt)), default=False)


def grade_grounded(llm, documents: str, generation: str) -> bool:
    """True if `generation` is supported by `documents` (hallucination check)."""
    prompt = GRADE_GROUNDED_PROMPT.format(documents=documents[:4000], generation=generation)
    return parse_binary(_content(llm.invoke(prompt)), default=True)
