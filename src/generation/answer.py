"""Answer a question from retrieved chunks only, with numbered citations.

Anti-hallucination measures (a common interview question):
1. The prompt tells the model to use ONLY the numbered sources.
2. Every claim must carry a [n] marker pointing at a source.
3. If retrieval is weak (best score < MIN_RELEVANCE) we skip the LLM and say so.
4. Temperature 0, so answers are reproducible.
5. We check that every [n] the model cites really exists.
"""
import re
from dataclasses import dataclass, field

import ollama

from src.config import LLM_MODEL, LLM_TEMPERATURE, MIN_RELEVANCE, OLLAMA_HOST, TOP_K
from src.retrieval.dense import DenseRetriever, Hit

NOT_FOUND = "I could not find this in the indexed documents."

SYSTEM_PROMPT = """You are a medical literature assistant. Answer the question using ONLY the numbered sources provided.
Rules:
- After every factual statement, cite the source number like [1] or [2][3].
- If the sources do not contain the answer, reply exactly: "I could not find this in the indexed documents."
- Do not use outside knowledge. Do not guess. Keep the answer short and clear.
- Do not give personal medical advice; suggest consulting a clinician or pharmacist where relevant."""

_client = ollama.Client(host=OLLAMA_HOST)


@dataclass
class Answer:
    question: str
    text: str
    sources: list[Hit] = field(default_factory=list)   # all retrieved, in [n] order
    cited: list[int] = field(default_factory=list)     # valid [n] the model used
    invalid_citations: list[int] = field(default_factory=list)
    answered: bool = True


def build_context(hits: list[Hit]) -> str:
    return "\n\n".join(f"[{i}] {h.citation}\n{h.chunk['text']}" for i, h in enumerate(hits, 1))


def answer_question(question: str, retriever: DenseRetriever, k: int = TOP_K) -> Answer:
    hits = retriever.search(question, k)
    if not hits or max(h.score for h in hits) < MIN_RELEVANCE:
        return Answer(question, NOT_FOUND, hits, answered=False)

    resp = _client.chat(
        model=LLM_MODEL,
        options={"temperature": LLM_TEMPERATURE},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Sources:\n{build_context(hits)}\n\nQuestion: {question}"},
        ],
    )
    text = resp["message"]["content"].strip()

    used = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text)})
    cited = [n for n in used if 1 <= n <= len(hits)]
    invalid = [n for n in used if n not in cited]
    answered = NOT_FOUND.lower() not in text.lower()
    return Answer(question, text, hits, cited, invalid, answered)
