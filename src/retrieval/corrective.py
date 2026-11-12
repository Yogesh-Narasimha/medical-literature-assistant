"""Corrective RAG (CRAG-style): grade retrieval, and if it is weak, rewrite the query and retry.

    retrieve -> grade --correct----------------------------> use hits
                  |--incorrect (out of domain)------------> "nothing relevant" (no rewrite)
                  |--ambiguous--> rewrite query -> retrieve again -> merge -> grade again
                                                                      |-- incorrect -> "nothing relevant"

The grade uses the best dense cosine score among the hits (no extra LLM call, so it is
instant). The rewrite is one short LLM call that turns lay wording ("blood thinner",
"Tylenol", "paracetamol") into the terms drug labels and abstracts actually use.
Thresholds come from the measured score gap: in-scope top-1 >= 0.75, off-topic <= 0.57.
"""
from dataclasses import dataclass

import ollama

from src.config import LLM_MODEL, MIN_RELEVANCE, OLLAMA_HOST, TOP_K
from src.retrieval.dense import Hit

CORRECT_SCORE = 0.72      # at or above: retrieval is good, answer from it
# between MIN_RELEVANCE and CORRECT_SCORE: ambiguous, try a rewrite; below MIN_RELEVANCE: incorrect

_client = ollama.Client(host=OLLAMA_HOST)

REWRITE_PROMPT = """Rewrite the user's medical question as a short search query for a database of FDA drug labels and PubMed abstracts.
Use generic drug names (acetaminophen, not Tylenol or paracetamol) and formal terms (anticoagulant, not blood thinner).
Output ONLY the query, no explanation, max 15 words.

Question: {q}
Query:"""


@dataclass
class Correction:
    grade_before: str
    rewritten: str | None = None
    grade_after: str | None = None

    def note(self) -> str:
        if self.rewritten is None:
            return "retrieval was incorrect (nothing close to the question); not rewriting" if self.grade_before == "incorrect" else ""
        return f"retrieval was {self.grade_before}; query rewritten to '{self.rewritten}' -> {self.grade_after}"


def grade(hits: list[Hit]) -> str:
    best = max((h.score for h in hits), default=0.0)
    if best >= CORRECT_SCORE:
        return "correct"
    return "ambiguous" if best >= MIN_RELEVANCE else "incorrect"


def rewrite_query(question: str) -> str:
    resp = _client.chat(
        model=LLM_MODEL, options={"temperature": 0.0, "num_predict": 40},
        messages=[{"role": "user", "content": REWRITE_PROMPT.format(q=question)}],
    )
    text = resp["message"]["content"].strip().splitlines()[0] if resp["message"]["content"].strip() else ""
    return text.strip(" \"'`")[:200] or question


def _merge(a: list[Hit], b: list[Hit], k: int) -> list[Hit]:
    seen, merged = set(), []
    for h in sorted(a + b, key=lambda h: h.score, reverse=True):
        key = (h.chunk["chunk_id"], h.chunk.get("drug"))
        if key not in seen:
            seen.add(key)
            merged.append(h)
    return merged[:k]


def corrective_search(question: str, retriever, k: int = TOP_K) -> tuple[list[Hit], Correction]:
    hits = retriever.search(question, k)
    g = grade(hits)
    corr = Correction(grade_before=g)
    if g == "correct":
        return hits, corr
    if g == "incorrect":
        # Clearly out of domain. Rewriting would force it into the medical domain (the rewriter
        # happily turns "fix a flat tire" into a drug query), so refuse instead.
        corr.grade_after = "incorrect"
        return [], corr
    corr.rewritten = rewrite_query(question)
    if corr.rewritten.lower() != question.lower():
        hits = _merge(hits, retriever.search(corr.rewritten, k), k)
    corr.grade_after = grade(hits)
    if corr.grade_after == "incorrect":
        return [], corr          # nothing relevant: let the agent say so instead of guessing
    return hits, corr
