"""Cross-encoder reranking.

A bi-encoder (the embedding model) encodes the question and each passage separately, then compares the
vectors: fast, but it cannot see how the question words interact with the passage words. A cross-encoder
reads (question, passage) TOGETHER and outputs a relevance score: slower, more accurate. So the usual
design is: cheap retrieval takes the top `pool` candidates, the cross-encoder reorders them.

The returned Hit.score stays the DENSE cosine of the original retrieval, so the refusal / corrective-RAG
thresholds keep their meaning; only the ORDER comes from the cross-encoder.

Needs torch + sentence-transformers (see requirements-finetune.txt). The model loads lazily on first use.
"""
from src.config import TOP_K
from src.retrieval.dense import Hit

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class RerankedRetriever:
    def __init__(self, base, model_name: str = DEFAULT_MODEL, pool: int = 20, max_chars: int = 1500):
        self.base, self.model_name, self.pool, self.max_chars = base, model_name, pool, max_chars
        self._model = None

    def _load(self):
        if self._model is None:
            import torch
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name, max_length=512,
                                       device="cuda" if torch.cuda.is_available() else "cpu")
        return self._model

    def search(self, query: str, k: int = TOP_K) -> list[Hit]:
        candidates = self.base.search(query, max(self.pool, k))
        if not candidates:
            return []
        scores = self._load().predict([(query, h.chunk["text"][: self.max_chars]) for h in candidates],
                                      show_progress_bar=False)
        order = sorted(range(len(candidates)), key=lambda i: -float(scores[i]))
        return [candidates[i] for i in order[:k]]
