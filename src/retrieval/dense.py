"""Dense (embedding) retriever over the FAISS index.

Every retriever in this project exposes the same `search(query, k)` method
returning a list of hits, so hybrid search and reranking (Day 5) can be
swapped in and compared by the eval harness without touching other code.
"""
import json
from dataclasses import dataclass

import faiss

from src.config import INDEX_DIR, TOP_K
from src.retrieval.embedder import embed_query


@dataclass
class Hit:
    score: float
    chunk: dict

    @property
    def citation(self) -> str:
        c = self.chunk
        return f"{c['title']} [{c['section']}]"


class DenseRetriever:
    def __init__(self, index_dir=INDEX_DIR):
        if not (index_dir / "faiss.index").exists():
            raise SystemExit("No index found. Run `python -m src.ingest.build_index` first.")
        self.index = faiss.read_index(str(index_dir / "faiss.index"))
        with open(index_dir / "chunks.jsonl", encoding="utf-8") as f:
            self.chunks = [json.loads(line) for line in f]

    def search(self, query: str, k: int = TOP_K) -> list[Hit]:
        q = embed_query(query)
        scores, ids = self.index.search(q, k)
        return [Hit(float(s), self.chunks[i]) for s, i in zip(scores[0], ids[0]) if i != -1]
