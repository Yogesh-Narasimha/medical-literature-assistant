"""Compare the approximate (HNSW) index with the exact (flat) one on the real corpus.

Measures, for the same query vectors: recall@10 (share of the exact top-10 that HNSW also returns), whether the
top-1 hit agrees, and the search time of each index (search only; the embedding call is the same for both).
Queries are every question in the eval sets plus the adversarial set, so they look like real traffic.

Run:  python -m scripts.bench_ann        (needs data/index/faiss.index and faiss_hnsw.index)
"""
import json
import statistics
import time
from pathlib import Path

import faiss
import numpy as np

from src.config import HNSW_EF_SEARCH, INDEX_DIR
from src.retrieval.embedder import embed_query

QUESTION_FILES = ["eval/questions.jsonl", "eval/questions_set2.jsonl", "eval/questions_set3.jsonl",
                  "eval/adversarial.jsonl"]


def questions() -> list[str]:
    out = []
    for f in QUESTION_FILES:
        for line in open(f, encoding="utf-8"):
            if line.strip():
                out.append(json.loads(line)["q"])
    return out


def timed_search(index, queries: np.ndarray, k: int, repeats: int = 5):
    ids, ms = None, []
    for _ in range(repeats):
        t = time.perf_counter()
        for q in queries:
            _, ids_one = index.search(q[None, :], k)
        ms.append((time.perf_counter() - t) / len(queries) * 1000)
    ids = np.vstack([index.search(q[None, :], k)[1] for q in queries])
    return ids, statistics.median(ms)


def main():
    qs = questions()
    queries = np.vstack([embed_query(q) for q in qs]).astype("float32")
    flat = faiss.read_index(str(INDEX_DIR / "faiss.index"))
    hnsw = faiss.read_index(str(INDEX_DIR / "faiss_hnsw.index"))
    print(f"{len(qs)} queries, {flat.ntotal} vectors, M={hnsw.hnsw.nb_neighbors(1)}, "
          f"flat file {Path(INDEX_DIR / 'faiss.index').stat().st_size / 1e6:.0f} MB, "
          f"hnsw file {Path(INDEX_DIR / 'faiss_hnsw.index').stat().st_size / 1e6:.0f} MB")
    exact_ids, exact_ms = timed_search(flat, queries, 10)
    print(f"exact (flat)               p50 {exact_ms:6.2f} ms/query")
    for ef in (16, 32, 64, HNSW_EF_SEARCH, 256):
        hnsw.hnsw.efSearch = ef
        ids, ms = timed_search(hnsw, queries, 10)
        recall = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(exact_ids, ids)])
        top1 = np.mean(exact_ids[:, 0] == ids[:, 0])
        print(f"hnsw efSearch={ef:<4}        p50 {ms:6.2f} ms/query   recall@10 {recall:.3f}   top-1 agrees {top1:.3f}")


if __name__ == "__main__":
    main()
