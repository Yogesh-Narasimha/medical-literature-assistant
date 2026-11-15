"""Embed every chunk and store the vectors in a FAISS index.

Output: data/index/faiss.index        exact (flat) vectors
        data/index/faiss_hnsw.index   approximate HNSW graph over the same vectors (with --hnsw)
        data/index/chunks.jsonl       chunk text + metadata, same row order
        data/index/meta.json          settings used (for reproducible experiments)
Run:    python -m src.ingest.build_index [--full] [--hnsw]

By default the build is incremental: vectors of chunks whose text is unchanged are read back out of the existing
index and only new or changed chunks are sent to the embedding model (embedding is the slow step: about 25 minutes
for 39,000 chunks). --full re-embeds everything.
"""
import hashlib
import json
import shutil
import sys
import time

import faiss
import numpy as np
from tqdm import tqdm

from src.config import EMBED_BATCH_SIZE, EMBED_MODEL, INDEX_DIR, PROCESSED_DIR
from src.retrieval.embedder import embed_documents

HNSW_M = 32               # graph neighbours per node
HNSW_EF_CONSTRUCTION = 200


def _key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def existing_vectors() -> dict[str, np.ndarray]:
    """text-hash -> vector for every chunk in the current exact index (empty if there is none)."""
    idx_path, chunks_path = INDEX_DIR / "faiss.index", INDEX_DIR / "chunks.jsonl"
    if not (idx_path.exists() and chunks_path.exists()):
        return {}
    index = faiss.read_index(str(idx_path))
    old = [json.loads(line) for line in open(chunks_path, encoding="utf-8")]
    if index.ntotal != len(old):
        return {}
    meta_path = INDEX_DIR / "meta.json"
    if meta_path.exists() and json.loads(meta_path.read_text()).get("embed_model") != EMBED_MODEL:
        return {}                                    # vectors from another model are not reusable
    matrix = index.reconstruct_n(0, index.ntotal)
    return {_key(c["text"]): matrix[i] for i, c in enumerate(old)}


def build_hnsw(matrix: np.ndarray) -> faiss.Index:
    index = faiss.IndexHNSWFlat(matrix.shape[1], HNSW_M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    index.add(matrix)
    return index


def main():
    chunks_path = PROCESSED_DIR / "chunks.jsonl"
    if not chunks_path.exists():
        raise SystemExit("Run `python -m src.ingest.chunk` first.")
    with open(chunks_path, encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]
    params_path = PROCESSED_DIR / "chunk_params.json"
    params = json.loads(params_path.read_text()) if params_path.exists() else {}

    reuse = {} if "--full" in sys.argv else existing_vectors()
    keys = [_key(c["text"]) for c in chunks]
    todo = [i for i, k in enumerate(keys) if k not in reuse]
    print(f"{len(chunks)} chunks: reusing {len(chunks) - len(todo)} vectors, embedding {len(todo)} with {EMBED_MODEL} ...")

    t0 = time.time()
    new_vecs: dict[int, np.ndarray] = {}
    for s in tqdm(range(0, len(todo), EMBED_BATCH_SIZE), desc="Embedding"):
        ids = todo[s:s + EMBED_BATCH_SIZE]
        vecs = embed_documents([chunks[i]["text"] for i in ids])
        for i, v in zip(ids, vecs):
            new_vecs[i] = v
    matrix = np.vstack([new_vecs[i] if i in new_vecs else reuse[keys[i]] for i in range(len(chunks))]).astype("float32")

    # Exact search with inner product on normalised vectors = cosine similarity.
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_DIR / "faiss.index"))
    if "--hnsw" in sys.argv:
        t1 = time.time()
        faiss.write_index(build_hnsw(matrix), str(INDEX_DIR / "faiss_hnsw.index"))
        print(f"HNSW graph built in {time.time() - t1:.0f} s")
    shutil.copy(chunks_path, INDEX_DIR / "chunks.jsonl")
    meta = {
        "embed_model": EMBED_MODEL,
        "dim": int(matrix.shape[1]),
        "n_chunks": len(chunks),
        "n_embedded_this_build": len(todo),
        **{k: v for k, v in params.items() if k.startswith("chunk_")},
        "build_seconds": round(time.time() - t0, 1),
    }
    (INDEX_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Index built: {meta}")


if __name__ == "__main__":
    main()
