"""Embed every chunk and store the vectors in a FAISS index.

Output: data/index/faiss.index   vectors
        data/index/chunks.jsonl  chunk text + metadata, same row order
        data/index/meta.json     settings used (for reproducible experiments)
Run:    python -m src.ingest.build_index
"""
import json
import shutil
import time

import faiss
import numpy as np
from tqdm import tqdm

from src.config import EMBED_BATCH_SIZE, EMBED_MODEL, INDEX_DIR, PROCESSED_DIR
from src.retrieval.embedder import embed_documents


def main():
    chunks_path = PROCESSED_DIR / "chunks.jsonl"
    if not chunks_path.exists():
        raise SystemExit("Run `python -m src.ingest.chunk` first.")
    with open(chunks_path, encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]
    params_path = PROCESSED_DIR / "chunk_params.json"
    params = json.loads(params_path.read_text()) if params_path.exists() else {}
    print(f"Embedding {len(chunks)} chunks with {EMBED_MODEL} ...")

    t0 = time.time()
    vectors = []
    for i in tqdm(range(0, len(chunks), EMBED_BATCH_SIZE), desc="Embedding"):
        batch = [c["text"] for c in chunks[i:i + EMBED_BATCH_SIZE]]
        vectors.append(embed_documents(batch))
    matrix = np.vstack(vectors)

    # Exact search with inner product on normalised vectors = cosine similarity.
    # Fine for tens of thousands of chunks; swap for IVF/HNSW if the corpus grows.
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_DIR / "faiss.index"))
    shutil.copy(chunks_path, INDEX_DIR / "chunks.jsonl")
    meta = {
        "embed_model": EMBED_MODEL,
        "dim": int(matrix.shape[1]),
        "n_chunks": len(chunks),
        **{k: v for k, v in params.items() if k.startswith("chunk_")},
        "build_seconds": round(time.time() - t0, 1),
    }
    (INDEX_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Index built: {meta}")


if __name__ == "__main__":
    main()
