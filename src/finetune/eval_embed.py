"""Compare embedding models on identical data.

Models: nomic-embed-text (current, via Ollama), bge-small untuned, bge-small fine-tuned.
All three search the SAME 1,837-chunk corpus (dense search only, no BM25/rerank) so the only thing that
differs is the embedding model.

Two test sets:
  1. Held-out synthetic: questions written for chunks of TEST-split documents (never trained on).
     Gold = the chunk the question was written from.
  2. The 19 hand-written questions in eval/questions.jsonl (strict + relaxed gold, as in eval/run_eval.py).

Run:  python -m src.finetune.eval_embed
"""
import json
import sys

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from eval.run_eval import is_gold_chunk, load
from src.config import DATA_DIR, INDEX_DIR, ROOT
from src.finetune.train import BASE_MODEL, OUT_DIR, QUERY_PREFIX
from src.retrieval.embedder import embed_query

KS = (1, 5, 10)


def load_chunks():
    with open(INDEX_DIR / "chunks.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


class NomicModel:
    name = "nomic-embed-text"

    def __init__(self):
        idx = faiss.read_index(str(INDEX_DIR / "faiss.index"))
        self.docs = idx.reconstruct_n(0, idx.ntotal)          # vectors already built, aligned to chunks

    def encode_queries(self, qs):
        return np.vstack([embed_query(q) for q in qs])


class STModel:
    def __init__(self, name, path, chunks, query_prefix):
        self.name = name
        self.prefix = query_prefix
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.m = SentenceTransformer(path, device=dev)
        self.m.max_seq_length = 256
        self.docs = self.m.encode([c["text"] for c in chunks], batch_size=64, normalize_embeddings=True,
                                  show_progress_bar=False)

    def encode_queries(self, qs):
        return self.m.encode([self.prefix + q for q in qs], batch_size=64, normalize_embeddings=True,
                             show_progress_bar=False)


def top_ids(model, queries, k):
    sims = model.encode_queries(queries) @ model.docs.T
    return np.argsort(-sims, axis=1)[:, :k]


def report(name, ranks):
    n = len(ranks)
    row = {f"R@{k}": sum(1 for r in ranks if r and r <= k) / n for k in KS}
    row["MRR"] = sum(1 / r for r in ranks if r) / n
    print(f"  {name:28} " + "  ".join(f"{m}={v:.3f}" for m, v in row.items()))
    return row


def main():
    chunks = load_chunks()
    models = [NomicModel(),
              STModel("bge-small (untuned)", BASE_MODEL, chunks, QUERY_PREFIX)]
    if OUT_DIR.exists():
        models.append(STModel("bge-small (ft, long qs)", str(OUT_DIR), chunks, QUERY_PREFIX))
    mixed = OUT_DIR.with_name("bge-small-medical-mixed")
    if mixed.exists():
        models.append(STModel("bge-small (ft, long+short)", str(mixed), chunks, QUERY_PREFIX))

    # 1) held-out synthetic questions
    test = []
    with open(DATA_DIR / "finetune" / "pairs.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["split"] == "test":
                test += [(q, r["chunk_id"]) for q in r["questions"]]
    ids_by_chunk = {}
    for i, c in enumerate(chunks):
        ids_by_chunk.setdefault(c["chunk_id"], set()).add(i)
    print(f"Held-out synthetic: {len(test)} questions on unseen documents, corpus {len(chunks)} chunks")
    results = {}
    for m in models:
        tops = top_ids(m, [q for q, _ in test], max(KS))
        ranks = []
        for (q, cid), row in zip(test, tops):
            gold = ids_by_chunk[cid]
            ranks.append(next((r for r, i in enumerate(row, 1) if i in gold), None))
        results[m.name + " | synthetic"] = report(m.name, ranks)

    # 2) hand-written questions
    items = [i for i in load() if i["drug"]]
    for relaxed in (False, True):
        print(f"Hand-written ({len(items)} questions), {'relaxed' if relaxed else 'strict'} gold")
        for m in models:
            tops = top_ids(m, [it["q"] for it in items], max(KS))
            ranks = []
            for it, row in zip(items, tops):
                ranks.append(next((r for r, i in enumerate(row, 1)
                                   if is_gold_chunk(it, chunks[i], relaxed)), None))
            results[m.name + f" | hand {'relaxed' if relaxed else 'strict'}"] = report(m.name, ranks)

    out = ROOT / "eval" / "embedding_results.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved {out}")


if __name__ == "__main__":
    sys.exit(main())
