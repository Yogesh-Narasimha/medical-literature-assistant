"""Split documents into overlapping chunks, keeping source metadata.

Each label section (e.g. drug_interactions) is chunked on its own, so a
chunk never mixes two sections. That keeps citations precise.

Output: data/processed/chunks.jsonl
Run:    python -m src.ingest.chunk  [--size 200 --overlap 40]
"""
import argparse
import json
import re

from src.config import CHUNK_OVERLAP, CHUNK_SIZE, PROCESSED_DIR, RAW_DIR

RAW_FILES = ["drug_labels.jsonl", "pubmed_abstracts.jsonl"]


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_words(text: str, size: int, overlap: int) -> list[str]:
    """Sliding window over words. Snaps the end of a window to a sentence
    boundary when one is close, so chunks rarely cut a sentence in half."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    words = text.split()
    if len(words) <= size:
        return [" ".join(words)] if words else []
    chunks, start = [], 0
    while start < len(words):
        end = min(start + size, len(words))
        if end < len(words):
            # look back up to 25% of the window for a sentence end
            for j in range(end, start + int(size * 0.75), -1):
                if words[j - 1].endswith((".", "?", "!")):
                    end = j
                    break
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = max(end - overlap, start + 1)
    return chunks


def chunk_document(doc: dict, size: int, overlap: int) -> list[dict]:
    out = []
    for section, text in doc["sections"].items():
        for i, piece in enumerate(split_words(clean(text), size, overlap)):
            out.append({
                "chunk_id": f"{doc['doc_id']}#{section}#{i}",
                "doc_id": doc["doc_id"],
                "source_type": doc["source_type"],
                "drug": doc.get("drug", ""),
                "title": doc["title"],
                "section": section,
                "url": doc.get("url", ""),
                "text": piece,
            })
    return out


def main(size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "chunks.jsonl"
    n_docs = n_chunks = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for name in RAW_FILES:
            path = RAW_DIR / name
            if not path.exists():
                print(f"  (skipping {name}: not downloaded yet)")
                continue
            with open(path, encoding="utf-8") as f:
                for line in f:
                    doc = json.loads(line)
                    n_docs += 1
                    for c in chunk_document(doc, size, overlap):
                        out.write(json.dumps(c, ensure_ascii=False) + "\n")
                        n_chunks += 1
    (PROCESSED_DIR / "chunk_params.json").write_text(
        json.dumps({"chunk_size": size, "chunk_overlap": overlap, "n_docs": n_docs, "n_chunks": n_chunks}))
    print(f"{n_docs} documents -> {n_chunks} chunks (size={size}, overlap={overlap})")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=CHUNK_SIZE)
    ap.add_argument("--overlap", type=int, default=CHUNK_OVERLAP)
    args = ap.parse_args()
    main(args.size, args.overlap)
