"""Generate synthetic (question, chunk) training pairs with the local LLM.

For every chunk, qwen2.5:3b writes 2 questions a patient or clinician might ask that the passage
answers. Pairs are split BY DOCUMENT (hash of doc_id), never by chunk, so no passage from a held-out
document is ever seen in training (a chunk-level split would leak neighbouring chunks of the same label).

Resumable: re-running skips chunks already written.
Run:  python -m src.finetune.make_pairs
Out:  data/finetune/pairs.jsonl   {chunk_id, split, drug, section, text, questions[]}
"""
import hashlib
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import ollama

from src.config import DATA_DIR, LLM_MODEL, OLLAMA_HOST, PROCESSED_DIR

OUT = DATA_DIR / "finetune" / "pairs.jsonl"
TEST_FRACTION_MOD = 5          # doc_id hash % 5 == 0 -> test (about 20% of documents)
MIN_WORDS = 40
QUESTIONS_PER_CHUNK = 2

PROMPT = """Below is a passage from a drug label or medical abstract.
Write {n} different questions that a patient or clinician might ask and that this passage answers.
Use plain wording, and mention the drug or condition by name. One question per line, no numbering, no extra text.

Passage:
{text}

Questions:"""

PROMPT_SHORT = """Below is a passage from a drug label or medical abstract.
Write {n} SHORT questions (under 12 words each) that a patient might type into a search box and that this
passage answers. Keep them simple and natural, like "Can I take X with Y?" or "What are the side effects of X?".
Name the drug or condition. One question per line, no numbering, no extra text.

Passage:
{text}

Questions:"""

SHORT = len(sys.argv) > 1 and sys.argv[1] == "short"     # python -m src.finetune.make_pairs short
if SHORT:
    OUT = OUT.with_name("pairs_short.jsonl")

_client = ollama.Client(host=OLLAMA_HOST)


def split_of(doc_id: str) -> str:
    h = int(hashlib.md5(doc_id.encode()).hexdigest(), 16)
    return "test" if h % TEST_FRACTION_MOD == 0 else "train"


def parse_questions(raw: str) -> list[str]:
    qs = []
    for line in raw.splitlines():
        line = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
        if len(line) > 15 and line.endswith("?"):
            qs.append(line)
    return qs[:QUESTIONS_PER_CHUNK]


def make(chunk: dict) -> dict | None:
    resp = _client.chat(
        model=LLM_MODEL, options={"temperature": 0.7, "num_predict": 100},
        messages=[{"role": "user", "content": (PROMPT_SHORT if SHORT else PROMPT).format(
            n=QUESTIONS_PER_CHUNK, text=chunk["text"][:2000])}],
    )
    qs = parse_questions(resp["message"]["content"])
    if not qs:
        return None
    return {"chunk_id": chunk["chunk_id"], "doc_id": chunk["doc_id"], "split": split_of(chunk["doc_id"]),
            "drug": chunk.get("drug"), "section": chunk["section"], "source_type": chunk["source_type"],
            "text": chunk["text"], "questions": qs}


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(PROCESSED_DIR / "chunks.jsonl", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]
    seen_ids = set()
    unique = []
    for c in chunks:                      # same label stored under 2 drugs shares chunk_ids
        if c["chunk_id"] not in seen_ids and len(c["text"].split()) >= MIN_WORDS:
            seen_ids.add(c["chunk_id"])
            unique.append(c)
    done = set()
    if OUT.exists():
        with open(OUT, encoding="utf-8") as f:
            done = {json.loads(line)["chunk_id"] for line in f}
    todo = [c for c in unique if c["chunk_id"] not in done]
    print(f"{len(unique)} usable chunks, {len(done)} done, {len(todo)} to go", flush=True)

    n_ok = 0
    with open(OUT, "a", encoding="utf-8") as out, ThreadPoolExecutor(max_workers=2) as pool:
        for i, rec in enumerate(pool.map(make, todo), 1):
            if rec:
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                n_ok += 1
            if i % 25 == 0:
                print(f"  {i}/{len(todo)} processed, {n_ok} written", flush=True)
    print(f"Done. {n_ok} chunks with questions -> {OUT}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
