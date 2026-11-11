"""Ask a question and get a cited answer.

Usage:
    python -m scripts.ask "Can I take ibuprofen with warfarin?"
    python -m scripts.ask            (interactive mode)
"""
import sys

from src.generation.answer import Answer, answer_question
from src.retrieval.hybrid import default_retriever


def show(ans: Answer):
    print(f"\nQ: {ans.question}\n\n{ans.text}\n")
    if ans.answered and ans.cited:
        print("Sources:")
        for n in ans.cited:
            hit = ans.sources[n - 1]
            print(f"  [{n}] {hit.citation}  (score {hit.score:.3f})")
            if hit.chunk.get("url"):
                print(f"      {hit.chunk['url']}")
    elif ans.answered:
        print("Warning: the answer has no citations, so treat it as unverified.")
    if ans.invalid_citations:
        print(f"Warning: model cited non-existent sources {ans.invalid_citations}.")
    print()


def main():
    retriever = default_retriever()
    if len(sys.argv) > 1:
        show(answer_question(" ".join(sys.argv[1:]), retriever))
        return
    print("Type a question (empty line to quit).")
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        show(answer_question(q, retriever))


if __name__ == "__main__":
    main()
