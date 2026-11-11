"""Try the retriever from the command line.

Usage:
    python -m scripts.search "Can I take ibuprofen with warfarin?"
    python -m scripts.search            (interactive mode)
"""
import sys
import textwrap

from src.config import TOP_K
from src.retrieval.dense import DenseRetriever


def show(retriever: DenseRetriever, query: str, k: int = TOP_K):
    print(f"\nQ: {query}\n")
    for rank, hit in enumerate(retriever.search(query, k), 1):
        print(f"{rank}. ({hit.score:.3f}) {hit.citation}")
        print(textwrap.indent(textwrap.shorten(hit.chunk["text"], 300), "   "))
        if hit.chunk.get("url"):
            print(f"   {hit.chunk['url']}")
        print()


def main():
    retriever = DenseRetriever()
    if len(sys.argv) > 1:
        show(retriever, " ".join(sys.argv[1:]))
        return
    print("Type a question (empty line to quit).")
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        show(retriever, q)


if __name__ == "__main__":
    main()
