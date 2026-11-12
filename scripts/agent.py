"""Ask the ReAct agent.

Usage:
    python -m scripts.agent "Can I take ibuprofen with warfarin?"
    python -m scripts.agent            (interactive)
"""
import sys

from src.agent.graph import run_agent


def show(r: dict):
    print(f"\nQ: {r['question']}\n")
    for t in r["trace"]:
        print(f"  tool: {t['tool']}({t['args']}) -> {t['n_results']} sources")
        if t.get("note"):
            print(f"        note: {t['note']}")
    print(f"\n{r['answer']}\n")
    if r["cited"]:
        print("Sources:")
        for n in r["cited"]:
            s = r["sources"][n - 1]
            print(f"  [{n}] {s['title']} [{s['section']}]  {s['url']}")
    elif r["answered"]:
        print("Warning: no citations, treat as unverified.")
    for s in r.get("uncited", []):
        print(f"Warning: no source for: {s}")
    if r["invalid"]:
        print(f"Warning: non-existent citations {r['invalid']}")
    print()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        show(run_agent(" ".join(sys.argv[1:])))
    else:
        print("Type a question (empty line to quit).")
        while q := input("> ").strip():
            show(run_agent(q))
