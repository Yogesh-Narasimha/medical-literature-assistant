"""Evaluation harness.

Retrieval (needs only Ollama embeddings, fast):
    python -m eval.run_eval retrieval
    -> Recall@k and MRR: does a top-k chunk come from the expected drug + section?

Agent (needs the LLM, slow on CPU):
    python -m eval.run_eval agent
    -> tool routing accuracy, citation rate, valid-citation rate, and refusal on
       out-of-scope questions.

Metrics are defined on "gold" = (drug, acceptable sections) per question in
eval/questions.jsonl. Out-of-scope questions (drug == null) should be refused.
"""
import json
import sys
from pathlib import Path

from src.config import MIN_RELEVANCE

QUESTIONS = Path(__file__).with_name("questions.jsonl")


def load() -> list[dict]:
    with open(QUESTIONS, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def is_gold(item: dict, drug: str, section: str) -> bool:
    return item["drug"] == drug and section in item["sections"]


def is_gold_chunk(item: dict, chunk: dict, relaxed: bool) -> bool:
    """Strict: the expected drug's label section. Relaxed: also a PubMed abstract on the topic
    (item["kw"]: all terms must appear), because a topical abstract is a legitimate source."""
    if is_gold(item, chunk.get("drug"), chunk["section"]):
        return True
    return bool(relaxed and item.get("kw") and chunk["source_type"] == "pubmed"
                and all(t in chunk["text"].lower() for t in item["kw"]))


class _CorrectiveAdapter:
    """Lets the corrective-RAG pipeline be scored like any other retriever."""

    def __init__(self, retriever):
        self.retriever = retriever

    def search(self, query, k):
        from src.retrieval.corrective import corrective_search
        return corrective_search(query, self.retriever, k)[0]


def _first_rank(item, hits, relaxed):
    return next((n for n, h in enumerate(hits, 1) if is_gold_chunk(item, h.chunk, relaxed)), None)


def _score_retriever(name, retriever, items, ks):
    strict, relaxed = [], []
    for it in items:
        hits = retriever.search(it["q"], max(ks))
        strict.append(_first_rank(it, hits, False))
        relaxed.append(_first_rank(it, hits, True))
    for label, ranks in (("strict", strict), ("relaxed", relaxed)):
        row = {f"R@{k}": sum(1 for x in ranks if x and x <= k) / len(items) for k in ks}
        row["MRR"] = sum(1 / x for x in ranks if x) / len(items)
        print(f"  {name:11} {label:7} " + "  ".join(f"{m}={v:.2f}" for m, v in row.items()))
    return strict


def eval_retrieval(ks=(1, 3, 5)):
    from src.retrieval.dense import DenseRetriever
    from src.retrieval.hybrid import BM25Retriever, HybridRetriever
    dense = DenseRetriever()
    final = HybridRetriever(dense, drug_boost=0.01)
    from src.retrieval.rerank import RerankedRetriever
    plain_hybrid = HybridRetriever(dense)
    retrievers = {"dense": dense, "bm25": BM25Retriever(dense.chunks), "hybrid": plain_hybrid,
                  "hyb+drug": final, "hyb+corr": _CorrectiveAdapter(final),
                  "hyb+rerank": RerankedRetriever(plain_hybrid), "drug+rerank": RerankedRetriever(final)}
    items = [i for i in load() if i["drug"]]
    print(f"Retrieval over {len(items)} in-scope questions")
    all_ranks = {n: _score_retriever(n, r, items, ks) for n, r in retrievers.items()}
    print()
    print("Per question strict rank (" + " / ".join(retrievers) + "):")
    for i, it in enumerate(items):
        print("  " + " / ".join(f"{all_ranks[n][i]!s:>4}" for n in retrievers) + f"   {it['q']}")

    # Out-of-scope: top dense scores help choose MIN_RELEVANCE.
    oos = [i for i in load() if not i["drug"]]
    print()
    print(f"Out-of-scope top-1 dense scores (MIN_RELEVANCE currently {MIN_RELEVANCE}):")
    for it in oos:
        print(f"  {dense.search(it['q'], 1)[0].score:.3f}  {it['q']}")
    inscope = [dense.search(i["q"], 1)[0].score for i in items]
    print(f"In-scope top-1 dense score: min {min(inscope):.3f}, mean {sum(inscope) / len(inscope):.3f}")


def eval_agent():
    from src.agent.graph import run_agent
    items = load()
    tool_ok = cited = valid = refused_ok = n_in = n_out = 0
    gold_cited = 0
    for it in items:
        try:
            res = run_agent(it["q"])
        except Exception as e:
            print(f"  ERROR on '{it['q']}': {e}")
            n_in += bool(it["drug"])
            n_out += not it["drug"]
            continue
        tools = [t["tool"] for t in res["trace"]]
        if it["drug"]:
            n_in += 1
            ok_tools = {it["tool"], "check_interaction"} if it["tool"] == "lookup_drug_label" else {it["tool"]}
            tool_ok += bool(ok_tools & set(tools))
            has_cit = bool(res["cited"]) and res["answered"]
            cited += has_cit
            valid += has_cit and not res["invalid"]
            gold_cited += any(is_gold(it, res["sources"][n - 1]["title"].split(": ")[-1],
                                      res["sources"][n - 1]["section"]) for n in res["cited"])
            status = f"tools={tools} cited={res['cited']}"
        else:
            n_out += 1
            ok = not res["answered"] or not res["cited"]
            refused_ok += ok
            status = f"refused={'yes' if ok else 'NO'}"
        print(f"  {it['q'][:50]:50}  {status}")
    print(f"\nAgent over {n_in} in-scope questions")
    print(f"  Tool routing accuracy:      {tool_ok / n_in:.2f}")
    print(f"  Answers with citations:     {cited / n_in:.2f}")
    print(f"  All citations valid:        {valid / n_in:.2f}")
    print(f"  Cites a gold source:        {gold_cited / n_in:.2f}")
    print(f"Out-of-scope refusal rate:    {refused_ok / n_out:.2f}  ({n_out} questions)")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "retrieval"
    {"retrieval": eval_retrieval, "agent": eval_agent}[mode]()
