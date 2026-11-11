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


def eval_retrieval(ks=(1, 3, 5)):
    from src.retrieval.dense import DenseRetriever
    r = DenseRetriever()
    items = [i for i in load() if i["drug"]]
    ranks = []
    for it in items:
        hits = r.search(it["q"], max(ks))
        rank = next((n for n, h in enumerate(hits, 1)
                     if is_gold(it, h.chunk.get("drug"), h.chunk["section"])), None)
        ranks.append(rank)
        print(f"  rank={rank!s:>4}  {it['q']}")
    print(f"\nRetrieval over {len(items)} in-scope questions")
    for k in ks:
        print(f"  Recall@{k}: {sum(1 for x in ranks if x and x <= k) / len(items):.2f}")
    print(f"  MRR:       {sum(1 / x for x in ranks if x) / len(items):.2f}")

    # Out-of-scope: what are the top scores? Helps choose MIN_RELEVANCE.
    oos = [i for i in load() if not i["drug"]]
    print(f"\nOut-of-scope top-1 scores (MIN_RELEVANCE currently {MIN_RELEVANCE}):")
    for it in oos:
        print(f"  {r.search(it['q'], 1)[0].score:.3f}  {it['q']}")
    inscope = [r.search(i["q"], 1)[0].score for i in items]
    print(f"In-scope top-1 score: min {min(inscope):.3f}, mean {sum(inscope) / len(inscope):.3f}")


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
