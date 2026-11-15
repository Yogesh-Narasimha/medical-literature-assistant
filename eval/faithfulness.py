"""Measure the claim-level faithfulness check (src/faithfulness.py) on a benchmark built from real answers.

Step 1 (--collect): run the agent on every in-scope question of sets 1-3 and save answers with their sources.
Step 2 (default):   build labelled claims from those answers and score the checker.

  real         a sentence the agent wrote, with the source(s) it cited. NOT guaranteed to be supported (the model
               can misquote), so the flag rate on these is an upper bound on false alarms; flagged ones are printed
               for reading.
  wrong-source the same sentence paired with a chunk from a DIFFERENT drug's label: unsupported by construction.
  wrong-number the same sentence with one number changed (500 -> 650): unsupported by construction.
  negated      the same sentence with a negation flipped ("should not" <-> "should"): contradicted by construction.

Questions are split in half by position (even = dev, odd = test). The cut-offs are chosen on dev; the numbers
reported are on test.

Run:  python -m eval.faithfulness --collect     (needs Ollama; ~10 minutes)
      python -m eval.faithfulness [--out eval/faithfulness_results.json]
"""
import json
import random
import re
import sys
from pathlib import Path

ANSWERS = Path("eval/faithfulness_answers.json")
QUESTION_FILES = ["eval/questions.jsonl", "eval/questions_set2.jsonl", "eval/questions_set3.jsonl"]
REPAIRS: list[int] = []
_NUM = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w]*\])")
_NEG = [(" should not ", " should "), (" is not ", " is "), (" are not ", " are "), (" not recommended", " recommended"),
        (" contraindicated", " recommended"), (" increase", " decrease"), (" increased", " decreased"),
        (" higher ", " lower "), (" avoid ", " use ")]


def collect():
    from src.agent.graph import run_agent, warmup
    warmup()
    out = []
    for f in QUESTION_FILES:
        for line in open(f, encoding="utf-8"):
            q = json.loads(line)
            if not q.get("drug"):
                continue
            r = run_agent(q["q"])
            out.append({"q": q["q"], "drug": q["drug"], "answer": r["answer"], "answered": r["answered"],
                        "sources": [{k: s.get(k, "") for k in ("title", "section", "text")} for s in r["sources"]]})
            print(f"  {len(out):3} {q['q'][:60]}", flush=True)
    ANSWERS.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Saved {len(out)} answers to {ANSWERS}")


def change_number(sentence: str, rng: random.Random) -> str | None:
    nums = [m for m in _NUM.finditer(sentence)]
    if not nums:
        return None
    m = rng.choice(nums)
    old = float(m.group(1))
    new = old * rng.choice([2, 3, 0.5, 4]) if old else 7
    text = str(int(new)) if new == int(new) else f"{new:g}"
    if text == m.group(1):
        text = str(int(old) + 3)
    return sentence[:m.start(1)] + text + sentence[m.end(1):]


def negate(sentence: str) -> str | None:
    low = sentence.lower()
    for a, b in _NEG:
        i = low.find(a)
        if i != -1:
            return sentence[:i] + b + sentence[i + len(a):]
    return None


def build(records: list[dict], seed: int = 0) -> list[dict]:
    from src.faithfulness import claims, repair_citations
    rng = random.Random(seed)
    label_chunks = [(r["drug"], s["text"]) for r in records for s in r["sources"] if s["title"].startswith("FDA label")]
    items = []
    for qi, r in enumerate(records):
        if not r["answered"]:
            continue
        split = "dev" if qi % 2 == 0 else "test"
        answer, fixes = repair_citations(r["answer"], r["sources"])     # the benchmark scores the answer as shipped
        REPAIRS.append(len(fixes))
        for sentence, cited in claims(answer):
            premises = [r["sources"][n - 1]["text"] for n in cited if 1 <= n <= len(r["sources"])]
            if not premises:
                continue
            base = {"q": r["q"], "split": split}
            items.append({**base, "kind": "real", "claim": sentence, "premises": premises})
            other = [t for d, t in label_chunks if d != r["drug"]]
            if other:
                items.append({**base, "kind": "wrong-source", "claim": sentence, "premises": [rng.choice(other)]})
            changed = change_number(sentence, rng)
            if changed:
                items.append({**base, "kind": "wrong-number", "claim": changed, "premises": premises})
            neg = negate(sentence)
            if neg:
                items.append({**base, "kind": "negated", "claim": neg, "premises": premises})
    return items


def rates(items, flagged) -> dict:
    out = {}
    for kind in ("real", "wrong-source", "wrong-number", "negated"):
        idx = [i for i, it in enumerate(items) if it["kind"] == kind]
        out[kind] = {"n": len(idx), "flagged": sum(flagged[i] for i in idx),
                     "rate": round(sum(flagged[i] for i in idx) / max(len(idx), 1), 3)}
    return out


def main():
    if "--collect" in sys.argv:
        return collect()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from src import faithfulness as fa
    records = json.loads(ANSWERS.read_text(encoding="utf-8"))
    items = build(records)
    sig = []
    for it in items:
        entail, contra = fa.nli(it["claim"], it["premises"])
        sig.append({"entail": entail, "contra": contra, "overlap": fa.overlap(it["claim"], it["premises"]),
                    "numbers": fa.unsupported_numbers(it["claim"], it["premises"])})
    dev = [i for i, it in enumerate(items) if it["split"] == "dev"]
    test = [i for i, it in enumerate(items) if it["split"] == "test"]

    def flags(entail_t, contra_t, overlap_t, numbers=True):
        return [(numbers and bool(s["numbers"])) or s["contra"] >= contra_t
                or (s["overlap"] < overlap_t and s["entail"] < entail_t) for s in sig]

    def score(idx, f):
        r = rates([items[i] for i in idx], [f[i] for i in idx])
        caught = (r["wrong-source"]["rate"] + r["wrong-number"]["rate"] + r["negated"]["rate"]) / 3
        return caught - r["real"]["rate"], r

    # choose the three cut-offs on dev: maximise (mean detection over the synthetic kinds) - (flag rate on real)
    grid = [(e, c, o) for e in (0.1, 0.3, 0.5, 0.7) for c in (0.5, 0.7, 0.9, 0.97, 1.01)
            for o in (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01)]
    best = max(grid, key=lambda g: score(dev, flags(*g))[0])
    print(f"{len(records)} answers -> {len(items)} labelled claims ({len(dev)} dev, {len(test)} test)")
    n_real = sum(1 for it in items if it["kind"] == "real")
    print(f"citation repair re-pointed {sum(REPAIRS)} of {n_real} cited sentences, in {sum(1 for x in REPAIRS if x)} answers")
    print(f"cut-offs chosen on dev: entailment < {best[0]}, contradiction >= {best[1]}, overlap < {best[2]}")
    report = {"cutoffs": {"entailment": best[0], "contradiction": best[1], "overlap": best[2]},
              "n_answers": len(records), "n_claims": len(items), "n_real_claims": n_real,
              "citations_repaired": sum(REPAIRS)}
    configs = [("numbers + contradiction + (overlap or entailment)  [used]", flags(*best)),
               ("entailment only (flag if < 0.5)", flags(0.5, 1.01, 1.01, numbers=False)),
               ("number check only", flags(0.0, 1.01, 0.0))]
    for name, f in configs:
        _, r = score(test, f)
        report[name] = r
        print(f"\n{name}  (test split)")
        for kind, v in r.items():
            what = "flagged (upper bound on false alarms)" if kind == "real" else "caught"
            print(f"  {kind:13} {v['flagged']:3}/{v['n']:<3} = {v['rate']:.2f}  {what}")
    f = flags(*best)
    print("\nReal claims flagged on the test split (read these: a flag is either a false alarm or a real problem):")
    for i in test:
        if items[i]["kind"] == "real" and f[i]:
            s = sig[i]
            print(f"  [entail {s['entail']:.2f} contra {s['contra']:.2f} overlap {s['overlap']:.2f}"
                  f"{' num:' + ','.join(s['numbers']) if s['numbers'] else ''}] {items[i]['claim'][:260]}")
    if "--out" in sys.argv:
        Path(sys.argv[sys.argv.index("--out") + 1]).write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
