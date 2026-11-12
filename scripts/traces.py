"""Summarise agent run traces (data/traces/traces.jsonl).

Usage:
    python -m scripts.traces            aggregate numbers
    python -m scripts.traces --last 5   also show the last 5 runs
"""
import json
import sys

from src.tracing import TRACE_FILE, load_traces, summarize


def main():
    records = load_traces()
    if not records:
        print(f"No traces yet ({TRACE_FILE}). Ask the agent something first.")
        return
    print(json.dumps(summarize(records), indent=2))
    if "--last" in sys.argv:
        n = int(sys.argv[sys.argv.index("--last") + 1])
        for r in records[-n:]:
            tools = ", ".join(f"{t['tool']}({t['ms']}ms)" for t in r["tools"]) or "none"
            print(f"\n{r['ts']}  {r['total_ms']}ms  answered={r['answered']}  tools: {tools}\n  Q: {r['question']}")


if __name__ == "__main__":
    main()
