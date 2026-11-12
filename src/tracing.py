"""Run tracing: every agent run is appended to data/traces/traces.jsonl.

Each record answers "what happened and how long did it take": the question, tools called (with arguments,
result counts, per-tool latency, corrective-RAG notes), LLM calls/latency/tokens, whether the answer was
a refusal, and the citation quality counters. `python -m scripts.traces` summarises the file.

Optional LangFuse export: if LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set and the `langfuse`
package is installed, each run is also sent as a LangFuse trace (see send_to_langfuse). Tracing never
raises: an observability failure must not break an answer.
"""
import json
import os
import time

from src.config import DATA_DIR

TRACE_FILE = DATA_DIR / "traces" / "traces.jsonl"


def to_record(result: dict) -> dict:
    """The compact, JSON-safe view of an agent result that is worth keeping."""
    return {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "question": result["question"],
        "answered": result["answered"],
        "n_sources": len(result["sources"]),
        "cited": result["cited"],
        "invalid_citations": result["invalid"],
        "uncited_statements": len(result.get("uncited", [])),
        "uncited_examples": [s[:200] for s in result.get("uncited", [])[:5]],     # to audit the checker
        "tools": [{"tool": t["tool"], "args": t["args"], "n_results": t["n_results"],
                   "ms": t.get("ms"), "note": t.get("note", "")[:200]} for t in result["trace"]],
        **result.get("stats", {}),
    }


def log_trace(result: dict, path=TRACE_FILE) -> None:
    try:
        rec = to_record(result)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"):
            send_to_langfuse(rec)
    except Exception:
        pass


def send_to_langfuse(rec: dict) -> None:
    """Send one finished run to LangFuse (SDK 4.x observation API): one root 'agent' observation with a
    child 'tool' observation per tool call. The run is exported after it finished, so span durations are
    near zero; the real per-tool latency is in each observation's metadata ("ms"). Verified against a local
    stand-in HTTP server (tests/test_tracing.py), not against LangFuse Cloud."""
    from langfuse import get_client
    client = get_client()
    meta = {k: rec[k] for k in ("answered", "n_sources", "total_ms", "llm_calls", "llm_ms", "tokens", "steps")
            if k in rec}
    with client.start_as_current_observation(name="medical-assistant-run", as_type="agent",
                                             input=rec["question"], metadata=meta) as root:
        for t in rec["tools"]:
            with client.start_as_current_observation(
                    name=t["tool"], as_type="tool", input=t["args"],
                    metadata={"ms": t["ms"], "note": t["note"]}) as span:
                span.update(output={"n_results": t["n_results"]})
        root.update(output={"answered": rec["answered"], "cited": rec["cited"],
                            "invalid_citations": rec["invalid_citations"]})
    client.flush()


def load_traces(path=TRACE_FILE) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def summarize(records: list[dict]) -> dict:
    """Aggregate numbers for a list of trace records."""
    if not records:
        return {"runs": 0}
    lat = sorted(r.get("total_ms", 0) for r in records)
    pct = lambda p: lat[min(len(lat) - 1, int(p * len(lat)))]          # noqa: E731
    tools: dict[str, int] = {}
    for r in records:
        for t in r["tools"]:
            tools[t["tool"]] = tools.get(t["tool"], 0) + 1
    n = len(records)
    return {
        "runs": n,
        "latency_ms_p50": pct(0.50), "latency_ms_p95": pct(0.95),
        "avg_llm_calls": round(sum(r.get("llm_calls", 0) for r in records) / n, 2),
        "avg_tokens": round(sum(r.get("tokens", 0) for r in records) / n),
        "refusal_rate": round(sum(1 for r in records if not r["answered"]) / n, 3),
        "corrective_rewrite_rate": round(sum(1 for r in records if any("rewritten" in t["note"] for t in r["tools"])) / n, 3),
        "runs_with_invalid_citations": sum(1 for r in records if r["invalid_citations"]),
        "runs_with_uncited_statements": sum(1 for r in records if r["uncited_statements"]),
        "tool_usage": tools,
    }
