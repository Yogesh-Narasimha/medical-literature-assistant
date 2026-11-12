"""Tracing tests: local JSONL traces, summary maths, and the LangFuse export against a stand-in server."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from src import tracing

RESULT = {
    "question": "Can I take ibuprofen with warfarin?", "answered": True, "cited": [1, 2], "invalid": [],
    "uncited": ["x"], "sources": [{}, {}],
    "trace": [{"tool": "check_interaction", "args": {"drug_a": "ibuprofen", "drug_b": "warfarin"},
               "n_results": 2, "note": "", "ms": 15}],
    "stats": {"total_ms": 6000, "llm_calls": 2, "llm_ms": 5800, "tokens": 1500, "steps": 2},
}


def test_log_trace_writes_one_jsonl_record(tmp_path):
    path = tmp_path / "t" / "traces.jsonl"
    tracing.log_trace(RESULT, path)
    tracing.log_trace(RESULT, path)
    recs = tracing.load_traces(path)
    assert len(recs) == 2
    assert recs[0]["question"] == RESULT["question"] and recs[0]["tools"][0]["ms"] == 15
    assert recs[0]["uncited_statements"] == 1 and recs[0]["tokens"] == 1500


def test_log_trace_never_raises(tmp_path):
    tracing.log_trace({"broken": True}, tmp_path / "x.jsonl")          # missing keys: must be swallowed


def test_summarize():
    refused = {**tracing.to_record(RESULT), "answered": False, "total_ms": 2000,
               "tools": [{"tool": "search_documents", "args": {}, "n_results": 0, "ms": 5,
                          "note": "query rewritten to 'x'"}]}
    s = tracing.summarize([tracing.to_record(RESULT), refused])
    assert s["runs"] == 2 and s["refusal_rate"] == 0.5 and s["corrective_rewrite_rate"] == 0.5
    assert s["latency_ms_p50"] in (2000, 6000) and s["tool_usage"] == {"check_interaction": 1, "search_documents": 1}
    assert tracing.summarize([]) == {"runs": 0}


def test_langfuse_export_reaches_a_server(monkeypatch):
    pytest.importorskip("langfuse")
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append((self.path, self.rfile.read(int(self.headers.get("Content-Length", 0)))))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *a):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("LANGFUSE_HOST", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    try:
        tracing.send_to_langfuse(tracing.to_record(RESULT))
    finally:
        server.shutdown()
    body = b"".join(b for _, b in received)
    assert any("/api/public/otel" in p for p, _ in received), [p for p, _ in received]
    assert b"medical-assistant-run" in body and b"check_interaction" in body
