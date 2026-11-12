"""LangGraph ReAct agent: think -> act (tool call) -> observe -> repeat -> answer.

        +-------+  tool_calls?  +-------+
 start->| agent |-------------->| tools |
        +-------+<--------------+-------+
            | no tool calls (or step limit)
            v
           END (final answer with [n] citations)
"""
import re
import time
from typing import TypedDict

import ollama
from langgraph.graph import END, START, StateGraph

from src.agent.coverage import missing_subject
from src.agent.tools import TOOL_FUNCS, TOOL_SCHEMAS
from src.citations import has_citation, parse_citations, strip_invalid, strip_markers, uncited_sentences
from src.config import AGENT_MAX_STEPS, LLM_MODEL, LLM_TEMPERATURE, OLLAMA_HOST
from src.tracing import log_trace

NOT_FOUND = "I could not find this in the indexed documents."

SYSTEM_PROMPT = f"""You are a medical literature assistant with three tools:
- check_interaction: use when the question names TWO drugs ("can I take A with B").
- lookup_drug_label: use ONLY when the user asks for a specific label section of a named drug:
  boxed warning, dosage, contraindications, or "what drugs interact with X".
- search_documents: use for everything else: side effects, risks, how a drug works, what it is used
  for, evidence, comparisons, questions about a drug class, and ANY question about pregnancy,
  breastfeeding, children, elderly, kidney or liver problems (the answer may sit in several label
  sections, and this tool searches all of them plus PubMed abstracts).
Call a tool first, read the results, and call another tool if needed.
If the user asks what THEY personally should take or do for a symptom, do not recommend a drug or
dose; instead summarize what the sources say and say that a clinician or pharmacist should decide.
Tool results are numbered sources like [1], [2]. Answer ONLY from those sources and cite every
factual statement with its number, e.g. [1][3]. If the sources do not answer the question, reply
exactly: "{NOT_FOUND}" Never use outside knowledge. Keep answers short. This is not personal
medical advice; suggest consulting a clinician or pharmacist where relevant."""

_client = ollama.Client(host=OLLAMA_HOST)


class State(TypedDict):
    messages: list[dict]
    sources: list[dict]          # global, in citation order: sources[n-1] is [n]
    trace: list[dict]            # tool calls made, for the UI / eval
    steps: int
    llm_calls: int               # observability: how many LLM calls, how long, how many tokens
    llm_ms: float
    tokens: int


def agent_node(state: State) -> dict:
    force_answer = state["steps"] >= AGENT_MAX_STEPS
    t0 = time.perf_counter()
    resp = _client.chat(
        model=LLM_MODEL,
        messages=state["messages"],
        tools=None if force_answer else TOOL_SCHEMAS,
        options={"temperature": LLM_TEMPERATURE},
    )
    llm_ms = (time.perf_counter() - t0) * 1000
    msg = resp["message"]
    entry = {"role": "assistant", "content": msg.get("content") or ""}
    if msg.get("tool_calls"):
        entry["tool_calls"] = [
            {"function": {"name": c["function"]["name"], "arguments": dict(c["function"]["arguments"])}}
            for c in msg["tool_calls"]
        ]
    return {"messages": state["messages"] + [entry], "steps": state["steps"] + 1,
            "llm_calls": state["llm_calls"] + 1, "llm_ms": state["llm_ms"] + llm_ms,
            "tokens": state["tokens"] + (resp.get("prompt_eval_count") or 0) + (resp.get("eval_count") or 0)}


def tools_node(state: State) -> dict:
    sources, messages, trace = list(state["sources"]), list(state["messages"]), list(state["trace"])
    for call in state["messages"][-1]["tool_calls"]:
        name, args = call["function"]["name"], call["function"]["arguments"]
        fn = TOOL_FUNCS.get(name)
        t0 = time.perf_counter()
        try:
            new, info = fn(**args) if fn else ([], f"Unknown tool {name}")
        except Exception as e:   # a bad tool call must not crash the agent; tell the model instead
            new, info = [], f"Tool {name} failed ({type(e).__name__}: {e}). Check the arguments and try again."
        lines = []
        for s in new:
            idx = next((i for i, o in enumerate(sources) if o["key"] == s["key"]), None)
            if idx is None:
                sources.append(s)
                idx = len(sources) - 1
            lines.append(f"[{idx + 1}] {s['title']} [{s['section']}]\n{s['text']}")
        # `info` is "" for plain successes, a corrective-RAG note ("query rewritten to ...") or an error message
        trace.append({"tool": name, "args": args, "n_results": len(new), "note": info,
                      "ms": round((time.perf_counter() - t0) * 1000)})
        messages.append({"role": "tool", "content": "\n\n".join(lines) or info, "tool_name": name})
    return {"messages": messages, "sources": sources, "trace": trace}


def route(state: State) -> str:
    return "tools" if state["messages"][-1].get("tool_calls") and state["steps"] <= AGENT_MAX_STEPS else END


def build_graph():
    g = StateGraph(State)
    g.add_node("agent", agent_node)
    g.add_node("tools", tools_node)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    return g.compile()


def _add_citations(messages: list[dict], answer: str, n_sources: int) -> str:
    """Small models forget citations or invent source numbers. One repair pass: rewrite with valid [n]."""
    resp = _client.chat(
        model=LLM_MODEL,
        options={"temperature": LLM_TEMPERATURE},
        messages=messages[:-1] + [{"role": "user", "content": (
            "Rewrite this answer so that every factual statement ends with the number of the source it "
            f"came from, like [1] or [1][2]. The only valid source numbers are 1 to {n_sources}. "
            "Use only the numbered sources above and keep it short. "
            "If a statement is not supported by any source, delete it.\n\n"
            f"Answer to rewrite:\n{answer}")}],
    )
    rewritten = resp["message"]["content"].strip()
    # The 3B model sometimes "repairs" an answer into a bare "[1]". Keep the original rather than lose the content.
    if len(strip_markers(rewritten)) < max(30, len(strip_markers(answer)) // 4):
        return answer
    return rewritten


_SPECIAL_POPULATION = re.compile(
    r"pregnan|breast-?feed|lactat|nursing|infant|child|pediatric|elderly|older adult|renal|kidney|"
    r"hepatic|liver", re.I)


_PERSONAL_ADVICE = re.compile(
    r"what (should|can|do) i (take|use|do)|which (one |drug |medicine |medication )?should i|should i (take|use)|"
    r"what('s| is) (best|safest) (for|to take)|i have (a |an )?\w+.*what", re.I)


def routing_hint(question: str) -> str:
    """Small models ignore general routing rules but follow a direct hint. Questions about special
    populations need several label sections (pregnancy limits can sit in contraindications AND
    use_in_specific_populations), which a single-section lookup would miss."""
    hint = ""
    if _SPECIAL_POPULATION.search(question):
        hint += "\n\n(Tool hint: use search_documents for this question; it covers every label section.)"
    if _PERSONAL_ADVICE.search(question):
        hint += ("\n\n(Important: do NOT recommend a specific drug or dose for this person. Summarize only what "
                 "the sources say about uses and cautions, and tell them a clinician or pharmacist should decide.)")
    return hint


_graph = None


def run_agent(question: str) -> dict:
    """Returns {question, answer, cited, invalid, sources, trace, answered}."""
    global _graph
    _graph = _graph or build_graph()
    t_start = time.perf_counter()
    missing = missing_subject(question)
    if missing:
        # The question is about a drug the index has no label or abstract for. Retrieval would return chunks
        # that merely mention it (lithium appears in other drugs' interaction sections), which would produce a
        # confident answer to a different question. Say so instead.
        result = {
            "question": question, "cited": [], "invalid": [], "uncited": [], "sources": [], "answered": False,
            "answer": f"{NOT_FOUND} The indexed documents have no label or abstract for {', '.join(missing)}.",
            "trace": [{"tool": "coverage_check", "args": {"missing": missing}, "n_results": 0,
                       "note": "named drug not in index; refused before retrieval", "ms": 0}],
            "stats": {"total_ms": round((time.perf_counter() - t_start) * 1000), "llm_calls": 1, "llm_ms": 0,
                      "tokens": 0, "steps": 0},
        }
        log_trace(result)
        return result
    state = _graph.invoke({
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": question + routing_hint(question)}],
        "sources": [], "trace": [], "steps": 0, "llm_calls": 0, "llm_ms": 0.0, "tokens": 0,
    })
    answer = state["messages"][-1]["content"].strip() or NOT_FOUND
    n = len(state["sources"])
    if n == 0:
        # Nothing was retrieved, so any answer would come from the model's own memory. The whole point of
        # the system is to answer only from documents, so refuse instead.
        answer = NOT_FOUND
    if n and NOT_FOUND.lower() not in answer.lower():
        _, bad = parse_citations(answer, n)
        if not has_citation(answer) or bad:        # missing or made-up citations: one repair pass
            answer = _add_citations(state["messages"], answer, n)
        answer = strip_invalid(answer, n)          # whatever made-up markers survive are removed
    if n and _PERSONAL_ADVICE.search(question) and NOT_FOUND.lower() not in answer.lower():
        # The 3B model sometimes still names a drug to take. A fixed notice makes the limit unmistakable.
        answer = ("I can't recommend what you personally should take; a clinician or pharmacist should decide. "
                  "What the indexed sources say:\n\n" + answer)
    cited, invalid = parse_citations(answer, n)
    result = {
        "question": question, "answer": answer, "cited": cited,
        "invalid": invalid,
        "uncited": uncited_sentences(answer) if n and NOT_FOUND.lower() not in answer.lower() else [],
        "sources": state["sources"], "trace": state["trace"],
        "answered": NOT_FOUND.lower() not in answer.lower(),
        "stats": {"total_ms": round((time.perf_counter() - t_start) * 1000), "llm_calls": state["llm_calls"],
                  "llm_ms": round(state["llm_ms"]), "tokens": state["tokens"], "steps": state["steps"]},
    }
    log_trace(result)
    return result
