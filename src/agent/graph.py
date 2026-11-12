"""LangGraph ReAct agent: think -> act (tool call) -> observe -> repeat -> answer.

        +-------+  tool_calls?  +-------+
 start->| agent |-------------->| tools |
        +-------+<--------------+-------+
            | no tool calls (or step limit)
            v
           END (final answer with [n] citations)
"""
import re
from typing import TypedDict

import ollama
from langgraph.graph import END, START, StateGraph

from src.agent.tools import TOOL_FUNCS, TOOL_SCHEMAS
from src.config import AGENT_MAX_STEPS, LLM_MODEL, LLM_TEMPERATURE, OLLAMA_HOST

NOT_FOUND = "I could not find this in the indexed documents."

SYSTEM_PROMPT = f"""You are a medical literature assistant with three tools:
- check_interaction: use when the question names TWO drugs ("can I take A with B").
- lookup_drug_label: use ONLY when the user asks for a specific label section of a named drug:
  boxed warning, dosage, contraindications, or "what drugs interact with X".
- search_documents: use for everything else: side effects, risks, how a drug works, what it is used
  for, evidence, comparisons, questions about a drug class. It searches FDA labels AND PubMed abstracts.
Call a tool first, read the results, and call another tool if needed.
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


def agent_node(state: State) -> dict:
    force_answer = state["steps"] >= AGENT_MAX_STEPS
    resp = _client.chat(
        model=LLM_MODEL,
        messages=state["messages"],
        tools=None if force_answer else TOOL_SCHEMAS,
        options={"temperature": LLM_TEMPERATURE},
    )
    msg = resp["message"]
    entry = {"role": "assistant", "content": msg.get("content") or ""}
    if msg.get("tool_calls"):
        entry["tool_calls"] = [
            {"function": {"name": c["function"]["name"], "arguments": dict(c["function"]["arguments"])}}
            for c in msg["tool_calls"]
        ]
    return {"messages": state["messages"] + [entry], "steps": state["steps"] + 1}


def tools_node(state: State) -> dict:
    sources, messages, trace = list(state["sources"]), list(state["messages"]), list(state["trace"])
    for call in state["messages"][-1]["tool_calls"]:
        name, args = call["function"]["name"], call["function"]["arguments"]
        fn = TOOL_FUNCS.get(name)
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
        trace.append({"tool": name, "args": args, "n_results": len(new), "note": info})
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


def _add_citations(messages: list[dict], answer: str) -> str:
    """Small models often forget citations. One repair pass: rewrite with [n] markers."""
    resp = _client.chat(
        model=LLM_MODEL,
        options={"temperature": LLM_TEMPERATURE},
        messages=messages[:-1] + [{"role": "user", "content": (
            "Rewrite this answer so that every factual statement ends with the number of the source it "
            "came from, like [1] or [1][2]. Use only the numbered sources above and keep it short. "
            "If a statement is not supported by any source, delete it.\n\n"
            f"Answer to rewrite:\n{answer}")}],
    )
    return resp["message"]["content"].strip() or answer


_graph = None


def run_agent(question: str) -> dict:
    """Returns {question, answer, cited, invalid, sources, trace, answered}."""
    global _graph
    _graph = _graph or build_graph()
    state = _graph.invoke({
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}],
        "sources": [], "trace": [], "steps": 0,
    })
    answer = state["messages"][-1]["content"].strip() or NOT_FOUND
    if state["sources"] and NOT_FOUND.lower() not in answer.lower() and not re.search(r"\[\d+\]", answer):
        answer = _add_citations(state["messages"], answer)
    used = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer)})
    cited = [n for n in used if 1 <= n <= len(state["sources"])]
    return {
        "question": question, "answer": answer, "cited": cited,
        "invalid": [n for n in used if n not in cited],
        "sources": state["sources"], "trace": state["trace"],
        "answered": NOT_FOUND.lower() not in answer.lower(),
    }
