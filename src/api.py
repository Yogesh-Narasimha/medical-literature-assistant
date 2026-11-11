"""FastAPI backend.

Run:  uvicorn src.api:app --reload
Docs: http://localhost:8000/docs
"""
from fastapi import FastAPI
from pydantic import BaseModel

from src.agent.graph import run_agent

app = FastAPI(title="Medical Literature Assistant")


class Ask(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(body: Ask):
    r = run_agent(body.question)
    return {
        "question": r["question"],
        "answer": r["answer"],
        "answered": r["answered"],
        "citations": [
            {"n": n, **{k: r["sources"][n - 1][k] for k in ("title", "section", "url")}}
            for n in r["cited"]
        ],
        "invalid_citations": r["invalid"],
        "tool_calls": r["trace"],
    }
