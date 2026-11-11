# Medical Literature Assistant

An agentic RAG system that answers medical and drug questions from FDA drug labels and PubMed abstracts, with citations for every answer. Everything runs locally with [Ollama](https://ollama.com), so no API keys are needed.

> Status: **working end to end.** Retrieval, cited answers, a LangGraph ReAct agent, an eval harness, a FastAPI backend and a Streamlit UI. Hybrid search, reranking and embedding fine-tuning are next (see roadmap).

## How it works

```
Documents (FDA labels, PubMed) -> chunk -> embed (nomic-embed-text) -> FAISS index
User question -> embed query -> top-k similar chunks -> (Day 2+) LLM answer with citations
```

- **Section-aware chunking:** each label section (for example `drug_interactions`) is chunked on its own, with overlap and sentence-boundary snapping, so a citation points to one precise section.
- **Task-prefixed embeddings:** `nomic-embed-text` gets `search_document:` / `search_query:` prefixes, as it was trained with them.
- **Cosine search:** vectors are L2-normalised and stored in a FAISS inner-product index.
- **One retriever interface:** every retriever exposes `search(query, k)`, so dense, hybrid and reranked versions can be compared by the same eval harness.

## Setup (Windows / macOS / Linux)

1. Install Ollama from https://ollama.com/download, then pull the models:
   ```bash
   ollama pull nomic-embed-text
   ollama pull qwen2.5:3b
   ```
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1  # Windows PowerShell (if blocked: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass)
   # source .venv/bin/activate   # macOS / Linux
   pip install -r requirements.txt
   ```

## Run (from the project root)

```bash
python -m src.data.fetch_openfda     # ~40 FDA drug labels
python -m src.data.fetch_pubmed      # ~200 PubMed abstracts
python -m src.ingest.chunk           # -> data/processed/chunks.jsonl
python -m src.ingest.build_index     # -> data/index/
python -m scripts.search "Can I take ibuprofen with warfarin?"   # retrieval only
python -m scripts.ask "Can I take ibuprofen with warfarin?"      # cited answer (needs qwen2.5:3b)
```

Agent, API and UI:

```bash
python -m scripts.agent "Can I take ibuprofen with warfarin?"   # ReAct agent with tool trace
uvicorn src.api:app --reload                                     # POST /ask, docs at /docs
streamlit run app.py                                             # web UI
python -m eval.run_eval retrieval                                # Recall@k, MRR, score thresholds
python -m eval.run_eval agent                                    # routing, citations, refusals
```

### The agent

A LangGraph graph with two nodes, `agent` (the LLM decides) and `tools`, looping think -> act -> observe
until the model answers or hits `AGENT_MAX_STEPS`. Tools: `lookup_drug_label` (exact label section of a
named drug, good for interactions and dosing) and `search_documents` (semantic search over labels and
PubMed). Every tool result is numbered `[n]`; the answer must cite those numbers, and invalid or missing
citations are flagged in the CLI, API and UI.

Try different chunk sizes: `python -m src.ingest.chunk --size 300 --overlap 60`, then rebuild the index.

## Results (qwen2.5:3b + nomic-embed-text, 1,837 chunks, 16 in-scope + 3 out-of-scope questions)

### Retrieval: dense vs. BM25 vs. hybrid (`python -m eval.run_eval retrieval`)

| Retriever | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---|---|---|---|
| Dense (nomic-embed-text) | 0.50 | 0.75 | 0.75 | 0.61 |
| BM25 | 0.50 | 0.62 | 0.69 | 0.57 |
| Hybrid (BM25 + dense, RRF) | 0.38 | 0.75 | 0.81 | 0.56 |
| **Hybrid + drug-aware rerank** (used) | **0.62** | **0.75** | **0.88** | **0.72** |

Plain hybrid raised Recall@5 but hurt the top result; boosting chunks of any drug named in the query
fixed that and recovered questions dense search missed entirely (warfarin interactions). Caveat: the gold
labels are defined by drug, so this boost is partly aligned with what the metric rewards, and the set is
small (16 questions), so treat differences of one question (about 0.06) as noise. A cross-encoder reranker
is not included: it needs PyTorch and a model download.

### Agent (`python -m eval.run_eval agent`)

| Metric | Score |
|---|---|
| Answers with citations | 1.00 |
| Answers citing a gold source | 0.88 |
| All citations valid | 0.88 |
| Out-of-scope refusal | 1.00 (3/3) |
| Tool routing vs. gold | 0.88 |

How it got here: with the first tool descriptions the 3B model always picked `lookup_drug_label` and
never `search_documents` (routing 0.50, PubMed unused). Rewriting the prompt so label lookup is only for
named sections raised routing to 0.88 and citations to 1.00. Swapping dense for the hybrid retriever
then raised "cites a gold source" from 0.81 to 0.88. The refusal threshold (0.65) sits in the measured
gap between out-of-scope (max 0.57) and in-scope (min 0.75) top-1 dense scores. Each agent answer takes
1-2 minutes on a laptop CPU.

Known weak spots: ciprofloxacin and long-term PPI risk questions are still missed by every retriever, and
some answers still carry an invalid `[n]` (flagged in the CLI, API and UI).

## Project structure

```
src/
  config.py              all settings (models, chunk size, drug list)
  data/                  downloaders for openFDA and PubMed
  ingest/                chunking and index building
  retrieval/             embedder, dense retriever, BM25 + hybrid retriever
  generation/            plain RAG answer with citations
  agent/                 LangGraph ReAct agent and its tools
  api.py                 FastAPI backend
app.py                   Streamlit UI
scripts/                 command-line search, ask, agent
eval/                    questions.jsonl + run_eval.py
```

## Roadmap

- [x] Day 1: data download, chunking, embeddings, FAISS retrieval
- [x] Day 2: answer generation with citations (local LLM)
- [x] Day 3: LangGraph ReAct agent with document-search and drug-label tools
- [x] Day 4: evaluation harness (Recall@k, MRR, tool routing, citation validity, refusals)
- [x] Day 5: hybrid search (BM25 + dense, RRF) with a drug-aware rerank, measured (cross-encoder still open)
- [ ] Days 6–7: fine-tuned biomedical embedding model, measured
- [ ] Day 8: corrective RAG, LangFuse tracing, Streamlit UI

## Disclaimer

This is a learning project, not medical advice.
