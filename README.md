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

## Results (qwen2.5:3b + nomic-embed-text, 1,837 chunks, 19 in-scope + 3 out-of-scope questions)

The question set is small, so treat a one-question difference (about 0.05) as noise. *Strict* counts only the
expected FDA label section; *relaxed* also counts a PubMed abstract on the topic (a legitimate source that the
strict gold labels ignore; only 4 questions carry topic keywords for this).

### Retrieval (`python -m eval.run_eval retrieval`)

| Retriever | Recall@1 strict / relaxed | Recall@5 strict / relaxed | MRR strict / relaxed |
|---|---|---|---|
| Dense (nomic-embed-text) | 0.47 / 0.68 | 0.74 / 0.89 | 0.60 / 0.79 |
| BM25 | 0.42 / 0.63 | 0.58 / 0.74 | 0.48 / 0.67 |
| Hybrid (BM25 + dense, RRF) | 0.37 / 0.58 | 0.79 / 0.95 | 0.54 / 0.73 |
| **Hybrid + drug-aware rerank** (used) | **0.58 / 0.68** | **0.84 / 0.95** | **0.68 / 0.79** |
| + corrective RAG | 0.58 / 0.68 | 0.84 / 0.95 | 0.68 / 0.80 |

Plain hybrid raised Recall@5 but hurt the top result; boosting chunks of any drug named in the query fixed
that (for example warfarin interaction questions that dense search missed). Caveat: the gold labels are defined
by drug, so this boost is partly aligned with what the metric rewards. Corrective RAG barely changes these
numbers because most questions already grade as "correct" and never trigger a rewrite; its value shows on
lay-language questions and out-of-domain refusals. A cross-encoder reranker is not included (it needs PyTorch
and a model download).

### Agent (`python -m eval.run_eval agent`)

| Metric | Score |
|---|---|
| Tool routing vs. gold | 0.89 |
| Answers with citations | 0.95 |
| All citations valid | 0.84 |
| Answers citing a gold source | 0.84 |
| Out-of-scope refusal | 1.00 (3/3) |

How it got here: with the first tool descriptions the 3B model always picked `lookup_drug_label` and never
`search_documents` (routing 0.50, PubMed unused). Rewriting the prompt raised routing to 0.88; the hybrid
retriever raised "cites a gold source" from 0.81 to 0.88 (on the original 16 questions). Adding three
lay-language questions (brand names, "blood thinner") lowered the aggregate slightly (citations 1.00 -> 0.95:
the "Tylenol with a blood thinner" answer came back without a citation). Corrective RAG grades retrieval by
similarity score, rewrites ambiguous queries into label terminology, and refuses without rewriting when
retrieval is clearly off-topic (an early version let the rewriter turn "fix a flat tire" into a drug query).
A drug-name normalizer maps brand and lay names (Tylenol, paracetamol, Coumadin) to the generic names the label
tools use. The refusal threshold (0.65) sits in the measured gap between out-of-scope (max 0.57) and in-scope
(min 0.66) top-1 dense scores, a narrow margin on the lay-language questions. Each agent answer takes 1-2
minutes on a laptop CPU.

Known weak spots: ciprofloxacin and PPI risk questions still miss under the strict metric (the cipro label
openFDA returned is short and has no boxed warning); "what painkiller is safest for a stomach ulcer" retrieves
nothing gold; some answers still carry an invalid `[n]` (flagged in the CLI, API and UI).

### Embedding fine-tuning experiment (`src/finetune/`, needs a GPU; RTX 3050 6 GB used here)

Goal: does fine-tuning a small embedding model on this domain beat the off-the-shelf one?

1. `make_pairs.py`: qwen2.5:3b writes 2 questions per chunk (3,345 questions over 1,731 chunks). The train/test
   split is by *document*, so no passage from a held-out document is seen in training (423 held-out questions).
2. `train.py`: BAAI/bge-small-en-v1.5 (33M params), in-batch-negatives contrastive loss, 3 epochs, batch 32,
   fp16. Training takes under a minute.
3. `eval_embed.py`: all models search the same 1,837-chunk corpus with dense search only.

| Model | Held-out synthetic R@5 / MRR | Hand-written strict R@5 / MRR | Hand-written relaxed R@5 / MRR |
|---|---|---|---|
| nomic-embed-text (used) | 0.816 / 0.640 | 0.737 / 0.646 | 0.895 / 0.821 |
| bge-small, untuned | 0.816 / 0.648 | 0.684 / 0.544 | 0.789 / 0.728 |
| bge-small, fine-tuned on long questions | 0.872 / 0.687 | 0.684 / 0.481 | 0.842 / 0.667 |
| bge-small, fine-tuned on long + short questions | 0.851 / 0.681 | 0.684 / 0.537 | 0.842 / 0.731 |

What happened, honestly: the first fine-tune (long, detailed synthetic questions) improved the held-out
synthetic questions but made the hand-written questions *worse* (strict R@1 0.47 -> 0.32): it learned the
style of the synthetic questions. A second attempt, made after seeing that result, added short search-box
style questions to the training data and recovered the hand-written score to the untuned level while keeping
most of the synthetic gain. Even so, **no fine-tuned model beats nomic-embed-text on the hand-written
questions, so the pipeline keeps nomic.** The hand-written set is only 19 questions (one question is about
0.05), the second attempt was chosen after seeing test results, and training questions are LLM-generated, so
these results are indicative, not conclusive. Likely next steps: hard-negative mining, more and
cleaner training questions, or fine-tuning a larger base model.

## Project structure

```
src/
  config.py              all settings (models, chunk size, drug list)
  data/                  downloaders for openFDA and PubMed
  ingest/                chunking and index building
  retrieval/             embedder, dense retriever, BM25 + hybrid retriever, corrective RAG
  generation/            plain RAG answer with citations
  agent/                 LangGraph ReAct agent and its tools
  finetune/              synthetic pairs, contrastive fine-tuning, embedding comparison
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
- [x] Days 6–7: embedding fine-tuning experiment, measured (did not beat nomic-embed-text; see results)
- [x] Day 8 (part): corrective RAG (grade, rewrite, refuse), Streamlit UI
- [ ] Day 8 (rest): LangFuse tracing

## Disclaimer

This is a learning project, not medical advice.
