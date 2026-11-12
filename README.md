# Medical Literature Assistant

An agentic RAG system that answers medical and drug questions from FDA drug labels and PubMed abstracts, with citations for every answer. Everything runs locally with [Ollama](https://ollama.com), so no API keys are needed.

> Status: **working end to end** (API and UI both run-tested locally). Retrieval, cited answers, a LangGraph ReAct agent, an eval harness, a FastAPI backend and a Streamlit UI. Hybrid search, reranking and embedding fine-tuning are next (see roadmap).

![Streamlit UI: cited answer, expandable sources, agent trace](docs/ui-screenshot.png)

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
3. Optional extras:
   - Cross-encoder reranker (on by default if installed; skipped silently otherwise). Needs PyTorch, then
     `pip install -r requirements-finetune.txt`. For an NVIDIA GPU:
     `pip install torch --index-url https://download.pytorch.org/whl/cu121` (the pins in
     `requirements-finetune.txt` match torch 2.5.1).
   - Tests: `pip install -r requirements-dev.txt`. LangFuse export: `pip install -r requirements-observability.txt`.

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
python -m scripts.traces                                         # latency / refusal / tool-usage summary of past runs
```

### The agent

Before the graph runs, a coverage check refuses questions about drugs the index does not contain. The graph
has two nodes, `agent` (the LLM decides) and `tools`, looping think -> act -> observe until the model answers
or hits `AGENT_MAX_STEPS`. Tools: `check_interaction` (what two drugs' labels say about each other, including
drug classes), `lookup_drug_label` (exact label section of a named drug) and `search_documents` (hybrid
search with a cross-encoder rerank and corrective RAG over labels and PubMed). Every tool result is numbered
`[n]`; the answer must cite those numbers, and invalid, missing or uncited claims are flagged in the CLI, API
and UI. Each run is traced (see Observability below).

Try different chunk sizes: `python -m src.ingest.chunk --size 300 --overlap 60`, then rebuild the index.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q          # 41 unit tests, no Ollama or index needed (chunking, BM25/hybrid fusion, reranking, corrective RAG,
                             # citation checks, coverage check, drug-name matching, tracing, train/test split)
```

The tests were checked by deliberately breaking the code (the hybrid fusion key and the out-of-domain
refusal): the matching tests failed, then passed again once the code was restored.

## Docker (written but not tested)

`Dockerfile` and `docker-compose.yml` run Ollama, the API (`:8000`) and the UI (`:8501`) as services. I could
not build or run them because Docker is not installed on my development machine, so treat them as an
untested starting point. The commands to pull models and build the index are in the compose file header.

## Results

Setup: qwen2.5:3b + nomic-embed-text, 1,837 chunks, **43 in-scope + 7 out-of-scope hand-written questions**
(`eval/questions.jsonl`; the out-of-scope set includes two medical questions about drugs that are not in the
index, lithium and adalimumab). With 43 questions one question is about 0.02-0.03, so treat differences of
one or two questions as noise; the LLM also varies slightly between runs. *Strict* counts only the expected FDA
label section; *relaxed* also counts a PubMed abstract on the topic (only 4 questions carry topic keywords).

### Retrieval (`python -m eval.run_eval retrieval`)

| Retriever | Recall@1 strict / relaxed | Recall@5 strict / relaxed | MRR strict / relaxed |
|---|---|---|---|
| Dense (nomic-embed-text) | 0.53 / 0.63 | 0.79 / 0.86 | 0.65 / 0.73 |
| BM25 | 0.58 / 0.67 | 0.72 / 0.79 | 0.62 / 0.71 |
| Hybrid (BM25 + dense, RRF) | 0.56 / 0.65 | 0.84 / 0.91 | 0.67 / 0.76 |
| Hybrid + drug-aware boost | 0.70 / 0.74 | 0.86 / 0.91 | 0.76 / 0.81 |
| + corrective RAG | 0.70 / 0.74 | 0.86 / 0.91 | 0.76 / 0.81 |
| Hybrid + cross-encoder rerank | 0.67 / 0.77 | 0.84 / 0.91 | 0.75 / 0.83 |
| **Hybrid + drug boost + cross-encoder (used)** | **0.74 / 0.84** | 0.84 / 0.91 | **0.78 / 0.87** |

- Plain hybrid raised Recall@5 but hurt the top result; boosting chunks of any drug named in the query fixed
  that. The cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`, top 20 candidates, GPU) adds a smaller,
  consistent gain on top: about 2 questions at rank 1 under strict gold, 4 under relaxed.
- Caveats: the gold labels are defined by drug, so the drug boost is partly aligned with what the metric
  rewards. The cross-encoder is a general web-search model, not a medical one, and it loses one question the
  boost alone got right ("what drugs interact with warfarin": it prefers a PubMed overview abstract, which
  only the relaxed gold accepts). Corrective RAG does not change retrieval metrics because most questions grade
  as "correct" and never trigger a rewrite; it matters for lay wording and refusals.
- The reranker is switched by `USE_RERANKER` in `src/config.py` and falls back silently if PyTorch is missing.

### Agent (`python -m eval.run_eval agent`)

| Metric | Score |
|---|---|
| Tool routing vs. gold (43 in-scope) | 0.95 |
| Answers with citations | 0.98 |
| All citations valid | 0.98 |
| Answers citing a gold source | 0.91 |
| Out-of-scope refusal (7) | 1.00 |

From the run traces of that eval (50 runs, RTX 3050 laptop GPU): median latency 4.2 s, p95 12.9 s, about
1.9 LLM calls and 2,200 tokens per question, 4% of runs triggered a corrective query rewrite, and 11 of 50
runs had at least one statement flagged as uncited. That flag is a heuristic (a citation at the start or end
of a paragraph covers it) and still fires on benign conclusion sentences, so read it as "check this", not
"this is wrong". One in-scope question was left without a citation in that run (the sacubitril/valsartan
boxed warning, because the model wrote the drug name without "and"); fuzzy drug-name matching was added
afterwards and a unit test covers it, but the eval was not re-run after that fix.

How it got here: with the first tool descriptions the 3B model always picked `lookup_drug_label` and never
`search_documents` (routing 0.50, PubMed unused); rewriting the prompt raised routing to 0.88. Adding hybrid
retrieval, the guardrails below and a larger question set moved the numbers to the table above. Each step was
re-measured; earlier smaller question sets gave slightly different numbers.

### Guardrails added after manual testing

Running varied questions by hand (not just the eval set) exposed failures the metrics had missed, so the
agent now has these safeguards, each with unit tests:

- **No sources, no answer:** if nothing was retrieved the answer is "not found", never the model's memory.
- **Coverage check:** an LLM call extracts the drug names in the question; if none of them is in the index,
  the system refuses and names the missing drug. Without it, "side effects of lithium" was answered from other
  drugs' interaction sections that merely mention lithium (a confident answer to a different question). A
  question naming at least one covered drug still proceeds ("lithium with ibuprofen" checks whether the
  ibuprofen label mentions lithium and says so if not).
- **Citation repair:** missing or non-existent `[n]` markers trigger one rewrite pass, and any made-up marker
  that survives is stripped; a rewrite that collapses into a bare `[1]` is rejected.
- **Uncited-statement flag:** factual sentences with no citation are listed as unverified in the CLI, API
  (`uncited_statements`) and UI.
- **Routing hints:** pregnancy/kidney/liver/child questions go to document search because they span several
  label sections (a single-section lookup had missed warfarin's pregnancy contraindication).
- **Tool robustness:** brand and lay names (Tylenol, Coumadin), drug classes ("blood thinner", "NSAID"),
  placeholder arguments ("nil"), a dropped "and" and close misspellings are resolved instead of failing.
- **Personal advice:** "what should I take for X" answers start with a fixed notice that a clinician or
  pharmacist must decide. The 3B model still sometimes names drugs in the body, so this is a safeguard, not
  a guarantee.

Still a 3B model: it can phrase things loosely (for example softening "contraindicated"), which is why the
sources are always shown next to the answer. Known weak spots: ciprofloxacin and PPI risk questions still
miss under the strict metric (the cipro label openFDA returned is short and has no boxed warning), and a
vague lay question like "what painkiller is safest for a stomach ulcer" retrieves nothing gold.

### Observability (`src/tracing.py`, `python -m scripts.traces`)

Every agent run appends a JSON record to `data/traces/traces.jsonl`: question, tool calls with arguments,
result counts and per-tool latency, corrective-RAG notes, LLM calls, tokens, total latency, whether it was a
refusal, and citation counters. `python -m scripts.traces` prints p50/p95 latency, refusal and rewrite rates
and tool usage. Optionally (`pip install -r requirements-observability.txt` and set `LANGFUSE_PUBLIC_KEY`,
`LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`) each run is also exported to LangFuse. The export is tested against a
local stand-in HTTP server that receives the spans, not against LangFuse Cloud, and it is sent after the run
finishes, so span durations are near zero (the real latency is in each span's metadata).

### Embedding fine-tuning experiment (`src/finetune/`, needs a GPU; RTX 3050 6 GB used here)

Goal: does fine-tuning a small embedding model on this domain beat the off-the-shelf one?

1. `make_pairs.py`: qwen2.5:3b writes 2 questions per chunk (3,345 questions over 1,731 chunks). The train/test
   split is by *document*, so no passage from a held-out document is seen in training (423 held-out questions).
2. `train.py`: BAAI/bge-small-en-v1.5 (33M params), in-batch-negatives contrastive loss, 3 epochs, batch 32,
   fp16. Training takes under a minute.
3. `eval_embed.py`: all models search the same 1,837-chunk corpus with dense search only.

| Model | Held-out synthetic R@5 / MRR | Hand-written (43) strict R@5 / MRR | Hand-written relaxed R@5 / MRR |
|---|---|---|---|
| nomic-embed-text (used) | 0.816 / 0.640 | 0.791 / 0.679 | 0.860 / 0.757 |
| bge-small, untuned | 0.816 / 0.648 | 0.744 / 0.630 | 0.791 / 0.712 |
| bge-small, fine-tuned on long questions | 0.872 / 0.687 | 0.767 / 0.581 | 0.837 / 0.663 |
| bge-small, fine-tuned on long + short questions | 0.851 / 0.681 | 0.721 / 0.612 | 0.791 / 0.698 |

What happened, honestly: fine-tuning improved the held-out *synthetic* questions (MRR 0.648 -> 0.687) but made
the hand-written questions *worse*: the model learned the style of the LLM-written questions. A second attempt,
made after seeing that result, added short search-box style questions to the training data; it recovered part of
the loss but still sits slightly below the untuned model on hand-written questions. **No fine-tuned model beats
nomic-embed-text, so the pipeline keeps nomic.** The training questions are LLM-generated and the hand-written
set is small, so these results are indicative, not conclusive. Likely next steps: hard-negative mining, more and
cleaner training questions, or fine-tuning a larger base model.

## Project structure

```
src/
  config.py              all settings (models, chunk size, drug list)
  data/                  downloaders for openFDA and PubMed
  ingest/                chunking and index building
  retrieval/             embedder, dense retriever, BM25 + hybrid, corrective RAG, cross-encoder rerank
  generation/            plain RAG answer with citations
  agent/                 LangGraph ReAct agent, its tools, drug-coverage check
  tracing.py             per-run JSON traces (+ optional LangFuse export)
  finetune/              synthetic pairs, contrastive fine-tuning, embedding comparison
  api.py                 FastAPI backend
app.py                   Streamlit UI
scripts/                 command-line search, ask, agent, trace summary
tests/                   unit tests
eval/                    questions.jsonl + run_eval.py
```

## Roadmap

- [x] Day 1: data download, chunking, embeddings, FAISS retrieval
- [x] Day 2: answer generation with citations (local LLM)
- [x] Day 3: LangGraph ReAct agent with document-search and drug-label tools
- [x] Day 4: evaluation harness (Recall@k, MRR, tool routing, citation validity, refusals)
- [x] Day 5: hybrid search (BM25 + dense, RRF), drug-aware boost and cross-encoder rerank, measured
- [x] Days 6–7: embedding fine-tuning experiment, measured (did not beat nomic-embed-text; see results)
- [x] Day 8 (part): corrective RAG (grade, rewrite, refuse), Streamlit UI
- [x] Day 8 (rest): run tracing (local JSON + optional LangFuse export, tested against a stand-in server)
- [ ] Not done: build and run the Docker setup (no Docker on my machine); validate on a larger, independently written question set

## Disclaimer

This is a learning project, not medical advice.
