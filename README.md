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
python -m src.data.fetch_openfda     # ~300 FDA drug labels (see DRUGS in src/config.py)
python -m src.data.fetch_pubmed      # ~1,900 PubMed abstracts (79 queries)
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
python -m pytest -q          # 55 unit tests, no Ollama or index needed (chunking, BM25/hybrid fusion, reranking, corrective RAG,
                             # citation checks, coverage check, drug-name matching, tracing, train/test split)
```

The tests were checked by deliberately breaking the code (the hybrid fusion key and the out-of-domain
refusal): the matching tests failed, then passed again once the code was restored.

## Docker (image not built; its commands were tested)

`Dockerfile` and `docker-compose.yml` run Ollama, the API (`:8000`) and the UI (`:8501`) as services. I could
not build the image or run `docker compose`: Docker Desktop needs admin rights and WSL2, neither of which my
development machine has in this session. What I did test, outside Docker: a fresh clone of the repo installs
from `requirements.txt` in a clean virtual environment and passes the tests; the exact container commands
(`uvicorn src.api:app --host 0.0.0.0 --port 8000` with `OLLAMA_HOST` set, and `streamlit run app.py
--server.address 0.0.0.0 --server.port 8501`) start, answer `/ask`, and serve the page. Not tested: the image
build itself, the volume mounts and the service networking, so treat the Docker files as an unverified starting
point. The commands to pull models and build the index are in the compose file header.

## Results

Setup: qwen2.5:3b + nomic-embed-text. The project was first built and measured on **40 drugs / 1,837 chunks**,
then scaled to **298 drugs / 1,907 PubMed abstracts / 17,693 chunks** (about 10x) to see what breaks; the tables
below show both, and everything after "Scaling" is on the 298-drug index. **Question set 1**
(`eval/questions.jsonl`): 43 in-scope + 7 out-of-scope hand-written questions; the out-of-scope set includes two
medical questions about drugs that are not in the index (lithium, adalimumab). **Question set 2**
(`eval/questions_set2.jsonl`): 16 + 3 questions about other drugs, written later and evaluated once before I
changed anything. **Adversarial set** (`eval/adversarial.jsonl`): 31 questions written to break the assistant.
All three sets were written by me, not by an independent person, and I fixed bugs that sets 2 and the adversarial
set exposed, so only each set's first run is a clean held-out result. With these sizes one question is about
0.02-0.06, so treat differences of one or two questions as noise; the LLM also varies slightly between runs.
*Strict* counts only the expected FDA label section; *relaxed* also counts a PubMed abstract on the topic (only 4
questions carry topic keywords).

### Retrieval (`python -m eval.run_eval retrieval`)

| Retriever | 40 drugs: R@1 / R@5 / MRR (strict) | 298 drugs: R@1 / R@5 / MRR (strict) | 298 drugs, relaxed |
|---|---|---|---|
| Dense (nomic-embed-text) | 0.53 / 0.79 / 0.65 | 0.42 / 0.70 / 0.53 | 0.49 / 0.74 / 0.60 |
| BM25 | 0.58 / 0.72 / 0.62 | 0.51 / 0.72 / 0.59 | 0.58 / 0.77 / 0.65 |
| Hybrid (BM25 + dense, RRF) | 0.56 / 0.84 / 0.67 | 0.49 / 0.81 / 0.61 | 0.56 / 0.86 / 0.68 |
| Hybrid + drug-aware boost | 0.70 / 0.86 / 0.76 | 0.67 / 0.86 / 0.75 | 0.72 / 0.88 / 0.79 |
| + corrective RAG | 0.70 / 0.86 / 0.76 | 0.67 / 0.86 / 0.75 | 0.72 / 0.88 / 0.79 |
| Hybrid + cross-encoder rerank | 0.74 / 0.84 / 0.78 | 0.70 / 0.86 / 0.76 | 0.77 / 0.93 / 0.83 |
| **Hybrid + drug boost + cross-encoder (used)** | **0.77 / 0.86 / 0.81** | **0.70 / 0.86 / 0.76** | **0.77 / 0.93 / 0.83** |

Scaling to 7.5x more drugs cost the plain retrievers a lot (dense MRR 0.65 to 0.53) and the final configuration
only a little (MRR 0.81 to 0.76, Recall@5 unchanged at 0.86): the drug-aware boost and the reranker are what make
the system hold up as the index grows. The 40-drug numbers are from the earlier runs of the same code (before the
brand-name and misspelling changes below); the 298-drug column is the final code.

- Plain hybrid raised Recall@5 but hurt the top result; boosting chunks of any drug named in the query fixed
  that. The cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`, top 20 candidates, GPU) adds a smaller,
  consistent gain on top. It reads each passage prefixed with its title and section ("FDA label: cetirizine
  (indications and usage): ..."): OTC labels often never name the drug (the cetirizine label does not contain the
  word "cetirizine"), and without the prefix the reranker buried the drug's own chunks. That change moved the
  final configuration from MRR 0.78 to 0.81 (strict).
- Caveats: the gold labels are defined by drug, so the drug boost is partly aligned with what the metric
  rewards. The cross-encoder is a general web-search model, not a medical one. Corrective RAG does not change
  retrieval metrics because most questions grade as "correct" and never trigger a rewrite; it matters for lay
  wording and refusals.
- The reranker is switched by `USE_RERANKER` in `src/config.py` and falls back silently if PyTorch is missing.

### Agent (`python -m eval.run_eval agent [--questions eval/questions_set2.jsonl]`)

qwen2.5:3b (the default), final code:

| Metric | Set 1, 40 drugs | Set 2, 40 drugs | **Set 1, 298 drugs** | **Set 2, 298 drugs** |
|---|---|---|---|---|
| Tool routing vs. gold | 0.95 | 1.00 | 0.95 | 1.00 |
| Answers with citations | 1.00 | 0.94 | 0.98 | 1.00 |
| All citations valid | 1.00 | 0.94 | 0.98 | 1.00 |
| Answers citing a gold source | 0.95 | 0.88 | 0.91 | 0.94 |
| Out-of-scope refusal | 7/7 | 3/3 | 7/7 | 3/3 |
| Median / p95 latency (RTX 3050 laptop GPU) | 4.3 s / 13.8 s | 5.0 s / 22.2 s | 4.5 s / 11.8 s | 8.2 s / 27.4 s |
| Runs with flagged uncited statements | 11 / 50 | 5 / 19 | 13 / 50 | 3 / 19 |

(The 298-drug agent numbers were measured one small tool-argument fix before the final commit, "tegre tol" below;
retrieval and the adversarial set were re-run after it.) At 298 drugs the agent lost about one question on set 1
(one more answer without a valid citation, gold-source rate 0.95 to 0.91) and set 2 held; set 2 got slower
(median 5.0 s to 8.2 s), which I did not investigate.

Set 2's *first* run, before any change, scored routing 1.00, citations 0.94, gold source 0.88 and 3/3 refusals, so
the headline numbers did not move after the fixes, but the questions that failed did. That run exposed three bugs
that are fixed and unit-tested: a one-word query ("cetirizine") was refused because short queries score low on
cosine (now a query that names a covered drug counts as relevant); the reranker buried drugs whose label never
names them (above); and one question made the model generate about 44,000 tokens over 10 minutes (every LLM call
now has a 700-token cap). A later run also showed the query rewriter turning "symptoms of the flu" into
"acetaminophen flu symptoms" and answering from that label; a rewrite may no longer add a drug the question did not
name or imply. Remaining miss on set 2: "what side effects can escitalopram have?" is answered without citations
even after the repair pass (flagged as unverified in the output).

The "uncited statement" flag is a heuristic (a citation at the start or end of a paragraph covers it); it fires in
roughly a quarter of runs, still catches benign conclusion sentences, and should be read as "check this", not "this
is wrong". Earlier versions of this README reported different numbers on smaller question sets; each step was
re-measured, and the LLM is not perfectly repeatable between runs.

#### Scaling to 298 drugs, and an adversarial question set

`python -m eval.adversarial` runs 31 questions written to break the assistant, each with a pass/fail rule checked
by code (and every answer is printed so a person can read it too): misspelled drugs, brand names (including
Novartis products: Entresto, Gleevec, Tegretol, Diovan, Cosentyx), a question naming one indexed and one
non-indexed drug, false premises ("why is aspirin safe for children with a viral infection?"), dose numbers, prompt
injection, drugs that are not indexed, questions that name no drug, an emergency, a Spanish question.

| | 40-drug index, first run (25 questions) | 298-drug index, first run (31) | 298-drug index, final code (31) |
|---|---|---|---|
| Passed | 20 / 25 | 22 / 31 | **30 / 31** |

The questions changed between columns (6 added; three out-of-scope drugs swapped for ones that were still absent
once the index grew), so read the columns as "what the same method found at each stage", not as a clean
before/after. What it found, all fixed and unit-tested:

- **Brand names were not understood.** "Zoloft side effects" returned zoledronic acid and "Entresto" returned
  enoxaparin, with invented claims (a heart-failure drug "indicated for deep vein thrombosis"). Two causes: my first
  brand-name collection rejected salt names ("sertraline hydrochloride"), and, more important, nothing in the
  index contains the word "Zoloft", so the drug's own chunks never entered the candidate pool and the drug boost
  (which only reorders candidates) could not help. Now brand names (1,657 in total: collected from the labels, plus a
  curated list for brands openFDA does not carry, such as Zoloft and Coumadin) are expanded to the generic name
  before searching.
- **A drug not in the index, asked together with one that is** ("can I take lithium with ibuprofen?"): the
  coverage check lets it through because ibuprofen is covered, and one 40-drug answer claimed an interaction
  with a drug that had no source. The answer now starts with a fixed note naming the uncovered drug.
- **Answers with a hole in them**: "the usual starting dose of lisinopril is [1]." (the citation was written, the fact
  was not). Detected per sentence; one retry, otherwise the agent says it could not state an answer.
- **A question that names no drug** ("drug interactions") was answered from whichever label scored highest; now it
  asks for a specific drug.
- **Misspellings** ("metformine") were fixed inside the tools but not inside retrieval; the retriever now matches
  close spellings of one drug (checked against 45 ordinary medical words: no false matches).
- **A 3B-model glitch**: it once wrote the tool argument "tegre tol" for Tegretol; names are now also tried with
  spaces removed.
- **Rules I changed after seeing results, which you should discount accordingly**: the judge now accepts a PubMed
  abstract that mentions the drug (it first compared abstract titles to drug names), and accepts "hypotensive" for
  "hypotension". It also fails an answer that cites an unrelated drug's label, which is stricter than the first
  version.

The one remaining failure: "what is the maximum daily dose of acetaminophen?" is answered with "one packet every
6 hours", from an OTC label, and never states a daily maximum. I did not relax the rule to hide it.

Also from scaling: adalimumab, semaglutide and tirzepatide now appear in a few PubMed titles, so they are only
partly out of scope; set 1's adalimumab question is still refused. Data fix found while scaling: the `.gitignore`
rule `data/` also matched `src/data/`, so the two download scripts had never been committed and a fresh clone could
not run the documented commands; fixed.

#### Does a bigger model help? qwen2.5:3b vs qwen2.5:7b (same code, same questions)

| | 3B set 1 | 7B set 1 | 3B set 2 | 7B set 2 |
|---|---|---|---|---|
| Tool routing | 0.95 | 0.93 | 1.00 | 0.94 |
| Citations / all valid | 1.00 / 1.00 | 1.00 / 1.00 | 0.94 / 0.94 | 1.00 / 1.00 |
| Cites a gold source | 0.95 | 0.91 | 0.88 | 0.88 |
| Refusals | 7/7 | 7/7 | 3/3 | 3/3 |
| Median / p95 latency | 4.3 s / 13.8 s | 11.7 s / 33.5 s | 5.0 s / 22.2 s | 13.4 s / 28.6 s |
| Runs with flagged uncited statements | 11 / 50 | 20 / 50 | 5 / 19 | 10 / 19 |

Not adopted. The 7B model fixed the one missing citation on set 2 but was no better on routing or gold sources
(one question lower on set 1), about 2.7x slower (the 5.2 GB model does not fit a 6 GB GPU: 18% of it ran on the
CPU) and flagged about twice as many uncited statements (I did not investigate why). With a larger GPU or a hosted
model the answer might differ; switch with `LLM_MODEL=qwen2.5:7b`.

### Guardrails added after manual testing

Running varied questions by hand (not just the eval set) exposed failures the metrics had missed, so the
agent now has these safeguards, each with unit tests:

- **No sources, no answer:** if nothing was retrieved the answer is "not found", never the model's memory.
- **Brand names and misspellings in retrieval:** a brand named in the query is expanded to its generic name before
  searching (dense, BM25 and reranker), and a close spelling of a drug name is treated as that drug.
- **Partly covered questions:** if a question names an indexed drug and a non-indexed one, the answer starts with a
  note that the non-indexed drug has no source.
- **Empty-answer retry:** an answer with a citation but no fact in a sentence ("... dose is [1].") gets one retry,
  then an honest "could not state an answer".
- **No drug named:** very short, purely generic questions ("drug interactions") ask for a specific drug.
- **Output cap:** every LLM call is limited to 700 new tokens; an uncapped small model once looped for 44,000
  tokens (10 minutes) on a single question.
- **Rewrite guard:** the corrective-RAG query rewriter may improve wording but may not add a drug the question did
  not name or imply (via brand names, lay names or drug classes); otherwise the rewrite is discarded and the
  question is refused.
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
eval/                    question sets (questions.jsonl, questions_set2.jsonl, adversarial.jsonl), run_eval.py,
                         adversarial.py, saved adversarial results (40-drug baseline and 298-drug final)
```

## What would change at production scale

This is a local portfolio project (17,693 chunks, 298 drugs, one laptop GPU). These are the gaps I would
close before anything like it served real users; none of them is built here.

- **Scale:** FAISS in memory is fine for tens of thousands of chunks (17,693 here; embedding them took about 7 minutes). The 40-to-298-drug step cost the plain retrievers a lot and the final configuration little; a real catalogue would be 10-100x larger again. For millions, use an approximate index (IVF/HNSW)
  or a vector database, and move BM25 to a real search engine instead of my in-process implementation.
- **Freshness:** labels and abstracts are downloaded once. A real system needs scheduled re-ingestion, versioned
  indexes, and answers that show the label's effective date, since labels change when new safety data appears.
- **Evaluation validity:** my question sets are small and written by me. I would use questions written by
  pharmacists or clinicians, a larger set so one question does not move a metric by 0.02-0.06, and regression
  runs on every change.
- **Answer faithfulness:** citation checks confirm that a cited source exists, not that it supports the
  sentence. A production system would add a claim-level entailment check and human review of a sample.
- **Model:** a 3B local model phrases things loosely. A larger or hosted model (or a larger GPU) might change the
  3B-vs-7B result measured above; that would need re-measuring, along with cost and data-privacy review.
- **Safety and governance:** this is not a medical device. Clinical use would need regulatory review, audit
  logging, access control, and a clear path to escalate to a human.
- **Operations:** the API has no authentication, rate limiting or monitoring beyond local trace files; the Docker
  image has not been built.

## Roadmap

- [x] Day 1: data download, chunking, embeddings, FAISS retrieval
- [x] Day 2: answer generation with citations (local LLM)
- [x] Day 3: LangGraph ReAct agent with document-search and drug-label tools
- [x] Day 4: evaluation harness (Recall@k, MRR, tool routing, citation validity, refusals)
- [x] Day 5: hybrid search (BM25 + dense, RRF), drug-aware boost and cross-encoder rerank, measured
- [x] Days 6–7: embedding fine-tuning experiment, measured (did not beat nomic-embed-text; see results)
- [x] Day 8 (part): corrective RAG (grade, rewrite, refuse), Streamlit UI
- [x] Day 8 (rest): run tracing (local JSON + optional LangFuse export, tested against a stand-in server)
- [x] Scaled to 298 drugs and 17,693 chunks, with a 31-question adversarial set (30/31 on the final code)
- [ ] Not done: build and run the Docker image (needs admin + WSL2, unavailable here); validate on a larger, independently written question set

## Disclaimer

This is a learning project, not medical advice.
