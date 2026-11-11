# Medical Literature Assistant

An agentic RAG system that answers medical and drug questions from FDA drug labels and PubMed abstracts, with citations for every answer. Everything runs locally with [Ollama](https://ollama.com), so no API keys are needed.

> Status: **Day 1, retrieval pipeline.** Answer generation, the agent, the eval harness and the retrieval experiments come next (see roadmap).

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
   .venv\Scripts\activate        # Windows
   # source .venv/bin/activate   # macOS / Linux
   pip install -r requirements.txt
   ```

## Run (from the project root)

```bash
python -m src.data.fetch_openfda     # ~40 FDA drug labels
python -m src.data.fetch_pubmed      # ~200 PubMed abstracts
python -m src.ingest.chunk           # -> data/processed/chunks.jsonl
python -m src.ingest.build_index     # -> data/index/
python -m scripts.search "Can I take ibuprofen with warfarin?"
```

Try different chunk sizes: `python -m src.ingest.chunk --size 300 --overlap 60`, then rebuild the index.

## Project structure

```
src/
  config.py              all settings (models, chunk size, drug list)
  data/                  downloaders for openFDA and PubMed
  ingest/                chunking and index building
  retrieval/             embedder + dense retriever
scripts/search.py        command-line search
eval/                    evaluation set and metrics (coming next)
```

## Roadmap

- [x] Day 1: data download, chunking, embeddings, FAISS retrieval
- [ ] Day 2: answer generation with citations (local LLM)
- [ ] Day 3: LangGraph ReAct agent with document-search and drug-label tools
- [ ] Day 4: evaluation harness (Recall@k, MRR, faithfulness)
- [ ] Day 5: hybrid search (BM25 + dense) and cross-encoder reranking, measured
- [ ] Days 6–7: fine-tuned biomedical embedding model, measured
- [ ] Day 8: corrective RAG, LangFuse tracing, Streamlit UI

## Disclaimer

This is a learning project, not medical advice.
