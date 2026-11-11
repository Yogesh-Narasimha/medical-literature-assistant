"""Thin wrapper around Ollama embeddings.

nomic-embed-text is trained with task prefixes: documents get
"search_document: " and queries get "search_query: ". Using them noticeably
improves retrieval, and it's a good detail to mention in interviews.
"""
import numpy as np
import ollama

from src.config import EMBED_MODEL, OLLAMA_HOST

DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "

_client = ollama.Client(host=OLLAMA_HOST)


def _normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, 1e-12, None)


def embed(texts: list[str], model: str = EMBED_MODEL) -> np.ndarray:
    """Return L2-normalised float32 vectors, so inner product == cosine."""
    resp = _client.embed(model=model, input=texts)
    return _normalize(np.asarray(resp["embeddings"], dtype="float32"))


def embed_documents(texts: list[str]) -> np.ndarray:
    return embed([DOC_PREFIX + t for t in texts])


def embed_query(text: str) -> np.ndarray:
    return embed([QUERY_PREFIX + text])
