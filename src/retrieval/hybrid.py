"""BM25 (lexical) retriever and hybrid retriever (BM25 + dense, fused with RRF).

Why hybrid: dense embeddings capture meaning but can blur exact tokens such as drug
names ("warfarin"); BM25 matches those tokens exactly but misses paraphrases.
Reciprocal Rank Fusion (RRF) combines the two rankings without needing comparable
scores: score(d) = sum over rankers of 1 / (RRF_K + rank_in_that_ranker).

All retrievers expose search(query, k) -> list[Hit], like DenseRetriever.
"""
import math
import re
from collections import Counter

from src.config import TOP_K, USE_RERANKER
from src.retrieval.dense import DenseRetriever, Hit

RRF_K = 60
DRUG_BOOST = 0.01   # chosen by the retrieval eval (eval/run_eval.py retrieval)
STOPWORDS = set("a an and are as at be by can for from how i in is it of on or the to what which with do does "
                "take taken use used".split())
_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


class BM25Retriever:
    """Okapi BM25 over chunk text. Title and drug name are indexed too, so a query
    naming a drug matches that drug's chunks even if the section text says "the product"."""

    def __init__(self, chunks: list[dict], k1: float = 1.5, b: float = 0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        docs = [tokenize(f"{c.get('drug', '')} {c['title']} {c['section'].replace('_', ' ')} {c['text']}")
                for c in chunks]
        self.tfs = [Counter(d) for d in docs]
        self.lens = [len(d) for d in docs]
        self.avg_len = sum(self.lens) / len(docs)
        df = Counter(t for tf in self.tfs for t in tf)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int = TOP_K) -> list[Hit]:
        terms = [t for t in tokenize(query) if t in self.idf]
        scored = []
        for i, tf in enumerate(self.tfs):
            s = 0.0
            for t in terms:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg_len))
            if s > 0:
                scored.append((s, i))
        scored.sort(reverse=True)
        return [Hit(s, self.chunks[i]) for s, i in scored[:k]]


class HybridRetriever:
    def __init__(self, dense: DenseRetriever | None = None, pool: int = 30, drug_boost: float = 0.0):
        self.dense = dense or DenseRetriever()
        self.bm25 = BM25Retriever(self.dense.chunks)
        self.pool = pool   # how deep each ranker is read before fusing
        # Entity-aware rerank: bonus added to chunks of a drug named in the query.
        # 0.0 = plain RRF. A rank-1 hit in one ranker is worth 1/61 = 0.016, so 0.01 is a strong nudge.
        self.drug_boost = drug_boost
        self.drugs = sorted({c["drug"].lower() for c in self.dense.chunks if c.get("drug")})

    def _query_drugs(self, query: str) -> set[str]:
        q = query.lower()
        return {d for d in self.drugs if re.search(r"\b" + re.escape(d) + r"\b", q)}

    def search(self, query: str, k: int = TOP_K) -> list[Hit]:
        dense_hits = self.dense.search(query, self.pool)
        fused: dict[tuple, float] = {}
        by_id: dict[tuple, Hit] = {}
        for hits in (dense_hits, self.bm25.search(query, self.pool)):
            for rank, h in enumerate(hits, 1):
                cid = (h.chunk["chunk_id"], h.chunk.get("drug"))
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank)
                by_id.setdefault(cid, h)
        if self.drug_boost:
            named = self._query_drugs(query)
            for cid in fused:
                if cid[1] and cid[1].lower() in named:
                    fused[cid] += self.drug_boost
        dense_score = {(h.chunk["chunk_id"], h.chunk.get("drug")): h.score for h in dense_hits}
        top = sorted(fused, key=fused.get, reverse=True)[:k]
        # Keep the dense cosine as .score so the MIN_RELEVANCE refusal check still means the same thing;
        # chunks found only by BM25 get 0.0 and so never rescue an off-topic question.
        return [Hit(dense_score.get(cid, 0.0), by_id[cid].chunk) for cid in top]


def default_retriever():
    """The retriever used by the agent and CLI: hybrid BM25 + dense with a drug-aware boost, then (if
    USE_RERANKER and torch are available) a cross-encoder reorders the top candidates."""
    retriever = HybridRetriever(drug_boost=DRUG_BOOST)
    if USE_RERANKER:
        try:
            import sentence_transformers  # noqa: F401  (also pulls in torch)
            from src.retrieval.rerank import RerankedRetriever
            return RerankedRetriever(retriever)
        except ImportError:
            pass          # no torch installed: run without the cross-encoder
    return retriever
