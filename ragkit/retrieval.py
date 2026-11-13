from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Protocol

import numpy as np

from .chunking import Chunk

STOPWORDS = frozenset("a an and are as at be by can do does for from how i if in is it me my of on or our the to we what when where which who will with you your".split())


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


class BM25Index:
    """Okapi BM25: keyword search that rewards rare terms and discounts long documents."""

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks, self.k1, self.b = chunks, k1, b
        self._tf = [Counter(tokenize(c.searchable)) for c in chunks]
        self._len = [sum(tf.values()) for tf in self._tf]
        self._avg_len = sum(self._len) / max(len(chunks), 1)
        df = Counter(term for tf in self._tf for term in tf)
        n = len(chunks)
        self._idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        scores = []
        terms = tokenize(query)
        for i, tf in enumerate(self._tf):
            score = 0.0
            for t in terms:
                f = tf.get(t, 0)
                if f:
                    norm = f + self.k1 * (1 - self.b + self.b * self._len[i] / self._avg_len)
                    score += self._idf[t] * f * (self.k1 + 1) / norm
            if score > 0:
                scores.append((self.chunks[i], score))
        return sorted(scores, key=lambda x: -x[1])[:k]


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


class HashingEmbedder:
    """Deterministic local embedder: hashed character n-grams, L2-normalised.

    This is NOT a semantic model. It matches on spelling, so it catches
    "reimbursable" vs "reimburse" or a typo, but not "vacation" vs "leave".
    It exists so the pipeline runs offline and tests are reproducible. Swap in
    a sentence-transformer or an embeddings API through the Embedder protocol.
    """

    def __init__(self, dim: int = 1024, ngram: tuple[int, int] = (3, 5)) -> None:
        self.dim, self.ngram = dim, ngram

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in tokenize(text):
                padded = f"<{word}>"
                for n in range(self.ngram[0], self.ngram[1] + 1):
                    for i in range(len(padded) - n + 1):
                        h = int.from_bytes(hashlib.blake2b(padded[i:i + n].encode(), digest_size=8).digest(), "big")
                        out[row, h % self.dim] += 1.0 if (h >> 63) else -1.0  # signed hashing reduces collision bias
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-12)


class DenseIndex:
    """Cosine similarity over embeddings (exact search; fine for thousands of chunks)."""

    def __init__(self, chunks: list[Chunk], embedder: Embedder) -> None:
        self.chunks, self.embedder = chunks, embedder
        self._matrix = embedder.embed([c.searchable for c in chunks])

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        scores = self._matrix @ self.embedder.embed([query])[0]
        top = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top]


def reciprocal_rank_fusion(rankings: list[list[Chunk]], k: int = 60) -> list[tuple[Chunk, float]]:
    """Merge ranked lists by rank, not score: sum of 1 / (k + rank).

    BM25 scores and cosine similarities live on different scales, so adding
    them is meaningless. Ranks are comparable, which is why RRF needs no tuning.
    """
    scores: dict[str, float] = {}
    by_id: dict[str, Chunk] = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking, start=1):
            scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (k + rank)
            by_id[chunk.id] = chunk
    return sorted(((by_id[i], s) for i, s in scores.items()), key=lambda x: -x[1])


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], embedder: Embedder | None = None, candidates: int = 20) -> None:
        self.bm25 = BM25Index(chunks)
        self.dense = DenseIndex(chunks, embedder or HashingEmbedder())
        self.candidates = candidates

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        lexical = [c for c, _ in self.bm25.search(query, self.candidates)]
        semantic = [c for c, _ in self.dense.search(query, self.candidates)]
        return reciprocal_rank_fusion([lexical, semantic])[:k]
