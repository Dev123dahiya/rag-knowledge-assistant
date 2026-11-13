"""Retrieval and answer metrics against a labelled question set."""
from __future__ import annotations

import json
import os

from .answer import ExtractiveLLM, RagPipeline
from .chunking import Chunk, chunk_markdown
from .retrieval import BM25Index, DenseIndex, HashingEmbedder, HybridRetriever


def load_corpus(directory: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    for name in sorted(os.listdir(directory)):
        if name.endswith(".md"):
            with open(os.path.join(directory, name), encoding="utf-8") as f:
                chunks.extend(chunk_markdown(name, f.read()))
    return chunks


def retrieval_metrics(retriever, questions: list[dict], ks=(1, 3, 5)) -> dict[str, float]:
    """Recall@k: was a chunk containing the answer in the top k? MRR: mean of 1/rank of the first such chunk."""
    hits = {k: 0 for k in ks}
    reciprocal = 0.0
    for q in questions:
        results = [c for c, _ in retriever.search(q["question"], max(ks))]
        rank = next((i for i, c in enumerate(results, 1)
                     if c.source == q["source"] and q["answer_contains"] in c.text), None)
        if rank:
            reciprocal += 1.0 / rank
            for k in ks:
                hits[k] += rank <= k
    n = len(questions)
    return {**{f"recall@{k}": hits[k] / n for k in ks}, "mrr": reciprocal / n}


def answer_metrics(pipeline: RagPipeline, questions: list[dict]) -> dict[str, float]:
    correct = cited_right = refused = 0
    for q in questions:
        answer = pipeline.ask(q["question"])
        refused += answer.refused
        correct += q["answer_contains"] in answer.text
        cited_right += any(c.source == q["source"] for c in answer.citations)
    n = len(questions)
    return {"answer_contains_fact": correct / n, "cites_correct_document": cited_right / n, "refused": refused / n}


def run(corpus_dir: str) -> dict:
    chunks = load_corpus(corpus_dir)
    with open(os.path.join(corpus_dir, "questions.json"), encoding="utf-8") as f:
        questions = json.load(f)
    embedder = HashingEmbedder()
    retrievers = {
        "BM25 (keywords)": BM25Index(chunks),
        "Dense (hashed n-grams)": DenseIndex(chunks, embedder),
        "Hybrid (RRF of both)": HybridRetriever(chunks, embedder),
    }
    return {
        "chunks": len(chunks), "questions": len(questions),
        "retrieval": {name: retrieval_metrics(r, questions) for name, r in retrievers.items()},
        "answers_extractive_baseline": answer_metrics(RagPipeline(retrievers["Hybrid (RRF of both)"], ExtractiveLLM()), questions),
    }
