import os

import numpy as np
import pytest

from ragkit import (BM25Index, Chunk, DenseIndex, ExtractiveLLM, HashingEmbedder, HybridRetriever, RagPipeline,
                    build_prompt, check_citations, chunk_markdown, reciprocal_rank_fusion)
from ragkit.answer import REFUSAL
from ragkit.evaluate import load_corpus, run

CORPUS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "corpus")

DOC = """# Handbook

Intro line before any section.

## Hotels
Stays are capped at 7,000 rupees per night. Book through the travel desk.

## Meals
The allowance is 1,200 rupees per day.
"""


def test_chunks_follow_headings_and_keep_the_heading_path():
    chunks = chunk_markdown("h.md", DOC)
    assert [c.heading for c in chunks] == ["Handbook", "Handbook > Hotels", "Handbook > Meals"]
    assert chunks[1].id == "h.md#1" and "7,000 rupees" in chunks[1].text
    assert "Hotels" in chunks[1].searchable  # the heading is searchable even though the body never says "hotel"


def test_long_sections_split_on_sentences_with_overlap():
    body = " ".join(f"Sentence number {i} has exactly seven words." for i in range(40))
    chunks = chunk_markdown("long.md", f"# T\n## S\n{body}", max_words=50, overlap_sentences=1)
    assert len(chunks) > 3
    assert all(len(c.text.split()) <= 57 for c in chunks)      # limit plus one overlapped sentence
    assert all(c.text.endswith(".") for c in chunks)            # never cut mid-sentence
    for a, b in zip(chunks, chunks[1:]):
        assert a.text.split(". ")[-1].rstrip(".") in b.text     # last sentence repeats at the start of the next
    joined = " ".join(c.text for c in chunks)
    assert all(f"number {i} " in joined for i in range(40))     # nothing lost


def chunks_fixture():
    return [Chunk("a#0", "a", "Pets", "Cats sleep most of the day."),
            Chunk("b#0", "b", "Pets", "Dogs need a walk every day and enjoy playing fetch."),
            Chunk("c#0", "c", "Finance", "Invoices are reimbursed within thirty days.")]


def test_bm25_ranks_by_term_match_and_rewards_rare_terms():
    index = BM25Index(chunks_fixture())
    assert index.search("dogs walk")[0][0].id == "b#0"
    assert index.search("day")[0][0].id == "a#0"          # same term in both, shorter chunk wins
    assert index.search("quantum physics") == []
    ranked = [c.id for c, _ in index.search("invoices day")]
    assert ranked[0] == "c#0"                              # "invoices" is rarer than "day"


def test_hashing_embedder_is_deterministic_normalised_and_typo_tolerant():
    e = HashingEmbedder()
    v = e.embed(["reimbursement policy", "reimbursment polcy", "kubernetes cluster"])
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)
    assert np.array_equal(v, e.embed(["reimbursement policy", "reimbursment polcy", "kubernetes cluster"]))
    assert v[0] @ v[1] > 0.5 > v[0] @ v[2]


def test_dense_index_finds_a_misspelled_query_that_bm25_misses():
    chunks = chunks_fixture()
    assert BM25Index(chunks).search("reimbursment of invoises") == []
    assert DenseIndex(chunks, HashingEmbedder()).search("reimbursment of invoises", k=1)[0][0].id == "c#0"


def test_rrf_rewards_agreement_between_rankers():
    a, b, c = chunks_fixture()
    fused = reciprocal_rank_fusion([[a, b, c], [b, c, a]])
    assert [x.id for x, _ in fused] == ["b#0", "a#0", "c#0"]
    assert fused[0][1] == pytest.approx(1 / 62 + 1 / 61)


def test_prompt_numbers_and_fences_sources():
    prompt = build_prompt("How long?", chunks_fixture()[:2])
    assert '<source id="1" file="a" section="Pets">' in prompt and '<source id="2"' in prompt
    assert prompt.rstrip().endswith("Answer with citations:")


def test_citation_checker_flags_invented_sources():
    assert check_citations("Yes [1], and also [3].", 3) == ([1, 3], True)
    assert check_citations("Trust me [7].", 3) == ([], False)
    assert check_citations("No citation at all.", 3) == ([], True)


class ScriptedLLM:
    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def complete(self, system, prompt):
        self.calls.append((system, prompt))
        return self.reply


def test_pipeline_maps_citations_back_to_chunks_and_flags_ungrounded_answers():
    retriever = HybridRetriever(chunks_fixture())
    good = RagPipeline(retriever, ScriptedLLM("Dogs need a daily walk [1]."), k=2).ask("How often do dogs need a walk?")
    assert good.grounded and [c.id for c in good.citations] == ["b#0"]

    invented = RagPipeline(retriever, ScriptedLLM("Dogs can fly [9]."), k=2).ask("Can dogs fly?")
    assert not invented.grounded and invented.citations == []
    uncited = RagPipeline(retriever, ScriptedLLM("Dogs are great."), k=2).ask("Are dogs great?")
    assert not uncited.grounded

    llm = ScriptedLLM(REFUSAL)
    refused = RagPipeline(retriever, llm, k=2).ask("What about dogs?")
    assert refused.refused and "only the numbered sources" in llm.calls[0][0]


def test_extractive_baseline_answers_and_refuses():
    pipeline = RagPipeline(HybridRetriever(load_corpus(CORPUS)), ExtractiveLLM())
    answer = pipeline.ask("How long is the probation period?")
    assert "3 months" in answer.text and answer.citations[0].source == "10-onboarding.md" and answer.grounded
    assert pipeline.ask("What is the capital of France?").refused


def test_retrieval_quality_on_the_sample_corpus_does_not_regress():
    report = run(CORPUS)
    assert report["questions"] == 28
    hybrid = report["retrieval"]["Hybrid (RRF of both)"]
    assert hybrid["recall@3"] >= 0.95 and hybrid["mrr"] >= 0.90
    assert hybrid["mrr"] >= report["retrieval"]["BM25 (keywords)"]["mrr"]
