"""python -m ragkit eval            evaluate retrieval on the sample corpus
   python -m ragkit ask "question"  answer from the sample corpus (offline extractive baseline)"""
from __future__ import annotations

import os
import sys

from .answer import ExtractiveLLM, RagPipeline
from .evaluate import load_corpus, run
from .retrieval import HybridRetriever

CORPUS = os.environ.get("RAG_CORPUS", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "corpus"))


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else "eval"
    if command == "ask":
        pipeline = RagPipeline(HybridRetriever(load_corpus(CORPUS)), ExtractiveLLM())
        answer = pipeline.ask(" ".join(sys.argv[2:]))
        print(answer.text)
        for i, c in enumerate(answer.citations, 1):
            print(f"  source: {c.source} ({c.heading})")
        return
    report = run(CORPUS)
    print(f"{report['chunks']} chunks, {report['questions']} labelled questions\n")
    print("| Retriever | Recall@1 | Recall@3 | Recall@5 | MRR |")
    print("|---|---:|---:|---:|---:|")
    for name, m in report["retrieval"].items():
        print(f"| {name} | {m['recall@1']:.3f} | {m['recall@3']:.3f} | {m['recall@5']:.3f} | {m['mrr']:.3f} |")
    a = report["answers_extractive_baseline"]
    print(f"\nExtractive baseline answers: fact present {a['answer_contains_fact']:.3f}, "
          f"correct document cited {a['cites_correct_document']:.3f}, refused {a['refused']:.3f}")


if __name__ == "__main__":
    main()
