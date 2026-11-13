from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

from .chunking import Chunk
from .retrieval import tokenize

SYSTEM_PROMPT = (
    "You answer questions using only the numbered sources provided. "
    "Cite every claim with its source number in square brackets, like [2]. "
    "If the sources do not contain the answer, reply exactly: I don't know based on the provided documents."
)
REFUSAL = "I don't know based on the provided documents."


class LLM(Protocol):
    def complete(self, system: str, prompt: str) -> str: ...


@dataclass
class Answer:
    text: str
    citations: list[Chunk] = field(default_factory=list)
    grounded: bool = True           # False when the model cited a source that was not provided
    refused: bool = False


def build_prompt(question: str, chunks: list[Chunk]) -> str:
    """Number the sources and fence them, so the model can cite and cannot confuse them with instructions."""
    blocks = [f'<source id="{i}" file="{c.source}" section="{c.heading}">\n{c.text}\n</source>' for i, c in enumerate(chunks, 1)]
    return "Sources:\n" + "\n".join(blocks) + f"\n\nQuestion: {question}\nAnswer with citations:"


def check_citations(text: str, n_sources: int) -> tuple[list[int], bool]:
    """Return the cited source numbers and whether all of them exist.

    A model can invent "[7]" when only 4 sources were given. Checking this is
    cheap and catches a whole class of hallucinated references.
    """
    cited = sorted({int(m) for m in re.findall(r"\[(\d+)\]", text)})
    valid = all(1 <= c <= n_sources for c in cited)
    return [c for c in cited if 1 <= c <= n_sources], valid


class ExtractiveLLM:
    """Offline stand-in for a language model.

    It does not generate text. It returns the source sentence that shares the
    most words with the question, with its citation. That keeps the pipeline,
    tests and evaluation runnable with no API key, and gives a baseline that a
    real model should beat.
    """

    def __init__(self, min_overlap: float = 1.0) -> None:
        self.min_overlap = min_overlap

    def complete(self, system: str, prompt: str) -> str:
        question = prompt.rsplit("Question:", 1)[1].split("\n", 1)[0]
        q_terms = set(tokenize(question))
        best: tuple[float, int, str] = (0.0, 0, "")
        for match in re.finditer(r'<source id="(\d+)"[^>]*section="([^"]*)">\n(.*?)\n</source>', prompt, flags=re.S):
            number, section, body = int(match.group(1)), match.group(2), match.group(3)
            section_terms = set(tokenize(section))
            for sentence in re.split(r"(?<=[.!?])\s+", body):
                s_terms = set(tokenize(sentence))
                overlap = len(q_terms & s_terms) + 0.5 * len(q_terms & section_terms - s_terms)
                # Earlier sources were ranked higher by the retriever: break ties in their favour.
                if overlap > best[0]:
                    best = (overlap, number, sentence.strip())
        if best[0] < self.min_overlap:
            return REFUSAL
        return f"{best[2]} [{best[1]}]"


class RagPipeline:
    def __init__(self, retriever, llm: LLM, k: int = 4) -> None:
        self.retriever, self.llm, self.k = retriever, llm, k

    def ask(self, question: str) -> Answer:
        chunks = [c for c, _ in self.retriever.search(question, self.k)]
        if not chunks:
            return Answer(REFUSAL, refused=True)
        text = self.llm.complete(SYSTEM_PROMPT, build_prompt(question, chunks)).strip()
        if text == REFUSAL:
            return Answer(text, refused=True)
        cited, valid = check_citations(text, len(chunks))
        return Answer(text, [chunks[i - 1] for i in cited], grounded=valid and bool(cited))
