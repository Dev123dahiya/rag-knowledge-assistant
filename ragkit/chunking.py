from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    id: str          # "<source>#<n>"
    source: str      # file name
    heading: str     # "Doc title > Section", kept with the text so the chunk stands alone
    text: str

    @property
    def searchable(self) -> str:
        return f"{self.heading}\n{self.text}"


def chunk_markdown(source: str, markdown: str, max_words: int = 120, overlap_sentences: int = 1) -> list[Chunk]:
    """Split a Markdown document along its headings, then by size.

    Splitting on headings keeps each chunk about one topic. The heading path is
    stored with every chunk: a paragraph saying "it is capped at 7,000 rupees"
    is useless to a retriever unless it also knows the section is about hotels.
    Long sections are split on sentence boundaries with a small overlap so a
    fact that straddles the cut is still found whole in one chunk.
    """
    title, section = "", ""
    sections: list[tuple[str, list[str]]] = []
    for line in markdown.splitlines():
        match = re.match(r"^(#{1,6})\s+(.*)", line)
        if match:
            if len(match.group(1)) == 1:
                title, section = match.group(2).strip(), ""
            else:
                section = match.group(2).strip()
            sections.append((" > ".join(p for p in (title, section) if p), []))
        elif line.strip():
            if not sections:
                sections.append(("", []))
            sections[-1][1].append(line.strip())

    chunks: list[Chunk] = []
    for heading, lines in sections:
        if not lines:
            continue
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", " ".join(lines)) if s]
        current: list[str] = []
        for sentence in sentences:
            if current and sum(len(s.split()) for s in current) + len(sentence.split()) > max_words:
                chunks.append(Chunk(f"{source}#{len(chunks)}", source, heading, " ".join(current)))
                current = current[-overlap_sentences:] if overlap_sentences else []
            current.append(sentence)
        if current:
            chunks.append(Chunk(f"{source}#{len(chunks)}", source, heading, " ".join(current)))
    return chunks
