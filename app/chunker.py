"""Section-aware chunking.

* Chunks never cross a section heading, so each chunk carries one section label.
* Paragraphs are packed up to `target` tokens; an oversized paragraph is split on sentences.
* Consecutive chunks of the same section share a short sentence overlap.
* Each table is its own chunk (split by rows, caption repeated, if it is too large).
"""
from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass

from .parser import Block, ParsedDoc
from .tokens import count_tokens

_SENT = re.compile(r"(?<=[.!?。！？])\s+(?=[A-Z0-9\"'(\[一-鿿])")


@dataclass
class Chunk:
    id: int
    text: str
    section: str
    page_start: int
    page_end: int
    kind: str  # "text" | "table"
    tokens: int

    @property
    def pages(self) -> str:
        return f"p. {self.page_start}" if self.page_start == self.page_end else f"pp. {self.page_start}-{self.page_end}"

    def search_text(self) -> str:
        return f"{self.section}\n{self.text}"

    def to_dict(self) -> dict:
        return asdict(self)


def split_sentences(text: str) -> list[str]:
    return [s for s in _SENT.split(text) if s.strip()]


def _split_long(text: str, max_tokens: int) -> list[str]:
    """Split one long paragraph into pieces of at most max_tokens (sentence boundaries)."""
    pieces, cur = [], ""
    for sent in split_sentences(text):
        while count_tokens(sent) > max_tokens:  # pathological: no sentence breaks
            words = sent.split(" ")
            half = max(1, len(words) // 2)
            head, sent = " ".join(words[:half]), " ".join(words[half:])
            if cur:
                pieces.append(cur)
                cur = ""
            pieces.append(head)
        candidate = f"{cur} {sent}".strip()
        if cur and count_tokens(candidate) > max_tokens:
            pieces.append(cur)
            cur = sent
        else:
            cur = candidate
    if cur:
        pieces.append(cur)
    return pieces


def _overlap_tail(text: str, max_tokens: int) -> str:
    tail: list[str] = []
    for sent in reversed(split_sentences(text)):
        if count_tokens(" ".join([sent, *tail])) > max_tokens:
            break
        tail.insert(0, sent)
    return " ".join(tail)


def chunk_document(doc: ParsedDoc, target: int = 350, max_tokens: int = 500,
                   overlap: int = 50, table_max: int = 1500) -> list[Chunk]:
    chunks: list[Chunk] = []
    section = "Front matter"
    buf: list[tuple[str, int]] = []  # (paragraph text, page)

    def emit(text: str, p0: int, p1: int, kind: str = "text") -> None:
        chunks.append(Chunk(id=len(chunks), text=text, section=section, page_start=p0,
                            page_end=p1, kind=kind, tokens=count_tokens(text)))

    def flush_text() -> None:
        nonlocal buf
        if not buf:
            return
        emit("\n".join(t for t, _ in buf), buf[0][1], buf[-1][1])
        buf = []

    def add_paragraph(text: str, page: int) -> None:
        nonlocal buf
        current = "\n".join(t for t, _ in buf)
        if buf and count_tokens(current + "\n" + text) > target:
            flush_text()
            tail = _overlap_tail(current, overlap) if overlap else ""
            if tail and count_tokens(tail + "\n" + text) <= max_tokens:
                buf = [(tail, page)]
        buf.append((text, page))

    for block in doc.blocks:
        if block.kind == "heading":
            flush_text()
            section = block.text
            continue
        if block.kind == "table":
            flush_text()
            _emit_table(block, table_max, emit)
            continue
        if count_tokens(block.text) > max_tokens:
            for piece in _split_long(block.text, target):
                add_paragraph(piece, block.page)
        else:
            add_paragraph(block.text, block.page)
    flush_text()
    return [c for c in chunks if c.text.strip()]


def _emit_table(block: Block, table_max: int, emit) -> None:
    caption, *rows = block.text.split("\n")
    if count_tokens(block.text) <= table_max:
        emit(block.text, block.page, block.page, "table")
        return
    # balanced parts (e.g. 16 rows -> 8 + 8) so row groups of a table are not cut unevenly
    n_parts = math.ceil(count_tokens(block.text) / table_max)
    while True:
        n_parts = min(n_parts, len(rows))
        base, extra = divmod(len(rows), n_parts)
        bounds = [i * base + min(i, extra) for i in range(n_parts + 1)]
        parts = [rows[bounds[i]:bounds[i + 1]] for i in range(n_parts)]
        texts = [("\n".join([caption if i == 0 else caption + " (continued)", *p])) for i, p in enumerate(parts)]
        if all(count_tokens(t) <= table_max for t in texts) or n_parts == len(rows):
            break
        n_parts += 1
    for t in texts:
        emit(t, block.page, block.page, "table")
