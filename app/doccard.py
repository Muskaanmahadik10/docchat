"""Document card: the facts any question may need regardless of retrieval.

Questions like "what is the title?", "who wrote it?", "how many tables are there?" or "what
sections does it have?" contain no words that appear in the relevant passage, so keyword or
embedding search alone can miss them. The card is built once from layout (no LLM) and placed in
every answer prompt (~500-900 tokens):

  title · PDF metadata · front matter (authors, affiliations, emails: page-1 text before the
  first section heading) · abstract (or the opening text) · outline (headings) · captions · counts
"""
from __future__ import annotations

import re

from .parser import CAPTION_RE, ParsedDoc
from .tokens import truncate_tokens

ABSTRACT_RE = re.compile(r"^\s*(abstract|summary|摘要)\b", re.I)


def build_doc_card(doc: ParsedDoc, max_tokens: int = 900) -> str:
    blocks = doc.blocks
    first_heading = next((i for i, b in enumerate(blocks) if b.kind == "heading"), len(blocks))

    front = [b.text for b in blocks[:first_heading] if b.kind == "text" and b.page <= 2]
    abstract = ""
    for i, b in enumerate(blocks):
        if b.kind == "heading" and ABSTRACT_RE.match(b.text):
            parts = []
            for x in blocks[i + 1:i + 8]:
                if x.kind == "heading":
                    break
                if x.kind == "text" and x.page <= 2:
                    parts.append(x.text)
            abstract = " ".join(parts)
            break
        if b.kind == "text" and ABSTRACT_RE.match(b.text):  # "Abstract—We propose ..." inline
            abstract = b.text
            break
    if not abstract:  # no abstract: use the opening text of the body
        abstract = " ".join(b.text for b in blocks[first_heading:first_heading + 6] if b.kind == "text")

    title_lc = (doc.title or "").lower()
    headings = list(dict.fromkeys(b.text for b in blocks if b.kind == "heading"
                                  and b.text.lower() not in title_lc))  # no repeated title lines
    captions = []
    for b in blocks:
        first_line = b.text.split("\n", 1)[0]
        if (b.kind == "table" or CAPTION_RE.match(first_line)) and first_line not in captions:
            captions.append(first_line)
    numbered = [m.group(0).lower() for c in captions if (m := re.match(r"(table|figure|fig\.)\s*\d+", c, re.I))]
    n_tables = len({n for n in numbered if n.startswith("table")})
    n_figures = len({n for n in numbered if n.startswith("fig")})

    meta_lines = [f"{k}: {v}" for k, v in doc.meta.items()
                  if k in ("author", "subject", "keywords", "creationDate", "modDate") and v]

    parts = [
        f"Title: {doc.title or '(not detected)'}",
        f"Pages: {doc.num_pages} · tables with captions: {n_tables} · figures with captions: {n_figures}",
    ]
    if meta_lines:
        parts.append("PDF metadata: " + "; ".join(meta_lines))
    if front:
        parts.append("Front matter (authors, affiliations, contact; first page, before the first heading):\n"
                     + truncate_tokens("\n".join(front), 220))
    if abstract:
        parts.append("Abstract / opening:\n" + truncate_tokens(abstract, 330))
    if headings:
        parts.append("Outline (section headings in order): " + truncate_tokens(" | ".join(headings), 160))
    if captions:
        parts.append("Table and figure captions:\n" + truncate_tokens("\n".join(c[:160] for c in captions), 200))
    return truncate_tokens("\n".join(parts), max_tokens)
