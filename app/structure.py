"""Deterministic helpers for questions that point at the document's structure or ask to count.

1. Structural references: "explain Table 2", "what is in Figure 3?", "summarize Section 4",
   "what is on page 8?", "what is reference [12]?", "the appendix". Search might find these;
   a direct lookup by ID always does, so the matching chunks are included before ranked results.
2. Term counts: "how many times is GraphRAG mentioned, and where?" is counted by code over the
   document's page text (not over chunks, which overlap and would double-count).
"""
from __future__ import annotations

import re
from collections import Counter

from .chunker import Chunk

TABLE_RE = re.compile(r"\btables?\s*(\d{1,3})\b", re.I)
FIGURE_RE = re.compile(r"\b(?:figures?|figs?\.?)\s*(\d{1,3})\b", re.I)
SECTION_RE = re.compile(r"\b(?:section|sec\.?|chapter)\s*(\d{1,2}(?:\.\d{1,2})*)\b", re.I)
PAGE_RE = re.compile(r"\b(?:page|p\.)\s*(\d{1,4})\b", re.I)
APPENDIX_RE = re.compile(r"\bappendix\b", re.I)
REF_NUM_RE = re.compile(r"\b(?:reference|ref\.?|citation)\s*\[?(\d{1,3})\]?|^\s*\[(\d{1,3})\]|\s\[(\d{1,3})\]\s*\?", re.I)
# Chinese forms: 表2, 圖3 / 图3, 第4節 / 第4节 / 第4章, 第8頁 / 第8页
ZH_TABLE, ZH_FIG = re.compile(r"表\s*(\d{1,3})"), re.compile(r"[圖图]\s*(\d{1,3})")
ZH_SECTION, ZH_PAGE = re.compile(r"第\s*(\d{1,2})\s*[節节章]"), re.compile(r"第\s*(\d{1,4})\s*[頁页]")


def find_refs(question: str) -> list[tuple[str, str]]:
    refs: list[tuple[str, str]] = []
    for kind, pats in (("table", (TABLE_RE, ZH_TABLE)), ("figure", (FIGURE_RE, ZH_FIG)),
                       ("section", (SECTION_RE, ZH_SECTION)), ("page", (PAGE_RE, ZH_PAGE))):
        for pat in pats:
            refs += [(kind, m.group(1)) for m in pat.finditer(question)]
    for m in REF_NUM_RE.finditer(question):
        refs.append(("reference", next(g for g in m.groups() if g)))
    if APPENDIX_RE.search(question) or "附錄" in question or "附录" in question:
        refs.append(("appendix", ""))
    return list(dict.fromkeys(refs))


def resolve(chunks: list[Chunk], refs: list[tuple[str, str]]) -> tuple[list[int], list[str]]:
    """Chunk ids that answer each structural reference (in priority order) + human-readable notes."""
    ids: list[int] = []
    notes: list[str] = []

    def add(found: list[Chunk], label: str) -> None:
        new = [c.id for c in found if c.id not in ids]
        ids.extend(new)
        notes.append(f"{label}: {'found on ' + ', '.join(sorted({c.pages for c in found})) if found else 'not found in the document'}")

    for kind, ident in refs:
        if kind in ("table", "figure"):
            word = r"(?:Table)" if kind == "table" else r"(?:Figure|Fig\.)"
            caption = re.compile(rf"(^|\n){word}\s*{ident}\s*[:.]", re.I)
            mention = re.compile(rf"\b{word}\s*{ident}\b", re.I)
            # the chunk holding the caption (and, for tables, all parts) first; then text that discusses it
            primary = [c for c in chunks if caption.search(c.text)]
            discuss = [c for c in chunks if c not in primary and mention.search(c.text)][:2]
            add(primary + discuss, f"{kind.capitalize()} {ident}")
        elif kind == "section":
            major = ident.split(".")[0]
            in_sec = [c for c in chunks if re.match(rf"\s*{major}(\.|\s|$)", c.section)]
            if "." in ident:  # subsection: chunks of that section that mention its number first
                sub = [c for c in in_sec if re.search(rf"(^|\s){re.escape(ident)}\s+[A-Z]", c.text)]
                in_sec = sub + [c for c in in_sec if c not in sub]
            add(in_sec, f"Section {ident}")
        elif kind == "page":
            n = int(ident)
            add([c for c in chunks if c.page_start <= n <= c.page_end], f"Page {ident}")
        elif kind == "appendix":
            add([c for c in chunks if re.search(r"appendix", c.section, re.I)], "Appendix")
        elif kind == "reference":
            entry = re.compile(rf"(^|\n)\s*\[{ident}\]\s")
            add([c for c in chunks if entry.search(c.text)], f"Reference [{ident}]")
    return ids, notes


def term_counts(page_texts: list[tuple[int, str]], terms: list[str], max_terms: int = 4) -> str:
    """Occurrences of each term (case-insensitive, whole words) with the pages they are on."""
    lines = []
    for term in [t.strip() for t in terms if t and t.strip()][:max_terms]:
        pat = re.compile(rf"(?<![\w-]){re.escape(term)}(?![\w-])", re.I)
        per_page = Counter()
        for page, text in page_texts:
            n = len(pat.findall(text))
            if n:
                per_page[page] += n
        total = sum(per_page.values())
        pages = ", ".join(f"p.{p} ({n})" for p, n in sorted(per_page.items()))
        lines.append(f'"{term}": {total} occurrences' + (f" on {len(per_page)} pages: {pages}" if total else ""))
    if not lines:
        return ""
    return "Term counts computed by code over the full text of every page (case-insensitive, whole words):\n" + "\n".join(lines)
