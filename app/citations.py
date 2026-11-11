"""Citation statistics, computed locally from the whole document (no LLM).

"Which paper is referenced the most?" is a counting question over every page: retrieval only
returns a few passages, so no model can answer it from search results. Here we count every
in-text citation and match it to the reference list, for both common styles:

  numeric      "... as in [13] and [2, 5-7]"            -> reference [13], [2], [5], [6], [7]
  author-year  "Edge et al. (2024)", "(Guo & Xia, 2024)" -> the entry whose first author is
                                                           Edge/Guo and whose year is 2024
A citation only counts if it matches an entry of the reference list, so stray numbers or
years in the text cannot inflate the counts.
"""
from __future__ import annotations

import re
from collections import Counter

from .parser import ParsedDoc

REF_HEADING = re.compile(r"^\s*(\d+\.?\s*)?(references|bibliography|works cited|參考文獻|参考文献)\b", re.I)
NUM_ENTRY = re.compile(r"(?m)^\s*\[(\d{1,3})\]\s*")
NUM_CITE = re.compile(r"\[(\d{1,3}(?:\s*[,;–\-]\s*\d{1,3})*)\]")
YEAR = re.compile(r"\b(19[5-9]\d|20[0-4]\d)[a-z]?\b")
SURNAME = r"[A-Z][A-Za-z'À-ſ\-]+"
AY_CITE = re.compile(rf"({SURNAME})(?:\s+et\s+al\.?|\s+(?:and|&)\s+{SURNAME})?,?\s*\(?\s*((?:19|20)\d\d)[a-z]?")


def _split_sections(doc: ParsedDoc) -> tuple[list[str], list[str]]:
    """(body texts, reference-list texts) in reading order."""
    body, refs, in_refs = [], [], False
    for b in doc.blocks:
        if b.kind == "heading":
            in_refs = bool(REF_HEADING.match(b.text))
            continue
        (refs if in_refs else body).append(b.text)
    return body, refs


def _expand(group: str) -> list[int]:
    out = []
    for part in re.split(r"\s*[,;]\s*", group):
        if re.fullmatch(r"\d+\s*[–\-]\s*\d+", part):
            a, b = (int(x) for x in re.split(r"\s*[–\-]\s*", part))
            if 0 < b - a < 30:
                out.extend(range(a, b + 1))
                continue
        if part.strip().isdigit():
            out.append(int(part))
    return out


def _first_author_surname(entry: str) -> str | None:
    head = re.split(r",| and | & |\.\s", entry.strip(), maxsplit=1)[0].strip()
    words = [w for w in head.split() if w[:1].isupper()]
    if not words:
        return None
    # "Lastname, F." puts the surname first (one word before the comma); "First Last," puts it last
    return words[0] if len(head.split()) == 1 else words[-1]


def citation_stats(doc: ParsedDoc, top: int = 10) -> dict:
    body, refs = _split_sections(doc)
    if not refs:
        return {"style": "none", "n_references": 0, "n_citations": 0, "top": []}
    ref_text = "\n".join(refs)
    body_text = "\n".join(body)

    numbered = {int(m.group(1)): None for m in NUM_ENTRY.finditer(ref_text)}
    if len(numbered) >= 3:  # numeric style
        parts = NUM_ENTRY.split(ref_text)  # ['', '1', 'entry', '2', 'entry', ...]
        entries = {int(parts[i]): " ".join(parts[i + 1].split()) for i in range(1, len(parts) - 1, 2)}
        counts = Counter(n for g in NUM_CITE.findall(body_text) for n in _expand(g) if n in entries)
        top_items = [{"key": f"[{n}]", "count": c, "entry": entries[n][:180]} for n, c in counts.most_common(top)]
        return {"style": "numeric", "n_references": len(entries), "n_citations": sum(counts.values()), "top": top_items}

    # author-year style. Reference blocks often merge several entries, so instead of splitting the
    # list into entries, count in-text "Surname et al. (Year)" citations and confirm each one by
    # finding that surname followed by that year in the reference list.
    counts: Counter = Counter()
    found: dict[tuple[str, str], int] = {}
    for surname, year in AY_CITE.findall(body_text):
        if surname.lower() in _NOT_NAMES:
            continue
        key = (surname, year)
        if key not in found:
            found[key] = -1
            for m in re.finditer(rf"\b{re.escape(surname)}\b", ref_text):
                y = YEAR.search(ref_text, m.end(), m.end() + 400)
                if y and y.group(1) == year:  # the first year after the name is this entry's year
                    # entry start: the ". " that ends the previous entry, if it is just before the name
                    lo = max(0, m.start() - 40)
                    ends = [e.end() for e in re.finditer(r"\.\s+", ref_text[lo:m.start()])]
                    found[key] = lo + ends[-1] if ends else (0 if lo == 0 else m.start())
                    break
        if found[key] >= 0:
            counts[key] += 1
    top_items = [{"key": f"{sn} et al. ({yr})", "count": c,
                  "entry": " ".join(ref_text[found[(sn, yr)]:found[(sn, yr)] + 180].split())}
                 for (sn, yr), c in counts.most_common(top)]
    return {"style": "author-year", "n_references": len(counts), "n_citations": sum(counts.values()), "top": top_items}


_NOT_NAMES = {"table", "figure", "fig", "section", "appendix", "in", "the", "since", "from", "until", "by", "of",
              "and", "or", "as", "at", "on", "for", "to", "with", "january", "february", "march", "april", "may",
              "june", "july", "august", "september", "october", "november", "december", "version", "page", "step"}


def format_stats(stats: dict) -> str:
    if not stats.get("top"):
        return ""
    lines = [f"Citation counts computed by code from the whole document ({stats['style']} style): "
             f"{stats['n_citations']} in-text citations matched to {stats['n_references']} "
             f"{'reference-list entries' if stats['style'] == 'numeric' else 'distinct cited references'}. "
             "Most cited, highest first:"]
    lines += [f"{i}. {t['key']} cited {t['count']} times: {t['entry']}" for i, t in enumerate(stats["top"], 1)]
    return "\n".join(lines)
