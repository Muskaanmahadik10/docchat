"""Table-cell attribution check for numbers in an answer.

`ungrounded_numbers` only asks "does this number appear somewhere in the evidence?". That passes an
answer that swaps two real values (e.g. reports the -Low score as the -High score). This module asks
the stronger question: "is this number attached to the right row and column?".

Tables are linearised by the parser as  `row label: col header / sub header = value; ...`,  so every
cell carries its labels (e.g. row "Overall", column "CS / -High"). For each number in the answer we
take the labels written next to it (the clause around it, or for a Markdown table the cell's column
header, the row's label cells and the heading above the table) and look for a source cell with that
value whose labels do not contradict them. A label contradicts when the answer names a different
value for the same slot of the same table (says "-High" where the cell is "-Low") and does not name
the cell's own value. Unnamed slots are not penalised, so the check flags clear misattributions, not
terse answers. It is deterministic and needs no LLM call.
"""
from __future__ import annotations

import re
from collections import defaultdict

_NUM = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?%?")
_CELL_LINE = re.compile(r"^(?P<row>[^:\n]{1,100}):\s(?P<rest>.*\s=\s.*)$")
_CAPTION = re.compile(r"^\s*(Table|Tab\.)\s*(\d+)", re.I)
_DASHES = "-−‑–"
_DASH_CLASS = r"\-−‑–"  # the same characters, for use inside [...]
_SENT_SPLIT = re.compile(r"(?<!\d)[.!?](?!\d)|\n")
_LIST_ITEM = re.compile(r"^\s*([-*•]|\d+[.)])\s+")
_NUM_CELL = re.compile(r"[\s*_]*[-+]?\d[\d,]*(?:\.\d+)?%?[\s*_]*")
_HEADING = re.compile(r"^\s*(#{1,6}\s|\*\*[^*]+\*\*:?\s*$)|:\s*(\*\*)?\s*$")
# A clause ends at these words. Before a number only the hard breaks count: "LightRAG against
# NaiveRAG on CS is 61.2%" keeps its subject; after a number, "61.2%, compared to 38.8% for X" does not.
_BOUNDARY_BEFORE = re.compile(r"[,;]|\b(?:while|whereas|but)\b", re.I)
_BOUNDARY_AFTER = re.compile(r"[,;]|\b(?:while|whereas|compared|versus|vs\.?|than|but|against)\b", re.I)


def _value(text: str) -> float | None:
    m = _NUM.search(text)
    if not m:
        return None
    try:
        return float(m.group().replace(",", "").rstrip("%").lstrip("+"))
    except ValueError:
        return None


def _usable_label(t: str) -> bool:
    t = t.strip()
    return bool(t) and t != "(row)" and not re.fullmatch(r"col\d+", t) and not re.fullmatch(r"[\d.,%\s]+", t)


def _term_regexes(term: str) -> list[re.Pattern]:
    """Patterns that must all match for a label to count as named in the answer."""
    t = term.strip()
    if t and t[0] in _DASHES:  # ablation-style labels: "-High" matches "-High", "−High", "High variant"
        core = re.escape(t[1:].strip())
        return [re.compile(rf"(?<!\w)[{_DASH_CLASS}]\s?{core}\b|\b{core}\s+variant\b", re.I)]
    out = []
    for i, m in enumerate(re.finditer(r"[A-Za-z][A-Za-z0-9]+(\.)?", t)):  # every word ("BERTBASE", "Mix")
        tok = m.group().rstrip(".")
        flags = 0 if len(tok) <= 4 else re.I  # short labels ("CS", "Mix") must keep their case
        # an abbreviation ("Seq.") also matches the full word ("Sequence"); other words allow a plural
        tail = r"\w*" if m.group(1) else r"(?:s|es|ed)?(?!\w)"
        # a label's first word must not follow a dash ("High" is not "-High"); later words may
        # ("Sequence" in "RAG-Sequence")
        head = rf"(?<![\w{_DASH_CLASS}])" if i == 0 else r"(?<!\w)"
        out.append(re.compile(rf"{head}{re.escape(tok)}{tail}", flags))
    return out


class _Cell:
    __slots__ = ("table", "value", "labels", "text")

    def __init__(self, table: str, value: float, labels: dict, text: str):
        self.table, self.value, self.labels, self.text = table, value, labels, text


def _parse_cells(sources: list[str]) -> tuple[list[_Cell], set[float]]:
    """All table cells in the evidence, plus the values that also occur in prose (not attributable)."""
    cells: list[_Cell] = []
    prose_values: set[float] = set()
    for k, src in enumerate(sources):
        lines = src.splitlines()
        cap = next((m for m in (_CAPTION.match(l) for l in lines[:2]) if m), None)
        table = f"table{cap.group(2)}" if cap else f"src{k}"  # "(continued)" chunks share their table
        for line in lines:
            m = _CELL_LINE.match(line.strip())
            if not m:
                if not _CAPTION.match(line):  # caption numbers ("Table 2") are not results
                    prose_values.update(v for v in (_value(n) for n in _NUM.findall(line)) if v is not None)
                continue
            row_parts = [p.strip() for p in m.group("row").split(" – ")]
            for part in m.group("rest").split("; "):
                if " = " not in part:
                    continue
                header, val = part.rsplit(" = ", 1)
                v = _value(val)
                if v is None:
                    continue
                labels = {("r", i): r for i, r in enumerate(row_parts) if _usable_label(r)}
                labels.update({("c", j): h.strip() for j, h in enumerate(header.split(" / ")) if _usable_label(h)})
                cells.append(_Cell(table, v, labels, f"{m.group('row')} · {header} = {val.strip()}"))
    return cells, prose_values


def _contexts(answer: str) -> list[tuple[float, str, str]]:
    """(value, label context, the number as written) for every number in the answer.

    Context = the labels written next to the number: for a Markdown table cell, its column header,
    the row's label cells, the nearest heading and the line introducing the table; for prose, the
    clause around the number (cut at commas and contrast words such as "compared to"), plus the
    heading a list item sits under."""
    text = re.sub(r"\[[^\]]*\]", " ", answer)  # citation brackets
    text = re.sub(r"(^|[.!?]\s+|\n)\s*(\*\*)?Overall(\*\*)?,", r"\1", text)  # discourse "Overall," is not a label
    out: list[tuple[float | None, str, str]] = []
    header: list[str] | None = None
    above: list[str] = []  # label cells of the previous table row (for grouped rows)
    heading = ""
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("|"):
            cells = [c.strip() for c in s.strip("|").split("|")]
            if header is None:
                header, above = cells, []
                continue
            if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                continue
            # grouped rows ("| -High | Agriculture |" then "|  | CS |"): a blank leading label cell
            # belongs to the group above it
            for j, c in enumerate(cells):
                if c or _NUM_CELL.fullmatch(c) or j >= len(above):
                    break
                cells[j] = above[j]
            above = [c if not _NUM_CELL.fullmatch(c) else "" for c in cells]
            labels = " ".join(c for c in cells if not _NUM_CELL.fullmatch(c))
            for j, c in enumerate(cells):
                if _NUM_CELL.fullmatch(c):
                    col = header[j] if j < len(header) else ""
                    # only the heading, not the prose above the table: an introduction such as
                    # "the full LightRAG model wins most" names labels of other rows
                    out.append((_value(c), f"{heading} | {col} | {labels}", c.strip("*_ ")))
            continue
        header = None
        if not s:
            continue
        is_item = bool(_LIST_ITEM.match(s))
        body = _LIST_ITEM.sub("", s) if is_item else s
        if _HEADING.search(body) and not _NUM.search(re.sub(r"^#+\s*\d+[.)]?", "", body)):
            heading = body  # a heading or an introducing line ("**CS dataset (Overall):**")
        for sent in _SENT_SPLIT.split(body):
            nums = list(_NUM.finditer(sent))
            for i, m in enumerate(nums):
                before = sent[nums[i - 1].end() if i else 0:m.start()]
                after = sent[m.end():nums[i + 1].start() if i + 1 < len(nums) else len(sent)]
                cuts = list(_BOUNDARY_BEFORE.finditer(before))
                before = before[cuts[-1].end():] if cuts else before
                cut = _BOUNDARY_AFTER.search(after)
                after = after[:cut.start()] if cut else after
                ctx = before + m.group() + after
                out.append((_value(m.group()), f"{heading} | {ctx}" if is_item else ctx, m.group()))
    return [(v, c, w) for v, c, w in out if v is not None and not (float(v).is_integer() and 0 <= v <= 12)]


def misattributed_numbers(answer: str, sources: list[str]) -> list[dict]:
    """Numbers whose labels in the answer contradict every source table cell holding that value."""
    cells, prose_values = _parse_cells(sources)
    if not cells:
        return []
    by_value: dict[float, list[_Cell]] = defaultdict(list)
    vocab: dict[tuple, set[str]] = defaultdict(set)  # (table, slot) -> all labels used in that slot
    for c in cells:
        by_value[c.value].append(c)
        for slot, lab in c.labels.items():
            vocab[(c.table, slot)].add(lab)
    patterns: dict[str, list[re.Pattern]] = {}

    def named(term: str, ctx: str) -> bool:
        ps = patterns.get(term)
        if ps is None:
            ps = patterns[term] = _term_regexes(term)
        return bool(ps) and all(p.search(ctx) for p in ps)

    def consistent(c: _Cell, ctx: str) -> bool:
        own = set(c.labels.values())
        for slot, lab in c.labels.items():
            others = [o for o in vocab[(c.table, slot)] if o not in own and named(o, ctx)]
            if others and not named(lab, ctx):
                return False
        return True

    flagged: list[dict] = []
    for v, ctx, written in _contexts(answer):
        cands = by_value.get(v)
        if not cands or v in prose_values:
            continue
        if not any(consistent(c, ctx) for c in cands) and written not in [f["value"] for f in flagged]:
            flagged.append({"value": written, "context": " ".join(ctx.split())[:160],
                            "cells": [c.text for c in cands[:3]]})
    return flagged
