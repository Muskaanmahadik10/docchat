"""Local document parsing — no LLM involved.

PDF parsing uses PyMuPDF layout information:
  * reading order: blocks are ordered top-to-bottom, with left-then-right column order on
    two-column pages;
  * headings: lines whose font is clearly larger than the body font become section headings;
  * noise: rotated text (e.g. the arXiv side stamp), page numbers and tiny figure labels are dropped;
  * tables: regions found by `page.find_tables()` that are mostly numeric are rebuilt from word
    coordinates (rows by y, columns by x) and linearised as "row label: column header = value"
    lines, which an LLM reads far more reliably than a flattened grid. Their words are removed
    from the running text so numbers are not duplicated out of context.
"""
from __future__ import annotations

import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

NUM_RE = re.compile(r"^[<>≤≥~≈±+\-−]?\s*\(?\d[\d,]*(\.\d+)?\)?\s*(%|x|×)?$")
CAPTION_RE = re.compile(r"^(Table|Figure|Fig\.)\s*\d+\s*[:.]", re.I)


@dataclass
class Block:
    kind: str  # "heading" | "text" | "table"
    text: str
    page: int  # 1-based
    bbox: tuple[float, float, float, float] = (0, 0, 0, 0)


@dataclass
class ParsedDoc:
    title: str
    num_pages: int
    blocks: list[Block] = field(default_factory=list)
    meta: dict = field(default_factory=dict)  # PDF metadata (author, subject, dates, ...)


# ----------------------------------------------------------------------------- tables


def _is_num(text: str) -> bool:
    return bool(NUM_RE.match(text.strip()))


def _group_rows(words: list[tuple]) -> list[list[tuple]]:
    rows: list[dict] = []
    for w in sorted(words, key=lambda w: ((w[1] + w[3]) / 2, w[0])):
        yc, h = (w[1] + w[3]) / 2, w[3] - w[1]
        if rows and abs(yc - rows[-1]["yc"]) <= max(2.0, 0.45 * h):
            rows[-1]["words"].append(w)
        else:
            rows.append({"yc": yc, "words": [w]})
    return [sorted(r["words"], key=lambda w: w[0]) for r in rows]


def _row_cells(row: list[tuple]) -> list[dict]:
    cells: list[dict] = []
    for w in row:
        h = w[3] - w[1]
        if cells and w[0] - cells[-1]["x1"] <= 0.6 * h:
            cells[-1]["text"] += " " + w[4]
            cells[-1]["x1"] = max(cells[-1]["x1"], w[2])
        else:
            cells.append({"x0": w[0], "x1": w[2], "text": w[4]})
    for c in cells:
        c["xc"] = (c["x0"] + c["x1"]) / 2
        c["num"] = _is_num(c["text"])
    return cells


def _cluster_1d(values: list[float], tol: float) -> list[float]:
    centers: list[list[float]] = []
    for v in sorted(values):
        if centers and v - statistics.mean(centers[-1]) <= tol:
            centers[-1].append(v)
        else:
            centers.append([v])
    return [statistics.mean(c) for c in centers]


def linearize_table(words: list[tuple]) -> tuple[list[str], float]:
    """Rebuild a table from word boxes. Returns (lines, numeric_ratio)."""
    rows = [_row_cells(r) for r in _group_rows(words)]
    cells = [c for r in rows for c in r]
    if not cells:
        return [], 0.0
    numeric_ratio = sum(c["num"] for c in cells) / len(cells)
    num_rows = [r for r in rows if sum(c["num"] for c in r) >= 2]
    if not num_rows:
        return [" | ".join(c["text"] for c in r) for r in rows], numeric_ratio

    anchors = _cluster_1d([c["xc"] for r in num_rows for c in r if c["num"]], tol=10)
    gap = min((b - a for a, b in zip(anchors, anchors[1:])), default=40)
    label_limit = anchors[0] - gap / 2

    def nearest(x: float) -> int:
        return min(range(len(anchors)), key=lambda i: abs(anchors[i] - x))

    top: dict[int, str] = {}
    sub: dict[int, str] = {}
    group = ""
    seen_data = False
    lines: list[str] = []
    for r in rows:
        value_cells = [c for c in r if c["xc"] >= label_limit]
        label = " ".join(c["text"] for c in r if c["xc"] < label_limit).strip()
        is_data = sum(c["num"] for c in value_cells) >= 1
        if not is_data:
            if not value_cells:  # a label-only row: treat as a row-group title
                group = label
                continue
            if len(value_cells) < 0.75 * len(anchors):  # spanning header (e.g. dataset names)
                for i, a in enumerate(anchors):
                    top[i] = min(value_cells, key=lambda c: abs(c["xc"] - a))["text"]
                sub.clear()
            else:
                if seen_data:  # a header row between data blocks starts a new group
                    sub.clear()
                for c in value_cells:  # stacked header rows above the data accumulate
                    i = nearest(c["xc"])
                    sub[i] = f"{sub[i]} {c['text']}" if sub.get(i) else c["text"]
            continue
        seen_data = True
        parts = []
        for c in value_cells:
            i = nearest(c["xc"])
            header = " / ".join(h for h in (top.get(i), sub.get(i)) if h) or f"col{i + 1}"
            parts.append(f"{header} = {c['text']}")
        row_label = " – ".join(x for x in (group, label) if x) or "(row)"
        lines.append(f"{row_label}: " + "; ".join(parts))
    return lines, numeric_ratio


# ----------------------------------------------------------------------------- PDF


def _body_font_size(doc: pymupdf.Document) -> float:
    sizes: Counter = Counter()
    for page in doc:
        for b in page.get_text("dict")["blocks"]:
            for line in b.get("lines", []):
                for s in line["spans"]:
                    sizes[round(s["size"])] += len(s["text"].strip())
    return float(sizes.most_common(1)[0][0]) if sizes else 10.0


def _inside(bbox, rect) -> bool:
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    return rect[0] - 1 <= cx <= rect[2] + 1 and rect[1] - 1 <= cy <= rect[3] + 1


def _order_blocks(blocks: list[dict], width: float) -> list[dict]:
    """Top-to-bottom; on genuine two-column pages, left column before right column."""
    mid = width / 2
    def col(b):
        x0, _, x1, _ = b["bbox"]
        if x1 <= mid + 10:
            return 0
        if x0 >= mid - 10:
            return 1
        return -1
    chars = Counter()
    for b in blocks:
        chars[col(b)] += b["chars"]
    total = sum(chars.values()) or 1
    two_column = chars[0] / total > 0.25 and chars[1] / total > 0.25
    if not two_column:
        return sorted(blocks, key=lambda b: (round(b["bbox"][1]), b["bbox"][0]))
    ordered, segment = [], []
    def flush():
        ordered.extend(sorted(segment, key=lambda b: (col(b), b["bbox"][1])))
        segment.clear()
    for b in sorted(blocks, key=lambda b: b["bbox"][1]):
        if col(b) == -1:
            flush()
            ordered.append(b)
        else:
            segment.append(b)
    flush()
    return ordered


def _line_text(spans: list[dict]) -> str:
    """Join spans, restoring spaces that PDFs encode as gaps (small caps, emails, labels)."""
    out = ""
    prev = None
    for s in spans:
        t = s["text"]
        if prev is not None and out and not out.endswith(" ") and not t.startswith(" "):
            if s["bbox"][0] - prev["bbox"][2] > 0.2 * s["size"]:
                out += " "
        out += t
        prev = s
    return out.strip()


HEADING_RE = re.compile(
    r"^((\d+(\.\d+)*|[A-Z](\.\d+)*)\.?\s+\S.*"  # "2 Related Work", "5.1 Effect of...", "A.1 Details"
    r"|(abstract|introduction|related work|background|method(s|ology)?|experiments?|results|discussion"
    r"|conclusions?|references|acknowledg(e)?ments?|appendix|limitations)\b.*)$", re.I)


def _is_bold(span: dict) -> bool:
    return bool(span["flags"] & 16) or any(k in span["font"] for k in ("Bold", "Medi", "Semibold", "Heavy", "Black"))


def _is_bold_heading(spans: list[dict], text: str, size: float, body: float) -> bool:
    """Body-size headings set in bold (common in ACL/IEEE templates): the whole line is bold,
    short, and numbered or a standard section name."""
    return (size >= body * 0.95 and len(text) <= 90 and all(_is_bold(s) for s in spans)
            and not text.endswith((".", ",", ":")) and bool(HEADING_RE.match(text)))


def _join_lines(lines: list[str]) -> str:
    out = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if out.endswith("-") and line[:1].islower():
            out = out[:-1] + line
        elif out:
            out += " " + line
        else:
            out = line
    return re.sub(r"\s+", " ", out).strip()


def _find_caption(text_blocks: list[dict], rect) -> dict | None:
    best, best_d = None, 120.0
    for b in text_blocks:
        if not CAPTION_RE.match(b["text"]):
            continue
        x0, y0, x1, y1 = b["bbox"]
        dy = max(rect[1] - y1, y0 - rect[3], 0)  # vertical gap (0 if overlapping)
        dx = max(rect[0] - x1, x0 - rect[2], 0)  # horizontal gap (side captions)
        d = dy + dx
        if d < best_d:
            best, best_d = b, d
    return best


def parse_pdf(path: str | Path) -> ParsedDoc:
    doc = pymupdf.open(path)
    body = _body_font_size(doc)
    meta = {k: v for k, v in (doc.metadata or {}).items() if isinstance(v, str) and v.strip()}
    title = meta.get("title", "")
    if re.search(r"\.(docx?|tex|pdf|pptx?)$|^untitled|^microsoft word", title, re.I) or len(title) < 4:
        title = ""
    out = ParsedDoc(title=title, num_pages=len(doc), meta=meta)

    for pno, page in enumerate(doc):
        width, height = page.rect.width, page.rect.height
        words = page.get_text("words")

        # 1) numeric tables → linearised table blocks
        tables: list[tuple[tuple, list[str]]] = []
        try:
            found = page.find_tables().tables
        except Exception:
            found = []
        for t in found:
            rect = tuple(t.bbox)
            twords = [w for w in words if _inside(w[:4], rect)]
            lines, ratio = linearize_table(twords)
            if ratio >= 0.3 and len(lines) >= 2:
                tables.append((rect, lines))

        # 2) text lines → blocks (headings separated out)
        raw_blocks: list[dict] = []
        for b in page.get_text("dict")["blocks"]:
            if b.get("type") != 0:
                continue
            cur: dict | None = None
            for line in b["lines"]:
                if abs(line["dir"][1]) > 0.1:  # rotated text (arXiv stamp, axis labels)
                    continue
                spans = [s for s in line["spans"] if s["text"].strip()]
                if not spans or any(_inside(line["bbox"], r) for r, _ in tables):
                    continue
                size = max(s["size"] for s in spans)
                if size < 0.65 * body:  # figure labels, sub/superscripts on their own
                    continue
                text = _line_text(spans)
                y0, y1 = line["bbox"][1], line["bbox"][3]
                if re.fullmatch(r"\d{1,4}", text) and (y0 > height * 0.9 or y1 < height * 0.08):
                    continue  # page number
                is_heading = (size >= body * 1.15 and len(text) < 120) or _is_bold_heading(spans, text, size, body)
                if is_heading:
                    if cur:
                        raw_blocks.append(cur)
                    raw_blocks.append({"kind": "heading", "lines": [text], "bbox": line["bbox"], "size": size})
                    cur = None
                    continue
                if cur is None:
                    cur = {"kind": "text", "lines": [], "bbox": list(line["bbox"])}
                cur["lines"].append(text)
                cur["bbox"] = [min(cur["bbox"][0], line["bbox"][0]), min(cur["bbox"][1], y0),
                               max(cur["bbox"][2], line["bbox"][2]), max(cur["bbox"][3], y1)]
            if cur:
                raw_blocks.append(cur)

        for rb in raw_blocks:
            rb["text"] = _join_lines(rb["lines"])
            rb["chars"] = len(rb["text"])
            rb["bbox"] = tuple(rb["bbox"])
        raw_blocks = [rb for rb in raw_blocks if rb["text"]]

        # merge "1" + "INTRODUCTION" heading pieces that sit on the same baseline
        merged: list[dict] = []
        for rb in raw_blocks:
            prev = merged[-1] if merged else None
            if (prev and prev["kind"] == rb["kind"] == "heading"
                    and abs(prev["bbox"][1] - rb["bbox"][1]) < 3):
                prev["text"] = f"{prev['text']} {rb['text']}"
                prev["bbox"] = (prev["bbox"][0], prev["bbox"][1], rb["bbox"][2], rb["bbox"][3])
                continue
            merged.append(rb)

        # 3) attach captions to tables
        text_blocks = [b for b in merged if b["kind"] == "text"]
        for rect, lines in tables:
            cap = _find_caption(text_blocks, rect)
            caption = cap["text"] if cap else "Table (no caption found)"
            if cap:
                cap["kind"] = "caption-used"
            body_text = caption + "\n" + "\n".join(lines)
            merged.append({"kind": "table", "text": body_text, "bbox": rect, "chars": len(body_text)})
        merged = [b for b in merged if b["kind"] != "caption-used"]

        # document title: the largest heading on page 1 (PDF metadata titles are often junk
        # like "Microsoft Word - draft.docx"); title lines are not section headings
        if pno == 0:
            heads = [b for b in merged if b["kind"] == "heading"]
            if heads:
                top_size = max(h["size"] for h in heads)
                if top_size >= body * 1.3 or not out.title:
                    for h in heads:
                        if h["size"] == top_size:
                            h["kind"] = "title"
                    out.title = " ".join(h["text"] for h in heads if h["kind"] == "title")

        for b in _order_blocks(merged, width):
            out.blocks.append(Block(kind=b["kind"], text=b["text"], page=pno + 1, bbox=b["bbox"]))
    doc.close()
    return out


# ----------------------------------------------------------------------------- plain text


def parse_text(path: str | Path) -> ParsedDoc:
    """.txt / .md: markdown '#' lines become headings, blank lines separate paragraphs."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    out = ParsedDoc(title=Path(path).stem, num_pages=1)
    para: list[str] = []
    def flush():
        if para:
            out.blocks.append(Block(kind="text", text=_join_lines(para), page=1))
            para.clear()
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            flush()
            out.blocks.append(Block(kind="heading", text=line.lstrip("# ").strip(), page=1))
        elif not line.strip():
            flush()
        else:
            para.append(line)
    flush()
    return out


SUPPORTED = {".pdf": parse_pdf, ".txt": parse_text, ".md": parse_text}


def parse_document(path: str | Path) -> ParsedDoc:
    suffix = Path(path).suffix.lower()
    if suffix not in SUPPORTED:
        raise ValueError(f"Unsupported file type {suffix!r}; use PDF, TXT or MD")
    return SUPPORTED[suffix](path)
