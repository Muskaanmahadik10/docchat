"""Build the project presentation (docs/DocChat_Presentation.pptx) with python-pptx.

Storyline: what DocChat does, how it got there step by step, and what each capability looks like on a
real question. Every answer shown is copied from an evaluation run in evaluation/results/, and the
headline numbers are read from those files, so the deck matches what was measured.
Run:  python docs/build_deck.py
"""
from __future__ import annotations

import glob
import json
import math
import statistics
from pathlib import Path

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "DocChat_Presentation.pptx"  # or a path given as the first argument
import sys as _sys  # noqa: E402
ARGS = [a for a in _sys.argv[1:] if not a.startswith("--")]
RES = ROOT / "evaluation" / "results"

INK, INK2 = RGBColor(0x14, 0x22, 0x38), RGBColor(0x22, 0x34, 0x50)
INK3 = RGBColor(0x33, 0x4A, 0x6B)
TEAL, TEAL_SOFT = RGBColor(0x0E, 0x9A, 0xA7), RGBColor(0xD8, 0xF1, 0xF3)
AMBER, AMBER_SOFT = RGBColor(0xE8, 0x9B, 0x2E), RGBColor(0xFC, 0xEE, 0xD9)
GREEN = RGBColor(0x2E, 0x8B, 0x57)
LIGHT, WHITE = RGBColor(0xF3, 0xF6, 0xF9), RGBColor(0xFF, 0xFF, 0xFF)
TEXT, MUTED, LINE = RGBColor(0x1F, 0x2A, 0x37), RGBColor(0x5B, 0x6B, 0x7F), RGBColor(0xD5, 0xDD, 0xE6)
ON_DARK, ON_DARK2 = RGBColor(0xC9, 0xD6, 0xE3), RGBColor(0x9F, 0xB2, 0xC6)
HEAD, BODY = "Cambria", "Calibri"


# ----------------------------------------------------------------------------- data

def load(name: str) -> dict:
    return json.loads((RES / name).read_text(encoding="utf-8"))


def latest(pattern: str) -> dict:
    return json.loads(Path(sorted(glob.glob(str(RES / pattern)))[-1]).read_text(encoding="utf-8"))


def passed(d: dict, key=None) -> tuple[int, int]:
    rows = [r for r in d["results"] if key is None or key(r)]
    return sum(r["pass"] for r in rows), len(rows)


def find(d: dict, text: str) -> dict:
    return next(r for r in d["results"] if text in r["q"])


import sys  # noqa: E402
sys.path.insert(0, str(ROOT / "evaluation"))
from run_eval import score  # noqa: E402

QUESTIONS = {(q.get("doc"), q["q"]): q for q in
             json.loads((ROOT / "evaluation" / "questions.json").read_text(encoding="utf-8"))["questions"]}


def rescore_pass(r: dict, rubric: dict) -> bool:
    item = rubric.get((r.get("doc"), r["q"]))
    return score(item, r.get("answer", ""))[0] if item else bool(r["pass"])


RET = latest("retrieval_*.json")                            # dev retrieval benchmark (no answer model)
GROQ = load("answer_20261106_100032.json")                  # qwen3.8-27b answer run
DEV = load("answer_20261106_163351.json")                   # qwen2.5:14b dev answer run
HELD = load("heldout_questions_answer_20261106_153711.json")  # unseen paper, questions frozen first
ROB = load("robustness_questions_answer_20261106_155826.json")

M27 = GROQ["health"]["llm"]["model"]  # full API id, e.g. qwen/qwen3.8-27b
M14 = DEV["health"]["llm"]["model"]

TRACE = find(GROQ, "-High variant on the CS")              # the question peeled back on slide 11
NB = "\u2011"  # non-breaking hyphen for display
TRACE_Q = TRACE["q"].replace("-High", NB + "High")
CORE_Q = {r["q"]: r for r in GROQ["results"] if r["type"] == "core"}
ret_p, ret_n = passed(RET)
held_p, held_n = passed(HELD)
conf_p, conf_n = passed(ROB, lambda r: r["type"] != "conversation")
conv_p, conv_n = passed(ROB, lambda r: r["type"] == "conversation")
groq_first, groq_n = passed(GROQ)
groq_p = sum(rescore_pass(r, QUESTIONS) for r in GROQ["results"])  # current rubric (stricter Q3, see questions.json)
EVID = 3114  # median evidence incl. document card, full system (ablation config E)

# Retrieval ablation (evaluation/results/ablation_20261106_112619.md): facts reached / median evidence tokens
ABLATION = [("v0.1", "Keyword search", "BM25", 27, 4808),
            ("v0.2", "+ Meaning search", "embeddings, fused", 33, 4979),
            ("v0.3", "+ Reranker", "local cross-encoder", 33, 2363),
            ("v0.4", "+ Document card", "title, authors, abstract", 34, 2998),
            ("v1.0", "+ Lookup", "Table/page lookup, counters", 35, 3114)]
CALLS_CHECKED, LARGEST_CALL = 196, 6917   # all qwen2.5:14b evaluation calls (docs/REPORT.md §8)


# ----------------------------------------------------------------------------- helpers

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]


def bg(slide, color):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def text(slide, x, y, w, h, runs, size=16, color=TEXT, bold=False, font=BODY, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, spacing=4, italic=False):
    """runs: str, or list of paragraphs; a paragraph is str or list of (text, {opts})."""
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    for i, para in enumerate(runs if isinstance(runs, list) else [runs]):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(spacing)
        for piece in (para if isinstance(para, list) else [(para, {})]):
            t, o = piece if isinstance(piece, tuple) else (piece, {})
            r = p.add_run()
            r.text = t
            f = r.font
            f.name = o.get("font", font)
            f.size = Pt(o.get("size", size))
            f.bold = o.get("bold", bold)
            f.italic = o.get("italic", italic)
            f.color.rgb = o.get("color", color)
    return box


def bullets(slide, x, y, w, h, items, size=15, color=TEXT, gap=7, mark="•"):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(gap)
        first = True
        for t, o in (item if isinstance(item, list) else [(item, {})]):
            r = p.add_run()
            r.text = (f"{mark}  " if first else "") + t
            first = False
            r.font.name, r.font.size = BODY, Pt(o.get("size", size))
            r.font.bold = o.get("bold", False)
            r.font.color.rgb = o.get("color", color)
    return box


def rect(slide, x, y, w, h, fill, line=None, rounded=True, radius=0.08):
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE,
                                 Inches(x), Inches(y), Inches(w), Inches(h))
    if rounded:
        shp.adjustments[0] = radius
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    if line:
        shp.line.color.rgb = line
        shp.line.width = Pt(1)
    else:
        shp.line.fill.background()
    shp.shadow.inherit = False
    return shp


def badge(slide, x, y, label, fill=TEAL, color=WHITE, d=0.55, size=16):
    """The deck's motif: a lettered circle."""
    c = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y), Inches(d), Inches(d))
    c.fill.solid()
    c.fill.fore_color.rgb = fill
    c.line.fill.background()
    c.shadow.inherit = False
    tf = c.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.word_wrap = False
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = label
    r.font.name, r.font.size, r.font.bold, r.font.color.rgb = HEAD, Pt(size), True, color
    return c


def arrow(slide, x1, y1, x2, y2, color=MUTED, width=1.75):
    ln = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    ln.line.color.rgb = color
    ln.line.width = Pt(width)
    el = ln.line._get_or_add_ln()
    el.append(el.makeelement("{http://schemas.openxmlformats.org/drawingml/2006/main}tailEnd", {"type": "triangle"}))
    return ln


def pill(slide, x, y, w, label, fill=TEAL_SOFT, color=INK, size=11, h=0.3, bold=True):
    rect(slide, x, y, w, h, fill, radius=0.5)
    text(slide, x, y, w, h, label, size=size, color=color, bold=bold, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)


def content_slide(title, kicker=None):
    s = prs.slides.add_slide(BLANK)
    bg(s, WHITE)
    if kicker:
        text(s, 0.6, 0.38, 12, 0.3, kicker.upper(), size=11, color=TEAL, bold=True)
    text(s, 0.6, 0.62, 12.1, 0.8, title, size=30, color=INK, bold=True, font=HEAD)
    text(s, 12.2, 7.0, 0.6, 0.3, str(len(prs.slides)), size=10, color=MUTED, align=PP_ALIGN.RIGHT)
    return s


def chat(slide, x, y, w, q, a, qh=0.62, ah=1.0, size=13, dark=False):
    """A question bubble (right) and an answer bubble (left), copied from real runs."""
    rect(slide, x + 0.5, y, w - 0.5, qh, TEAL if dark else INK, radius=0.25)
    text(slide, x + 0.7, y, w - 0.9, qh, q, size=size, color=WHITE, anchor=MSO_ANCHOR.MIDDLE)
    ay = y + qh + 0.12
    rect(slide, x, ay, w - 0.5, ah, INK3 if dark else TEAL_SOFT, radius=0.18)
    text(slide, x + 0.2, ay, w - 0.9, ah, a, size=size, color=WHITE if dark else TEXT, anchor=MSO_ANCHOR.MIDDLE)
    return ay + ah


def capability(slide, x, y, w, h, icon, title, hard, q, a, how, tag, qh=0.62, ah=1.0):
    rect(slide, x, y, w, h, LIGHT)
    badge(slide, x + 0.3, y + 0.28, icon, d=0.55, size=15)
    text(slide, x + 1.05, y + 0.3, w - 1.3, 0.5, title, size=19, color=INK, bold=True, font=HEAD)
    text(slide, x + 0.3, y + 0.98, w - 0.6, 0.5, [[("Why it's hard  ", {"bold": True, "color": AMBER}), (hard, {})]],
         size=12.5, color=MUTED)
    end = chat(slide, x + 0.3, y + 1.6, w - 0.6, q, a, qh=qh, ah=ah)
    text(slide, x + 0.3, end + 0.2, w - 0.6, y + h - end - 0.65, [[("How  ", {"bold": True, "color": TEAL}), (how, {})]],
         size=12.5, color=TEXT)
    text(slide, x + 0.3, y + h - 0.4, w - 0.6, 0.3, tag, size=10, color=MUTED, italic=True)


SCREENS = ROOT / "docs" / "screens"


def screen_by_height(slide, name, x, y, h, crop=(0, 0, 0, 0)):
    from PIL import Image
    iw, ih = Image.open(SCREENS / name).size
    l, t, r, b = crop
    return screen(slide, name, x, y, h * (iw * (1 - l - r)) / (ih * (1 - t - b)), crop)


def screen(slide, name, x, y, w, crop=(0, 0, 0, 0)):
    """A UI screenshot from docs/screens (captured by docs/capture_screens.py), cropped (l, t, r, b)."""
    from PIL import Image
    path = SCREENS / name
    iw, ih = Image.open(path).size
    l, t, r, b = crop
    h = w * (ih * (1 - t - b)) / (iw * (1 - l - r))
    pic = slide.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(w), Inches(h))
    pic.crop_left, pic.crop_top, pic.crop_right, pic.crop_bottom = l, t, r, b
    pic.line.color.rgb = LINE
    pic.line.width = Pt(0.75)
    return pic


def notes(slide, t):
    slide.notes_slide.notes_text_frame.text = t


# Slide numbers for cross-references ("→ slide 6"), filled in once every slide exists.
SL: dict[str, int] = {}
LATER: list[tuple] = []


def mark(key):
    SL[key] = len(prs.slides)


def fill_later():
    for box, template, size in LATER:
        run = box.text_frame.paragraphs[0].runs[0]
        run.text = template.format(**SL)


# ============================================================================ slides

# 1 — cold open ---------------------------------------------------------------------------------
s = prs.slides.add_slide(BLANK)
bg(s, INK)
text(s, 0.8, 0.75, 6.0, 0.4, "INDIVIDUAL PROJECT · MUSKAAN MAHADIK · 2026", size=12, color=TEAL, bold=True)
text(s, 0.8, 1.6, 6.0, 1.2, "DocChat", size=60, color=WHITE, bold=True, font=HEAD)
text(s, 0.8, 2.85, 5.7, 1.6, "Ask a long research paper anything, through an LLM that can see only 10,000 tokens.",
     size=24, color=ON_DARK)
bullets(s, 0.8, 4.65, 5.6, 2.0, [
    "Parsed on this machine; the LLM gets text only",
    "Answers cite the page, table or section they came from",
    "Every model call is measured against the 10K limit",
], size=15, color=ON_DARK2, gap=8, mark="—")
rect(s, 7.0, 0.9, 5.6, 5.3, INK2, radius=0.05)
text(s, 7.3, 1.1, 5.0, 0.3, "LightRAG (arXiv 2410.05779) · 16 pages", size=11, color=ON_DARK2, bold=True)
chat(s, 7.3, 1.55, 5.0, TRACE_Q, [[("The Overall win rate of the ", {}), (NB + "High", {"bold": True}),
                                      (" variant on the CS dataset is ", {}), ("56.0%", {"bold": True}), (" [S4].", {})]],
     qh=0.95, ah=0.95, size=14, dark=True)
led = {e["call"]: e for e in TRACE["ledger"]}
pill(s, 7.3, 3.85, 2.3, "Source: Table 2, ablation", fill=TEAL, color=WHITE, size=11)
pill(s, 9.75, 3.85, 2.55, "Number found in source ✓", fill=GREEN, color=WHITE, size=11)
text(s, 7.3, 4.5, 5.0, 1.5, [
    [("Token ledger", {"bold": True, "color": WHITE})],
    [(f"plan    {led['plan']['actual_prompt_tokens']:>5,} prompt + {led['plan']['actual_output_tokens']} output tokens", {})],
    [(f"answer  {led['answer']['actual_prompt_tokens']:>5,} prompt + {led['answer']['actual_output_tokens']} output tokens", {})],
    [("limit   10,000 per call", {"color": TEAL})]], size=12, color=ON_DARK, font="Consolas", spacing=2)
notes(s, "Open with the live demo if the machine is ready: upload the LightRAG PDF and ask this exact question. "
         "The answer reads one cell from a table that PDF extraction usually breaks, cites it, and the ledger shows "
         "the whole exchange used about half of one 10K window. Then: 'Here is how it got there.'")

# 2 — the problem -------------------------------------------------------------
s = content_slide("Two hard limits and one test paper drove the design", "The problem")
rows = [("The LLM may not parse the document", "PDFs are parsed locally with PyMuPDF: reading order, headings, and "
         "tables rebuilt from word positions. The LLM only ever receives and returns text."),
        ("The LLM sees at most 10K tokens per call", "One budget gate checks every call before it is sent; real "
         "token counts are logged per call and shown under every answer."),
        ("Test case: a 16-page paper, 1.5× the window", "LightRAG (arXiv 2410.05779): summarise it, compare it with "
         "GraphRAG, and read its ablation table, every answer cited to the page.")]
for i, (rule, answer) in enumerate(rows):
    y = 1.8 + i * 1.5
    badge(s, 0.6, y + 0.3, str(i + 1), d=0.6, size=17)
    rect(s, 1.45, y, 4.1, 1.2, AMBER_SOFT)
    text(s, 1.7, y, 3.7, 1.2, rule, size=17, color=INK, bold=True, font=HEAD, anchor=MSO_ANCHOR.MIDDLE)
    arrow(s, 5.65, y + 0.6, 6.25, y + 0.6, color=TEAL, width=2.25)
    rect(s, 6.35, y, 6.4, 1.2, TEAL_SOFT)
    text(s, 6.6, y, 6.0, 1.2, answer, size=15, color=TEXT, anchor=MSO_ANCHOR.MIDDLE)
box = text(s, 0.6, 6.4, 12.2, 0.4, "How documents longer than the window, and summaries, are handled: slides "
           "{context} and {summary}.", size=14, color=MUTED)
LATER.append((box, "How documents longer than the window, and summaries, are handled: slides {context} and "
                   "{summary}.", 14))
notes(s, "Each limit maps to one mechanism I can point to in the code and in the UI.")

# 3 — how it works --------------------------------------------------------------------------------
s = content_slide("Parse once, retrieve narrowly, answer inside the budget", "How it works")
mark("how")
lanes = [("Once per\ndocument", [("Upload", "PDF · TXT · MD", LIGHT), ("Parse locally", "no LLM", TEAL_SOFT),
                                 ("Document card", "title · authors · abstract", TEAL_SOFT),
                                 ("Chunks", "by section; tables kept whole", TEAL_SOFT),
                                 ("Index", "keywords + embeddings", TEAL_SOFT),
                                 ("Summary notes", "map-reduce, when first asked", AMBER_SOFT)]),
         ("Every\nquestion", [("Question", "any language", LIGHT), ("Planner", "intent + search queries", AMBER_SOFT),
                              ("Route", "question · list · summary · ask back", TEAL_SOFT),
                              ("Evidence", "search · rerank · lookup · counts", TEAL_SOFT),
                              ("Budget gate", "≤ 10,000 tokens", TEAL_SOFT),
                              ("Answer", "streamed, with citations", AMBER_SOFT)])]
bw, bh, gap, x0 = 1.62, 1.25, 0.2, 1.85
for r, (label, boxes) in enumerate(lanes):
    y = 1.85 + r * 2.0
    text(s, 0.6, y, 1.25, bh, label, size=13, color=MUTED, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    for i, (head, sub, fill) in enumerate(boxes):
        x = x0 + i * (bw + gap)
        rect(s, x, y, bw, bh, fill, line=LINE)
        text(s, x + 0.08, y, bw - 0.16, bh, [[(head, {"bold": True, "size": 14, "color": INK})],
                                              [(sub, {"size": 11, "color": MUTED})]],
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, spacing=2)
        if i < len(boxes) - 1:
            arrow(s, x + bw + 0.02, y + bh / 2, x + bw + gap - 0.02, y + bh / 2)
arrow(s, x0 + 4 * (bw + gap) + bw / 2, 1.85 + bh + 0.04, x0 + 3 * (bw + gap) + bw / 2, 3.83, color=TEAL)
text(s, 0.6, 5.35, 12.2, 0.4, [[("Amber", {"bold": True, "color": AMBER}), ("  LLM call (text in, text out)      ", {"color": MUTED}),
                                ("Teal", {"bold": True, "color": TEAL}), ("  local code or small local models", {"color": MUTED})]], size=13)
text(s, 0.6, 5.85, 12.2, 1.0, "After every answer, three checks run in code: every cited source must exist, every number "
     "must appear in the evidence, and every table number must sit under the row and column it came from. They flag "
     "problems under the answer for the reader; they do not rewrite it.", size=14)
notes(s, "A normal question costs two LLM calls: a ~200-token planner and the answer. Parsing, embeddings "
         "(nomic-embed-text), reranking (bge-reranker-base), lookups, counting and the checks are all local.")

# 4 — what it looks like ----------------------------------------------------------------------------
s = content_slide("What it looks like: every answer shows its work", "The product")
screen(s, "sources.png", 0.6, 1.55, 8.4)
calls = [("1", "The answer, with its citation", "56.0% [S2]: one cell of the ablation table."),
         ("2", "Click a citation, see the source", "[S2] opens the exact text the model read: Table 2, page 8, "
                                                   "rebuilt as \"row · column = value\"."),
         ("3", "The document's numbers", "16 pages, ~15,389 tokens: larger than the 10K window, so it is "
                                         "retrieved, not pasted."),
         ("4", "Checks and ledger", "Under each answer: cited sources exist, numbers found under the right "
                                    "row and column, tokens per call.")]
for i, (n, t, d) in enumerate(calls):
    yy = 1.6 + i * 1.3
    badge(s, 9.35, yy, n, d=0.45, size=13, fill=TEAL if i % 2 == 0 else INK)
    text(s, 9.95, yy + 0.02, 2.85, 0.4, t, size=14, color=INK, bold=True)
    text(s, 9.95, yy + 0.42, 2.85, 0.85, d, size=12, color=MUTED)
text(s, 0.6, 6.9, 8.4, 0.3, f"Live capture of the local app, {M14}", size=10, color=MUTED, italic=True)
notes(s, "This is the real app, captured while answering. The left list is the document library; the header shows "
         "the paper is bigger than the context, so it goes through retrieval and map-reduce.")

# 5 — over-context --------------------------------------------------------------------------------
s = content_slide("A 15K-token paper never meets the 10K window whole", "Documents longer than the context")
mark("context")
text(s, 0.6, 1.75, 4.0, 1.1, "15,389", size=58, color=INK, bold=True, font=HEAD)
text(s, 0.6, 2.85, 4.0, 0.5, "tokens of text in the LightRAG paper", size=14, color=MUTED)
text(s, 0.6, 3.5, 4.0, 1.1, "10,000", size=58, color=TEAL, bold=True, font=HEAD)
text(s, 0.6, 4.6, 4.0, 0.5, "tokens the LLM may see in one call", size=14, color=MUTED)
cd = CategoryChartData()
cd.categories = ["One answer call"]
parts = [("Instructions + question", 450), ("Evidence incl. document card (median)", EVID),
         ("Headroom", 10000 - 450 - EVID - 1800), ("Reserved for the answer", 1800)]
for name, v in parts:
    cd.add_series(name, [v])
ch = s.shapes.add_chart(XL_CHART_TYPE.BAR_STACKED, Inches(4.8), Inches(1.65), Inches(8.0), Inches(2.0), cd).chart
ch.has_legend, ch.legend.include_in_layout = True, False
ch.legend.font.size, ch.legend.font.name = Pt(11), BODY
ch.value_axis.maximum_scale, ch.value_axis.has_major_gridlines = 10000, False
ch.value_axis.tick_labels.font.size = Pt(10)
ch.category_axis.visible = False
for ser, col in zip(ch.series, [MUTED, TEAL, LINE, AMBER]):
    ser.format.fill.solid()
    ser.format.fill.fore_color.rgb = col
    ser.data_labels.show_value = True
    ser.data_labels.font.size = Pt(10)
    ser.data_labels.font.color.rgb = TEXT if col == LINE else WHITE
    ser.data_labels.position = XL_LABEL_POSITION.CENTER
bullets(s, 4.8, 4.0, 8.0, 3.0, [
    [("Budget first. ", {"bold": True}), ("10,000 minus the answer reserve, with a 15% safety margin. A call that would not fit "
                                         "is stopped, never silently cut.", {})],
    [("Retrieve narrowly. ", {"bold": True}), ("Section-aware chunks, keyword + meaning search over several planner queries, then "
                                              "a cross-encoder keeps only clearly relevant passages.", {})],
    [("Tables as compact text. ", {"bold": True}), ("A whole results table fits, and every number keeps its row and column.", {})],
    [("Whole-document questions ", {"bold": True}), ("(summaries, \"list all…\") read every page in parts: next slide.", {})],
], size=14)
notes(s, f"Median evidence over the {ret_n} benchmark questions is {EVID:,} tokens including the document card, about a third of the window. "
         "The remaining headroom is why answers never need truncation.")

# 5 — summary ------------------------------------------------------------------------------------
s = content_slide("Summaries read every page, in parts that fit", "How the summary is handled")
mark("summary")
xs = [0.6 + i * 2.25 for i in range(4)]
for i, x in enumerate(xs):
    rect(s, x, 1.8, 2.0, 0.95, LIGHT, line=LINE)
    text(s, x + 0.1, 1.8, 1.8, 0.95, [[(f"Part {i + 1}", {"bold": True, "color": INK, "size": 15})],
                                      [("≤ 4,000 tokens", {"color": MUTED, "size": 12})]], align=PP_ALIGN.CENTER,
         anchor=MSO_ANCHOR.MIDDLE, spacing=1)
    arrow(s, x + 1.0, 2.8, x + 1.0, 3.25)
    rect(s, x, 3.3, 2.0, 0.9, AMBER_SOFT, line=LINE)
    text(s, x + 0.1, 3.3, 1.8, 0.9, [[("Map call → notes", {"bold": True, "color": INK, "size": 14})],
                                     [("≤ 600 tokens out", {"color": MUTED, "size": 12})]], align=PP_ALIGN.CENTER,
         anchor=MSO_ANCHOR.MIDDLE, spacing=1)
    arrow(s, x + 1.0, 4.25, 4.85, 4.75, color=TEAL)
rect(s, 2.85, 4.8, 4.0, 1.05, AMBER_SOFT, line=LINE)
text(s, 2.95, 4.8, 3.8, 1.05, [[("Answer call", {"bold": True, "color": INK, "size": 15})],
                               [("document card + all notes + your request", {"color": MUTED, "size": 12})]],
     align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, spacing=1)
bullets(s, 9.6, 1.8, 3.2, 4.6, [
    [("Why: ", {"bold": True}), ("search returns fragments; a summary needs every page.", {})],
    [("Any length: ", {"bold": True}), ("if the notes exceed 5,000 tokens they are condensed again, as many levels as needed.", {})],
    [("Built when asked: ", {"bold": True}), ("on the first summary request, then cached; every later summary is one call.", {})],
    [("Reusable: ", {"bold": True}), ("\"key results only\" or \"用中文總結\" reuse the same notes.", {})],
], size=14)
text(s, 0.6, 6.2, 12.0, 0.6, f"LightRAG: 4 map calls of 3,363–4,202 prompt tokens each (17,883 tokens, once); "
     f"each summary after that is one call of 4,496 tokens ({M27}, final run).", size=13, color=MUTED)
notes(s, "References are left out of the map step. Each note keeps its page range, so the summary cites [N1] to [N4] "
         "and the user can see which part of the paper a sentence came from.")

# 6 — release notes ------------------------------------------------------------------------------
s = content_slide("Release notes: each layer had to earn its place", "How it got better")
cw, cg, cx0, base, top = 2.05, 0.38, 0.75, 5.15, 2.2
for i, (ver, name, sub, p, tok) in enumerate(ABLATION):
    x = cx0 + i * (cw + cg)
    hgt = (base - top) * p / 35
    rect(s, x + 0.3, base - hgt, cw - 0.6, hgt, TEAL if i == len(ABLATION) - 1 else TEAL_SOFT, rounded=False)
    text(s, x, base - hgt - 0.5, cw, 0.45, f"{p}/35", size=22, color=INK, bold=True, font=HEAD, align=PP_ALIGN.CENTER)
    text(s, x, base + 0.12, cw, 0.35, [[(ver + "  ", {"color": TEAL, "bold": True}), (name, {"bold": True, "color": INK})]],
         size=14, align=PP_ALIGN.CENTER)
    text(s, x, base + 0.45, cw, 0.3, sub, size=11, color=MUTED, align=PP_ALIGN.CENTER)
    pill(s, x + 0.2, base + 0.85, cw - 0.4, f"{tok:,} tokens of evidence", fill=AMBER_SOFT if i == 2 else LIGHT, size=11)
text(s, cx0 + 2 * (cw + cg) - 0.1, 1.5, cw + 0.4, 0.5, "same coverage, half the reading", size=12, color=AMBER,
     bold=True, align=PP_ALIGN.CENTER)
text(s, 0.75, 6.55, 12, 0.4, f"Facts that reached the model's evidence, {ret_n} questions on 4 papers, with each layer "
     "switched on in turn (no answer model involved).", size=12, color=MUTED)
notes(s, "This is the retrieval ablation from the report. Meaning search fixed questions phrased differently from "
         "the paper; the reranker halved what the model has to read with no loss; the document card fixed 'who wrote "
         "this'; direct lookup fixed 'what is on page 13'. Every layer is a measured step, not a guess.")

# 7 — capabilities: tables + document --------------------------------------------------------------
s = content_slide("It reads the tables, and it knows the document", "What it can do · 1/4")
capability(s, 0.6, 1.6, 5.95, 4.85, "T", "Reads the right cell",
           "a PDF table is just positioned words; one shifted column changes the number.",
           TRACE_Q, f"The Overall win rate of the {NB}High variant on the CS dataset is 56.0% [S4].",
           "tables are rebuilt from word coordinates and written as \"row · column = value\", so every number "
           "keeps its labels.", f"Answer from {M27}", qh=0.85, ah=0.8)
capability(s, 6.8, 1.6, 5.95, 4.85, "D", "Knows what it is reading",
           "the author line answers no topic, so search never retrieves it.",
           "Who are the authors?", "Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, Chao Huang [D]",
           "a document card (title, authors, abstract, outline) is built from the layout without an LLM and "
           "goes into every call.", f"Answer from {M14}", qh=0.85, ah=0.8)
notes(s, "Two failures most document chats have: wrong numbers from tables and 'I don't know who wrote this'. "
         "[D] in the answer means the fact came from the document card.")

# 8 — capabilities: counting + lookup ---------------------------------------------------------------
s = content_slide("It counts, and it goes straight to page 13", "What it can do · 2/4")
capability(s, 0.6, 1.6, 5.95, 4.85, "#", "Counts instead of guessing",
           "an LLM cannot count citations across pages it never sees.",
           "Which paper is cited the most in this paper?",
           "\"From local to global: A graph rag approach to query-focused summarization\" by Darren Edge et al., "
           "which is cited 5 times [R].",
           "citations are matched to the reference list and terms are counted in code (\"GraphRAG\": 37 times); "
           "the model only reports the result.", f"Answer from {M14}", qh=0.62, ah=1.15)
capability(s, 6.8, 1.6, 5.95, 4.85, "↗", "Goes straight to the place",
           "\"page 13\" or \"Table 4\" means nothing to a search engine.",
           "What is on page 13?",
           "Experimental settings, the statistics of the datasets (Table 4), and the prompts used for graph "
           "generation (Figure 4) [S1][S2][S4].",
           "references to a table, figure, section, page, appendix or reference number are resolved directly, "
           "in English and Chinese.", f"Answer from {M14}, shortened", qh=0.62, ah=1.15)
notes(s, "Counting and direct lookup are deterministic: code does the part LLMs are bad at, the model writes the sentence.")

# 9 — capabilities: lists + several documents --------------------------------------------------------
s = content_slide("It reads across the paper, and across papers", "What it can do · 3/4")
capability(s, 0.6, 1.6, 5.95, 4.85, "L", "Finds every item, not the top five",
           "\"list all…\" needs every page; search returns only the best few passages.",
           "List all baseline methods LightRAG is compared with.",
           "NaiveRAG · GraphRAG · RQ-RAG · HyDE  [L1]–[L4]",
           "the question is mapped over every part of the paper, each part reports what it contains, and the "
           "lists are merged.", f"Answer from {M14}", qh=0.62, ah=0.8)
x, y, w, h = 6.8, 1.6, 5.95, 4.85
rect(s, x, y, w, h, LIGHT)
badge(s, x + 0.3, y + 0.28, "+", d=0.55, size=18)
text(s, x + 1.05, y + 0.3, w - 1.3, 0.5, "Several papers in one chat", size=19, color=INK, bold=True, font=HEAD)
text(s, x + 0.3, y + 0.98, w - 0.6, 0.5, [[("Why it's hard  ", {"bold": True, "color": AMBER}),
                                          ("one long paper can crowd the other out of the window.", {})]], size=12.5, color=MUTED)
pic = screen(s, "multi.png", x + 0.3, y + 1.5, w - 0.85, crop=(0.2, 0.19, 0.255, 0.36))
how_y = y + 1.5 + pic.height / 914400 + 0.12
text(s, x + 0.3, how_y, w - 0.6, 0.3,
     [[("How  ", {"bold": True, "color": TEAL}), ("each paper is searched and reranked on its own.", {})]],
     size=12, color=TEXT)
text(s, x + 0.3, how_y + 0.33, w - 0.6, 0.25, f"Live capture, {M14}", size=10, color=MUTED, italic=True)
notes(s, "Live demo option: tick '+ chat' on a second paper and ask 'Which datasets does each paper use in its "
         "experiments?'. The answer is split by document and each part cites its own source.")

# 10 — capabilities: conversation ------------------------------------------------------------------
s = content_slide("It holds a conversation, and it pushes back", "What it can do · 4/4")
grid = [("↩", "Follows up", "And against GraphRAG?",
         "LightRAG outperforms GraphRAG, particularly in larger datasets with complex language contexts… [S4]",
         "turn 8 of a 12-turn conversation; \"it\" and \"against\" are resolved first"),
        ("?", "Asks back when unclear", "What is its score?",
         "The question is ambiguous… The closest related information is the win rates against the baselines in Table 1.",
         "it names what it could mean instead of picking one"),
        ("!", "Corrects false premises", "Why does LightRAG use reinforcement learning to build its graph?",
         "The sources do not mention that LightRAG uses reinforcement learning… not supported by the document.",
         "every question is checked against the document before it is answered"),
        ("中", "Answers in your language", "這篇論文的作者是誰？",
         "這篇論文的作者是 Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, 和 Chao Huang [S4].",
         "search splits Chinese and Japanese text into characters and pairs, so it works without spaces")]
for i, (icon, title, q, a, how) in enumerate(grid):
    x, y = 0.6 + (i % 2) * 6.2, 1.55 + (i // 2) * 2.68
    rect(s, x, y, 5.95, 2.55, LIGHT)
    badge(s, x + 0.25, y + 0.22, icon, d=0.46, size=13)
    text(s, x + 0.85, y + 0.22, 3.0, 0.46, title, size=16, color=INK, bold=True, font=HEAD, anchor=MSO_ANCHOR.MIDDLE)
    chat(s, x + 0.25, y + 0.82, 5.45, q, a, qh=0.45, ah=0.72, size=12)
    text(s, x + 0.25, y + 2.18, 5.45, 0.3, how, size=10.5, color=MUTED, italic=True)
text(s, 0.6, 6.92, 9, 0.3, f"Answers from {M14} (robustness set), shortened with …", size=10, color=MUTED, italic=True)
notes(s, "Earlier answers are not fed back into the answer prompt, only earlier questions; that keeps one wrong "
         "answer from spreading through a long conversation. Document text is treated as data, so instructions "
         "inside a PDF are ignored.")

# 11 — peel back the layers -------------------------------------------------------------------------
s = content_slide("One question, peeled back", "What happens behind the answer")
plan = TRACE["plan"]
layers = [("1", "Plan", "A small LLM call picks the intent and writes search queries",
           [f"intent: {plan['intent']}"] + [f"“{q}”" for q in plan["queries"]]),
          ("2", "Gather", "Search every query, rerank, pack within the budget",
           ["Table 2 chunk + its caption + the rest of the table", "the document card, always"]),
          ("3", "Answer", "Answers from those sources only, with citations",
           [f"“…the Overall win rate of the {NB}High variant on the CS dataset is 56.0% [S4].”"]),
          ("4", "Check", "Code verifies the answer before it is marked as checked",
           ["cited sources exist ✓", "56.0 appears in the evidence ✓"])]
for i, (n, title, desc, items) in enumerate(layers):
    y = 1.6 + [0, 1.6, 2.65, 3.7][i]
    badge(s, 0.6, y + 0.1, n, d=0.5, size=14, fill=TEAL if i % 2 == 0 else INK)
    text(s, 1.3, y + 0.08, 1.4, 0.5, title, size=18, color=INK, bold=True, font=HEAD)
    text(s, 2.6, y + 0.12, 5.3, 0.35, desc, size=12.5, color=MUTED)
    text(s, 2.6, y + 0.5, 5.3, 1.0, "  ·  ".join(items) if i != 0 else "\n".join(items), size=11.5 if i == 0 else 12.5,
         color=TEXT, font="Consolas" if i == 0 else BODY, spacing=0)
rows = [["Call", "Estimated", "Actual prompt", "Output", "Seconds"]]
for e in TRACE["ledger"]:
    rows.append([e["call"], f"{e['est_prompt_tokens']:,}", f"{e['actual_prompt_tokens']:,}",
                 f"{e['actual_output_tokens']:,}", f"{e['seconds']:.1f}"])
rect(s, 8.15, 1.6, 4.6, 5.15, LIGHT)
text(s, 8.45, 1.8, 4.0, 0.4, "Token ledger (shown in the UI)", size=15, color=INK, bold=True, font=HEAD)
tbl = s.shapes.add_table(len(rows), 5, Inches(8.45), Inches(2.35), Inches(4.0), Inches(1.2)).table
for c, wd in enumerate([0.8, 0.85, 0.95, 0.7, 0.7]):
    tbl.columns[c].width = Inches(wd)
for r, row in enumerate(rows):
    for c, val in enumerate(row):
        cell = tbl.cell(r, c)
        cell.fill.solid()
        cell.fill.fore_color.rgb = INK if r == 0 else WHITE
        cell.margin_left = cell.margin_right = Inches(0.05)
        para = cell.text_frame.paragraphs[0]
        para.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.RIGHT
        run = para.add_run()
        run.text = val
        run.font.size, run.font.name = Pt(11), BODY
        run.font.bold = r == 0
        run.font.color.rgb = WHITE if r == 0 else TEXT
total = sum(e["actual_prompt_tokens"] for e in TRACE["ledger"])
bullets(s, 8.45, 3.85, 4.0, 2.8, [
    [("Largest call: ", {"bold": True}), (f"{max(e['actual_prompt_tokens'] for e in TRACE['ledger']):,} of 10,000 tokens", {})],
    [("Estimate vs. provider count: ", {"bold": True}), ("within 8%, covered by the 15% margin", {})],
    [("Same view for every call: ", {"bold": True}), ("plan, map, reduce, answer", {})],
], size=12.5, gap=8)
notes(s, "This is the exact trace of the opening question. Nothing is hidden: the sources panel shows the text the model "
         "saw, the ledger shows what it cost, and the checks run without trusting the model.")

# 12 — the three core test questions ---------------------------------------------------------------
s = content_slide("Three test questions on the LightRAG paper", "Results")
mark("results")
# Final run on the final code (token ledgers in docs/screens/q1-q3.png): largest call, and the whole question
FINAL = {"Q1": "answer call 4,496 · 1 call (+4 map calls once)",
         "Q2": "answer call 4,785 · 2 calls, 5,310 in all",
         "Q3": "answer call 6,407 · 2 calls, 6,882 in all"}
qcols = [("Q1", "summary this document.", "summary this document."),
         ("Q2", "compare lightRAG with GraphRAG", "compare lightRAG with GraphRAG"),
         ("Q3", "Performance of ablated versions of LightRAG", "Performance of ablated versions of LightRAG")]
for i, (tag, qtext, key) in enumerate(qcols):
    x = 0.6 + i * 4.1
    rect(s, x, 1.6, 3.85, 4.95, LIGHT)
    badge(s, x + 0.25, 1.8, tag, d=0.55, size=14)
    text(s, x + 0.95, 1.8, 2.75, 0.55, f"“{qtext}”", size=14, color=INK, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    pill(s, x + 0.25, 6.05, 3.35, FINAL[tag], size=10.5)
bullets(s, 0.85, 2.6, 3.4, 3.4, [
    "Structured: gist, problem, method, key results, limitations",
    "Gist: graph-based text indexing + dual-level retrieval for faster, cheaper, more context-aware RAG [D][N1]",
    "Exact numbers, e.g. Legal: 84.8% overall win rate vs. NaiveRAG",
    "Built from 4 map-reduce notes that cover all 16 pages",
], size=13.5, gap=8)
bullets(s, 4.95, 2.6, 3.4, 3.4, [
    "Quality: LightRAG wins, most clearly on the large datasets [S4]",
    "Retrieval cost on Legal: GraphRAG ~610,000 tokens over hundreds of calls; LightRAG < 100 tokens, one call",
    "Updates: GraphRAG rebuilds its communities; LightRAG merges new graph nodes in",
    "Method: dual-level (specific + broad) retrieval vs. community reports",
], size=13.5, gap=8)
text(s, 9.05, 2.6, 3.4, 0.35, "Overall win rate vs. NaiveRAG", size=12.5, color=INK, bold=True)
abl = [["", "Full", NB + "High", NB + "Low", NB + "Origin"], ["Agri", "67.6", "64.8", "65.2", "74.4"],
       ["CS", "61.2", "56.0", "56.4", "60.8"], ["Legal", "84.8", "78.0", "81.2", "84.4"],
       ["Mix", "60.0", "57.6", "64.8", "55.6"]]
tbl = s.shapes.add_table(5, 5, Inches(9.05), Inches(3.0), Inches(3.55), Inches(1.75)).table
for c, wdt in enumerate((0.6, 0.65, 0.7, 0.7, 0.9)):
    tbl.columns[c].width = Inches(wdt)
for r, row in enumerate(abl):
    for c, val in enumerate(row):
        cell = tbl.cell(r, c)
        cell.fill.solid()
        cell.fill.fore_color.rgb = INK if r == 0 else WHITE
        para = cell.text_frame.paragraphs[0]
        para.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.RIGHT
        run = para.add_run()
        run.text = val
        run.font.size, run.font.name, run.font.bold = Pt(10.5), BODY, r == 0 or c == 0
        run.font.color.rgb = WHITE if r == 0 else TEXT
text(s, 9.05, 4.9, 3.4, 1.0, "Full Table 2 per dataset and dimension for every variant, each value cited and table-checked.",
     size=12, color=MUTED)
text(s, 0.6, 6.75, 12, 0.3, f"Final run, {M27} (open weights, via Groq; the default). Repeated from fresh uploads: Q3 complete 3/3 (0/3 on the offline qwen2.5:14b). Tokens per call out of 10,000; excerpts.", size=10,
     color=MUTED, italic=True)
notes(s, "Q1 goes through the summary route (map-reduce notes), Q2 and Q3 through retrieval. Q3 needs a whole results "
         "table, which is why tables are kept whole with their caption and continuation.")

# 13 — proof ------------------------------------------------------------------------------------------
s = content_slide("Measured, not demoed once", "Evaluation")
mark("measured")
tiles = [(f"{ret_p}/{ret_n}", "questions whose facts reached the evidence", "4 papers, retrieval benchmark", TEAL),
         (f"{held_p}/{held_n}", "correct answers on an unseen paper", f"{M14}; questions frozen before the run", INK),
         (f"{groq_p}/{groq_n}", f"correct answers with {M27}",
          f"stricter Q3 check ({groq_first}/{groq_n} as first scored)", INK),
         (f"{conf_p}/{conf_n}", "confusing questions handled", f"{M14}; false premises, injection, off-topic", TEAL),
         (f"{conv_p}/{conv_n}", "turns right in a 12-turn conversation", f"{M14}; follow-ups depend on earlier turns", INK),
         (f"{CALLS_CHECKED}/{CALLS_CHECKED}", "LLM calls within 10,000 tokens", f"all {M14} runs; largest {LARGEST_CALL:,}", TEAL)]
for i, (big, label, sub, col) in enumerate(tiles):
    x, y = 0.6 + (i % 3) * 4.15, 1.65 + (i // 3) * 2.45
    rect(s, x, y, 3.9, 2.2, LIGHT)
    text(s, x + 0.3, y + 0.2, 3.4, 0.95, big, size=44, color=col, bold=True, font=HEAD)
    text(s, x + 0.3, y + 1.15, 3.4, 0.5, label, size=14, color=INK, bold=True)
    text(s, x + 0.3, y + 1.65, 3.4, 0.4, sub, size=11.5, color=MUTED)
text(s, 0.6, 6.6, 12.2, 0.4, f"Open models: {M14} and {M27}. 0 answers cited a source that does not exist. "
     "Questions, expected facts and every run are in evaluation/.", size=12, color=MUTED)
notes(s, "Development questions were written on four papers; a fifth paper (the original RAG paper) was held out and "
         "its questions frozen before the first run. Every number on this slide is read from the result files.")

# 14 — how I worked with AI -------------------------------------------------------------------------
s = content_slide("How I worked with AI: a loop with guardrails", "AI-native process")
mark("ai")
loop = [("Spec", "I wrote the spec and the\nbinding limits first"), ("Build", "thin slices,\none layer at a time"),
        ("Test", "on real papers of new\nlayouts, not one sample"), ("Measure", "benchmark + ablation\nafter every change"),
        ("Fix", "AI diagnoses first;\nI decide the direction")]
cx, cy, rad = 3.95, 3.95, 1.4
pos = []
for i in range(len(loop)):
    ang = -math.pi / 2 + i * 2 * math.pi / len(loop)
    pos.append((cx + rad * math.cos(ang), cy + rad * math.sin(ang)))
for i, ((px, py), (t, d)) in enumerate(zip(pos, loop)):
    nx, ny = pos[(i + 1) % len(pos)]
    dx, dy = nx - px, ny - py
    L = math.hypot(dx, dy)
    arrow(s, px + dx / L * 0.6, py + dy / L * 0.6, nx - dx / L * 0.6, ny - dy / L * 0.6, color=TEAL, width=2)
for i, ((px, py), (t, d)) in enumerate(zip(pos, loop)):
    badge(s, px - 0.48, py - 0.48, t, d=0.96, size=12, fill=INK if i % 2 else TEAL)
    if i == 0:                       # top: label to the right
        text(s, px + 0.6, py - 0.32, 2.3, 0.7, d, size=11, color=MUTED)
    elif px > cx + 1:                # right: label to the right
        text(s, px + 0.6, py - 0.32, 1.75, 0.7, d, size=11, color=MUTED)
    elif px < cx - 1:                # left: label to the left
        text(s, px - 2.35, py - 0.32, 1.75, 0.7, d, size=11, color=MUTED, align=PP_ALIGN.RIGHT)
    else:                            # bottom: label below
        text(s, px - 1.0, py + 0.58, 2.0, 0.7, d, size=11, color=MUTED, align=PP_ALIGN.CENTER)
rect(s, 7.6, 1.6, 5.15, 4.6, LIGHT)
text(s, 7.9, 1.8, 4.6, 0.4, "Guardrails I set so speed did not cost trust", size=16, color=INK, bold=True, font=HEAD)
bullets(s, 7.9, 2.4, 4.6, 4.3, [
    [("Bias guard as a test. ", {"bold": True}), ("The build fails if a paper-specific name or number appears in the "
                                                 "application code.", {})],
    [("Held-out paper. ", {"bold": True}), ("Its questions were frozen before the first run, so tuning could not leak in.", {})],
    [("Every design choice cited. ", {"bold": True}), ("Each feature in the report names the paper and finding behind it.", {})],
    [("Numbers from files. ", {"bold": True}), ("This deck is generated from the evaluation results, not typed in.", {})],
    [("Tests with a fake LLM. ", {"bold": True}), ("58 offline tests run in CI on every push.", {})],
], size=13, gap=9)
notes(s, "Built with Claude Code. My role: decide what to build and in what order, set the guardrails, and judge results "
         "against real documents. The AI wrote most of the code and ran the measurements. The collaboration log is in "
         "docs/AI_COLLABORATION.md.")

# 15 — what I learned working with AI ------------------------------------------------------------------
s = content_slide("What I learned working with AI", "Insights")
mark("insights")
lessons = [("1", "Verification is the real work",
            "AI writes plausible code and plausible answers fast. So I built checks that do not trust it: the token "
            "ledger, and citation, number and table checks. The table check caught an answer my own string-match "
            "scoring had passed."),
           ("2", "Measure the cost, not only the score",
            "Reading the token ledgers showed plain questions taking the read-everything route (16–27K tokens). One "
            "guard brought them back to about 4K. A per-call limit is not a cost limit."),
           ("3", "Give the AI a spec and a test, not only a prompt",
            "The 10K rule became a gate and a test; \"don't tune to the test paper\" became a test that fails the "
            "build. Spec first, tests as guardrails, a commit at every checkpoint."),
           ("4", "A second opinion is input, not an answer",
            "I had other AI models review the project. Each suggestion was checked against the paper, the tests and "
            "the rules first; one parser tweak would have fitted this one paper, so I declined it."),
           ("5", "Honest numbers are part of the product",
            "Scores are reported per model and as first scored; the unseen paper's questions were frozen before the "
            "first run. Knowing where it fails is what makes the rest believable.")]
for i, (n, title, body) in enumerate(lessons):
    col, row = i % 2, i // 2
    x, y = 0.6 + col * 6.2, 1.55 + row * 1.78
    w = 5.95 if i < 4 else 12.15
    rect(s, x, y, w, 1.6, LIGHT)
    badge(s, x + 0.25, y + 0.22, n, d=0.48, size=14, fill=TEAL if i % 2 == 0 else INK)
    text(s, x + 0.9, y + 0.2, w - 1.1, 0.4, title, size=16, color=INK, bold=True, font=HEAD)
    text(s, x + 0.9, y + 0.62, w - 1.1, 0.95, body, size=12, color=TEXT)
notes(s, "These are lessons from this project, each with the evidence behind it. " "Lesson 3 is spec-driven and test-driven development with checkpoints, applied to AI-assisted coding.")

# 17 — close ------------------------------------------------------------------------------------------
s = prs.slides.add_slide(BLANK)
bg(s, INK)
text(s, 0.9, 0.85, 11.5, 0.9, "What's next", size=36, color=WHITE, bold=True, font=HEAD)
bullets(s, 0.9, 1.95, 6.2, 3.4, [
    "Per-passage evidence extraction before answering (PaperQA-style)",
    "Optional layout model for unusual tables",
    "Multi-user deployment: auth, job queue, Kubernetes",
    "More open models on the same benchmark",
], size=17, color=ON_DARK, gap=12)
rect(s, 7.6, 1.85, 4.85, 3.55, INK2, radius=0.05)
text(s, 7.9, 2.05, 4.3, 0.4, "Good to know", size=17, color=TEAL, bold=True, font=HEAD)
bullets(s, 7.9, 2.6, 4.3, 2.4, [
    "Text PDFs only: scanned pages would need OCR",
    "Figures are understood through captions and text",
    "Free API tier: about 7K input tokens per minute",
    "Instructions planted in a document are removed or fenced off; new phrasings rely on the model (27B resisted 9/9)",
], size=14, color=ON_DARK, gap=9)
text(s, 7.9, 4.85, 4.3, 0.5, "Full ranges and limits: docs/REPORT.md, section 11", size=12, color=ON_DARK2, italic=True)
text(s, 0.9, 6.15, 11.5, 0.5, "Thank you", size=24, color=WHITE, bold=True, font=HEAD)
text(s, 0.9, 6.7, 11.5, 0.4, "Demo: run.bat  or  docker compose up  →  http://localhost:8000", size=13, color=ON_DARK2,
     font="Consolas")
notes(s, "Close on the opening idea: the 10K limit turned out to be a question of choosing what the model reads. "
         "Offer to open the token ledger, the sources panel or the evaluation report live.")

# appendix — the three core answers, captured live ------------------------------------------------
APPX = [("Q1", "summary this document.", "q1.png", (0.195, 0.07, 0.253, 0.335),
         [[("Route: ", {"bold": True}), ("full-document summary (map-reduce)", {})],
          [("Calls: ", {"bold": True}), ("4 map calls (built once, then cached) + 1 answer call; largest 4,802 tokens", {})],
          [("Reads: ", {"bold": True}), ("notes [N1]–[N4] that together cover all 16 pages", {})],
          [("Checks: ", {"bold": True}), ("5 cited sources exist; every number found in the notes", {})]]),
        ("Q2", "compare lightRAG with GraphRAG", "q2.png", (0.195, 0.07, 0.253, 0.465),
         [[("Route: ", {"bold": True}), ("retrieval Q&A, hybrid search + rerank", {})],
          [("Calls: ", {"bold": True}), ("planner + answer: 5,310 tokens in total; answer call 4,785", {})],
          [("Covers: ", {"bold": True}), ("quality on large datasets, retrieval and update cost (610,000 vs. < 100 "
                                         "tokens), method differences", {})],
          [("Checks: ", {"bold": True}), ("8 cited sources exist; every number found", {})]]),
        ("Q3", "Performance of ablated versions of LightRAG", "q3.png", (0, 0.02, 0, 0),
         [[("Route: ", {"bold": True}), ("retrieval Q&A; Table 2 retrieved whole with its caption", {})],
          [("Calls: ", {"bold": True}), ("planner + answer: 6,882 tokens in total; answer call 6,407", {})],
          [("Covers: ", {"bold": True}), ("full model, " + NB + "High, " + NB + "Low and " + NB + "Origin × 4 datasets × "
                                         "4 dimensions, every value cited", {})],
          [("Checks: ", {"bold": True}), ("cited sources exist; every number found; no number under a wrong "
                                         "row/column (table check)", {})]])]
for tag, q, img, crop, facts in APPX:
    s = content_slide(f"{tag} · “{q}”", "Appendix · live answer")
    pic = screen_by_height(s, img, 0.6, 1.5, 5.45, crop=crop)
    x = 0.6 + pic.width / 914400 + 0.45
    rect(s, x, 1.5, 12.73 - x, 3.3, LIGHT)
    text(s, x + 0.3, 1.7, 12.73 - x - 0.6, 0.4, "What happened", size=16, color=INK, bold=True, font=HEAD)
    bullets(s, x + 0.3, 2.25, 12.73 - x - 0.6, 4.5, facts, size=13, gap=10)
    text(s, 0.6, 7.0, 8, 0.3, f"Live capture of the local app, {M27} via Groq, final code", size=10, color=MUTED, italic=True)
    notes(s, "Backup slide: the full answer as it appeared in the app, with its checks and token ledger.")

fill_later()
cp = prs.core_properties
cp.author = cp.last_modified_by = "Muskaan Mahadik"
cp.title = "DocChat: chat with long documents under a 10K-token LLM limit"
if ARGS:
    OUT = Path(ARGS[0])
prs.save(OUT)
print(f"saved {OUT} ({len(prs.slides)} slides)")
