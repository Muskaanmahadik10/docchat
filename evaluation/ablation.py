"""Ablation study on the retrieval benchmark (no LLM calls).

Adds components one at a time and measures, for each configuration, how many questions get the facts
they need into the evidence, and how many evidence tokens that takes:

    A  BM25 only
    B  + embeddings (hybrid, RRF)
    C  + cross-encoder rerank
    D  + document card
    E  + direct lookup and citation counts (= the full system)

    python evaluation/ablation.py --pdf lightrag=PATH --pdf bert=PATH ...   (documents already uploaded are reused)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from app.config import Settings  # noqa: E402
from app.llm import LLM, Embedder  # noqa: E402
from app.pipeline import Pipeline  # noqa: E402
from app.store import DocumentStore  # noqa: E402
from app.tokens import count_tokens  # noqa: E402
from run_eval import score  # noqa: E402

CONFIGS = [
    ("A  BM25 only", {"embeddings", "rerank", "card", "lookup", "citations"}),
    ("B  + embeddings (hybrid)", {"rerank", "card", "lookup", "citations"}),
    ("C  + rerank", {"card", "lookup", "citations"}),
    ("D  + document card", {"lookup", "citations"}),
    ("E  + lookup & citation counts (full)", set()),
]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # questions include Chinese; Windows consoles default to cp1252
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", action="append", default=[], help="name=path")
    args = ap.parse_args()
    pdfs = dict(p.split("=", 1) for p in args.pdf)

    settings = Settings.from_env()
    settings.summary_on_upload = False
    llm, embedder = LLM(settings), Embedder(settings)
    store = DocumentStore(settings, llm, embedder)
    pipeline = Pipeline(settings, llm, embedder, store)

    docs = {}
    for name, path in pdfs.items():
        data = Path(path).read_bytes()
        rec = store.get(hashlib.sha256(data).hexdigest()[:16]) or store.add(Path(path).name, data)
        while rec.status == "processing":
            time.sleep(2)
        docs[name] = rec

    spec = json.loads((HERE / "questions.json").read_text(encoding="utf-8"))
    questions = [q for q in spec["questions"] if q["doc"] in docs and not q.get("abstain")
                 and q["type"] not in ("list", "count", "follow-up")
                 and not (q["type"] == "core" and q["q"].startswith("summary"))]

    rows = []
    for label, off in CONFIGS:
        pipeline.ablate = set(off)
        passed, tokens, by_type = 0, [], {}
        for q in questions:
            ev = pipeline.gather_evidence(docs[q["doc"]], q["q"], [], [])
            ok, _ = score(q, "\n".join(s["text"] for s in ev["sources"]))
            passed += ok
            tokens.append(sum(count_tokens(s["text"]) for s in ev["sources"]))  # chunks + card + counts
            t = by_type.setdefault(q["type"], [0, 0])
            t[0] += ok
            t[1] += 1
        rows.append((label, passed, len(questions), statistics.median(tokens), by_type))
        print(f"{label:40} {passed}/{len(questions)}  median evidence ≈{statistics.median(tokens):.0f} tokens", flush=True)

    types = sorted({t for r in rows for t in r[4]})
    lines = ["| Configuration | Facts reached the evidence | Median evidence tokens | " + " | ".join(types) + " |",
             "|---|---|---|" + "---|" * len(types)]
    for label, p, n, med, bt in rows:
        lines.append(f"| {label} | {p}/{n} | {med:.0f} | " + " | ".join(f"{bt[t][0]}/{bt[t][1]}" if t in bt else "–" for t in types) + " |")
    out = HERE / "results" / f"ablation_{time.strftime('%Y%m%d_%H%M%S')}.md"
    out.write_text("# Retrieval ablation\n\n" + "\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nreport: {out.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()
