"""Run the table check (app/numcheck.py) over every saved answer, without calling any model.

Saved runs keep the answers but not the evidence, so each answer is checked against every chunk of its
own document. That is stricter than the live check, which sees only the evidence the model was given:
a number is flagged only if no table cell anywhere in the document carries it under compatible labels.

Needs the documents parsed once by DocChat (upload them in the app, or point --data at a data folder):
    python evaluation/tablecheck_saved.py --data data --data C:/path/to/other/data
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from app.numcheck import misattributed_numbers  # noqa: E402

RUBRICS = [("heldout_questions_answer_", "heldout_questions.json"),
           ("robustness_questions_answer_", "robustness_questions.json"),
           ("answer_", "questions.json")]


def stem(url: str) -> str:
    """'https://arxiv.org/pdf/2410.05779' -> '2410.05779'; drops a version suffix like 'v4'."""
    return re.sub(r"v\d+$", "", Path(url).name.removesuffix(".pdf"))


def load_docs(data_dirs: list[str]) -> dict[str, list[str]]:
    """filename -> chunk texts, first data folder wins."""
    out: dict[str, list[str]] = {}
    for data in data_dirs:
        for d in sorted((Path(data) / "docs").glob("*")):
            meta, chunks = d / "meta.json", d / "chunks.json"
            if meta.exists() and chunks.exists():
                name = json.loads(meta.read_text(encoding="utf-8")).get("filename", "")
                out.setdefault(name, [c["text"] for c in json.loads(chunks.read_text(encoding="utf-8"))])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", action="append", default=[], help="a DocChat data folder (repeatable)")
    args = ap.parse_args()
    docs = load_docs(args.data or [str(HERE.parent / "data")])
    total = flagged = 0
    for path in sorted((HERE / "results").glob("*answer_*.json")):
        rubric = next(f for prefix, f in RUBRICS if path.name.startswith(prefix))
        sources = {key: stem(v["source"]) for key, v in
                   json.loads((HERE / rubric).read_text(encoding="utf-8"))["documents"].items()}
        for r in json.loads(path.read_text(encoding="utf-8"))["results"]:
            want = sources.get(r.get("doc"), "")
            chunks = next((c for name, c in docs.items() if want and want in name), None)
            if chunks is None:
                print(f"  skipped (document '{r.get('doc')}' not parsed): {path.name}: {r['q'][:60]}")
                continue
            total += 1
            got = misattributed_numbers(r.get("answer") or "", chunks)
            if got:
                flagged += 1
                print(f"- {path.name}: \"{r['q']}\" (rubric: {'pass' if r['pass'] else 'fail'})")
                for g in got:
                    print(f"    {g['value']} next to \"{g['context'][:90]}\"; source: {g['cells'][0][:90] if g['cells'] else '?'}")
    print(f"\nTable check over saved answers: {flagged} of {total} flagged.")


if __name__ == "__main__":
    main()
