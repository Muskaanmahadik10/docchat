"""Re-score saved answer runs with the current rubric, without calling any model.

A rubric change (see `_scoring_changes` in each question file) should never silently rewrite history: this
prints, for every saved answer run, the score as first recorded next to the score under today's rubric,
and names every answer whose verdict changed.

Run:  python evaluation/rescore.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_eval import score  # noqa: E402

RESULTS = HERE / "results"
RUBRICS = [("heldout_questions_answer_", "heldout_questions.json"),
           ("robustness_questions_answer_", "robustness_questions.json"),
           ("answer_", "questions.json")]


def items(question_file: str) -> dict[tuple, dict]:
    data = json.loads((HERE / question_file).read_text(encoding="utf-8"))
    out = {(q.get("doc"), q["q"]): q for q in data.get("questions", [])}  # the same question recurs per document
    for conv in data.get("conversations", []):
        for turn in conv.get("turns", []):
            out.setdefault((conv.get("doc"), turn["q"]), turn)
    return out


def main() -> None:
    rows, changed = [], []
    for path in sorted(RESULTS.glob("*answer_*.json")):
        rubric = next(f for prefix, f in RUBRICS if path.name.startswith(prefix))
        by_q = items(rubric)
        data = json.loads(path.read_text(encoding="utf-8"))
        first = now = 0
        for r in data["results"]:
            item = by_q.get((r.get("doc"), r["q"]))
            ok = score(item, r.get("answer", ""))[0] if item else r["pass"]
            first += bool(r["pass"])
            now += bool(ok)
            if bool(ok) != bool(r["pass"]):
                changed.append((path.name, r["q"], r["pass"], ok, score(item, r.get("answer", ""))[1]))
        model = data.get("health", {}).get("llm", {}).get("model", "?")
        rows.append((path.name, model, f"{first}/{len(data['results'])}", f"{now}/{len(data['results'])}"))
    print("| Run | Model | As first scored | Current rubric |\n|---|---|---|---|")
    for row in rows:
        print("| " + " | ".join(row) + " |")
    print()
    for name, q, was, now, missing in changed:
        print(f"- {name}: \"{q}\" {'pass' if was else 'fail'} -> {'pass' if now else 'fail'}"
              + (f" (missing: {'; '.join(missing)})" if missing else ""))
    if not changed:
        print("No verdict changed.")


if __name__ == "__main__":
    main()
