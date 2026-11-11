"""Ask the three core questions several times, each round from a fresh upload (no cached summary).

One good screenshot does not show that an answer is reliable; this does. Each round deletes every
document on the server, uploads the paper again, and asks Q1-Q3; answers are scored with the rubric in
questions.json and saved with their checks and token ledgers.

    python evaluation/repeat_core.py --server http://127.0.0.1:8000 --pdf path/to/2410.05779.pdf --rounds 3

--plan q3-heavy asks Q1 once, Q2 twice and Q3 three times (saves tokens on a rate-limited API).
Saved to evaluation/results/core_repeat_<tag>_<date>.json.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import date
from pathlib import Path

import httpx

from run_eval import ask, score, upload

HERE = Path(__file__).resolve().parent
# Table 2, Overall vs. NaiveRAG (Agriculture, CS, Legal, Mix), as parsed from the paper
OVERALL = {"Full": ["67.6", "61.2", "84.8", "60.0"], "-High": ["64.8", "56.0", "78.0", "57.6"],
           "-Low": ["65.2", "56.4", "81.2", "64.8"], "-Origin": ["74.4", "60.8", "84.4", "55.6"]}
PLANS = {"all": None, "q3-heavy": {1: ("summary", "compare", "ablated"), 2: ("compare", "ablated"), 3: ("ablated",)}}


def record(res: dict, item: dict) -> dict:
    done = res.get("done") or {}
    keep = ("call", "actual_prompt_tokens", "actual_output_tokens", "max_output_tokens", "within_limit", "seconds")
    row = {"answer": res.get("answer", ""), "error": res.get("error"), "intent": (res.get("plan") or {}).get("intent"),
           "invalid_citations": (done.get("citations") or {}).get("invalid"),
           "ungrounded_numbers": done.get("ungrounded_numbers"), "misattributed_numbers": done.get("misattributed_numbers"),
           "ledger": [{k: e.get(k) for k in keep} for e in done.get("ledger") or []],
           "summary_ledger": [{k: e.get(k) for k in keep} for e in done.get("summary_ledger") or []]}
    row["rubric_pass"], row["rubric_missing"] = score(item, row["answer"])
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8000")
    ap.add_argument("--pdf", required=True, help="the LightRAG paper (arXiv 2410.05779)")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--plan", choices=list(PLANS), default="all")
    ap.add_argument("--tag", default="run")
    args = ap.parse_args()
    data = json.loads((HERE / "questions.json").read_text(encoding="utf-8"))
    assign = [q for q in data["questions"] if q.get("type") == "core"]
    plan, runs = PLANS[args.plan], []
    with httpx.Client(base_url=args.server, timeout=3600) as c:
        for rnd in range(1, args.rounds + 1):
            for d in c.get("/api/documents").json():
                c.delete(f"/api/documents/{d['id']}")
            doc = upload(c, args.pdf, need_summary=False)
            for item in assign:
                if plan and not any(k in item["q"] for k in plan.get(rnd, ())):
                    continue
                t0 = time.time()
                row = record(ask(c, doc["id"], item["q"]), item)
                row.update(round=rnd, q=item["q"], seconds=round(time.time() - t0, 1))
                if "ablated" in item["q"]:
                    row["overall_values_present"] = {v: [x for x in xs if x in row["answer"]] for v, xs in OVERALL.items()}
                runs.append(row)
                print(f"round {rnd}: {item['q'][:40]:40} rubric={row['rubric_pass']} "
                      f"misattributed={len(row['misattributed_numbers'] or [])} {row['seconds']}s", flush=True)
    out = HERE / "results" / f"core_repeat_{args.tag}_{date.today():%Y%m%d}.json"
    out.write_text(json.dumps({"runs": runs}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved", out)


if __name__ == "__main__":
    main()
