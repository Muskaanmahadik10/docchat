"""Run the generic evaluation against a running DocChat server.

    python evaluation/run_eval.py --pdf lightrag=PATH --pdf bert=PATH --pdf nist=PATH [--mode retrieval|answer]

--mode retrieval  (no LLM, seconds): did the facts each question needs reach the evidence given to
                  the model? Scores the pipeline (parsing, chunking, search, rerank, card) on its own.
--mode answer     (uses the configured LLM): scores the final answers, plus citation validity,
                  the numeric grounding check and the 10K token limit of every call.
Writes evaluation/results/<mode>_<timestamp>.json and a Markdown report next to it.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
ABSTAIN_RE = re.compile(
    r"(does not|doesn't|do not|did not|not) (mention|specify|provide|contain|include|state|report|say|discuss|describe)"
    r"|no (information|mention|details?|data)|not (mentioned|specified|provided|stated|reported|available|found|included|covered)"
    r"|cannot (find|determine|answer)|could not find|isn't (mentioned|specified)|is not (mentioned|specified|given)", re.I)


def norm(text: str) -> str:
    text = text.lower().replace("−", "-")
    return re.sub(r"(?<=\d),(?=\d{3})", "", text)  # 5,081,069 -> 5081069


def score(item: dict, text: str) -> tuple[bool, list[str]]:
    t = norm(text)
    if item.get("abstain"):
        ok = bool(ABSTAIN_RE.search(text))
        return ok, [] if ok else ["no abstention phrase"]
    missing = [s for s in item.get("all", []) if norm(s) not in t]
    for group in item.get("any", []):
        if not any(norm(s) in t for s in group):
            missing.append(" | ".join(group))
    missing += [f"must not contain: {s}" for s in item.get("none", []) if norm(s) in t]
    return not missing, missing


class QuotaExhausted(RuntimeError):
    pass


def check_quota(res: dict) -> None:
    """A provider quota error is not a wrong answer: stop instead of scoring it."""
    err = (res.get("error") or {}).get("text", "")
    if "per day" in err.lower() or "daily" in err.lower() or "TPD" in err:
        raise QuotaExhausted(err[:200])


def upload(client: httpx.Client, path: str, need_summary: bool) -> dict:
    with open(path, "rb") as f:
        doc = client.post("/api/documents", files={"file": (Path(path).name, f, "application/pdf")}).json()
    while True:
        try:
            doc = client.get(f"/api/documents/{doc['id']}").json()
        except httpx.TransportError:  # server busy (CPU-bound ingestion): poll again
            time.sleep(5)
            continue
        if doc["status"] == "error":
            raise RuntimeError(f"{path}: {doc['error']}")
        # with SUMMARY_ON_UPLOAD=false the summary stays "pending" and is built on the first summary question
        if doc["status"] == "ready" and (not need_summary or doc["summary_status"] != "running"):
            return doc
        time.sleep(3)


def ask(client: httpx.Client, doc_id: str, q: str, history: list | None = None) -> dict:
    answer, out = [], {"events": []}
    with client.stream("POST", "/api/chat", json={"doc_id": doc_id, "message": q, "history": history or []}, timeout=3600) as r:
        buf = ""
        for chunk in r.iter_text():
            buf += chunk
            while "\n\n" in buf:
                raw, buf = buf.split("\n\n", 1)
                ev = raw.split("\n")[0][7:]
                data = json.loads(raw.split("\n")[1][6:])
                if ev == "token":
                    answer.append(data["text"])
                elif ev in ("plan", "sources", "done", "error"):
                    out[ev] = data
    out["answer"] = "".join(answer)
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # questions include Chinese; Windows consoles default to cp1252
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8000")
    ap.add_argument("--pdf", action="append", default=[], help="name=path, name as in questions.json")
    ap.add_argument("--mode", choices=["retrieval", "answer"], default="retrieval")
    ap.add_argument("--only", help="comma-separated question types to run")
    ap.add_argument("--questions", default="questions.json", help="question file in evaluation/")
    args = ap.parse_args()

    spec = json.loads((HERE / args.questions).read_text(encoding="utf-8"))
    tag = "" if args.questions == "questions.json" else Path(args.questions).stem + "_"
    pdfs = dict(p.split("=", 1) for p in args.pdf)
    questions = [q for q in spec["questions"] if q["doc"] in pdfs
                 and (not args.only or q["type"] in args.only.split(","))]
    if args.mode == "retrieval":  # summary answers come from map-reduce notes, not retrieval
        questions = [q for q in questions if not (q["type"] == "core" and q["q"].startswith("summary"))
                     and not q.get("abstain") and q["type"] not in ("list", "count", "follow-up")]
    if not questions:
        sys.exit("no questions selected (check --pdf names)")

    client = httpx.Client(base_url=args.server, timeout=600)
    docs = {name: upload(client, path, need_summary=args.mode == "answer") for name, path in pdfs.items()}
    results = []
    for i, item in enumerate(questions, 1):
        doc = docs[item["doc"]]
        t0 = time.time()
        if args.mode == "retrieval":
            ev = client.post("/api/retrieve", json={"doc_id": doc["id"], "message": item["q"]}).json()
            text = "\n".join(s["text"] for s in ev["sources"])
            ok, missing = score(item, text)
            row = {"evidence_tokens": ev["evidence_tokens"], "n_sources": len(ev["sources"])}
        else:
            res = ask(client, doc["id"], item["q"], item.get("history"))
            check_quota(res)
            ok, missing = score(item, res["answer"])
            done = res.get("done", {})
            row = {"answer": res["answer"], "error": res.get("error"), "plan": res.get("plan"),
                   "invalid_citations": done.get("citations", {}).get("invalid", []),
                   "ungrounded_numbers": done.get("ungrounded_numbers", []),
                   "misattributed_numbers": done.get("misattributed_numbers", []),
                   "all_calls_within_limit": all(e["within_limit"] for e in done.get("ledger", [])),
                   "ledger": done.get("ledger", [])}
        row.update({"doc": item["doc"], "type": item["type"], "q": item["q"], "pass": ok, "missing": missing,
                    "seconds": round(time.time() - t0, 1)})
        results.append(row)
        print(f"[{i}/{len(questions)}] {'PASS' if ok else 'FAIL'} {item['doc']:8} {item['type']:10} {item['q'][:60]}"
              + (f"  missing: {missing}" if missing else ""), flush=True)

    # scripted conversations: every turn sends the real earlier answers as history
    for conv in spec.get("conversations", []) if args.mode == "answer" else []:
        if conv["doc"] not in docs or (args.only and "conversation" not in args.only.split(",")):
            continue
        history: list[dict] = []
        for k, turn in enumerate(conv["turns"], 1):
            t0 = time.time()
            res = ask(client, docs[conv["doc"]]["id"], turn["q"], history)
            check_quota(res)
            ok, missing = score(turn, res["answer"])
            done = res.get("done", {})
            results.append({"doc": conv["doc"], "type": "conversation", "turn": k, "q": turn["q"], "pass": ok,
                            "missing": missing, "answer": res["answer"], "error": res.get("error"),
                            "plan": res.get("plan"), "invalid_citations": done.get("citations", {}).get("invalid", []),
                            "ungrounded_numbers": done.get("ungrounded_numbers", []),
                            "misattributed_numbers": done.get("misattributed_numbers", []),
                            "all_calls_within_limit": all(e["within_limit"] for e in done.get("ledger", [])),
                            "ledger": done.get("ledger", []), "seconds": round(time.time() - t0, 1)})
            print(f"[turn {k}/{len(conv['turns'])}] {'PASS' if ok else 'FAIL'} {turn['q'][:60]}"
                  + (f"  missing: {missing}" if missing else ""), flush=True)
            history += [{"role": "user", "content": turn["q"]}, {"role": "assistant", "content": res["answer"]}]

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    health = client.get("/api/health").json()
    (out_dir / f"{tag}{args.mode}_{stamp}.json").write_text(json.dumps(
        {"mode": args.mode, "health": health, "docs": docs, "results": results}, ensure_ascii=False, indent=1),
        encoding="utf-8")

    by_type, by_doc = defaultdict(list), defaultdict(list)
    for r in results:
        by_type[r["type"]].append(r["pass"])
        by_doc[r["doc"]].append(r["pass"])
    total = sum(r["pass"] for r in results)
    lines = [f"# DocChat evaluation — {args.mode} mode ({stamp})", "",
             f"Model: `{health['llm']['model']}` · context limit {health['max_context_tokens']} tokens", "",
             f"**Overall: {total}/{len(results)} passed ({100 * total / len(results):.0f}%)**", "",
             "| Group | Passed |", "|---|---|"]
    lines += [f"| type: {k} | {sum(v)}/{len(v)} |" for k, v in sorted(by_type.items())]
    lines += [f"| doc: {k} | {sum(v)}/{len(v)} |" for k, v in sorted(by_doc.items())]
    if args.mode == "answer":
        lines += ["", f"Invalid citations: {sum(bool(r['invalid_citations']) for r in results)} answers · "
                      f"answers with ungrounded numbers: {sum(bool(r['ungrounded_numbers']) for r in results)} · "
                      f"answers with misattributed table numbers: {sum(bool(r.get('misattributed_numbers')) for r in results)} · "
                      f"all LLM calls within limit: {all(r['all_calls_within_limit'] for r in results)}"]
    lines += ["", "| | Doc | Type | Question | Missing |", "|---|---|---|---|---|"]
    lines += [f"| {'✅' if r['pass'] else '❌'} | {r['doc']} | {r['type']} | {r['q']} | {', '.join(r['missing'])} |"
              for r in results]
    (out_dir / f"{tag}{args.mode}_{stamp}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n{total}/{len(results)} passed — report: evaluation/results/{tag}{args.mode}_{stamp}.md")


if __name__ == "__main__":
    main()
