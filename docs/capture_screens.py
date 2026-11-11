"""Capture high-resolution screenshots of the running DocChat UI for the presentation.

Drives a headless Edge/Chrome over the DevTools protocol: selects documents, asks real questions,
waits for the streamed answer, then saves PNGs to docs/screens/. Answers are whatever the configured
model returns at capture time; nothing is staged. Requires: pip install websocket-client
Usage:  python docs/capture_screens.py --server http://127.0.0.1:8001 [--only answer,multi]
"""
from __future__ import annotations

import argparse
import base64
import json
import shutil
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

import websocket

OUT = Path(__file__).resolve().parent / "screens"
BROWSERS = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe"]
PORT = 9333
W, H = 1600, 1000

# scene -> (active document title prefix, extra document title prefixes, question, details to open)
SCENES = {
    "answer": ("LIGHTRAG", [], "In the ablation study, what is the Overall win rate of the -High variant on the CS dataset?",
               ["ledger"]),
    "sources": ("LIGHTRAG", [], "In the ablation study, what is the Overall win rate of the -High variant on the CS dataset?",
                ["sources", "src"]),
    "multi": ("LIGHTRAG", ["Retrieval-Augmented Generation"], "Which datasets does each paper use in its experiments?",
              ["ledger"]),
    "chinese": ("LIGHTRAG", [], "這篇論文的作者是誰？", []),
    # the three core test questions; long answers, so a taller page
    "q1": ("LIGHTRAG", [], "summary this document.", ["ledger"]),
    "q2": ("LIGHTRAG", [], "compare lightRAG with GraphRAG", ["ledger"]),
    "q3": ("LIGHTRAG", [], "Performance of ablated versions of LightRAG", ["ledger"]),
}
TALL = {"q1": 2600, "q2": 2600, "q3": 4600}  # viewport height per scene (default H)


class CDP:
    def __init__(self, ws_url: str):
        self.ws = websocket.create_connection(ws_url, timeout=600, suppress_origin=True)
        self.n = 0

    def call(self, method: str, **params):
        self.n += 1
        self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == self.n:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})

    def js(self, expr: str):
        r = self.call("Runtime.evaluate", expression=expr, awaitPromise=True, returnByValue=True)
        if r.get("exceptionDetails"):
            raise RuntimeError(r["exceptionDetails"])
        return r.get("result", {}).get("value")


def find_doc(cdp: CDP, prefix: str) -> str:
    doc_id = cdp.js(f"(state.docs.find(d => (d.title || d.filename).startsWith({json.dumps(prefix)})) || {{}}).id")
    if not doc_id:
        raise SystemExit(f"no document starting with {prefix!r} on the server")
    return doc_id


def run_scene(cdp: CDP, name: str, server: str):
    active, extras, question, opens = SCENES[name]
    cdp.call("Emulation.setDeviceMetricsOverride", width=W, height=TALL.get(name, H), deviceScaleFactor=2, mobile=False)
    cdp.call("Page.navigate", url=server)
    for _ in range(60):
        time.sleep(0.5)
        if cdp.js("typeof state !== 'undefined' && state.docs.length > 0"):
            break
    doc = find_doc(cdp, active)
    cdp.js(f"selectDoc({json.dumps(doc)})")
    for e in extras:
        cdp.js(f"state.extra.add({json.dumps(find_doc(cdp, e))}); renderDocs();")
    cdp.js(f"send({json.dumps(question)})")
    started = time.time()
    time.sleep(2)
    while cdp.js("state.busy"):
        time.sleep(2)
        if time.time() - started > 900:
            raise SystemExit(f"{name}: no answer after 15 minutes")
    for kind in opens:  # open the panels to show, in the last answer
        if kind == "src":
            cdp.js("(() => { const m = [...document.querySelectorAll('.msg.assistant')].pop();"
                   " const c = m.querySelector('a.cite'); const id = c ? c.dataset.src : null;"
                   " const s = (id && m.querySelector(`details.src[data-id=\"${id}\"]`)) || m.querySelector('details.src'); if (s) s.open = true; })()")
        else:
            cdp.js(f"(() => {{ const m = [...document.querySelectorAll('.msg.assistant')].pop();"
                   f" const d = m.querySelector('details.{kind}'); if (d) d.open = true; }})()")
    time.sleep(0.5)
    if "src" in opens:  # scroll so the opened source is in view
        cdp.js("(() => { const s = [...document.querySelectorAll('details.src[open]')].pop(); if (s) s.scrollIntoView({block: 'center'}); })()")
    else:
        cdp.js("(() => { const box = document.getElementById('messages'); box.scrollTop = 0; })()")
    time.sleep(0.5)
    png = cdp.call("Page.captureScreenshot", format="png")["data"]
    path = OUT / f"{name}.png"
    path.write_bytes(base64.b64decode(png))
    answer = cdp.js("[...document.querySelectorAll('.msg.assistant .answer')].pop().innerText")
    print(f"{name}: {time.time() - started:.0f}s -> {path.name}\n  {answer[:300]!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8000")
    ap.add_argument("--only", default=",".join(SCENES))
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    exe = next((b for b in BROWSERS if Path(b).exists()), None) or shutil.which("msedge") or shutil.which("chrome")
    profile = tempfile.mkdtemp(prefix="docchat-cdp-")
    proc = subprocess.Popen([exe, "--headless=new", f"--remote-debugging-port={PORT}", f"--user-data-dir={profile}",
                             f"--window-size={W},{H}", "--hide-scrollbars", "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(40):
            try:
                tabs = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json", timeout=2).read())
                page = next(t for t in tabs if t.get("type") == "page")
                break
            except Exception:
                time.sleep(0.5)
        else:
            raise SystemExit("browser did not start")
        cdp = CDP(page["webSocketDebuggerUrl"])
        cdp.call("Page.enable")
        cdp.call("Emulation.setDeviceMetricsOverride", width=W, height=H, deviceScaleFactor=2, mobile=False)
        for name in args.only.split(","):
            run_scene(cdp, name.strip(), args.server)
    finally:
        proc.terminate()
        shutil.rmtree(profile, ignore_errors=True)


if __name__ == "__main__":
    main()
