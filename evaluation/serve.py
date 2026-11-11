"""Start a second DocChat instance with settings overridden, e.g. to evaluate another model while the
main app keeps its own configuration and data:

    python evaluation/serve.py --port 8001 --set LLM_PROVIDER=ollama --set LLM_MODEL=qwen2.5:14b \\
        --set LLM_BASE_URL=https://xxxx.trycloudflare.com --set DATA_DIR=C:/path/to/data_eval

Environment variables win over .env, so the main configuration file is not touched.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--set", action="append", default=[], help="KEY=VALUE environment override")
    args = ap.parse_args()
    for kv in args.set:
        key, value = kv.split("=", 1)
        os.environ[key] = value
    sys.path.insert(0, str(ROOT))
    import uvicorn
    uvicorn.run("app.main:create_app", factory=True, host="127.0.0.1", port=args.port, app_dir=str(ROOT))


if __name__ == "__main__":
    main()
