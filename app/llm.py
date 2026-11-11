"""Text-to-text LLM and embedding clients.

The LLM is only ever given plain text (the document is parsed locally). Two backends:
  * "ollama" — local models via Ollama's /api/chat. `num_ctx` is set to the context limit, so
    the model itself cannot attend to more than 10K tokens.
  * "openai" — any OpenAI-compatible /v1/chat/completions server (vLLM, LM Studio, OpenAI, ...).

Every call is checked by ContextBudget before it is sent and recorded in a Ledger with the
server-reported token usage.
"""
from __future__ import annotations

import json
import re
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import httpx

from . import metrics
from .config import Settings
from .tokens import ContextBudget, Ledger, LedgerEntry


class LLMError(RuntimeError):
    pass


class LLM:
    def __init__(self, settings: Settings):
        self.s = settings
        self.timeout = httpx.Timeout(settings.request_timeout_s, connect=10)
        model_ctx = settings.model_context_tokens or self._discover_model_context()
        self.context_limit = min(settings.max_context_tokens, model_ctx) if model_ctx else settings.max_context_tokens
        self.budget = ContextBudget(self.context_limit)
        metrics.set_gauge("docchat_context_limit_tokens", self.context_limit)
        # Interactive questions take priority over background jobs (summary map-reduce):
        # a single local model serves one request at a time, so a background call started
        # while a user waits would delay their answer by minutes on CPU.
        self._interactive = 0
        self._sent: deque[tuple[float, int]] = deque()  # (time, input tokens) of the last minute
        self._rate_lock = threading.Lock()
        self._cond = threading.Condition()

    @contextmanager
    def interactive(self):
        with self._cond:
            self._interactive += 1
        try:
            yield
        finally:
            with self._cond:
                self._interactive -= 1
                self._cond.notify_all()

    def _throttle(self, tokens: int) -> float:
        """Wait until sending `tokens` more input tokens keeps us under the provider's per-minute
        limit (sliding 60 s window, shared by all threads). Returns the seconds waited."""
        limit = self.s.llm_input_tpm
        waited = 0.0
        while limit:
            with self._rate_lock:
                now = time.time()
                while self._sent and now - self._sent[0][0] > 60:
                    self._sent.popleft()
                used = sum(t for _, t in self._sent)
                if used + tokens <= limit or not self._sent:
                    self._sent.append((now, tokens))
                    return waited
                pause = 60 - (now - self._sent[0][0]) + 0.5
            time.sleep(pause)
            waited += pause
            metrics.inc("docchat_rate_limit_wait_seconds_total", pause)
        return waited

    def _wait_for_idle(self, still_background: Callable[[], bool]) -> None:
        """Hold a background call while a user is waiting for an answer, unless the job has been
        promoted (a user is now waiting for this job itself, e.g. a summary they asked for)."""
        with self._cond:
            while self._interactive and still_background():
                self._cond.wait(timeout=1.0)

    def _discover_model_context(self) -> int:
        """Best effort: read the model's trained context length from Ollama."""
        if self.s.llm_provider != "ollama":
            return 0
        try:
            r = httpx.post(f"{self.s.llm_base_url}/api/show", json={"model": self.s.llm_model}, timeout=5)
            info = r.json().get("model_info", {})
            return next((int(v) for k, v in info.items() if k.endswith(".context_length")), 0)
        except Exception:
            return 0

    # -- public API ---------------------------------------------------------------------

    def complete(self, messages: list[dict], *, max_tokens: int, ledger: Ledger, call: str,
                 temperature: float = 0.2, json_schema: dict | None = None,
                 background: bool | Callable[[], bool] = False) -> str:
        if background:
            self._wait_for_idle(background if callable(background) else lambda: True)
        return "".join(self.stream(messages, max_tokens=max_tokens, ledger=ledger, call=call,
                                   temperature=temperature, json_schema=json_schema))

    def stream(self, messages: list[dict], *, max_tokens: int, ledger: Ledger, call: str,
               temperature: float = 0.2, json_schema: dict | None = None) -> Iterator[str]:
        est = self.budget.check(messages, max_tokens)  # raises ContextOverflow
        self._throttle(int(est * 1.1) + 50)  # provider tokenizers count slightly more
        entry = ledger.add(LedgerEntry(call=call, est_prompt_tokens=est, max_output_tokens=max_tokens,
                                       context_limit=self.context_limit))
        start = time.time()
        kind = call.split(" ")[0]  # "summary-map 2/4" -> "summary-map"
        failed = False
        try:
            if self.s.llm_provider == "ollama":
                yield from self._ollama(messages, max_tokens, temperature, json_schema, entry)
            elif self.s.llm_provider == "openai":
                yield from self._openai(messages, max_tokens, temperature, json_schema, entry)
            else:
                raise LLMError(f"Unknown LLM_PROVIDER {self.s.llm_provider!r}")
        except httpx.HTTPError as e:
            failed = True
            raise LLMError(f"LLM request failed ({self.s.llm_provider} at {self.s.llm_base_url}): {e}") from e
        except LLMError:
            failed = True
            raise
        finally:
            entry.seconds = time.time() - start
            metrics.inc("docchat_llm_calls_total", call=kind)
            metrics.inc("docchat_llm_call_seconds_total", entry.seconds)
            if failed:
                metrics.inc("docchat_llm_errors_total", call=kind)
            prompt, output = entry.actual_prompt_tokens or 0, entry.actual_output_tokens or 0
            metrics.inc("docchat_llm_tokens_total", prompt, kind="prompt")
            metrics.inc("docchat_llm_tokens_total", output, kind="output")
            metrics.max_gauge("docchat_largest_call_tokens", prompt + output)

    def health(self) -> dict:
        try:
            if self.s.llm_provider == "ollama":
                r = httpx.get(f"{self.s.llm_base_url}/api/tags", timeout=20)
                names = [m["name"] for m in r.json().get("models", [])]
                ok = any(n == self.s.llm_model or n.split(":")[0] == self.s.llm_model for n in names)
                return {"ok": ok, "detail": "model available" if ok else f"model {self.s.llm_model} not pulled"}
            if not self.s.llm_api_key:
                return {"ok": False, "detail": "LLM_API_KEY is not set: add your key to .env "
                                               "(free Groq key: https://console.groq.com/keys), or use the local Ollama option"}
            r = httpx.get(f"{self.s.llm_base_url.rstrip('/')}/models", headers=self._headers(), timeout=10)
            if r.status_code in (401, 403):
                return {"ok": False, "detail": f"HTTP {r.status_code}: the API key in .env was rejected"}
            return {"ok": r.status_code == 200, "detail": f"HTTP {r.status_code}"}
        except httpx.HTTPError as e:
            return {"ok": False, "detail": str(e)}

    # -- backends -----------------------------------------------------------------------

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.s.llm_api_key}"} if self.s.llm_api_key else {}

    def _ollama(self, messages, max_tokens, temperature, json_schema, entry) -> Iterator[str]:
        body = {
            "model": self.s.llm_model,
            "messages": messages,
            "stream": True,
            "options": {"num_ctx": self.context_limit, "num_predict": max_tokens,
                        "temperature": temperature},
        }
        if json_schema:
            body["format"] = json_schema
        with httpx.stream("POST", f"{self.s.llm_base_url}/api/chat", json=body, timeout=self.timeout) as r:
            if r.status_code != 200:
                r.read()
                raise LLMError(f"Ollama HTTP {r.status_code}: {r.text[:300]}")
            for line in r.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                if data.get("error"):
                    raise LLMError(f"Ollama error: {data['error']}")
                piece = data.get("message", {}).get("content", "")
                if piece:
                    yield piece
                if data.get("done"):
                    entry.actual_prompt_tokens = data.get("prompt_eval_count")
                    entry.actual_output_tokens = data.get("eval_count")

    def _openai(self, messages, max_tokens, temperature, json_schema, entry) -> Iterator[str]:
        body = {
            "model": self.s.llm_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if json_schema:
            body["response_format"] = {"type": "json_object"}
        url = f"{self.s.llm_base_url.rstrip('/')}/chat/completions"
        for attempt in range(6):  # free tiers rate-limit by tokens/minute: wait and retry on 429/503
            with httpx.stream("POST", url, json=body, headers=self._headers(), timeout=self.timeout) as r:
                if r.status_code == 429:
                    r.read()
                    if "per day" in r.text.lower() or "(tpd)" in r.text.lower():  # waiting minutes will not help
                        raise LLMError("The model provider's daily token quota is used up. Try again later, or set "
                                       "another model in .env (LLM_MODEL / LLM_BASE_URL). Provider message: " + r.text[:200])
                    # "Request too large ... output tokens per minute (OTPM): Limit 1000, Requested 1435": the
                    # provider will not accept this max_tokens right now, so waiting does not help. Ask for what
                    # it allows; the ledger records the lower cap and the UI flags an answer that reaches it.
                    m = re.search(r"output tokens per minute.*?Limit (\d+)", r.text, re.I | re.S)
                    if m and "too large" in r.text.lower() and int(m.group(1)) - 50 < body["max_tokens"]:
                        body["max_tokens"] = max(200, int(m.group(1)) - 50)
                        entry.max_output_tokens = body["max_tokens"]
                        metrics.inc("docchat_llm_output_cap_reduced_total")
                        continue
                if r.status_code in (429, 503) and attempt < 5:
                    r.read()
                    try:
                        wait = float(r.headers.get("retry-after", "")) + 1
                    except ValueError:
                        wait = 10.0 * (attempt + 1)
                    time.sleep(min(wait, 90))
                    continue
                if r.status_code != 200:
                    r.read()
                    raise LLMError(f"LLM HTTP {r.status_code}: {r.text[:300]}")
                yield from self._openai_events(r, entry)
                return

    @staticmethod
    def _openai_events(r, entry) -> Iterator[str]:
        for line in r.iter_lines():
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            data = json.loads(payload)
            if data.get("usage"):
                entry.actual_prompt_tokens = data["usage"].get("prompt_tokens")
                entry.actual_output_tokens = data["usage"].get("completion_tokens")
            for choice in data.get("choices", []):
                piece = (choice.get("delta") or {}).get("content")
                if piece:
                    yield piece


class Embedder:
    """Dense embeddings. provider "none" disables them (retrieval falls back to BM25 only)."""

    def __init__(self, settings: Settings):
        self.s = settings
        self.enabled = settings.embed_provider != "none"

    def _prefix(self, kind: str) -> str:
        # nomic-embed-text is trained with task prefixes
        if "nomic" in self.s.embed_model:
            return "search_query: " if kind == "query" else "search_document: "
        return ""

    def embed(self, texts: list[str], kind: str = "document", batch: int = 16) -> list[list[float]]:
        if not self.enabled:
            return []
        out: list[list[float]] = []
        prefix = self._prefix(kind)
        for i in range(0, len(texts), batch):
            out.extend(self._embed_batch([prefix + t for t in texts[i:i + batch]]))
        return out

    def _embed_batch(self, part: list[str], attempts: int = 4) -> list[list[float]]:
        """One batch, retried with backoff: a local server that is busy swapping models can
        refuse transiently, and silently degrading the whole index to BM25 is worse than waiting."""
        last = ""
        for attempt in range(attempts):
            try:
                if self.s.embed_provider == "ollama":
                    r = httpx.post(f"{self.s.embed_base_url}/api/embed",
                                   json={"model": self.s.embed_model, "input": part, "truncate": True},
                                   timeout=self.s.request_timeout_s)
                    if r.status_code == 200:
                        return r.json()["embeddings"]
                elif self.s.embed_provider == "openai":
                    headers = {"Authorization": f"Bearer {self.s.embed_api_key}"} if self.s.embed_api_key else {}
                    r = httpx.post(f"{self.s.embed_base_url.rstrip('/')}/embeddings",
                                   json={"model": self.s.embed_model, "input": part},
                                   headers=headers, timeout=self.s.request_timeout_s)
                    if r.status_code == 200:
                        return [d["embedding"] for d in sorted(r.json()["data"], key=lambda d: d["index"])]
                else:
                    raise LLMError(f"Unknown EMBED_PROVIDER {self.s.embed_provider!r}")
                last = f"HTTP {r.status_code}: {r.text[:200]}"
            except httpx.HTTPError as e:
                last = str(e)
            time.sleep(2 * (attempt + 1))
        raise LLMError(f"Embedding request failed after {attempts} attempts: {last}")
