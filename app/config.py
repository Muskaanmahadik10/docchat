"""Runtime configuration, read from environment variables (and an optional .env file)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (KEY=VALUE lines); real environment variables win."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


@dataclass
class Settings:
    # LLM (text-to-text only). provider: "ollama" or "openai" (any OpenAI-compatible server).
    # Default: open-weights Qwen 3.8 27B on Groq (free API key), the configuration that answered all three
    # three core test questions completely. Offline alternative: qwen2.5:14b on Ollama (see .env.example).
    llm_provider: str = "openai"
    llm_model: str = "qwen/qwen3.8-27b"
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_api_key: str = ""
    # Embeddings. provider: "ollama", "openai" or "none" (BM25 only).
    embed_provider: str = "ollama"
    embed_model: str = "nomic-embed-text"
    embed_base_url: str = "http://localhost:11434"
    embed_api_key: str = ""
    # The design constraint: the LLM can see at most this many tokens per call
    # (prompt + generated answer). Every call is checked against it before it is sent.
    # Local cross-encoder reranker (FastEmbed/ONNX); "none" disables the second stage.
    rerank_model: str = "BAAI/bge-reranker-base"
    max_context_tokens: int = 10_000
    # The model's own context window, if known (0 = ask the backend / unknown). The effective
    # limit is min(max_context_tokens, model_context_tokens).
    model_context_tokens: int = 0
    answer_max_tokens: int = 1_800  # room for a full results table (Table 2 alone is 64 numbers)
    # Upper bound for retrieved evidence in one answer call (also capped by what is left of the context).
    retrieval_token_budget: int = 5_500
    # Chunking
    chunk_target_tokens: int = 350
    chunk_max_tokens: int = 500
    chunk_overlap_tokens: int = 50
    # Summary map-reduce
    summary_map_input_tokens: int = 4_000
    summary_map_output_tokens: int = 600
    summary_reduce_input_tokens: int = 5_000
    request_timeout_s: int = 1800
    # Provider limit on input tokens per minute (e.g. 7000 on Groq's free tier); 0 = no limit.
    # Calls wait client-side instead of colliding with HTTP 429.
    llm_input_tpm: int = 7000  # Groq free tier; set 0 for a local model
    # Build the map-reduce summary notes right after upload. Off by default: the notes are built on
    # the first summary request and cached, so a user who never asks for a summary costs no LLM calls.
    summary_on_upload: bool = False
    data_dir: Path = field(default_factory=lambda: ROOT / "data")

    @classmethod
    def from_env(cls) -> "Settings":
        _load_dotenv(ROOT / ".env")
        e = os.environ.get
        s = cls(
            llm_provider=e("LLM_PROVIDER", cls.llm_provider),
            llm_model=e("LLM_MODEL", cls.llm_model),
            llm_base_url=e("LLM_BASE_URL", cls.llm_base_url),
            llm_api_key=e("LLM_API_KEY", ""),
            embed_provider=e("EMBED_PROVIDER", cls.embed_provider),
            embed_model=e("EMBED_MODEL", cls.embed_model),
            # embeddings stay on local Ollama unless pointed elsewhere (a cloud LLM key does not
            # imply a cloud embedding endpoint)
            embed_base_url=e("EMBED_BASE_URL", e("LLM_BASE_URL", cls.embed_base_url)
                             if e("EMBED_PROVIDER", cls.embed_provider) == e("LLM_PROVIDER", cls.llm_provider)
                             else cls.embed_base_url),
            embed_api_key=e("EMBED_API_KEY", e("LLM_API_KEY", "")),
            rerank_model=e("RERANK_MODEL", cls.rerank_model),
            max_context_tokens=_int("MAX_CONTEXT_TOKENS", cls.max_context_tokens),
            model_context_tokens=_int("MODEL_CONTEXT_TOKENS", cls.model_context_tokens),
            answer_max_tokens=_int("ANSWER_MAX_TOKENS", cls.answer_max_tokens),
            retrieval_token_budget=_int("RETRIEVAL_TOKEN_BUDGET", cls.retrieval_token_budget),
            chunk_target_tokens=_int("CHUNK_TARGET_TOKENS", cls.chunk_target_tokens),
            chunk_max_tokens=_int("CHUNK_MAX_TOKENS", cls.chunk_max_tokens),
            chunk_overlap_tokens=_int("CHUNK_OVERLAP_TOKENS", cls.chunk_overlap_tokens),
            summary_map_input_tokens=_int("SUMMARY_MAP_INPUT_TOKENS", cls.summary_map_input_tokens),
            summary_map_output_tokens=_int("SUMMARY_MAP_OUTPUT_TOKENS", cls.summary_map_output_tokens),
            summary_reduce_input_tokens=_int("SUMMARY_REDUCE_INPUT_TOKENS", cls.summary_reduce_input_tokens),
            request_timeout_s=_int("REQUEST_TIMEOUT_S", cls.request_timeout_s),
            llm_input_tpm=_int("LLM_INPUT_TPM", cls.llm_input_tpm),
            summary_on_upload=e("SUMMARY_ON_UPLOAD", "false").lower() in ("1", "true", "yes"),
        )
        if e("DATA_DIR"):
            s.data_dir = Path(e("DATA_DIR"))
        return s
