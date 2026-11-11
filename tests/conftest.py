import json
import os
from pathlib import Path

import pytest

from app.config import Settings
from app.llm import LLM, Embedder
from app.tokens import LedgerEntry

PAPER_CANDIDATES = [
    os.environ.get("DOCCHAT_TEST_PDF", ""),
    str(Path(__file__).resolve().parent.parent / "samples" / "2410.05779.pdf"),
]


@pytest.fixture
def paper_pdf() -> Path:
    for p in PAPER_CANDIDATES:
        if p and Path(p).exists():
            return Path(p)
    pytest.skip("LightRAG paper not found; set DOCCHAT_TEST_PDF or put it in samples/2410.05779.pdf")


class FakeLLM(LLM):
    """Records every prompt; still goes through the real ContextBudget check."""

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.calls: list[tuple[str, list[dict]]] = []

    def stream(self, messages, *, max_tokens, ledger, call, temperature=0.2, json_schema=None):
        est = self.budget.check(messages, max_tokens)
        entry = ledger.add(LedgerEntry(call=call, est_prompt_tokens=est, max_output_tokens=max_tokens,
                                       context_limit=self.context_limit))
        self.calls.append((call, messages))
        if json_schema:
            text = getattr(self, "plan_raw", None) or json.dumps(
                getattr(self, "plan_response", None) or {"intent": "question", "queries": ["ablation variants", "table results"]})
        elif call.startswith("summary-map") or call.startswith("summary-reduce"):
            text = "- note: " + messages[-1]["content"][:200].replace("\n", " ")
        else:
            text = "Answer citing [S1]."
        entry.actual_prompt_tokens, entry.actual_output_tokens, entry.seconds = est, 10, 0.0
        yield text


class FakeEmbedder(Embedder):
    """Deterministic bag-of-letters vectors: enough to exercise the dense path."""

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.enabled = True

    def embed(self, texts, kind="document", batch=16):
        out = []
        for t in texts:
            v = [0.0] * 26
            for ch in t.lower():
                if "a" <= ch <= "z":
                    v[ord(ch) - 97] += 1
            out.append(v)
        return out


@pytest.fixture
def settings(tmp_path) -> Settings:
    s = Settings()
    s.data_dir = tmp_path / "data"
    s.model_context_tokens = 32_768  # don't query a live backend in tests
    s.rerank_model = "none"  # don't download the cross-encoder in unit tests
    s.llm_input_tpm = 0  # the fake LLM has no provider rate limit
    return s
