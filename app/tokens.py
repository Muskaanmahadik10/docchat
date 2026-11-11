"""Token counting and the per-call context budget.

The design assumes the LLM's max context is 10K tokens. We enforce that at a single
choke point: every LLM call goes through `ContextBudget.check()` before it is sent, and the
server-reported prompt token count is recorded afterwards in a `Ledger` so the limit can be
verified, not just assumed.

Counting uses tiktoken's cl100k_base. The served model (e.g. Qwen) has its own tokenizer, so
estimates are multiplied by SAFETY_FACTOR; the ledger compares estimate vs. actual.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

try:
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:  # offline / not installed: fall back to a conservative heuristic
    _ENC = None

SAFETY_FACTOR = 1.15
MESSAGE_OVERHEAD = 8  # chat-template tokens per message (role markers etc.)

_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")


def count_tokens(text: str) -> int:
    if not text:
        return 0
    if _ENC is not None:
        return len(_ENC.encode(text, disallowed_special=()))
    # Fallback without tiktoken: err high. cl100k spends ~1.6 tokens per Traditional Chinese character
    # and ~1 per 4 Latin characters, so count 2 per CJK character and 1 per 3 other characters.
    cjk = len(_CJK.findall(text))
    return 2 * cjk + (len(text) - cjk) // 3 + 1


def truncate_tokens(text: str, max_tokens: int) -> str:
    """Cut text to at most max_tokens (estimated), on a word boundary when possible."""
    if count_tokens(text) <= max_tokens:
        return text
    if _ENC is not None:
        ids = _ENC.encode(text, disallowed_special=())[:max_tokens]
        cut = _ENC.decode(ids)
    else:
        cut = text[: max_tokens * 3]
        while len(cut) > 1 and count_tokens(cut) > max_tokens:  # CJK text holds fewer characters per token
            cut = cut[: int(len(cut) * 0.9)]
    space = cut.rfind(" ")
    if space > len(cut) * 0.8:
        cut = cut[:space]
    return cut.rstrip() + " …"


def messages_tokens(messages: list[dict]) -> int:
    return sum(count_tokens(m["content"]) + MESSAGE_OVERHEAD for m in messages) + 4


class ContextOverflow(RuntimeError):
    pass


@dataclass
class ContextBudget:
    max_context: int

    def prompt_allowance(self, max_output: int) -> int:
        """Largest raw (pre-safety-factor) prompt token estimate that still fits with max_output."""
        return int((self.max_context - max_output) / SAFETY_FACTOR)

    def check(self, messages: list[dict], max_output: int) -> int:
        est = messages_tokens(messages)
        if est * SAFETY_FACTOR + max_output > self.max_context:
            raise ContextOverflow(
                f"Prompt ~{est} tokens (x{SAFETY_FACTOR} safety) + {max_output} output tokens "
                f"exceeds the {self.max_context}-token context limit"
            )
        return est


@dataclass
class LedgerEntry:
    call: str
    est_prompt_tokens: int
    max_output_tokens: int
    context_limit: int
    actual_prompt_tokens: int | None = None
    actual_output_tokens: int | None = None
    seconds: float | None = None

    @property
    def within_limit(self) -> bool:
        prompt = self.actual_prompt_tokens or int(self.est_prompt_tokens * SAFETY_FACTOR)
        output = self.actual_output_tokens if self.actual_output_tokens is not None else self.max_output_tokens
        return prompt + output <= self.context_limit

    def to_dict(self) -> dict:
        return {
            "call": self.call,
            "est_prompt_tokens": self.est_prompt_tokens,
            "actual_prompt_tokens": self.actual_prompt_tokens,
            "actual_output_tokens": self.actual_output_tokens,
            "max_output_tokens": self.max_output_tokens,
            "context_limit": self.context_limit,
            "within_limit": self.within_limit,
            "seconds": round(self.seconds, 1) if self.seconds is not None else None,
        }


@dataclass
class Ledger:
    entries: list[LedgerEntry] = field(default_factory=list)

    def add(self, entry: LedgerEntry) -> LedgerEntry:
        self.entries.append(entry)
        return entry

    def to_list(self) -> list[dict]:
        return [e.to_dict() for e in self.entries]
