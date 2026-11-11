"""Prometheus metrics in the text exposition format, without extra dependencies.

Served at /metrics; scrape with Prometheus and chart in Grafana. Thread-safe: LLM calls and summary
jobs run in worker threads.
"""
from __future__ import annotations

import threading
from collections import defaultdict

_lock = threading.Lock()
_counters: dict[tuple[str, tuple], float] = defaultdict(float)
_HELP = {
    "docchat_questions_total": ("counter", "Questions answered, by route"),
    "docchat_llm_calls_total": ("counter", "LLM calls, by call type"),
    "docchat_llm_tokens_total": ("counter", "Tokens reported by the LLM provider, by kind"),
    "docchat_llm_call_seconds_total": ("counter", "Total seconds spent in LLM calls"),
    "docchat_llm_errors_total": ("counter", "Failed LLM calls"),
    "docchat_rate_limit_wait_seconds_total": ("counter", "Seconds spent waiting for the provider's per-minute token limit"),
    "docchat_llm_output_cap_reduced_total": ("counter", "Calls retried with a lower max_tokens because the provider capped output"),
    "docchat_context_limit_tokens": ("gauge", "Effective context limit per LLM call"),
    "docchat_largest_call_tokens": ("gauge", "Largest prompt + output seen in one call since start"),
    "docchat_documents": ("gauge", "Documents loaded"),
}


def inc(name: str, value: float = 1.0, **labels: str) -> None:
    with _lock:
        _counters[(name, tuple(sorted(labels.items())))] += value


def set_gauge(name: str, value: float, **labels: str) -> None:
    with _lock:
        _counters[(name, tuple(sorted(labels.items())))] = value


def max_gauge(name: str, value: float) -> None:
    with _lock:
        key = (name, ())
        _counters[key] = max(_counters.get(key, 0.0), value)


def render() -> str:
    with _lock:
        items = sorted(_counters.items())
    lines, seen = [], set()
    for (name, labels), value in items:
        if name not in seen:
            kind, help_text = _HELP.get(name, ("untyped", name))
            lines += [f"# HELP {name} {help_text}", f"# TYPE {name} {kind}"]
            seen.add(name)
        label_str = ",".join(f'{k}="{v}"' for k, v in labels)
        lines.append(f"{name}{{{label_str}}} {value:g}" if label_str else f"{name} {value:g}")
    return "\n".join(lines) + "\n"


def reset() -> None:  # tests
    with _lock:
        _counters.clear()
