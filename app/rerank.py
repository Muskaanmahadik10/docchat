"""Second-stage reranking with a local cross-encoder (FastEmbed / ONNX, CPU).

Hybrid search (BM25 + embeddings) is good at recall but returns loosely related chunks too.
A cross-encoder reads (question, chunk) together and scores relevance much more precisely, so
the answer prompt gets the few chunks that matter instead of a dozen that might. Smaller,
cleaner evidence helps every model, and small models most of all.
"""
from __future__ import annotations

import threading

from .tokens import truncate_tokens


class Reranker:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.enabled = bool(model_name) and model_name.lower() != "none"
        self._model = None
        self._lock = threading.Lock()
        self.error = ""

    def _load(self):
        with self._lock:
            if self._model is None:
                import os
                from pathlib import Path
                from fastembed.rerank.cross_encoder import TextCrossEncoder
                # persistent (FastEmbed's default lives in %TEMP%) and short: Hugging Face cache paths
                # are deeply nested and break Windows' 260-character limit inside long project paths
                cache = os.environ.get("MODEL_CACHE_DIR") or str(Path.home() / ".cache" / "docchat" / "models")
                self._model = TextCrossEncoder(model_name=self.model_name, cache_dir=cache)
        return self._model

    def score(self, query: str, texts: list[str]) -> list[float] | None:
        """Relevance logits (higher = more relevant), or None if reranking is unavailable."""
        if not self.enabled or not texts:
            return None
        try:
            model = self._load()
            # bge-reranker-base reads 512 tokens: keep the head of each chunk (captions/headers first)
            return [float(s) for s in model.rerank(query, [truncate_tokens(t, 380) for t in texts])]
        except Exception as e:  # missing package / offline model download: fall back to fused ranking
            self.error = f"{type(e).__name__}: {e}"
            self.enabled = False
            return None


def select(scored: list[tuple[int, float]], max_keep: int = 8, min_keep: int = 3, drop: float = 5.0) -> list[int]:
    """Keep the strongest chunks: everything within `drop` logits of the best one (or above 0),
    at least `min_keep`, at most `max_keep`."""
    ranked = sorted(scored, key=lambda x: -x[1])
    if not ranked:
        return []
    top = ranked[0][1]
    # nothing clearly relevant (best score negative): keep only the few closest to the best
    floor = top - drop if top > 0 else top - drop / 2
    keep = [i for i, s in ranked if s >= min(floor, 0.0) and (top > 0 or s >= floor)]
    if len(keep) < min_keep:
        keep = [i for i, _ in ranked[:min_keep]]
    return keep[:max_keep]
