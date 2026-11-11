"""Hybrid retrieval: BM25 (exact terms, numbers, names) + dense embeddings (meaning),
fused with Reciprocal Rank Fusion over every (query x retriever) ranking."""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

from .chunker import Chunk

_WORD = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")
# scripts written without spaces (Chinese, Japanese kana): character unigrams and bigrams
_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿]")
# all other letters (accented Latin, Cyrillic, Greek, Arabic, Hangul, ...): whole words
_OTHER_WORD = re.compile(r"[^\W\d_぀-ヿ㐀-䶿一-鿿]+")
_STOP = set("""a an and are as at be by for from has have how in is it its of on or that the this
to was were what when where which who why will with do does did can could should would about into
than then there these those their they them we our you your i me my not no""".split())
REFERENCE_SECTIONS = re.compile(r"reference|bibliograph|acknowledg", re.I)
# questions about the bibliography itself must not have the reference list down-weighted
ABOUT_REFERENCES = re.compile(r"cit(e|ed|es|ing|ation)|reference|bibliograph|引用|參考文獻|参考文献", re.I)


def tokenize(text: str) -> list[str]:
    text = text.lower()
    toks = [t for t in _WORD.findall(text) if t not in _STOP]
    # split hyphenated/dotted compounds too, so "-high" matches "high"
    toks += [p for t in toks if ("-" in t or "." in t) for p in re.split(r"[.\-]", t) if p and p not in _STOP]
    cjk = _CJK.findall(text)
    toks += cjk + [a + b for a, b in zip(cjk, cjk[1:])]
    toks += [w for w in _OTHER_WORD.findall(text) if not w.isascii() and w not in _STOP]
    return toks


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = [len(d) for d in docs]
        self.avg = (sum(self.len) / len(docs)) if docs else 0.0
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> np.ndarray:
        out = np.zeros(len(self.tf))
        for i, tf in enumerate(self.tf):
            s = 0.0
            for t in set(query):
                f = tf.get(t)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.len[i] / (self.avg or 1))
                s += self.idf[t] * f * (self.k1 + 1) / denom
            out[i] = s
        return out


class HybridIndex:
    def __init__(self, chunks: list[Chunk], embeddings: np.ndarray | None):
        self.chunks = chunks
        self.bm25 = BM25([tokenize(c.search_text()) for c in chunks])
        self.emb = None
        if embeddings is not None and len(embeddings):
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            self.emb = embeddings / np.maximum(norms, 1e-9)
        self.penalty = np.array([0.5 if REFERENCE_SECTIONS.search(c.section) else 1.0 for c in chunks])

    def search(self, queries: list[str], query_embeddings: list[list[float]] | None = None,
               top_k: int = 12, per_list: int = 20, rrf_k: int = 60) -> list[tuple[int, float]]:
        """Return [(chunk_index, fused_score)] best first."""
        fused = np.zeros(len(self.chunks))
        rankings: list[np.ndarray] = []
        for q in queries:
            s = self.bm25.scores(tokenize(q))
            if s.max() > 0:
                order = np.argsort(-s)
                rankings.append(order[: min(per_list, int((s > 0).sum()))])
        if self.emb is not None and query_embeddings:
            for qe in query_embeddings:
                v = np.asarray(qe, dtype=np.float32)
                v = v / max(np.linalg.norm(v), 1e-9)
                rankings.append(np.argsort(-(self.emb @ v))[:per_list])
        for order in rankings:
            for rank, idx in enumerate(order):
                fused[idx] += 1.0 / (rrf_k + rank + 1)
        if not any(ABOUT_REFERENCES.search(q) for q in queries):
            fused *= self.penalty
        best = [int(i) for i in np.argsort(-fused) if fused[i] > 0][:top_k]
        return [(i, float(fused[i])) for i in best]
