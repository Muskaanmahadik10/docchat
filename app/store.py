"""Document registry: ingestion (parse -> chunk -> embed -> summary notes) and persistence.

Each document lives in data/docs/<sha256-prefix>/ so re-uploading the same file reuses the
cached index and summary notes.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .chunker import Chunk, chunk_document
from .config import Settings
from .index import HybridIndex
from .llm import LLM, Embedder
from .citations import citation_stats, format_stats
from .doccard import build_doc_card
from .parser import SUPPORTED, parse_document
from .summary import build_summary_notes
from .tokens import Ledger


@dataclass
class DocRecord:
    id: str
    filename: str
    sha256: str = ""
    title: str = ""
    status: str = "processing"  # processing | ready | error
    stage: str = "queued"
    error: str = ""
    num_pages: int = 0
    total_tokens: int = 0
    created: float = field(default_factory=time.time)
    retrieval: str = "hybrid"  # hybrid | bm25
    card: str = ""  # document card (title, authors, abstract, outline, captions): see doccard.py
    citations: str = ""  # most-cited references, counted by code over the whole document
    page_texts: list = field(default_factory=list, repr=False)  # [(page, text)], no overlap: for term counts
    summary_status: str = "pending"  # pending | running | ready | error
    summary_stage: str = ""
    summary_ledger: list = field(default_factory=list)
    summary_wanted: bool = False  # a user is waiting for the notes: build at interactive priority
    chunks: list[Chunk] = field(default_factory=list, repr=False)
    index: HybridIndex | None = field(default=None, repr=False)
    notes: list[dict] = field(default_factory=list, repr=False)
    summary_done: threading.Event = field(default_factory=threading.Event, repr=False)

    def public(self) -> dict:
        return {
            "id": self.id, "filename": self.filename, "sha256": self.sha256, "title": self.title, "status": self.status,
            "stage": self.stage, "error": self.error, "num_pages": self.num_pages,
            "num_chunks": len(self.chunks), "num_tables": sum(c.kind == "table" for c in self.chunks),
            "total_tokens": self.total_tokens, "retrieval": self.retrieval,
            "card": self.card, "summary_status": self.summary_status, "summary_stage": self.summary_stage,
            "created": self.created,
        }


def _page_texts(parsed) -> list[tuple[int, str]]:
    pages: dict[int, list[str]] = {}
    for b in parsed.blocks:
        pages.setdefault(b.page, []).append(b.text)
    return [(p, "\n".join(t)) for p, t in sorted(pages.items())]


class DocumentStore:
    def __init__(self, settings: Settings, llm: LLM, embedder: Embedder):
        self.s, self.llm, self.embedder = settings, llm, embedder
        self.root = settings.data_dir / "docs"
        self.root.mkdir(parents=True, exist_ok=True)
        self.docs: dict[str, DocRecord] = {}
        self.lock = threading.Lock()
        self._load_existing()

    # -- persistence ----------------------------------------------------------------------

    def _dir(self, doc_id: str) -> Path:
        return self.root / doc_id

    def _load_existing(self) -> None:
        for d in sorted(self.root.iterdir()):
            meta_path = d / "meta.json"
            if not meta_path.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                rec = DocRecord(id=meta["id"], filename=meta["filename"], sha256=meta.get("sha256", ""), title=meta.get("title", ""),
                                num_pages=meta.get("num_pages", 0), total_tokens=meta.get("total_tokens", 0),
                                created=meta.get("created", time.time()), retrieval=meta.get("retrieval", "hybrid"))
                rec.card, rec.citations = meta.get("card", ""), meta.get("citations", "")
                pages_path = d / "pages.json"
                if "card" not in meta or "citations" not in meta or not pages_path.exists():
                    src = next(d.glob("source.*"), None)  # stored by an older version: rebuild
                    if src:
                        parsed = parse_document(src)
                        rec.card, rec.citations = build_doc_card(parsed), format_stats(citation_stats(parsed))
                        meta.update(card=rec.card, citations=rec.citations)
                        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
                        pages_path.write_text(json.dumps(_page_texts(parsed), ensure_ascii=False), encoding="utf-8")
                if pages_path.exists():
                    rec.page_texts = [tuple(x) for x in json.loads(pages_path.read_text(encoding="utf-8"))]
                if not rec.sha256:
                    src = next(d.glob("source.*"), None)
                    rec.sha256 = hashlib.sha256(src.read_bytes()).hexdigest() if src else ""
                rec.chunks = [Chunk(**c) for c in json.loads((d / "chunks.json").read_text(encoding="utf-8"))]
                emb_path = d / "embeddings.npy"
                emb = np.load(emb_path) if emb_path.exists() else None
                made_by = meta.get("embed_model", self.s.embed_model)  # older indexes: assume current model
                if emb is not None and self.embedder.enabled and made_by != self.s.embed_model:
                    # vectors from another embedding model are not comparable: re-embed with the current one
                    try:
                        emb = np.asarray(self.embedder.embed([c.search_text() for c in rec.chunks]), dtype=np.float32)
                        np.save(emb_path, emb)
                        meta["embed_model"] = self.s.embed_model
                        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
                    except Exception as e:
                        print(f"[docchat] re-embedding {rec.filename} failed, using BM25 only: {e}")
                        emb, rec.retrieval = None, "bm25"
                rec.index = HybridIndex(rec.chunks, emb)
                rec.status, rec.stage = "ready", "ready"
                summary_path = d / "summary.json"
                if summary_path.exists():
                    data = json.loads(summary_path.read_text(encoding="utf-8"))
                    if data.get("key") == self._summary_key():
                        rec.notes, rec.summary_ledger = data["notes"], data.get("ledger", [])
                        rec.summary_status = "ready"
                        rec.summary_done.set()
                self.docs[rec.id] = rec
                if rec.summary_status != "ready" and self.s.summary_on_upload:
                    threading.Thread(target=self._build_summary, args=(rec,), daemon=True).start()
            except Exception:
                traceback.print_exc()

    def _summary_key(self) -> str:
        s = self.s
        return f"{s.llm_provider}|{s.llm_model}|{s.max_context_tokens}|{s.summary_map_input_tokens}|{s.summary_map_output_tokens}|{s.summary_reduce_input_tokens}|v2"

    # -- public API -----------------------------------------------------------------------

    def list(self) -> list[dict]:
        return [d.public() for d in sorted(self.docs.values(), key=lambda d: d.created, reverse=True)]

    def get(self, doc_id: str) -> DocRecord | None:
        return self.docs.get(doc_id)

    def delete(self, doc_id: str) -> bool:
        with self.lock:
            rec = self.docs.pop(doc_id, None)
        if rec:
            shutil.rmtree(self._dir(doc_id), ignore_errors=True)
        return rec is not None

    def add(self, filename: str, data: bytes) -> DocRecord:
        suffix = Path(filename).suffix.lower()
        if suffix not in SUPPORTED:
            raise ValueError(f"Unsupported file type {suffix!r}. Upload a PDF, TXT or MD file.")
        sha = hashlib.sha256(data).hexdigest()
        doc_id = sha[:16]
        with self.lock:
            existing = self.docs.get(doc_id)
            if existing and existing.status != "error":
                return existing
            rec = DocRecord(id=doc_id, filename=filename, sha256=sha)
            self.docs[doc_id] = rec
        d = self._dir(doc_id)
        d.mkdir(parents=True, exist_ok=True)
        (d / f"source{suffix}").write_bytes(data)
        threading.Thread(target=self._ingest, args=(rec, d / f"source{suffix}"), daemon=True).start()
        return rec

    # -- ingestion ------------------------------------------------------------------------

    def _ingest(self, rec: DocRecord, path: Path) -> None:
        try:
            rec.stage = "parsing (local, no LLM)"
            parsed = parse_document(path)
            rec.title = parsed.title or Path(rec.filename).stem
            rec.num_pages = parsed.num_pages
            rec.card = build_doc_card(parsed)
            rec.citations = format_stats(citation_stats(parsed))
            rec.page_texts = _page_texts(parsed)
            (self._dir(rec.id) / "pages.json").write_text(json.dumps(rec.page_texts, ensure_ascii=False), encoding="utf-8")
            rec.stage = "chunking"
            chunks = chunk_document(parsed, self.s.chunk_target_tokens, self.s.chunk_max_tokens,
                                    self.s.chunk_overlap_tokens)
            if not chunks:
                raise ValueError("No extractable text found (is it a scanned PDF without a text layer?)")
            rec.total_tokens = sum(c.tokens for c in chunks)
            emb = None
            if self.embedder.enabled:
                rec.stage = f"embedding {len(chunks)} chunks"
                try:
                    emb = np.asarray(self.embedder.embed([c.search_text() for c in chunks]), dtype=np.float32)
                except Exception as e:  # degrade gracefully to keyword retrieval
                    print(f"[docchat] embeddings unavailable, using BM25 only: {e}")
                    emb = None
            rec.retrieval = "hybrid" if emb is not None else "bm25"
            d = self._dir(rec.id)
            (d / "chunks.json").write_text(json.dumps([c.to_dict() for c in chunks], ensure_ascii=False, indent=1), encoding="utf-8")
            if emb is not None:
                np.save(d / "embeddings.npy", emb)
            rec.chunks, rec.index = chunks, HybridIndex(chunks, emb)
            (d / "meta.json").write_text(json.dumps({
                "id": rec.id, "filename": rec.filename, "sha256": rec.sha256, "title": rec.title, "num_pages": rec.num_pages,
                "total_tokens": rec.total_tokens, "created": rec.created, "retrieval": rec.retrieval,
                "card": rec.card, "citations": rec.citations,
                "embed_model": self.s.embed_model if emb is not None else None,
            }, ensure_ascii=False, indent=1), encoding="utf-8")
            rec.status, rec.stage = "ready", "ready"
        except Exception as e:
            traceback.print_exc()
            rec.status, rec.stage, rec.error = "error", "error", str(e)
            rec.summary_status = "error"
            rec.summary_done.set()
            return
        if self.s.summary_on_upload:
            self._build_summary(rec)

    _summary_slot = threading.Semaphore(1)

    def _build_summary(self, rec: DocRecord) -> None:
        rec.summary_status, rec.summary_stage = "running", "queued behind another summary"
        with self._summary_slot:
            self._build_summary_now(rec)

    def _build_summary_now(self, rec: DocRecord) -> None:
        rec.summary_stage = "starting"
        ledger = Ledger()
        try:
            def progress(msg: str) -> None:
                rec.summary_stage = msg
            notes = build_summary_notes(rec.chunks, self.llm, self.s, ledger, progress,
                                        cache_path=self._dir(rec.id) / "summary_calls.json",
                                        background=lambda: not rec.summary_wanted)
            rec.notes, rec.summary_ledger = notes, ledger.to_list()
            (self._dir(rec.id) / "summary.json").write_text(json.dumps(
                {"key": self._summary_key(), "notes": notes, "ledger": rec.summary_ledger},
                ensure_ascii=False, indent=1), encoding="utf-8")
            rec.summary_status, rec.summary_stage = "ready", "ready"
        except Exception as e:
            traceback.print_exc()
            rec.summary_ledger = ledger.to_list()
            rec.summary_status, rec.summary_stage = "error", str(e)
        finally:
            rec.summary_done.set()

    def retry_summary(self, rec: DocRecord) -> None:
        with self.lock:
            if rec.summary_status not in ("error", "pending") or rec.status != "ready":
                return
            rec.summary_status = "running"
            rec.summary_done.clear()
            threading.Thread(target=self._build_summary, args=(rec,), daemon=True).start()
