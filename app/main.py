"""FastAPI app: upload documents, chat over them (Server-Sent Events), inspect chunks."""
from __future__ import annotations

import json
import traceback
from collections.abc import Iterator

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import metrics
from .config import ROOT, Settings
from .llm import LLM, Embedder
from .pipeline import Pipeline
from .store import DocumentStore

MAX_UPLOAD_BYTES = 50 * 1024 * 1024


class ChatTurn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    doc_id: str
    doc_ids: list[str] = []  # extra documents to chat with at the same time
    message: str = Field(min_length=1, max_length=4000)
    history: list[ChatTurn] = []


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def create_app(settings: Settings | None = None, llm: LLM | None = None,
               embedder: Embedder | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    llm = llm or LLM(settings)
    embedder = embedder or Embedder(settings)
    store = DocumentStore(settings, llm, embedder)
    pipeline = Pipeline(settings, llm, embedder, store)

    app = FastAPI(title="DocChat", version="1.0")
    app.state.store, app.state.pipeline, app.state.settings = store, pipeline, settings

    @app.get("/metrics", response_class=PlainTextResponse)
    def prometheus_metrics():
        """Prometheus scrape endpoint (questions, LLM calls, tokens, latency, rate-limit waits)."""
        metrics.set_gauge("docchat_documents", len(store.docs))
        return PlainTextResponse(metrics.render(), media_type="text/plain; version=0.0.4")

    @app.get("/api/health")
    def health():
        return {
            "llm": {"provider": settings.llm_provider, "model": settings.llm_model, **llm.health()},
            "embeddings": {"provider": settings.embed_provider, "model": settings.embed_model},
            "rerank": {"model": settings.rerank_model, "enabled": pipeline.reranker.enabled,
                       "error": pipeline.reranker.error},
            "max_context_tokens": llm.context_limit,
            "answer_max_tokens": settings.answer_max_tokens,
        }

    @app.get("/api/documents")
    def list_documents():
        return store.list()

    @app.post("/api/documents")
    async def upload(file: UploadFile = File(...)):
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "File too large (max 50 MB)")
        if not data:
            raise HTTPException(400, "Empty file")
        try:
            rec = store.add(file.filename or "document", data)
        except ValueError as e:
            raise HTTPException(400, str(e))
        return rec.public()

    def _get(doc_id: str):
        rec = store.get(doc_id)
        if not rec:
            raise HTTPException(404, "Document not found")
        return rec

    @app.get("/api/documents/{doc_id}")
    def get_document(doc_id: str):
        return _get(doc_id).public()

    @app.get("/api/documents/{doc_id}/chunks")
    def get_chunks(doc_id: str):
        rec = _get(doc_id)
        return [c.to_dict() for c in rec.chunks]

    @app.get("/api/documents/{doc_id}/summary")
    def get_summary(doc_id: str):
        rec = _get(doc_id)
        return {"status": rec.summary_status, "stage": rec.summary_stage, "notes": rec.notes,
                "ledger": rec.summary_ledger}

    @app.delete("/api/documents/{doc_id}")
    def delete_document(doc_id: str):
        if not store.delete(doc_id):
            raise HTTPException(404, "Document not found")
        return {"deleted": doc_id}

    @app.post("/api/chat")
    def chat(req: ChatRequest):
        rec = _get(req.doc_id)
        if rec.status != "ready":
            raise HTTPException(409, f"Document is not ready yet ({rec.stage})")
        history = [t.model_dump() for t in req.history]
        extra = [_get(i) for i in dict.fromkeys(req.doc_ids) if i != req.doc_id]
        if any(r.status != "ready" for r in extra):
            raise HTTPException(409, "One of the selected documents is not ready yet")
        recs = [rec, *extra]

        def events() -> Iterator[str]:
            try:
                stream = (pipeline.answer(rec, req.message.strip(), history) if len(recs) == 1
                          else pipeline.answer_multi(recs, req.message.strip(), history))
                for event, data in stream:
                    yield _sse(event, data)
            except Exception as e:
                traceback.print_exc()
                yield _sse("error", {"text": f"{type(e).__name__}: {e}"})

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/api/retrieve")
    def retrieve(req: ChatRequest):
        """Evidence the answer call would receive (no LLM call): for debugging and retrieval evals."""
        rec = _get(req.doc_id)
        if rec.status != "ready":
            raise HTTPException(409, f"Document is not ready yet ({rec.stage})")
        ev = pipeline.gather_evidence(rec, req.message.strip(), [], [])
        return {k: ev[k] for k in ("sources", "retrieval", "queries", "evidence_tokens", "evidence_budget")}

    @app.middleware("http")
    async def no_stale_ui(request, call_next):
        """Browsers must revalidate the page and its scripts, or an updated build runs stale JS."""
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    static = ROOT / "static"
    app.mount("/static", StaticFiles(directory=static), name="static")

    @app.get("/")
    def index():
        return FileResponse(static / "index.html")

    return app
