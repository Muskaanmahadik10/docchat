"""MCP server: every tool declares all four annotations, they match the behaviour, and no tool calls the LLM."""
import asyncio
import json
import time

from app import mcp_server
from app.llm import LLM
from app.pipeline import Pipeline
from app.store import DocumentStore

from .conftest import FakeEmbedder, FakeLLM

HINTS = ("read_only_hint", "destructive_hint", "idempotent_hint", "open_world_hint")
DOC = """# Introduction
LightRAG is a retrieval-augmented generation framework that uses graph structures.

# Evaluation
Compared with GraphRAG, LightRAG needs fewer than 100 tokens and a single API call for retrieval.

# Ablation
Removing high-level retrieval (the -High variant) reduces performance across datasets.
"""


def _tools():
    return {t.name: t for t in asyncio.run(mcp_server.mcp.list_tools())}


def _runtime(settings):
    settings = mcp_server._mcp_settings(settings)
    llm, emb = FakeLLM(settings), FakeEmbedder(settings)
    store = DocumentStore(settings, llm, emb)
    rec = store.add("lightrag.md", DOC.encode())
    for _ in range(200):
        if store.get(rec.id).status == "ready":
            break
        time.sleep(0.05)
    mcp_server.configure(store, Pipeline(settings, llm, emb, store))
    return rec.id, llm


def test_every_tool_declares_all_four_hints_and_is_read_only():
    tools = _tools()
    assert set(tools) == {"list_documents", "get_document_card", "search_document"}
    for name, tool in tools.items():
        a = tool.annotations
        assert a is not None, name
        for hint in HINTS:
            assert isinstance(getattr(a, hint), bool), f"{name} is missing {hint}"
        assert (a.read_only_hint, a.destructive_hint, a.idempotent_hint, a.open_world_hint) == (True, False, True, False), name


def test_tools_return_cited_passages_without_calling_the_llm(settings):
    doc_id, llm = _runtime(settings)
    docs = json.loads(mcp_server.list_documents())
    assert [d["id"] for d in docs] == [doc_id] and docs[0]["status"] == "ready"
    card = json.loads(mcp_server.get_document_card(doc_id))
    assert card["id"] == doc_id and "LightRAG" in card["card"]
    hits = json.loads(mcp_server.search_document(doc_id, "What happens without high-level retrieval?", 3))
    assert 1 <= len(hits["passages"]) <= 3
    assert all(p["id"].startswith("S") and p["pages"] and p["section"] for p in hits["passages"])
    assert any("-High" in p["text"] for p in hits["passages"])
    assert llm.calls == []  # read-only: retrieval only, no generation


def test_bad_input_is_rejected_before_any_lookup(settings):
    doc_id, _ = _runtime(settings)
    for bad in ("../etc/passwd", "ABC", "", doc_id + "0"):
        assert "error" in json.loads(mcp_server.get_document_card(bad))
    assert "error" in json.loads(mcp_server.search_document(doc_id, "   "))
    assert "error" in json.loads(mcp_server.search_document("0" * 16, "anything"))


def test_mcp_settings_never_start_summaries_or_query_the_provider(settings):
    settings.summary_on_upload, settings.model_context_tokens = True, 0
    s = mcp_server._mcp_settings(settings)
    assert s.summary_on_upload is False and s.model_context_tokens == s.max_context_tokens
    assert LLM(s).context_limit == s.max_context_tokens
