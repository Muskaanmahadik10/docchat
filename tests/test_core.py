"""Unit tests: token budget, table linearisation, chunking, retrieval, summary map-reduce."""
import pytest

from app.chunker import chunk_document
from app.index import HybridIndex, tokenize
from app.parser import Block, ParsedDoc, linearize_table
from app.summary import build_summary_notes
from app.tokens import ContextBudget, ContextOverflow, Ledger, count_tokens, truncate_tokens

from .conftest import FakeLLM


# ---------------------------------------------------------------- tokens

def test_budget_rejects_oversized_prompt():
    budget = ContextBudget(10_000)
    small = [{"role": "user", "content": "hello " * 100}]
    assert budget.check(small, max_output=1000) > 0
    huge = [{"role": "user", "content": "word " * 9_000}]
    with pytest.raises(ContextOverflow):
        budget.check(huge, max_output=1000)


@pytest.mark.parametrize("counter", ["tiktoken", "fallback"])
def test_prompt_allowance_is_safe(counter, monkeypatch):
    """A prompt sized to the allowance passes the gate, with tiktoken and with the fallback counter."""
    import app.tokens as tokens
    if counter == "fallback":
        monkeypatch.setattr(tokens, "_ENC", None)
    elif tokens._ENC is None:
        pytest.skip("tiktoken not installed")
    budget = ContextBudget(10_000)
    allowance = budget.prompt_allowance(1800)
    target = allowance - 40  # leave room for message overhead
    words = target
    while count_tokens("token " * words) > target:  # size the prompt with the active counter
        words -= max(1, (count_tokens("token " * words) - target) // 2)
    budget.check([{"role": "user", "content": "token " * words}], max_output=1800)


def test_fallback_counter_is_conservative(monkeypatch):
    """Without tiktoken the estimate must not undercount: it errs toward rejecting a call, never overrunning."""
    import app.tokens as tokens
    if tokens._ENC is None:
        pytest.skip("tiktoken not installed")
    english = "The Overall win rate of the variant on the dataset is 56.0% [S4]. " * 50
    chinese = "總結這篇論文的作者是誰？請用中文說明表二的消融實驗結果與圖三的架構。" * 20
    exact = [count_tokens(t) for t in (english, chinese)]
    monkeypatch.setattr(tokens, "_ENC", None)
    assert count_tokens(english) >= exact[0] and count_tokens(chinese) >= exact[1]
    assert count_tokens(truncate_tokens(chinese, 100)) <= 102


def test_truncate_tokens():
    text = "alpha beta gamma " * 500
    out = truncate_tokens(text, 50)
    assert count_tokens(out) <= 52 and out.endswith("…")


# ---------------------------------------------------------------- tables

def _w(x0, y0, text, h=7.0):
    return (x0, y0, x0 + 4 * len(text), y0 + h, text, 0, 0, 0)


def test_linearize_table_multilevel_headers():
    words = [
        _w(200, 10, "Agriculture"), _w(300, 10, "CS"),
        _w(180, 20, "NaiveRAG"), _w(230, 20, "LightRAG"), _w(280, 20, "NaiveRAG"), _w(330, 20, "LightRAG"),
        _w(100, 30, "Overall"), _w(185, 30, "32.4%"), _w(235, 30, "67.6%"), _w(285, 30, "38.8%"), _w(335, 30, "61.2%"),
        _w(180, 40, "NaiveRAG"), _w(235, 40, "-High"), _w(280, 40, "NaiveRAG"), _w(335, 40, "-High"),
        _w(100, 50, "Overall"), _w(185, 50, "35.2%"), _w(235, 50, "64.8%"), _w(285, 50, "44.0%"), _w(335, 50, "56.0%"),
    ]
    lines, ratio = linearize_table(words)
    assert ratio > 0.3
    assert lines[0] == ("Overall: Agriculture / NaiveRAG = 32.4%; Agriculture / LightRAG = 67.6%; "
                        "CS / NaiveRAG = 38.8%; CS / LightRAG = 61.2%")
    assert "Agriculture / -High = 64.8%" in lines[1] and "CS / -High = 56.0%" in lines[1]


# ---------------------------------------------------------------- chunking

def _doc():
    para = ("LightRAG uses a dual-level retrieval paradigm. " * 12).strip()
    blocks = [Block("heading", "1 INTRODUCTION", 1)]
    blocks += [Block("text", para, 1 + i // 3) for i in range(9)]
    blocks += [Block("heading", "4 EVALUATION", 4)]
    blocks += [Block("table", "Table 2: Ablation.\n" + "\n".join(f"Row{i}: A = {i}.0%; B = {100 - i}.0%" for i in range(5)), 4)]
    blocks += [Block("text", "The -High variant removes high-level retrieval and performs worse.", 4)]
    return ParsedDoc(title="Test", num_pages=4, blocks=blocks)


def test_chunks_respect_sections_sizes_and_tables():
    chunks = chunk_document(_doc(), target=150, max_tokens=250, overlap=30)
    assert all(c.tokens <= 250 for c in chunks if c.kind == "text")
    assert {c.section for c in chunks} == {"1 INTRODUCTION", "4 EVALUATION"}
    tables = [c for c in chunks if c.kind == "table"]
    assert len(tables) == 1 and tables[0].text.startswith("Table 2")
    assert [c.id for c in chunks] == list(range(len(chunks)))


def test_large_table_is_split_evenly_with_caption():
    rows = "\n".join(f"Metric{i}: Agriculture / NaiveRAG = {i}.1%; Agriculture / LightRAG = {i}.9%" for i in range(40))
    doc = ParsedDoc("T", 1, [Block("table", "Table 1: Win rates.\n" + rows, 1)])
    chunks = chunk_document(doc, table_max=400)
    assert len(chunks) >= 2 and all(c.tokens <= 400 for c in chunks)
    assert all(c.text.startswith("Table 1: Win rates.") for c in chunks)
    sizes = [c.text.count("\n") for c in chunks]
    assert max(sizes) - min(sizes) <= 1


# ---------------------------------------------------------------- retrieval

def test_tokenize_handles_variants_and_cjk():
    toks = tokenize("The -High variant 比較 LightRAG")
    assert "high" in toks and "lightrag" in toks and "比較" in toks


def test_hybrid_search_finds_table_chunk():
    chunks = chunk_document(_doc(), target=150, max_tokens=250, overlap=30)
    idx = HybridIndex(chunks, None)
    hits = idx.search(["ablation table -High variant"], top_k=3)
    kinds = [chunks[i].kind for i, _ in hits]
    assert "table" in kinds or any("-High" in chunks[i].text for i, _ in hits)


# ---------------------------------------------------------------- summary map-reduce

def test_summary_map_reduce_respects_small_context(settings):
    settings.max_context_tokens = 2_000  # force many map calls and a reduce level
    settings.summary_map_output_tokens = 300
    settings.summary_reduce_input_tokens = 120
    llm = FakeLLM(settings)
    chunks = chunk_document(_doc(), target=150, max_tokens=250, overlap=30)
    ledger = Ledger()
    notes = build_summary_notes(chunks, llm, settings, ledger)
    calls = [e.call for e in ledger.entries]
    assert any(c.startswith("summary-map") for c in calls)
    assert any(c.startswith("summary-reduce") for c in calls)
    assert all(e.within_limit for e in ledger.entries)
    assert notes and notes[0]["page_start"] == 1


# ---------------------------------------------------------------- trap guards

def test_effective_context_is_min_of_limit_and_model(settings):
    from app.llm import LLM
    settings.model_context_tokens = 4096
    assert LLM(settings).context_limit == 4096
    settings.model_context_tokens = 32768
    assert LLM(settings).context_limit == 10_000


def test_citation_parsing_flags_unknown_ids():
    from app.pipeline import _citations
    assert _citations("A [S1]. B [S2, S4]. C [N3; 5]. not a cite: S9") == ["S1", "S2", "S4", "N3", "N5"]


def test_split_table_parts_are_siblings():
    from app.pipeline import _table_siblings
    rows = "\n".join(f"Metric{i}: A / X = {i}.1%; A / Y = {i}.9%" for i in range(40))
    doc = ParsedDoc("T", 3, [Block("table", "Table 2: Ablation.\n" + rows, 3),
                             Block("table", "Table 3: Other.\nR: A = 1.0%; B = 2.0%", 3)])
    chunks = chunk_document(doc, table_max=400)
    parts = [c for c in chunks if c.text.startswith("Table 2")]
    assert len(parts) >= 2
    assert {c.id for c in _table_siblings(chunks, parts[-1])} == {c.id for c in parts[:-1]}


def test_background_calls_wait_for_interactive_requests(settings):
    import threading, time
    llm = FakeLLM(settings)
    order = []
    def background():
        llm.complete([{"role": "user", "content": "x"}], max_tokens=10, ledger=Ledger(), call="summary-map 1/1", background=True)
        order.append("background")
    with llm.interactive():
        t = threading.Thread(target=background)
        t.start()
        time.sleep(0.2)
        order.append("interactive")
    t.join(timeout=5)
    assert order == ["interactive", "background"]


def test_linearize_table_stacked_headers():
    words = [
        _w(200, 10, "Dev"), _w(214, 10, "Set"),
        _w(180, 20, "MNLI-m"), _w(240, 20, "QNLI"),
        _w(182, 30, "(Acc)"), _w(242, 30, "(Acc)"),
        _w(100, 40, "BERTBASE"), _w(185, 40, "84.4"), _w(245, 40, "88.4"),
        _w(100, 50, "No"), _w(110, 50, "NSP"), _w(185, 50, "83.9"), _w(245, 50, "84.9"),
    ]
    lines, _ = linearize_table(words)
    assert lines[0] == "BERTBASE: Dev Set / MNLI-m (Acc) = 84.4; Dev Set / QNLI (Acc) = 88.4"
    assert lines[1].startswith("No NSP: Dev Set / MNLI-m (Acc) = 83.9")


def test_summary_resumes_from_call_cache(settings, tmp_path):
    llm = FakeLLM(settings)
    chunks = chunk_document(_doc(), target=150, max_tokens=250, overlap=30)
    cache = tmp_path / "calls.json"
    first = build_summary_notes(chunks, llm, settings, Ledger(), cache_path=cache)
    n_calls = len(llm.calls)
    second = build_summary_notes(chunks, llm, settings, Ledger(), cache_path=cache)
    assert second == first and len(llm.calls) == n_calls  # nothing re-run


def test_ungrounded_numbers_flags_invented_values():
    from app.pipeline import ungrounded_numbers
    sources = ["Overall: CS / -High = 56.0%; CS / LightRAG = 61.2%", "610,000 tokens in total"]
    assert ungrounded_numbers("CS -High is 56.0% [S1] vs 61.2%, using 610000 tokens.", sources) == []
    assert ungrounded_numbers("It reaches 99.9% [S2].", sources) == ["99.9"]


TABLE2 = [
    "Table 2: Performance of ablated versions of LightRAG, using NaiveRAG as reference.\n"
    "Overall: CS / NaiveRAG = 44.0%; CS / -High = 56.0%; Legal / NaiveRAG = 22.0%; Legal / -High = 78.0%",
    "Table 2: Performance of ablated versions of LightRAG, using NaiveRAG as reference. (continued)\n"
    "Overall: CS / NaiveRAG = 43.6%; CS / -Low = 56.4%; Legal / NaiveRAG = 18.8%; Legal / -Low = 81.2%",
]


def test_table_check_accepts_correct_attribution():
    from app.numcheck import misattributed_numbers
    ok = ["On CS, -High reaches an Overall win rate of 56.0% [S1], while -Low reaches 56.4% [S2].",
          "| Dataset | -High | -Low |\n|---|---|---|\n| CS | 56.0% | 56.4% |\n| Legal | 78.0% | 81.2% |",
          "**CS (Overall):**\n- -High: 56.0%\n- -Low: 56.4%",
          "The values are 56.0% and 56.4%.",  # unlabelled numbers are not penalised
          # grouped rows (variant written once per group) under prose that names other systems
          "**Win rates vs NaiveRAG**\nThe full LightRAG model generally wins most comparisons.\n\n"
          "| Variant | Dataset | Overall |\n|---|---|---|\n| -High | CS | 56.0% |\n| | Legal | 78.0% |\n"
          "| -Low | CS | 56.4% |\n| | Legal | 81.2% |"]
    for answer in ok:
        assert misattributed_numbers(answer, TABLE2) == [], answer


def test_table_check_flags_swapped_values():
    """The bag-of-numbers check passes all of these (every value exists in the table)."""
    from app.numcheck import misattributed_numbers
    from app.pipeline import ungrounded_numbers
    swapped = {"On CS, -High reaches 56.4% [S1], while -Low reaches 56.0% [S2].": {"56.4%", "56.0%"},
               "| Dataset | -High | -Low |\n|---|---|---|\n| Legal | 81.2% | 78.0% |": {"81.2%", "78.0%"},
               "| Variant | CS |\n|---|---|\n| -Low | 78.0% |": {"78.0%"},  # wrong dataset
               "- -Low on Legal (Overall): 18.8%": {"18.8%"},  # the baseline's share reported as the variant's
               "| Variant | Dataset | Overall |\n|---|---|---|\n| -High | CS | 56.4% |\n| | Legal | 81.2% |": {"56.4%", "81.2%"}}
    for answer, values in swapped.items():
        assert ungrounded_numbers(answer, TABLE2) == []
        assert {f["value"] for f in misattributed_numbers(answer, TABLE2)} == values, answer


def test_table_check_reads_hyphenated_and_abbreviated_row_labels():
    """Held-out failure: the answer gave RAG-Token's 54.8 as RAG-Sequence's score."""
    from app.numcheck import misattributed_numbers
    src = ["Table 6: Ablations on the dev set.\n"
           "RAG-Token: Exact Match = 43.5; Exact Match = 54.8\nRAG-Seq.: Exact Match = 44.5; Exact Match = 56.8"]
    wrong = misattributed_numbers("RAG-Sequence achieves an Exact Match score of 54.8 [S6].", src)
    assert [f["value"] for f in wrong] == ["54.8"]
    assert misattributed_numbers("RAG-Sequence achieves an Exact Match score of 44.5 [S6].", src) == []
    assert misattributed_numbers("RAG-Token reaches 54.8 [S6].", src) == []


def test_rerank_selection_keeps_strong_drops_weak():
    from app.rerank import select
    scored = [(1, 6.0), (2, 4.5), (3, -2.0), (4, -7.0), (5, 0.5)]
    assert select(scored) == [1, 2, 5]  # within 5 logits of the best, or above 0
    assert select([(1, -6.0), (2, -7.0), (3, -12.0), (4, -13.0)]) == [1, 2, 3]  # min_keep


def test_rerank_text_exposes_all_table_headers():
    from app.chunker import Chunk
    from app.pipeline import _rerank_text
    rows = ["Overall: CS / NaiveRAG = 38.8%; CS / LightRAG = 61.2%"] * 30 + ["Overall: CS / NaiveRAG = 44.0%; CS / -High = 56.0%"]
    c = Chunk(id=0, text="Table 2: Ablation.\n" + "\n".join(rows), section="4 EVALUATION", page_start=8,
              page_end=8, kind="table", tokens=900)
    text = _rerank_text(c)
    assert "CS / -High" in text and text.count("CS / LightRAG") == 1 and len(text) < 300


def test_openai_provider_waits_and_retries_on_rate_limit(settings, monkeypatch):
    import httpx
    from app.llm import LLM
    settings.llm_provider, settings.llm_base_url = "openai", "https://api.example.test/v1"
    calls = {"n": 0}
    sse = ('data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
           'data: {"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":1}}\n\ndata: [DONE]\n\n')

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"retry-after": "0"}, text="rate limited")
        return httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})

    from contextlib import contextmanager
    transport = httpx.MockTransport(handler)

    @contextmanager
    def fake_stream(method, url, **kw):
        kw.pop("timeout", None)
        with httpx.Client(transport=transport) as client, client.stream(method, url, **kw) as r:
            yield r

    monkeypatch.setattr(httpx, "stream", fake_stream)
    monkeypatch.setattr("app.llm.time.sleep", lambda s: None)
    from app import metrics
    metrics.reset()
    ledger = Ledger()
    out = LLM(settings).complete([{"role": "user", "content": "hi"}], max_tokens=10, ledger=ledger, call="t")
    assert out == "ok" and calls["n"] == 2 and ledger.entries[0].actual_prompt_tokens == 12
    text = metrics.render()  # the real client records Prometheus metrics for every call
    assert 'docchat_llm_calls_total{call="t"} 1' in text and 'docchat_llm_tokens_total{kind="prompt"} 12' in text


def test_reference_sections_not_penalised_for_citation_questions():
    from app.chunker import Chunk
    mk = lambda i, sec, txt: Chunk(id=i, text=txt, section=sec, page_start=1, page_end=1, kind="text", tokens=20)
    chunks = [mk(0, "Introduction", "We build on graph retrieval work by Edge et al. 2024."),
              mk(1, "References", "Edge, D. et al. 2024. From local to global: a graph RAG approach.")]
    idx = HybridIndex(chunks, None)
    ordinary = dict(idx.search(["Edge graph"], top_k=2))
    about_refs = dict(idx.search(["Edge graph cited"], top_k=2))
    assert about_refs[1] == 2 * ordinary[1]  # reference list no longer halved
    assert about_refs[0] == ordinary[0]


def test_input_token_rate_limiter_waits_instead_of_overrunning(settings, monkeypatch):
    from app.llm import LLM
    clock = {"t": 1000.0}
    monkeypatch.setattr("app.llm.time.time", lambda: clock["t"])
    monkeypatch.setattr("app.llm.time.sleep", lambda s: clock.__setitem__("t", clock["t"] + s))
    settings.llm_input_tpm = 7000
    llm = LLM(settings)
    assert llm._throttle(4300) == 0          # first call goes at once
    waited = llm._throttle(4300)             # second would exceed 7000/min: waits out the window
    assert 59 < waited < 62
    settings.llm_input_tpm = 0
    assert LLM(settings)._throttle(10**6) == 0  # no limit configured


def test_citation_counts_numeric_and_author_year():
    from app.citations import citation_stats
    numeric = ParsedDoc("T", 2, [
        Block("heading", "1 Introduction", 1),
        Block("text", "Attention [2] helps. Recurrent models [1, 2] and convolutions [3-4] were used before [2].", 1),
        Block("heading", "References", 2),
        Block("text", "[1] A. Alpha. Recurrent nets. 2014.\n[2] B. Beta. Attention. 2015.\n[3] C. Gamma. Conv. 2017.\n[4] D. Delta. More conv. 2017.", 2),
    ])
    s = citation_stats(numeric)
    assert s["style"] == "numeric" and s["top"][0]["key"] == "[2]" and s["top"][0]["count"] == 3
    assert s["n_citations"] == 6  # [2] x3, [1], [3], [4]; the range 3-4 expands

    author_year = ParsedDoc("T", 2, [
        Block("heading", "Introduction", 1),
        Block("text", "GraphRAG (Edge et al., 2024) builds communities. Edge et al. (2024) also report costs. "
                      "Dense retrieval (Gao & Callan, 2022) differs. Table 2 (2024) is not a citation.", 1),
        Block("heading", "REFERENCES", 2),
        Block("text", "Darren Edge, Ha Trinh, and Jonathan Larson. From local to global. arXiv, 2024. "
                      "Luyu Gao and Jamie Callan. Precise zero-shot dense retrieval. arXiv, 2022.", 2),
    ])
    s = citation_stats(author_year)
    assert s["style"] == "author-year" and s["top"][0]["key"] == "Edge et al. (2024)" and s["top"][0]["count"] == 2
    assert s["top"][0]["entry"].startswith("Darren Edge") and s["n_citations"] == 3  # "Table 2 (2024)" ignored


def test_find_structural_references_english_and_chinese():
    from app.structure import find_refs
    refs = find_refs("Explain Table 2 and Figure 3 on page 8, Section 4.2 and reference [12]; also the appendix")
    assert {("table", "2"), ("figure", "3"), ("page", "8"), ("section", "4.2"), ("reference", "12"), ("appendix", "")} <= set(refs)
    assert {("table", "2"), ("page", "8")} <= set(find_refs("請解釋表2，第8頁寫了什麼？"))
    assert find_refs("How does retrieval work?") == []


def test_resolve_finds_table_caption_and_page_chunks():
    from app.structure import resolve
    chunks = chunk_document(_doc(), target=150, max_tokens=250, overlap=30)
    ids, notes = resolve(chunks, [("table", "2"), ("page", "4"), ("figure", "9")])
    assert chunks[ids[0]].text.startswith("Table 2")
    assert any(chunks[i].page_start <= 4 <= chunks[i].page_end for i in ids)
    assert notes[-1] == "Figure 9: not found in the document"


def test_term_counts_whole_words_per_page():
    from app.structure import term_counts
    pages = [(1, "GraphRAG builds communities. graphrag again."), (3, "Unlike GraphRAG, LightRAG ... GraphRAG-style")]
    out = term_counts(pages, ["GraphRAG"])
    assert '"GraphRAG": 3 occurrences on 2 pages: p.1 (2), p.3 (1)' in out  # "GraphRAG-style" is not the word


def test_no_document_specific_strings_in_application_behaviour():
    """Bias guard: string constants the app uses (prompts, regexes, labels) must not mention the
    evaluation documents, their methods, datasets, variants or values. Comments and docstrings are
    excluded because they never reach the model."""
    import ast
    import pathlib
    import re
    banned = re.compile(r"lightrag|graphrag|naiverag|rq-rag|hyde|\bbert\b|nist|vaswani|attention is all|"
                        r"agriculture|\blegal\b|-high\b|-low\b|-origin|win rate|1,399|56\.0%|84\.8|edge et al|"
                        r"karpukhin|lewis et al", re.I)
    hits = []
    for f in sorted(pathlib.Path(__file__).resolve().parent.parent.joinpath("app").glob("*.py")):
        tree = ast.parse(f.read_text(encoding="utf-8"))
        docs = {id(n.body[0].value) for n in ast.walk(tree)
                if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef)) and n.body
                and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
        hits += [f"{f.name}:{n.lineno}" for n in ast.walk(tree)
                 if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs and banned.search(n.value)]
    ui = pathlib.Path(__file__).resolve().parent.parent / "static"
    hits += [p.name for p in ui.iterdir() if p.suffix in (".js", ".html") and banned.search(p.read_text(encoding="utf-8"))]
    assert hits == [], f"document-specific strings in application behaviour: {hits}"


def test_tokenizer_covers_all_scripts():
    assert "modèles" in tokenize("Évaluation des modèles")
    assert tokenize("Оценка модели") == ["оценка", "модели"]
    assert "결과" in tokenize("평가 결과")
    assert "拡張" in tokenize("検索拡張生成")  # Japanese/Chinese: character bigrams


def test_daily_quota_fails_fast_with_a_clear_message(settings, monkeypatch):
    import httpx
    from contextlib import contextmanager
    from app.llm import LLM, LLMError
    settings.llm_provider, settings.llm_base_url = "openai", "https://api.example.test/v1"
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(429, text='{"error":{"message":"Rate limit reached on tokens per day (TPD)"}}')

    transport = httpx.MockTransport(handler)

    @contextmanager
    def fake_stream(method, url, **kw):
        kw.pop("timeout", None)
        with httpx.Client(transport=transport) as client, client.stream(method, url, **kw) as r:
            yield r

    monkeypatch.setattr(httpx, "stream", fake_stream)
    monkeypatch.setattr("app.llm.time.sleep", lambda s: None)
    with pytest.raises(LLMError, match="daily token quota"):
        LLM(settings).complete([{"role": "user", "content": "hi"}], max_tokens=10, ledger=Ledger(), call="t")
    assert calls["n"] == 1  # no pointless retries


def test_provider_output_cap_lowers_max_tokens_instead_of_failing(settings, monkeypatch):
    """Groq answers 'Request too large ... output tokens per minute (OTPM): Limit 1000' for a large
    max_tokens; waiting does not help, so the client retries once with the cap the provider allows."""
    import httpx
    import json as _json
    from contextlib import contextmanager
    from app.llm import LLM
    settings.llm_provider, settings.llm_base_url = "openai", "https://api.example.test/v1"
    asked = []
    sse = ('data: {"choices":[{"delta":{"content":"ok"}}]}' + chr(10) + chr(10) +
           'data: {"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":1}}' + chr(10) + chr(10) +
           "data: [DONE]" + chr(10) + chr(10))

    def handler(request):
        asked.append(_json.loads(request.content)["max_tokens"])
        if len(asked) == 1:
            return httpx.Response(429, text='{"error":{"message":"Request too large for model on output tokens per '
                                            'minute (OTPM): Limit 1000, Requested 1435. Reduce max_tokens"}}')
        return httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})

    transport = httpx.MockTransport(handler)

    @contextmanager
    def fake_stream(method, url, **kw):
        kw.pop("timeout", None)
        with httpx.Client(transport=transport) as client, client.stream(method, url, **kw) as r:
            yield r

    monkeypatch.setattr(httpx, "stream", fake_stream)
    monkeypatch.setattr("app.llm.time.sleep", lambda s: None)
    ledger = Ledger()
    out = LLM(settings).complete([{"role": "user", "content": "hi"}], max_tokens=1800, ledger=ledger, call="answer")
    assert out == "ok" and asked == [1800, 950]
    assert ledger.entries[0].max_output_tokens == 950  # the ledger shows the cap actually used
