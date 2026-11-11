"""Checks against the default test document (arXiv 2410.05779, LightRAG).
Skipped when the PDF is not available locally. Values verified against the PDF's Table 2."""
from app.chunker import chunk_document
from app.index import HybridIndex
from app.parser import parse_pdf


def test_parser_extracts_sections_and_ablation_table(paper_pdf):
    doc = parse_pdf(paper_pdf)
    headings = [b.text for b in doc.blocks if b.kind == "heading"]
    assert any("INTRODUCTION" in h for h in headings)
    assert any("EVALUATION" in h for h in headings)
    assert "LIGHTRAG" in doc.title.upper()
    tables = [b.text for b in doc.blocks if b.kind == "table"]
    table2 = next(t for t in tables if t.startswith("Table 2"))
    # first data row of the -High block (Agriculture) and of the -Origin block (Mix)
    assert "Agriculture / NaiveRAG = 34.8%; Agriculture / -High = 65.2%" in table2
    assert "Mix / -Origin = 55.6%" in table2
    # table numbers are not duplicated into running text
    body = " ".join(b.text for b in doc.blocks if b.kind == "text")
    assert "34.8%" not in body


def test_retrieval_ranks_ablation_table_first(paper_pdf):
    chunks = chunk_document(parse_pdf(paper_pdf))
    idx = HybridIndex(chunks, None)
    top = [chunks[i] for i, _ in idx.search(["Performance of ablated versions of LightRAG"], top_k=3)]
    assert any(c.kind == "table" and c.text.startswith("Table 2") for c in top)


def test_generalises_to_two_column_bold_heading_paper():
    """BERT (arXiv 1810.04805): two-column ACL layout, bold body-size headings, stacked table headers."""
    import os
    from pathlib import Path
    import pytest
    path = os.environ.get("DOCCHAT_TEST_PDF2", "")
    if not path or not Path(path).exists():
        pytest.skip("set DOCCHAT_TEST_PDF2 to the BERT paper PDF")
    doc = parse_pdf(path)
    headings = [b.text for b in doc.blocks if b.kind == "heading"]
    assert {"Introduction", "Related Work", "Conclusion"} <= set(headings)
    table5 = next(b.text for b in doc.blocks if b.kind == "table" and b.text.startswith("Table 5"))
    assert "No NSP: Dev Set / MNLI-m (Acc) = 83.9" in table5
