# DocChat — chat with your documents under a 10K-token LLM

A website with a chat interface: upload a document, ask questions, get answers grounded in the
document with page citations. An individual project by Muskaan Mahadik.

Constraints honoured:

1. **No ready-made GPT/Gemini document parsing.** PDFs are parsed locally with PyMuPDF (text, reading
   order, headings, tables). The LLM is used **only text-to-text**.
2. **LLM max context = 10K tokens.** Every LLM call is checked *before* it is sent
   (prompt + reserved output ≤ 10,000) and the server-reported token counts are shown in a per-answer
   **token ledger**. With Ollama, `num_ctx` is also set to 10,000, so the model physically cannot see more.

## Start here

| You want to… | Open |
|---|---|
| Run the demo | [Quick start](#quick-start-windows): `run.bat`, then <http://127.0.0.1:8000> |
| See the presentation | [`docs/DocChat_Presentation.pptx`](docs/DocChat_Presentation.pptx) (slide 2 states the problem) |
| Read how documents beyond 10K and summaries are handled | [below](#q-how-do-you-handle-documents-larger-than-the-llms-10k-context), or [`docs/PROJECT_OVERVIEW.md`](docs/PROJECT_OVERVIEW.md) sections 6–7 |
| Check the results and where each number comes from | [`evaluation/results/RESULTS_INDEX.md`](evaluation/results/RESULTS_INDEX.md) |
| Read the engineering report (stages, evaluation, bias audit, limits) | [`docs/REPORT.md`](docs/REPORT.md) |
| See how the work was done with AI | [`docs/AI_COLLABORATION.md`](docs/AI_COLLABORATION.md) |
| Find a piece of code | [file map](docs/PROJECT_OVERVIEW.md#21-file-map); the core is `app/pipeline.py` (routes, evidence, answer), `app/tokens.py` (the 10K gate), `app/summary.py` (map-reduce), `app/numcheck.py` (table check) |
| Run the tests | `python -m pytest -q` (offline, no model needed) |

## Quick start (Windows)

Prerequisites: Python 3.10+, [Ollama](https://ollama.com) for the local embedding model, and a free
[Groq API key](https://console.groq.com/keys) for the default LLM, the open-weights **Qwen 3.8 27B**.
That is the configuration that answered all three core test questions completely (see
[the results](evaluation/results/RESULTS_INDEX.md)); a fully offline option is below.

```bash
ollama pull nomic-embed-text
```

Then double-click **`run.bat`**. On the first start it creates `.venv`, installs `requirements.txt`,
copies `.env.example` → `.env` and opens it: paste your key after `LLM_API_KEY=`, save, and press a key.
DocChat then opens at <http://127.0.0.1:8000>. On Windows, unzip to a short path such as `C:\DocChat`:
in deeply nested folders the install can hit Windows' 260-character path limit.

Manual start:

```bash
python -m venv .venv && .venv\Scripts\pip install -r requirements.txt
```

```bash
.venv\Scripts\python -m uvicorn app.main:create_app --factory --port 8000
```

Upload `https://arxiv.org/pdf/2410.05779` (the LightRAG paper) and click the suggested questions.
The header shows whether the model is reachable; a missing or rejected key is named there.

### Offline: qwen2.5:14b on Ollama, or on a free Colab GPU

No key and nothing leaving the machine: set the Option B lines in `.env` (`LLM_PROVIDER=ollama`,
`LLM_MODEL=qwen2.5:14b`, `LLM_INPUT_TPM=0`) and `ollama pull qwen2.5:14b` (a GPU with ~10 GB is
recommended). Measured honestly: it answered Q1 3/3 and Q2 2/3, but its Q3 (the ablation table) was
incomplete in all 3 runs, and it is easier to steer with instructions planted in a document; use the
default for the three core test questions.

No local GPU: open `colab/docchat_gpu_backend.ipynb` in Google Colab (T4 GPU, *Run all*). It runs
Ollama with `qwen2.5:14b` on the GPU and prints three lines to paste into `.env`:

```
LLM_MODEL=qwen2.5:14b
LLM_BASE_URL=https://xxxx.trycloudflare.com
EMBED_BASE_URL=https://xxxx.trycloudflare.com
```

Embeddings and reranking always stay local, and every option runs under the same 10K-per-call limit.

## How it works

```
upload ─► parse (PyMuPDF, local) ─► document card (title, authors, abstract, outline, captions)
                                 ─► section-aware chunks ─► BM25 + embeddings index
                                 └► map-reduce summary notes (on the first summary request, then cached)
question ─► planner (LLM, JSON): intent + search queries
             ├─ "summary"  ─► card + notes covering the whole document ─► answer [D] [N#]
             └─ "question" ─► hybrid search (all queries) ─► cross-encoder rerank (local)
                              ─► pack the relevant chunks into the token budget
                              ─► answer with citations [D] [S#] ─► citation, number and table checks
```

| Stage | What happens | File |
|---|---|---|
| Parse | Reading order (two-column aware), headings by font size, page numbers/rotated stamps/figure labels dropped. **Tables** found by `find_tables()` are rebuilt from word coordinates and linearised as `row: column header = value` (e.g. `Overall: CS / -High = 56.0%`) so multi-level headers survive. | `app/parser.py` |
| Chunk | Never crosses a section; ~350 tokens, max 500, 50-token sentence overlap; each table is its own chunk (split evenly by rows with the caption repeated if large). | `app/chunker.py` |
| Card | Built from layout, no LLM: title (largest page-1 font), front matter (authors, affiliations), abstract or opening text, outline, captions, counts, PDF metadata. Included in **every** prompt, so "what is the title / who wrote it / how many tables" work even though those words never appear in the passage. | `app/doccard.py` |
| Index | BM25 (exact words, numbers, variant names like `-High`) + `nomic-embed-text` vectors, fused with Reciprocal Rank Fusion over every query. Reference sections are down-weighted. | `app/index.py` |
| Rerank | A local cross-encoder (`BAAI/bge-reranker-base`, FastEmbed/ONNX on CPU) rescores the top 20 and keeps only clearly relevant chunks (at most 8). For the ablation question this keeps exactly both halves of Table 2 and the two paragraphs that explain the variants, instead of 12 loosely related chunks. Idea from QAnything/kotaemon two-stage retrieval. | `app/rerank.py` |
| Plan | One small LLM call returns `{intent, queries}`; comparisons get one query per item plus facet queries (architecture, retrieval, results, cost, updates). Non-English questions are turned into English queries. | `app/pipeline.py` |
| Answer | Chunks are packed by relevance until the budget is used (a split table travels with its sibling parts), then sorted in reading order and labelled `[S1]…`. The answer is streamed. Afterwards three deterministic checks run: cited IDs must exist, every number must appear in the evidence, and every number must sit under the row/column labels the answer gives it (the table check, `app/numcheck.py`). Failures are flagged in the UI. | `app/pipeline.py` |
| Budget | `ContextBudget.check()` is the single gate in front of every call; `Ledger` records estimated vs. actual tokens. | `app/tokens.py`, `app/llm.py` |

### Q: How do you handle documents larger than the LLM's 10K context?

The LightRAG paper is ~15K tokens of text (plus prompt and answer), so it never goes to the LLM whole.

* **Budget first.** Each call has a fixed allowance: `10,000 − reserved answer tokens (1,800)`, divided by
  a 1.15 safety factor because tiktoken's count differs slightly from the model's own tokenizer. System
  prompt, chat history (≤800 tokens, old turns dropped first) and the question are counted, and what
  remains (capped at 5,500) is the **evidence budget**. A call that would not fit raises `ContextOverflow`
  instead of being silently truncated. The UI's token ledger shows the real `prompt_eval_count` from Ollama
  for every call; for example, the LightRAG summary map calls measured 3,363–4,202 prompt tokens.
* **Retrieve only what matters.** Section-aware chunks + hybrid BM25/embedding search with several
  planner queries, so a comparison question gathers evidence from the method, experiment and cost sections,
  not just the top-k neighbours of one query.
* **Tables as compact text.** A table becomes one self-describing chunk, so a whole results table fits in the
  evidence budget and every number keeps its row/column meaning.
* **Effective context = min(10K, the model's real context).** The smaller value wins if a model has a smaller window.

### Q: How do you handle "summary"?

A summary needs the *whole* document, which top-k retrieval cannot give. DocChat uses
**hierarchical map-reduce**, run on the first summary request for a document and cached on disk
(`SUMMARY_ON_UPLOAD=true` runs it right after upload instead):

1. **Map:** the document (in reading order, references excluded) is cut into consecutive groups of
   ≤4,000 tokens; each group gets one call that writes faithful notes (claims, method, setup, key numbers,
   conclusions). The LightRAG paper becomes 4 map calls.
2. **Reduce:** if the notes together exceed 5,000 tokens, they are grouped and condensed again, as many
   levels as needed, and every call is within budget.
3. **Answer:** at question time the final notes (covering every page) and the user's actual request go into
   one call, so "summary this document", "key results only" or "用中文總結" are all answered with
   full-document coverage. Notes are cited as `[N1]…` with their page ranges.

Intent detection: obvious summary requests ("summary this document", "總結這篇") are routed by a rule
without an LLM call; anything else is decided by the planner. The whole-document "list" route runs only
for explicit "list all / every …" requests.

### Q: What does a question cost?

The 10K limit bounds each call; the total is measured too (`qwen/qwen3.8-27b`, LightRAG paper):

| Question | LLM calls | Tokens, all calls | Largest call |
|---|---|---|---|
| Q2 compare / Q3 ablation | 2 | 5,310 / 6,882 | 4,785 / 6,407 |
| Any other question (26 dev questions) | 2 | median 4,072 | 5,519 |
| Q1 summary | 1 (+ 4 map calls the first time) | 4,496 (+ 17,883 once) | 4,802 |
| *The whole paper in one prompt* | 1 | ≈ 17,000 | does not fit 10K |

Specific questions read 2.5–4× less than the whole paper; search, reranking and checks run on the CPU.
A summary reads everything once (≈1.3× one whole-paper read), then every later summary costs one call.

## Trust and correctness guards

* Answer prompt rules: cite every claim, copy numbers exactly, explain what a metric measures
  (e.g. a *win rate vs. a baseline* is not accuracy, and differences between two win rates against the same
  baseline are not head-to-head results), and say what is missing rather than guess.
* Document text is treated as **untrusted**: every prompt that carries document text (answer, summary
  map/reduce, list map) says instructions inside it are ignored; a test plants an instruction in an
  uploaded file and checks it reaches the model only inside the sources block.
* Citation check: any `[S#]` the model cites that was not provided is flagged in the UI.
* Number check and table check: a number found in no source is flagged; so is a real number placed under
  the wrong row or column (e.g. the -Low score reported as the -High score). Over all 109 saved evaluation
  answers the table check flagged 3, all genuine errors, one of which the string rubric had passed.
* An answer that stops at its output cap (e.g. a provider's per-minute output limit) is continued
  automatically in a follow-up call, each within the 10K limit; only if it still ends at its cap is it
  marked as possibly cut off.
* The checks flag problems for the reader; they do not rewrite the answer.
* Provenance: each document is identified by its SHA-256 (shown in the header); re-uploading the same file
  reuses its cached index and summary.
* No question-specific or paper-specific logic in the code; `tests/test_paper.py` only *checks* parsing
  of the test paper (e.g. Table 2 cell `Agriculture / -High = 65.2%`).

## Tested on more than the test paper

To check the parser is not tuned to one document, it is also tested on BERT (arXiv 1810.04805), which has a
different layout: two columns (ACL), bold body-size headings, and stacked three-level table headers. That run
found two real biases that are now fixed and covered by tests: headings were only detected by font size, and
stacked header rows overwrote each other. Both LightRAG revisions (v2 and v3 on arXiv) pass the paper tests.

## Evaluation (proof of generality)

`evaluation/questions.json` has 27 questions over three documents with different layouts: the
LightRAG paper (single column), BERT (two-column ACL) and the NIST AI RMF 1.0 (48-page government
report). They cover metadata (title, authors, affiliations, publisher, date), facts, single table
cells, the three core test questions and questions the document cannot answer. Expected strings were
checked against each document's extracted text.

```bash
.venv\Scripts\python evaluation/run_eval.py --mode retrieval --pdf lightrag=PATH --pdf bert=PATH --pdf nist=PATH
```

`--mode retrieval` needs no LLM: it checks that the facts each question needs reach the evidence given
to the model, so it measures parsing, chunking, search, reranking and the card on their own.
`--mode answer` scores the final answers of the configured model and also records citation validity,
the number and table checks and the token ledger. Reports go to `evaluation/results/`.

| Run | Model | Result |
|---|---|---|
| Retrieval, 35 dev questions (no LLM) | – | 35/35 (33/35 from hybrid search + reranker alone) |
| Answers, 27 dev questions | `qwen/qwen3.8-27b` (Groq) | 26/27 (27/27 as first scored; the stricter Q3 check fails one answer that defined -Origin wrongly) |
| Answers, 41 dev questions | `qwen2.5:14b` | 36/41 as first scored (39/41 after 3 accepted wordings were widened) |
| Answers, unseen paper (18 questions frozen before the run) | `qwen2.5:14b` | 16/18 |
| Q1–Q3 repeated from fresh uploads | `qwen/qwen3.8-27b` (default) | Q1 1/1, Q2 2/2, Q3 3/3 complete |
| Q1–Q3 repeated from fresh uploads | `qwen2.5:14b` (offline) | Q1 3/3, Q2 2/3, Q3 0/3 complete |
| Instructions planted in a document | both | 27B resisted 9/9; 14B resisted the 4 known phrasings, partly followed 3 of 5 unseen |

Every number and the file behind it: [`evaluation/results/RESULTS_INDEX.md`](evaluation/results/RESULTS_INDEX.md).

## Tests

```bash
.venv\Scripts\python -m pytest
```

Unit and API tests run offline with a fake LLM (token-budget gate, table linearisation, chunking,
hybrid search, map-reduce under a tiny 2K context, upload → chat → summary flow, persistence). Paper tests
run when the PDF is available (`samples/2410.05779.pdf` or `DOCCHAT_TEST_PDF=path`; BERT via
`DOCCHAT_TEST_PDF2=path`).

## Docker

```bash
docker compose up --build
```

Starts DocChat and Ollama together and pulls the models into a volume on first start (~5 GB), then serves
<http://localhost:8000>. Runs on CPU by default; uncomment the `deploy` block in `docker-compose.yml` to use
an NVIDIA GPU. To use a hosted model instead, set `LLM_PROVIDER`, `LLM_BASE_URL`, `LLM_MODEL` and
`LLM_API_KEY` in `.env`; embeddings and reranking still run in the container.

## CI

`.gitlab-ci.yml` runs the test suite on every push and merge request (JUnit report attached) and builds the
Docker image on the default branch. The tests use a fake LLM and fake embeddings, so CI needs no model
server or API key.

## Monitoring

`GET /metrics` serves Prometheus metrics: questions by route, LLM calls by type, prompt and output tokens,
time spent in LLM calls, LLM errors, time spent waiting for the provider's rate limit, the effective
context limit, the largest call seen and the number of documents. Point a Prometheus scrape job at it and
chart it in Grafana.

## API

| Method | Path | |
|---|---|---|
| POST | `/api/documents` | upload (PDF/TXT/MD, ≤50 MB) |
| GET | `/api/documents`, `/api/documents/{id}` | status, stats |
| GET | `/api/documents/{id}/chunks` | how the document was parsed and chunked |
| GET | `/api/documents/{id}/summary` | map-reduce notes + their token ledger |
| DELETE | `/api/documents/{id}` | remove |
| POST | `/api/chat` | `{doc_id, doc_ids?, message, history}` → Server-Sent Events: `status`, `plan`, `sources`, `token`, `done` |
| POST | `/api/retrieve` | evidence a question would receive, without an LLM call (debugging, retrieval evaluation) |
| GET | `/api/health` | model, embeddings and reranker status, effective context limit |
| GET | `/metrics` | Prometheus metrics |

## Limitations

* Scanned PDFs without a text layer are not OCR'd (an error is shown).
* Two tables printed side by side under one caption (e.g. BERT's SQuAD table) are linearised
  imperfectly; the values are present but some column headers are wrong.
* Table reconstruction targets numeric tables; formula-heavy layouts (e.g. the paper's cost figure) stay as
  running text, which is still retrievable together with the paragraph that explains it.
* Citation validation checks IDs, not semantic entailment; the Sources panel shows the exact text so a
  reader can verify each claim. The table check catches numbers under contradicting labels, not vague
  claims that name both sides, and no check verifies paraphrased claims.
* A 7B model on CPU is slow; use the Colab GPU backend for interactive demos.
