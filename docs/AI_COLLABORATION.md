# How this project was built with AI (Claude Code)

Built in one AI-native session with Claude Code (desktop app). The human set direction and made the
product decisions; the AI explored, implemented, tested and reported measured results. Everything below
happened in that session. Nothing was written retroactively to look better.

## Workflow

1. **Write the spec first.** The binding requirements were fixed before any code: a chat website, no
   GPT/Gemini parsing, a 10K context, three core test questions (Q1–Q3), an explanation of over-context and
   summary handling, and code + presentation + local demo.
2. **Survey before deciding.** It checked the machine (Python, Ollama models, 4-core CPU, no NVIDIA GPU),
   found an earlier over-engineered attempt (26 modules, still no demo), and asked the human two decisions:
   *fresh lean build vs. continue* → fresh; *local model vs. cloud API* → local `qwen2.5:7b`.
3. **Probe the data before writing code.** Before the parser existed, it inspected the real paper's
   fonts, headings and table geometry. It found that PyMuPDF's built-in table extractor corrupts decimals
   (`32.4%` → `324%` + a stray `.`), while word boxes are clean, so tables are rebuilt from word coordinates.
4. **Build in thin vertical slices**, each verified on the real paper: parser → chunker → hybrid index →
   LLM client with token gate → map-reduce summary → pipeline → API → UI.
5. **Tests with a fake LLM** (offline, seconds), plus paper-specific *checks* (never paper-specific *logic*).
6. **Live runs on the real model**, reading the token ledger and the latency, then fixing what they exposed.
7. **Cross-check with another AI's review.** The human pointed to a "traps" list from an earlier
   Gemini/Claude research session; each trap was mapped to code, and missing guards were added with tests.
8. **Benchmark, then audit the benchmark.** A development set, a retrieval ablation, a held-out paper
   with questions frozen before the first run, a robustness set and a bias scan turned into a test.
9. **Independent reviews near the end.** The human brought critiques from a separate ChatGPT
   conversation. Each claim was checked against the code, the token ledgers and the paper text before
   anything changed: some led to fixes (stage 12 below), some were wrong, and one suggestion (loosening the
   table rule so the paper's cost figure parses as a table) was declined because it would fit the system to
   one paper.

The work moved in twelve stages; [REPORT.md, section 2](REPORT.md#2-how-the-project-progressed) lists them
with what was observed and what changed.

## Problems found and how they were caught

| Problem | Caught by | Fix |
|---|---|---|
| Built-in table extraction corrupts decimals | inspecting raw extractor output on page 8 | rebuild tables from word boxes; linearise as `row: column = value` |
| Small-caps headings lost spaces (`LIGHTRAG:SIMPLEAND`) | reading parser output | insert spaces from span gaps |
| Formula-style cost "table" (Figure 2) would be mangled | numeric-ratio check on the page-9 table | keep it as running text; the next paragraph explains every number |
| Large table split 1,412 + 115 tokens, cutting row groups | retrieval test (GraphRAG rows split) | balanced split; unit test caught a 14/14/12 imbalance → exact even split |
| 7B on CPU: ~16 tok/s prompt, ~3 tok/s output | measured benchmark | Colab GPU backend notebook; 3B fallback documented |
| Live question timed out behind a background summary call | first live Q3 run | interactive requests get priority over background jobs (test added) |
| A split table returned only half its rows | first live Q3 run (`-Origin` rows missing) | sibling parts of a selected table are added (test added) |
| Too many marginal chunks in the evidence | first live Q3 run | candidates cut to top 12 before packing |
| Effective context could be < 10K | traps review | `min(10K, model context)` (test added) |
| Prompt injection via document text | traps review | sources marked untrusted in prompts |
| Fabricated citations | traps review | cited IDs validated against provided IDs; shown in UI |
| "Win rate" read as accuracy | traps review | generic rule on metric semantics in the answer prompt |
| Paper revisions differ (v2 local vs v3 online) | traps review | SHA-256 provenance; parser tested on both |
| Live answers had no citations (7B ignored the rule) | reading live Q2/Q3 output | citation instruction repeated at the end of the prompt |
| Planner added unrelated "cost" queries to a non-comparison question | live Q3 plan | facet queries limited to comparison questions |
| Parser biased to LightRAG's layout: headings missed on bold-heading papers | human asked "is it document-biased?" → tested on BERT | bold + numbered/standard-name heading rule (test added) |
| Stacked 3-level table headers overwrote each other | BERT Table 5 | header rows above data accumulate (test added) |
| Windows long-path limit broke `pip install` in a deep folder | install error | venv at a short path (no system setting changed) |
| "Who wrote it?" retrieved nothing (the author line matches no topic) | testing in the UI | document card built from the layout, in every prompt |
| Twelve loosely related chunks per question; later rows of wide tables cut off for the reranker | reading evidence and ledgers | cross-encoder reranker; tables shown to it as caption + row labels + column headers |
| "Most cited paper", "page 13", "how many times…" unanswerable from fragments | testing in the UI | citation counter, direct lookup, term counts in code |
| A fix found on one paper could quietly encode that paper | human asked for a bias audit | scan of all runtime strings; two prompt phrases reworded; the scan became a test that fails the build |
| The daily free-tier quota ran out mid-benchmark | quota error | clear error instead of long retries; benchmark stops on quota errors; evaluation moved to a Colab GPU |
| The 14B planner sent specific questions to the summary route and once returned malformed JSON | reading planner output in the 14B run | planner guards; JSON repair and one retry; follow-up-aware fallback |
| Plain questions sent down the read-everything list route (16,610–27,041 tokens vs. ~4K) | reading the token ledgers | list route only for explicit "list all / every …" (test added) |
| Summaries built for documents nobody summarised | cost review | summaries built on the first request, then cached (test added) |
| Real numbers under the wrong row or column passed the number check | reading saved answers | table check (`app/numcheck.py`): 3 of 109 saved answers flagged, all genuine; one had passed the string rubric |
| A Q3 answer cut off at the 1,200-token output cap | reading the live answer | compact-table rule, 1,800-token reserve, a UI warning when an answer hits its cap |
| A summary request could wait forever behind its own background build | a test that hung | a waited-for summary stops yielding to user questions (test added) |
| A score quoted after the accepted wordings were widened | honesty review | every score reported per model and as first scored (36/41), with the widened version beside it |
| A saved Q3 answer defined `-Origin` wrongly and still passed the rubric | an outside review (ChatGPT) reading saved answers | Q3 must now state what `-Origin` removes; all saved answers re-scored by `evaluation/rescore.py` (27B: 27/27 → 26/27) |
| Without tiktoken, Chinese text was counted at ~0.6× its real size | checking the review's fallback-counter finding | fallback counts 2 tokens per CJK character; a test checks it never counts below tiktoken |
| The summary map and reduce prompts had no "untrusted text" rule | a new test that plants an instruction in an uploaded file | rule added to both prompts |
| Slide 13 quoted token counts from an earlier run | the same review | every figure on the slide now from the final run, labelled with model and run |
| The default 14B left Q3 incomplete in 3 of 3 fresh runs (the 27B: 3/3 complete) | repeating Q1–Q3 from fresh uploads, as the review asked | default switched to 27B; 14B documented as the offline option with this limit |
| The 14B answered "BANANA" to an instruction planted in a document | a live injection test | `app/guard.py` in three measured steps; then five new phrasings frozen before testing showed the patterns only catch known phrasings (27B resisted all; 14B partly followed 3) |

## What the human decided

* Make the default the model that answers all three questions (27B), keeping the 14B as the offline option.
* Fresh build over continuing the old project; local model first; Colab as the GPU path; bringing in the
  earlier traps list.
* Free, open-weights models only, and protect the free quotas: "if the Colab resources are gone, don't
  waste compute on it"; never spend the daily Groq quota on demo day.
* Ask whether the system was biased toward the test paper, which led to the bias audit and its test.
* Report every number per model and as first scored, including the failures.
* Keep the remaining Q2 error documented rather than tune the parser to this paper.

The AI asked when a choice was the human's. It did not ask about things it could verify itself (versions,
data layout, speed, token counts).

## Insights

* **Verification is the real work.** AI writes plausible code and plausible answers fast. The checks that
  do not trust it (token ledger, citation, number and table checks) found what reading alone missed: the
  table check flagged an answer that the string-match scoring had passed.
* **Measure the cost, not only the score.** A per-call limit is not a cost limit. Reading the token ledgers
  exposed a route reading the whole paper for plain questions; one guard brought them back to ~4K tokens.
* **Measure, don't assume.** The decisive facts (decimal corruption, 3 tok/s on CPU, timeouts behind
  background jobs) only appeared when real data and the real model were run. Unit tests alone passed throughout.
* **Give the AI a spec and a test, not only a prompt.** The 10K rule became a gate and a test; "don't tune
  to the test paper" became a test that fails the build. Spec first, tests as guardrails, a commit at each
  checkpoint.
* **A second opinion is input, not an answer.** Other models' reviews caught real gaps (prompt injection,
  citation validation, revision provenance, cost per question), and also proposed changes that were wrong
  or would have fitted one paper. Every claim was checked before it changed the code.
* **Test your own defences on cases you did not tune them on.** The injection patterns stopped every
  phrasing they were built against and none of five new ones; only a held-out set showed that.
* **Keep it small enough to explain.** About 2,900 lines of Python in 16 modules; each design choice maps to
  one requirement.
