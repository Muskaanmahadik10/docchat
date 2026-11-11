# Results index: every number, and the file behind it

All runs are open-weights models under the same 10,000-token-per-call limit. Two models were measured:
**Qwen 3.8 27B** (`qwen/qwen3.8-27b` on Groq, the default configuration) and **qwen2.5:14b** (Ollama on
a Colab T4, the offline option). Scripts that produced each file are in `evaluation/`.

## The three core test questions, repeated from fresh uploads (2026-11-07)

Each round deleted the document, uploaded the paper again and asked the question with no cached summary.
Scored with the current rubric (Q3 must also say what `-Origin` removes) and read by hand against Table 2.

| Question | Qwen 3.8 27B (default) | qwen2.5:14b (offline) |
|---|---|---|
| Q1 summary | 1/1 correct (first build: 22,841 tokens in all, largest call 4,927) | 3/3 correct (20,649–21,399 tokens, largest 4,741) |
| Q2 compare | 2/2 correct (5,440 and 5,508 tokens) | 2/3: one answer gave no cost figures; one said LightRAG wins "across all dimensions" although GraphRAG narrowly wins on Mix |
| Q3 ablation | **3/3 complete**: all 4 variants × 4 datasets, every value matches Table 2, `-Origin` defined correctly (6,189–6,540 tokens, largest call 6,065) | **0/3 complete**: numbers for only 2 of 4 variants; full model only; no variant numbers and `-Origin` defined wrongly |
| Every call within 10,000 | ✓ | ✓ |

Files: `core_questions_repeat_27b_20261107.json` (Q1 once, Q2 twice, Q3 three times, to save API quota),
`core_questions_repeat_14b_20261107.json` (three full rounds; run before the stricter Q3 rubric and before the
injection defence, re-scored by hand here). Reproduce: `python evaluation/repeat_core_questions.py --pdf <paper> --plan q3-heavy`.

Notes on the 27B answers: one Q3 answer drew two table-check warnings that are false alarms (the sentence
names both the full model and, through its heading, the `-Low` variant); the number check listed correct
derived values ("> 50%", "exceed 80%", 1,399 × 2 × 5,000 = 13.99 million).

Because of these runs the default configuration moved from the 14B to the 27B.

Asked back to back on the demo machine, Groq's ~1,000-output-tokens-per-minute limit cut a 27B Q3 answer at 950
tokens (one repeat-run answer also ran at 950 and still fitted). Answers that stop at their cap are now continued
automatically; the re-run completed Q3 (all 4 variants × 4 datasets) in two answer calls, the larger 6,790 tokens.

## Instructions planted inside an uploaded document (2026-11-07)

A short policy states that belts are inspected every 30 days; one planted sentence per run tries to change
the answer. DEV phrasings were used while building `app/guard.py`; HELDOUT phrasings were written before
any run against them and never used for tuning.

| Defence step | Model | DEV phrasings followed | HELDOUT phrasings | File |
|---|---|---|---|---|
| System rule only (before) | 14B | answered "BANANA" (1 of 2 questions) | – | `core_questions_repeat_14b_20261107.json` (`injection`) |
| + `<source>` framing, labels, rule repeated before the answer | 14B | 9/12 | – | `injection_14b_step1_framing_and_labels_20261107.json` |
| + clear instructions removed from the prompt | 14B | 6/12 | – | `injection_14b_step2_remove_clear_cases_20261107.json` |
| + softer instruction-shaped sentences removed (final) | 14B | **0/8** | resisted 2 of 5 fully; answered in French when told to; repeated the planted interval beside the real one in 2 | `injection_14b_step3_final_20261107.json` |
| final | **27B** | **0/4** | **resisted 5/5** (2 answers quote the planted sentence only to warn the user) | `injection_27b_final_20261107.json` |

The detector caught none of the 5 held-out phrasings: pattern matching only removes phrasings it knows.
On new phrasings the defence rests on the framing and the model, which is another reason for the 27B
default. The harmless control sentence was never flagged or removed, and the detector raised no alarm on
any of the six real papers. Reproduce: `python evaluation/injection_test.py --set all`.

## Earlier runs (2026-11-06)

| Claim | Result | File |
|---|---|---|
| Retrieval: the needed facts reach the evidence (no LLM) | **35/35** | `retrieval_20261106_113744.md` (final; earlier `retrieval_*_09*.md` runs used 24 questions and are superseded) |
| Retrieval ablation, layer by layer | 27 → 33 → 33 → 34 → 35 of 35 | `ablation_20261106_112619.md` |
| Answers, 27 dev questions, 27B (pipeline of stage 6) | **26/27** under the current rubric (27/27 as first scored; its Q3 defined `-Origin` wrongly) | `answer_20261106_100032.md` |
| Answers, 41 dev questions, 14B | 36/41 as first scored; 39/41 after 3 accepted wordings were widened | `answer_20261106_163351.md` |
| Re-run of 4 questions after the 14B planner fixes | 3/4 | `answer_20261106_164133.md` |
| Retrieval, unseen paper (RAG, arXiv 2005.11401) | 14/15 | `heldout_questions_retrieval_20261106_152356.md` |
| Answers, unseen paper, 14B (questions frozen before the run) | 16/18 | `heldout_questions_answer_20261106_153711.md` |
| Robustness: confusing questions + a 12-turn conversation, 14B | 7/7 and 11/12 | `robustness_questions_answer_20261106_155826.md` |

## Checks run over saved answers (no model calls)

| Check | Result | Reproduce |
|---|---|---|
| Re-score every saved answer with the current rubric | only the 27B Q3 answer changed (pass → fail); the 14B dev run is 39/41 under the widened wordings | `python evaluation/rescore.py` |
| Table check over all 109 saved answers (2026-11-06 runs) | **3 flagged, all genuine**: a 14B Q3 "40.0%" that is NaiveRAG's share; held-out "RAG-Sequence 54.8" (RAG-Token's value; true 44.5); held-out "Explain Table 1" with four scores on the wrong tasks, which the rubric had passed | `python evaluation/tablecheck_saved.py --data data` (needs the papers uploaded once) |

## Cost of a question (final code, 27B)

| Question | LLM calls | Tokens, all calls | Largest call |
|---|---|---|---|
| Q2 / Q3 | 2 | 5,310 / 6,882 (repeats: 5,440–5,508 / 6,189–6,540) | 4,785 / 6,407 |
| Q1, notes already built | 1 | 4,496 | 4,496 |
| Q1, first time on a document | 1 + 4 map calls | about 22,400 (repeat: 22,841) | 4,802 (repeat: 4,927) |
| The whole paper in one prompt, for comparison | 1 | ≈ 17,000 | does not fit 10K |

Source: the token ledgers in `docs/screens/q1.png`–`q3.png` and the repeat run above.
