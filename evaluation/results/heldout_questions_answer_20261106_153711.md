# DocChat evaluation — answer mode (20261106_153711)

Model: `qwen2.5:14b` · context limit 10000 tokens

**Overall: 16/18 passed (89%)**

| Group | Passed |
|---|---|
| type: abstain | 1/1 |
| type: core | 1/1 |
| type: citation | 1/1 |
| type: comparison | 1/1 |
| type: fact | 6/6 |
| type: list | 1/1 |
| type: metadata | 3/3 |
| type: multilingual | 1/1 |
| type: structure | 1/1 |
| type: table | 0/2 |
| doc: rag | 16/18 |

Invalid citations: 0 answers · answers with ungrounded numbers: 0 · all LLM calls within limit: True

| | Doc | Type | Question | Missing |
|---|---|---|---|---|
| ✅ | rag | metadata | What is the title of this paper? |  |
| ✅ | rag | metadata | Who is the first author? |  |
| ✅ | rag | metadata | Which institutions are the authors from? |  |
| ✅ | rag | fact | What are the two RAG formulations proposed in the paper? |  |
| ✅ | rag | fact | Which retriever does RAG use? |  |
| ✅ | rag | fact | Which generator model is used, and how many parameters does it have? |  |
| ✅ | rag | fact | How many documents are in the retrieval index? |  |
| ✅ | rag | fact | Which Wikipedia dump is used as the knowledge source? |  |
| ❌ | rag | table | What exact match score does RAG-Sequence get on Natural Questions? | 44.5 |
| ❌ | rag | table | What score does RAG-Token get on WebQuestions? | 45.5 |
| ✅ | rag | structure | Explain Table 1. |  |
| ✅ | rag | fact | What hardware was used for training? |  |
| ✅ | rag | comparison | Compare RAG-Sequence with RAG-Token. |  |
| ✅ | rag | core | summary this document. |  |
| ✅ | rag | list | List all the tasks RAG is evaluated on. |  |
| ✅ | rag | citation | Which paper is cited the most? |  |
| ✅ | rag | multilingual | 這篇論文的第一作者是誰？ |  |
| ✅ | rag | abstain | What score does RAG achieve on WMT14 English-German translation? |  |
