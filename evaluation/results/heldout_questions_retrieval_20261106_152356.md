# DocChat evaluation — retrieval mode (20261106_152356)

Model: `qwen2.5:14b` · context limit 10000 tokens

**Overall: 14/15 passed (93%)**

| Group | Passed |
|---|---|
| type: citation | 1/1 |
| type: comparison | 1/1 |
| type: fact | 6/6 |
| type: metadata | 3/3 |
| type: multilingual | 1/1 |
| type: structure | 1/1 |
| type: table | 1/2 |
| doc: rag | 14/15 |

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
| ✅ | rag | table | What exact match score does RAG-Sequence get on Natural Questions? |  |
| ❌ | rag | table | What score does RAG-Token get on WebQuestions? | 45.5 |
| ✅ | rag | structure | Explain Table 1. |  |
| ✅ | rag | fact | What hardware was used for training? |  |
| ✅ | rag | comparison | Compare RAG-Sequence with RAG-Token. |  |
| ✅ | rag | citation | Which paper is cited the most? |  |
| ✅ | rag | multilingual | 這篇論文的第一作者是誰？ |  |
