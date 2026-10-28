# Lab 05: How to rebuild a RAG pipeline with LangChain

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatwan/rag_for_nlp/blob/main/05_Rebuild_With_LangChain/05_rebuild_with_langchain.ipynb)

**CPU | API key: `OPENAI_API_KEY` (chat and embedding calls) | Practises: building a RAG pipeline from a framework's
components and comparing it with your own | Demo**

## Goal

Rebuild our plain Python RAG pipeline with LangChain and check which behaviour is preserved. Compare retrieval results question by question, generate cited answers from shared evidence, and use metadata to scope a customer's search.

The first comparison shows why this matters: both searches find a labelled source for every development question, yet their complete result lists differ because one returns an extra chunk. A total alone hides the default that changed the evidence. Parts A–D compare retrieval; Parts E–H continue with answers, product filtering and the reusable pipeline.

## What you will learn

- How to turn chunk records into LangChain documents that keep their IDs and fields.
- How LangChain's embeddings and chat model are built from `models.env`, and what the embeddings object sends to the
  API.
- How to index documents in a vector store and search them with scores.
- How to compare a framework's results with your own question by question, and find the default that made them
  differ.
- How to get a cited answer back as a Python object, check it with Lab 02's quote check, and explain why its answers
  can differ from your own.
- How to leave documents out of a search with a metadata filter.

**In class:** Part A, Part B through Checkpoint B, then Parts C and D, shown as a demo. Part B first uses
`OpenAIEmbeddings` directly, then introduces the shared helper. The remaining embedding comparison in Part B and
Parts E to H are self-study.

## Coming from

You built the search (Lab 01) and the cited answer with its checks (Lab 02) in plain Python, and you can measure
both. This lab moves the same pipeline onto LangChain, the framework the labs after it use, and compares the two on
your own questions, step by step, so you know what changed before you build on it.

## What you will do

**Part A: Represent Our Chunks as LangChain Documents.** Inspect an original dictionary, map its text, ID and metadata to one `Document`, then convert all 277 records.

**Part B: Configure LangChain Embeddings and Compare Their Vectors.** Construct `OpenAIEmbeddings` directly, embed one question and two short policy statements, then use `lc_embeddings()` to reuse the configuration. As self-study, compare with Lab 01's `embed()` and distinguish input token IDs from output vectors.

**Part C: Store Document Embeddings and Retrieve Similar Chunks.** Embed the documents into `InMemoryVectorStore`, then compare one search's returned objects, IDs, scores and row counts with our plain search.

**Part D: Compare Retrieval Results and Check Search Defaults.** Compare all 30 question results, find the default `k=4`, measure the extra content tokens, and check retrieval again with an explicit `k=3`.

**Part E: Generate Cited Answers as Structured Python Objects (self-study).** Inspect a chat reply, describe claims with Pydantic, generate a `CitedAnswer`, and reuse our quote check.

**Part F: Compare Answer Behaviour Using the Same Evidence (self-study).** Give both answer steps identical lines, inspect differing replies, repeat the plain implementation as a control, and test abstention.

**Part G: Restrict Retrieval to the Customer's Product (self-study).** Use the customer's product to filter eligible documents, then compare evidence and answers for the same question.

**Part H: Reuse the LangChain Pipeline and Compare Its Evidence (self-study).** Run the shared pipeline, then compare its three-document evidence with the earlier budgeted five-hit pack.

**Challenge.** Compare a retriever against our search. Optional: scope retrieval to Boreal and test a question its policy cannot answer.

## Checkpoints

| After | You should see |
|---|---|
| Part A | 277 documents; H05 with `version` and `access` in its metadata |
| Part B | direct `OpenAIEmbeddings` usage: one vector for a question, two for the document list (1,536 values each under the default model); helper `send token IDs: True`; as self-study, 1,536 numbers with largest difference from `embed()` 0.000000, and the question as eight token IDs `[644, 1148, ...]` |
| Part C | the same top three for the Tesla question, scores within a thousandth; the store prints a fourth row |
| Part D | initial source-hit counts 30 and 30 of 30 (three results vs four); same lists 0 of 30; `k: int = 4`; 9,946 evidence tokens against 7,764; with `k=3`, same lists 30 of 30, largest score difference 0.0021 |
| Part E | `temperature: 0.0`; the same required fields as Lab 02's `SCHEMA`, with `$defs` and `title` in Pydantic's; *Nikola Tesla was born in the year 1856.*, quote found |
| Part F | both refuse 0 of 30 and every quote is found (30 and 32 claims); same cited IDs 28 of 30, same text 28 of 30 (plain against itself 30 of 30); the Moon question abstained 3 of 3 in each |
| Part G | without the filter H02, H04, H01 and both products' times; with `atlas_only` H02, H01, H05 and Atlas only, 3 of 3 |
| Part H | the module: Tesla with its quote found, *Insufficient evidence.* for the Moon; `pack()` 125 lines and 10,761 tokens, `evidence()` 90 lines and 8,382 tokens, more lines from `pack()` for 23 of 30 |

**About the saved outputs.** They come from `gpt-4o-mini` and `text-embedding-3-small`. Embedding scores can move
slightly, and model replies can change in wording, citations and refusal behaviour. Compare the returned evidence and
check results, using a repeated run of one implementation to put differences between implementations in context.

## Before you start

- **Lab 00 first,** then Labs 01 and 02: this lab uses `shared/search.py`, `shared/answer.py` and the cached chunk
  vectors.
- **Open the notebook:** on Colab with the badge above (add `OPENAI_API_KEY` under Secrets first); locally or in a
  codespace, open `05_rebuild_with_langchain.ipynb` and choose the `.venv` kernel. Run Setup and Config first.
- **Data:** `data/squad-chunks.json`, `data/support-case.json` and `data/development-questions.json`, read in place.
- **Packages:** `langchain-core` and `langchain-openai`, already installed by Lab 00's setup. No `langchain-community`.
- **Cost:** about 110 short chat calls and two or three embeddings of the corpus (about 21,000 tokens each), well
  under one US cent on `gpt-4o-mini`.

The notebook ends with a troubleshooting table: an import from an older tutorial, a missing key, `temperature` on a
reasoning model, a dictionary schema without a name, and a filter that reads a field some documents lack.

## Resources

- [LangChain: build a semantic search engine](https://docs.langchain.com/oss/python/langchain/knowledge-base):
  documents, embeddings, vector stores and retrievers
- [LangChain: models, structured output](https://docs.langchain.com/oss/python/langchain/models#structured-output):
  `with_structured_output` with a Pydantic class, its methods, and `include_raw=True` for the token counts
- [LangChain reference: `InMemoryVectorStore`](https://reference.langchain.com/python/langchain-core/vectorstores/in_memory/InMemoryVectorStore):
  every search method, its defaults and the filter argument
- [LangChain reference: `OpenAIEmbeddings`](https://reference.langchain.com/python/langchain-openai/embeddings/base/OpenAIEmbeddings):
  `check_embedding_ctx_length` and the settings for other OpenAI-compatible servers
- [LangChain: OpenAIEmbeddings integration](https://docs.langchain.com/oss/python/integrations/embeddings/openai):
  constructing the embeddings object and calling `embed_query()` and `embed_documents()` directly
- [Pydantic: models](https://docs.pydantic.dev/latest/concepts/models/): fields, validation and `model_dump()`

Next: [Lab 06](../06_Evaluation_Set/README.md) (build an evaluation set). Back to [the lab list](../README.md).
