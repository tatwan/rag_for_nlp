# Lab 08: How to improve retrieval with hybrid search and reranking

**CPU | API key: `OPENAI_API_KEY` (embedding and chat calls) | Practises: comparing retrieval changes | In class**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatwan/rag_for_nlp/blob/main/08_Hybrid_Search_And_Reranking/08_hybrid_search_and_reranking.ipynb)

## Goal

Build a keyword search and a reranker beside the existing vector search, then decide whether either change earns
its cost. You will see keyword matching rescue one question, fusion lower another, and a chat model reorder a policy
question's candidates correctly while sometimes returning invalid IDs.

Lab 06 gave us a fixed question set and separate retrieval and answer measures. Lab 07 showed why a judge needs
checking too. Here we change the retrieval order and measure the gains and losses on the same questions. A better
rank is useful only if the retrieved text supports the answer we need.

## What you will learn

- How BM25, the classic keyword ranking, scores a chunk from the words it shares with a question.
- How reciprocal rank fusion merges two ranked lists by their ranks instead of their scores.
- How to compare a change with the baseline question by question, and read a sign test on the questions that moved.
- How to rerank candidates with a chat model, and check in code that the reply names only the candidates.
- How to price a retrieval stage in tokens, dollars and seconds.
- How to write a decision from the paired results and the cost.

**In class:** Parts B to D, then Part E up to Checkpoint E (one rerank call). Part A, the ID check after
Checkpoint E, and Parts F to H are self-study.


## Coming from

Lab 05 demonstrates the LangChain Documents, embeddings and vector store reused here. Lab 06 explains known-support
ranks and golden records; Lab 07 separates a model's verdict from the evidence behind it. Here we compare retrieval
orders on the same questions. We inspect candidate text because a source hit or valid ID list does not establish
that an answer is factually supported.

## What you will do

| Part | What it establishes |
|---|---|
| A: Score the five policies with BM25 by hand (self-study) | Inspect policy texts, calculate a word's contribution and compare direct `bm25s` scores with the hand formula. |
| B: Compare keyword and vector rankings on the same corpus | Read the MVP candidates, construct and search a BM25 index, then keep each of 30 questions aligned with its source ranks. |
| C: Combine rankings and trace why fusion can lower an answer | Work a small RRF sum, explain `c=60` and depth 10, then inspect the Norman question's candidates and moved source. |
| D: Compare hybrid search with the baseline, question by question | Reconcile totals with the paired table, inspect gains/losses and interpret a sign test with its assumptions. |
| E: Let a chat model reorder the recovery question’s candidates | Inspect question/passage inputs, use structured output directly and interpret one rerank call. The candidate-permutation check after Checkpoint E is self-study. |
| F: Compare validated rerankings and measure their added cost (self-study) | Separate proposed order from fallback, inspect invalid IDs, repeat one fixed query and measure the 30-question comparison's tokens and time. |
| G: Inspect policy questions before choosing a retrieval change (self-study) | Read five policy questions and the distinctions their candidates must preserve, then write a decision with its limits. |
| H: Reuse the demonstrated search and comparison functions (self-study) | Read `shared/hybrid.py` and compare it with the direct calls and functions already demonstrated. |

Going further uses maximal marginal relevance (MMR) to select less repetitive candidates. The challenge repairs a
rejected ranking; an optional task tests candidate-order sensitivity on the same recovery query.

## Checkpoints

These describe the student notebook's saved outputs. The reranker's exact replies and timings can vary.

| After | What to inspect |
|---|---|
| A | Hand scores H01 1.705, H02 1.519, H04 1.412; the library score differs by a constant factor of 2.20. |
| B | 277 records and 4,083 distinct words; MVP's known source is vector rank 2 and BM25 rank 1. Vector hit@1: 29; hit@3: 30. BM25 hit@1: 22; hit@3: 28, all out of 30. |
| C | A 0.032522, B 0.032266, C 0.016129 in the small example. Norman's labelled source falls to fourth after fusion. |
| D | Hybrid hit@1/hit@3: 27/29, each out of 30. At hit@1: 26 both, 1 gained, 3 lost, 0 neither; p = 0.625. At hit@3: 0 gained, 1 lost; p = 1.000. |
| E | H02 ranks 3, 6 and 4 before reranking (vector/BM25/hybrid), then first in the model order; 561 input and 54 output tokens. The ID permutation check returns True. |
| F | 28 of 30 replies pass the ID check; the five repeats of one Saxon Garden query fail four times. At hit@1 the validated/fallback policy gains MVP and loses the computational measure. Reranking averages 961 input tokens, 74 output tokens and 1,617 ms per question. |
| G | Known policy first: vector 3/5, hybrid 3/5, reranked/fallback 4/5. The mobile-app proposal starts with H01 but omits H03, so fallback retains H04. |
| H | Exact BM25 and fused ID-list equality for 30/30 questions; p = 0.625 both ways; the additional recovery call puts H02 first and passes validation. |

The 30 development questions share five articles, and the five policy cases are closely related. Paired rows
show which outcomes changed. A large p-value does not establish equivalence, and repeated calls on one question
do not create independent questions. These are development comparisons rather than deployment reliability estimates.

**About the saved outputs.** They use `gpt-4o-mini` and `text-embedding-3-small`. Checkpoints interpret the displayed
results; model replies, similarity scores and timings can vary. The dollar estimate uses your configured prices and
covers reranking calls only. Read invalid replies and changed candidate texts before judging the totals.

## Before you start

- Run Lab 00's pre-check and read Lab 05's direct library examples. Packages include `bm25s` and LangChain's core components.
- Open `08_hybrid_search_and_reranking.ipynb`, select the project `.venv` kernel and run Setup and Config. On Colab, use the badge and add `OPENAI_API_KEY` under Secrets.
- The corpus uses `data/squad-chunks.json` and `data/support-case.json`. The 30 question labels are in `data/development-questions.json`; answer-key fields stay with the evaluator.
- Through Part H, there are 42 rerank calls, plus five for the optional shuffle challenge. Corpus and query embeddings also incur usage; repeated cells make new calls. The in-class reranking demonstration makes one chat call.

Part F's saved estimate is 0.19 US dollars per 1,000 reranked questions, projected from the 30 calls at the configured
prices. It excludes indexing, embedding charges, answer generation, infrastructure and provider-specific discounts.
Change the prices when you change the rerank model or provider. Search timers use already-built indexes and do not
measure full end-to-end answer latency.

The notebook ends with troubleshooting for tokenizer defaults, an oversized `k`, JSON mode and model settings.

## Resources

- [`bm25s` documentation](https://bm25s.github.io/): the BM25 package, its tokenizer options and scoring variants
- [Stanford IR book: Okapi BM25](https://nlp.stanford.edu/IR-book/html/htmledition/okapi-bm25-a-non-binary-model-1.html): term weighting, IDF, `k1` and `b`
- [Cormack, Clarke and Büttcher (2009)](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf): reciprocal rank
  fusion and `c = 60`
- [Elasticsearch: reciprocal rank fusion](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion)
  and [Qdrant: hybrid queries](https://qdrant.tech/documentation/concepts/hybrid-queries/): hybrid search inside a
  search engine and a vector database
- [Sun and others (2023), RankGPT](https://arxiv.org/abs/2304.09542): listwise reranking with a chat model
- [Cohere: rerank overview](https://docs.cohere.com/docs/rerank-overview): a production cross-encoder reranker

Next: [Lab 09: How to answer follow-up questions with retrieval](../09_Follow_Up_Questions/README.md). Back to
[the lab list](../README.md).

- [LangChain models: structured output](https://docs.langchain.com/oss/python/langchain/models#structured-output): parsed and raw replies from a model.
