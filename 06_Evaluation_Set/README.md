# Lab 06: How to build an evaluation set for RAG

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatwan/rag_for_nlp/blob/main/06_Evaluation_Set/06_evaluation_set.ipynb)

**CPU | API key: `OPENAI_API_KEY` (chat and embedding calls) | Practises: writing a golden set and scoring
retrieval, answers and refusals per question | In class**

## Goal

Build a fixed set of questions and use it to compare retrieval, answers and refusals one question at a time. You will
see a wrong duration score higher on word overlap than a correct sentence, then inspect the answers behind a pipeline's
aggregate scores.

Lab 04 helped us diagnose individual failures. Lab 05 rebuilt the pipeline with LangChain. Before changing that
pipeline, we need a repeatable way to see what improves and what breaks. The test questions, expected answers and
scoring rules give us that comparison. They also need inspection: a mistaken answer key can reward the wrong behaviour.

Parts A–D build the measures and ten evaluation records, then keep a row for every reply. The self-study sections
version the records, report the scores with their denominators and examine policy failures in a larger set.

## What you will learn

- Compute hit@3 and mean reciprocal rank (MRR@3), including misses in the denominator.
- Compute exact match (EM) and token F1 against accepted answers, and explain their limits on generated sentences.
- Write golden records with known support and accepted answers or an expected refusal, then version the set.
- Score a pipeline one question at a time and preserve the relationship between each question and its result.
- Report answer checks and refusals separately, and inspect each kind of question as its own slice.

**In class:** Parts A to D, except the versioning subsection at the end of Part C. That subsection and Parts E and F
are self-study.

## Coming from

[Lab 05](../05_Rebuild_With_LangChain/README.md) rebuilt the pipeline with LangChain and kept it in
`shared/lc_pipeline.py`, the baseline for the labs after it. Before you change that pipeline, you need questions to
measure every change on. This lab writes them, as a golden set kept outside the index, and scores the baseline on it.
The measures stay in plain Python, so every number can be checked by hand.

## What you will do

| Part | What it adds |
|---|---|
| A: Measure support ranks and answer word overlap by hand | Calculate hit@3, reciprocal rank, MRR@3, EM and word-token F1 on visible inputs; keep misses in the denominator. |
| B: Reuse the measures and test what they can miss | Turn the arithmetic into functions and compare a wrong duration with a right full sentence. |
| C: Define questions, accepted answers and required refusals | Inspect a Tesla record, convert seven development questions and add three policy cases. The final subsection fingerprints records, instructions and corpus (self-study). |
| D: Trace one evaluated answer, then score all ten questions | Reuse Lab 05's demonstrated LangChain components; inspect G02's ranked sources, evidence and answer, then preserve one result per record. |
| E: Choose checks and denominators that match the expected behaviour (self-study) | Report retrieval and answer measures separately; screen for accepted phrases and count required and unwanted refusals. |
| F: Find policy failures hidden by the whole-set score (self-study) | Score 38 records with the demonstrated evaluation helpers, distinguish a fixed evidence pack from search, inspect overlapping slices and repeat the eight policy cases. |
| Challenge: Test Boreal billing and reset questions | Decide the labels from the policies before scoring two cases. Optionally expand a reference list and compare its hash and scores. |

## Checkpoints

| After | You should see |
|---|---|
| Part A | reciprocal ranks 1, 0.333 and 0; MRR@3 0.444 (0.667 if the miss leaves the denominator); *about 30 minutes*: overlap 2, F1 0.800, EM 0 |
| Part B | the wrong *30 minutes* F1 0.500, the right Tesla sentence 0.222; the best of two references 0.800 |
| Part C | ten records, 1 to refuse; versioning (self-study): `.cache/golden-b5789bcc0fdd.json` |
| Part D | G02 retrieved H01, H04, H02 (rank 3), F1 0.333; ten rows, EM 1 only for *1884*, F1 0.000 for the complexity answer and G05, the weather question refused |
| Part E | hit@3 9 of 9 (1 not counted), MRR@3 0.870; EM 1 of 9, F1 0.371; `found()` 7 of 9; refused 1 of 1 and 0 of 9 |
| Part F | passed 34 of 38: development 29 of 30, policy cases 5 of 8; exception 0 of 1, wrong product 0 of 2, refusal slice 3 of 4; G05, G08, G09 failed 3 of 3 runs |

These are the saved student notebook's checkpoints. A pass means reference-phrase presence for an answer record or
an exact refusal for a refusal record; it does not certify factual support or citation quality. The 30 benchmark
questions dominate the 38-record total. Part F reports policy types and the overlapping refusal slice separately;
these small constructed sets do not estimate accuracy on new users' questions.

**About the saved outputs.** They come from `gpt-4o-mini` and `text-embedding-3-small`. Model replies vary from run to
run, so compare the pattern (which records pass and fail), not every word. On another model the failures can move to other questions; the checks, and the way to measure them,
stay the same.

## Before you start

- **Lab 00 first,** then Lab 05 Parts B–D for the direct embeddings/store examples. Lab 06 explains the cited-answer
  object and pipeline helpers where they are used; it also reuses Lab 02's `RULES`.
- **Open the notebook:** on Colab with the badge above (add `OPENAI_API_KEY` under Secrets first); locally or in a
  codespace, open `06_evaluation_set.ipynb` and choose the `.venv` kernel. Run Setup and Config first.
- **Data:** `data/development-questions.json`, `data/squad-chunks.json` and `data/support-case.json`, read in place.
  The eight policy cases are in `resources/golden-policy-cases.json`, constructed for these labs and reviewed (see
  Part F); they are not human annotations and not a benchmark.
- **Packages:** none beyond Lab 00's setup; the measures use only Python's standard library.
- **API usage:** through Part F, one corpus embedding (about 21,000 tokens), embeddings for search questions and
  65 chat calls. Challenge Task 1 adds six searches and six chat calls. Re-running cells makes new calls; cost
  depends on the configured provider, models and token usage.

The notebook writes two files to `.cache/` (gitignored): the ten records and their version metadata (Part C's
self-study subsection), then the first 38-row evaluation and its version (Part F). Filenames include the records
hash; the results filename also includes the chat model. Re-running can overwrite the same path. Inspect metadata
inside the file when comparing runs; the policy repeat counts are printed but are not saved in that 38-row file. It ends with a troubleshooting table: an F1 that divides by zero, chunk IDs compared with source
IDs, a reciprocal rank of a miss, and an average over refusal records.

## Resources

- [Rajpurkar et al., *SQuAD*, 2016](https://arxiv.org/abs/1606.05250): the dataset of the development questions, its
  several answers per question, and EM and F1
- [Voorhees, *The TREC-8 Question Answering Track Report*, 1999](https://trec.nist.gov/pubs/trec8/papers/qa_report.pdf):
  the question-answering evaluation that scored systems by mean reciprocal rank
- [Gao et al., *Enabling Large Language Models to Generate Text with Citations* (ALCE), 2023](https://arxiv.org/abs/2305.14627):
  exact match recall, a short reference found inside a long generated answer
- [LangSmith: evaluation concepts](https://docs.langchain.com/langsmith/evaluation-concepts): datasets, reference
  outputs, evaluators and experiments as a hosted service

Next: [Lab 07](../07_LLM_Judge_Calibration/README.md) (check an LLM judge against your own labels). Back to [the lab list](../README.md).
