# Lab 07: How to check an LLM judge against your own labels

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatwan/rag_for_nlp/blob/main/07_LLM_Judge_Calibration/07_llm_judge_calibration.ipynb)

**CPU | API key: `OPENAI_API_KEY` (chat, judge and embedding calls) | Practises: auditing claims and qualifiers, and
checking an LLM judge against human labels | In class**

## Goal

Check a model that grades RAG answers before relying on its verdicts. You will label five answers yourself, give a
judge the same questions and evidence, then inspect where it disagrees. In the saved example, the judge accepts two
answers that apply one product's policy to the other, even though its quoted words really occur in the evidence.

Lab 06 checked answer strings. Here we ask whether the evidence supports what an answer means, including the product
and the conditions of a rule. The judge helps with that reading, but its verdict and explanation both need checking.
The self-study sections also examine summary omissions and twelve recorded answers from a small open model.

## What you will learn

- Write a summary's required claims before reading it, and distinguish changed qualifiers from omissions.
- Label answers for faithfulness and save those human labels before seeing judge outputs.
- Give a judge a rubric, a reply schema and visible question/evidence/answer inputs.
- Compare human labels and judge verdicts in a confusion matrix, with explicit denominators for agreement and false passes.
- Check quoted spans and product names in code, and explain what each check leaves unresolved.
- Inspect the claims and reasons behind a verdict, then reuse the demonstrated judge through a shared helper.

**In class:** Parts B to E, without the two self-study subsections at the end of Part D. Part B's labels are filled in
for you: read each one against your own answers and change any that does not fit. Run every cell from the top, Part
A's included (it needs no input), and read Part A later. Parts A, F and G are self-study, and Part F's twelve
labels are the only ones you type: the notebook stops at that cell until you fill them.


## Coming from

[Lab 06](../06_Evaluation_Set/README.md) scored retrieval, answer strings and refusals. Those checks leave product
relationships and qualifiers for a reader to assess. Here a judge reads the actual question, evidence and answer;
human labels stay outside its prompt so we can compare its verdicts afterwards. The answer pipeline is the one
built in Lab 05 and explained on the path in Lab 06 Part D. Judge calls use LangChain; the counts and code checks
stay in plain Python.

## What you will do

| Part | Purpose |
|---|---|
| A: Check a summary's claims, qualifiers and omissions (self-study) | Write four requirements first, compare two summaries and distinguish the coarse tally from strict faithfulness |
| B: Label five answers from their actual evidence | Read a golden record and result row, inspect all five input triples and save human labels separately from model outputs |
| C: Give the judge a rubric and inspect one structured verdict | Interpret the rubric/schema, display G01's exact input and inspect LangChain's parsed/raw reply |
| D: Compare judge verdicts with human labels and inspect false passes | Trace individual label pairs into a confusion matrix, choose rate denominators and inspect the wrong-product claims; repeats and sample size are self-study |
| E: Check quoted spans and product matching separately | Inspect normalised text, test span occurrence and product names, and explain empty-list and missing-citation limits |
| F: Inspect judge decisions on recorded answers from another model (self-study) | Label twelve fixed answers, inspect claims behind agreeing verdicts and check source IDs |
| G: Reuse the demonstrated judge through a shared module (self-study) | Connect each helper to the public library calls and checks already used, then judge G08 again |
| Going further | Reverse fixed evidence lists and test verdict stability |
| Challenge | Test an added product instruction with a supported control; optionally compare an instruction to the grader with a planted fact |

Human labels, judge verdicts and string-check passes answer different questions. The matrix measures final-label
agreement on selected examples; it cannot validate the judge's reasons. A found span establishes word occurrence,
and valid structure makes a reply readable by code. Neither establishes factual support. The product-name check
catches the saved mismatches but can pass empty comparisons, missing citations or a line naming both products.

## Checkpoints

| After | You should see |
|---|---|
| Part A | your summary: supported 4 of 4, qualifiers kept 4 of 4, faithful 1; A03: supported 3 of 4, qualifiers kept 2 of 4, faithful 0 |
| Part B | 277 documents; five answers; G08 *Only workspace owners can export all reports in Boreal.*; G09 *Yes, password reset uses a mobile app.*; labels `['1', '1', '0', '0', 'NA']` saved in `.cache/labels-<hash>.json` |
| Part C | G01 question/evidence/answer JSON printed; one judge claim, S, H02; F 1; parsing error None; 512 input tokens |
| Part D | the matrix: you 1 row 2 and 0, you 0 row 2 and 0, 1 pair with an NA; agreement 2 of 4, false-pass rate 2 of 2; G08 and G09 passed in 3 of 3 runs (self-study) |
| Part E | every span found; `product_ok()` `False` for G08 and G09 only |
| Part F | the matrix: 7 and 0, 0 and 4, 1 pair with an NA, `differ: {}`; A05's one claim is H02's sentence; `A05 evidence not in the index: ['H99']` |
| Part G | `same rubric: True`; G08 F 1 again |

**About the saved outputs.** They come from `gpt-4o-mini` (answers and judge) and `text-embedding-3-small`. Your
labels are your own: read the answers in front of you, and compare which answers the judge gets wrong, not our counts.
The pipeline's answer to G09 moves between runs: about one run in ten it is *Insufficient evidence.*, G09 is then NA on
both sides, and the matrix holds one false pass (G08) instead of two. Which errors a judge makes depends on the model;
the way to find them, your labels first, does not.

## Before you start

- **Lab 00 first,** then Lab 06: this lab uses `shared/evals.py` and Lab 05's `shared/lc_pipeline.py`.
- **Open the notebook:** on Colab with the badge above (add `OPENAI_API_KEY` under Secrets first); locally or in a
  codespace, open `07_llm_judge_calibration.ipynb` and choose the `.venv` kernel. Run Setup and Config first.
- **Labels:** Parts A and B hold our labels, filled in; read them against what your run printed and change any that
  does not fit. Part F's label cell starts empty and stops the notebook until you label the twelve recorded answers.
- **Data:** `data/support-case.json` and the corpus, read in place. `resources/recorded-answers.json` holds 12 answers
  recorded from Qwen2.5-0.5B-Instruct (revision 7ae5576, CPU, greedy decoding) on 5 October 2026, each with the
  evidence it was given: recorded outputs, unchanged, with no labels. The SQuAD passages in their evidence are
  CC BY-SA 4.0 (see `data/README.md`).
- **API use:** one embedding of the corpus (about 21,000 tokens), then answer and judge calls. Repeats, order tests
  and completed challenges add calls. Check your configured provider's token prices before running them.

The notebook writes your labels to `.cache/` (gitignored). It ends with a troubleshooting table: an empty label cell,
`json_object` with a rubric that never says JSON, a reasoning model as judge without `REASONING_EFFORT=none`, and the
evidence passed to `span_found()` as one string.

## Resources

- [LangChain: models and structured output](https://docs.langchain.com/oss/python/langchain/models#structured-output):
  `with_structured_output()`, Pydantic schemas and the parsed/raw return values
- [Zheng et al., *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*, 2023](https://arxiv.org/abs/2306.05685):
  position, verbosity and self-preference bias in model judges, and agreement with human labels
- [OpenAI: evaluation best practices](https://developers.openai.com/api/docs/guides/evaluation-best-practices):
  model graders and how to check them against human judgement
- [LangSmith: evaluation concepts](https://docs.langchain.com/langsmith/evaluation-concepts): LLM-as-judge evaluators
  and human review as a hosted service
- [Es et al., *RAGAS*, 2023](https://arxiv.org/abs/2309.15217): faithfulness as the share of an answer's claims the
  context supports

Next: [Lab 08](../08_Hybrid_Search_And_Reranking/README.md) (improve retrieval with hybrid search and reranking). Back
to [the lab list](../README.md).
