# Lab 09: How to answer follow-up questions with retrieval

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatwan/rag_for_nlp/blob/main/09_Follow_Up_Questions/09_follow_up_questions.ipynb)

**CPU | API key: `OPENAI_API_KEY` (chat and embedding calls) | Practises: turning a follow-up question into a search
that finds the right evidence | In class**

## Goal

Carry a user's meaning from earlier turns into a search query, then inspect whether the retrieved evidence supports
an answer. After a Boreal password-reset conversation, *How long is the link valid?* retrieves Atlas's policy first.
The answer gives Atlas's 30 minutes with a real quote. Rewriting the question to name Boreal produces the relevant
10-minute answer in the saved examples.

Search receives a string, not the conversation in your head. We will first see what the last turn loses, then make
the missing product and task explicit. The self-study parts let a model propose its own search through a tool,
compare those queries on seven conversations, and ask for clarification when the product remains uncertain.

## What you will learn

- Trace how a follow-up can lose its product or task while a quoted citation still passes its check.
- Rewrite a conversation's last turn as a standalone query while retaining its account type and conditions.
- Check product names and added digits, and explain why these checks cannot establish meaning or factual support.
- Inspect a public LangChain tool's schema, bind it to a model, and execute a proposed call with its matching result.
- Compare routes on the same conversations and distinguish repeated calls from additional test cases.
- Ask which product the user means, and separate ambiguity from a gap in the available evidence.

**In class:** Parts A to C, except the ten-run count at the end of Part A. That subsection and Parts D, E and F are
self-study.

## Coming from

[Lab 05](../05_Rebuild_With_LangChain/README.md) demonstrated LangChain documents, embeddings, a vector store and
structured answers. Lab 06 Part D connects those operations in `shared/lc_pipeline.py`. Here we reuse that pipeline
and Lab 02's quote check to locate a new failure: the meaning of the last turn can be missing from the search input.

## What you will do

| Part | What it adds |
|---|---|
| A: Trace what a follow-up loses when searched alone | Inspect history, candidate text, evidence, one answer and quote checks before repeating calls |
| B: Carry the product and task into a standalone query | Inspect rewrite messages and one query before reusing the call on another conversation |
| C: Screen rewritten queries for missing products and added digits | Test two checks on good and bad inputs; examine the temporary-account condition they cannot test |
| D: Inspect and execute a model-proposed search (self-study) | Read tool schema and arguments, execute one call, return its result, then combine the steps in a loop |
| E: Compare query failures on the same seven conversations (self-study) | Read actual queries, compare outcomes per case and distinguish query checks from final-answer support |
| F: Ask for the product when history permits two readings (self-study) | Compare a prompt's unnecessary asks with a limited rule, then join clarification and answering |
| Challenge | Compare three answer routes on a new product switch; optionally test a less specific rewrite instruction |

The ten-call subsection in A is self-study. Repeated calls measure stability on fixed inputs, not additional
independent conversations. The two query screens print diagnostics; they do not block retrieval. A matching policy
ID tests a retrieval label, while support for the answer still requires reading the product, task and conditions.

## Checkpoints

| After | You should see |
|---|---|
| Part A | *How long is the link valid?*: H02 first (about 0.59), H04 second (about 0.38); one answer *Atlas reset links expire after 30 minutes.* citing H02, quote found; three answers: Atlas's 30 minutes alone 1 of 3, both products' times 2 of 3, every quote found. *What about Boreal?*: H04 about 0.33, just above a rainforest chunk (0.32); *Insufficient evidence.* 3 of 3 |
| Part A, ten runs (self-study) | Atlas's 30 minutes alone 8 of 10, both products' times 2 of 10 |
| Part B | the rewrite *Boreal password reset link validity duration.*, H04 first (about 0.78), Boreal's 10 minutes 3 of 3; *What about Boreal?* rewritten as *How long does a Boreal reset link remain valid?*, Boreal's 10 minutes 3 of 3 |
| Part C | the rewrite passes both checks; *How long is the link valid?* fails the product check; a query with *30 minutes* fails the number check with `['30']` |
| Part D | the tool call's query *link validity*, product kept `False`; Atlas's 30 minutes with H02, 3 of 3; with the product rule, *Boreal password reset link validity* and Boreal's 10 minutes, 3 of 3 |
| Part E | Rewrite: products present 21/21 across seven cases, answering policy first 15/15 across five; H05 copied 2/3. Original tool drops products on three cases. Product rule passes applicable query checks, including the unresolved two-product case. On the evidence gap, rewrite refuses 3/3; tool states no Boreal billing information 3/3 |
| Part F | the prompt asks on 5/7 cases: one expected and four unnecessary asks; the code rule asks on `two_products` only |

**About the saved outputs.** They use `gpt-4o-mini` and `text-embedding-3-small`. Replies can vary; compare the
query's meaning, retrieved policy and answer with their checks. The student copy's initial three calls give Atlas
alone once and both products twice; its ten-call example gives Atlas alone eight times and both twice.

## Before you start

- **Lab 00 first.** The pipeline comes from Lab 05 and Lab 06 Part D; the quote check comes from Lab 02.
- **Open the notebook:** use the Colab badge and add `OPENAI_API_KEY` under Secrets, or open
  `09_follow_up_questions.ipynb` locally or in a codespace and choose the `.venv` kernel. Run Setup and Config first.
- **Data:** the 272 SQuAD chunks and five policies under `data/`, plus seven constructed conversations in
  `resources/conversations.json`. The conversation labels are evaluation keys, not model inputs.
- **Packages:** `langchain`, `langchain-core` and `langchain-openai`, installed by Lab 00's setup.
- **API work and cost:** one corpus embedding, roughly 21,000 tokens, plus query embeddings on every search.
  The full notebook makes roughly 120 chat calls assuming one search per tool interaction; the challenges add calls.
  Each search round trip normally needs a proposal call and a later answer call. Rewriting also costs a call.
  The notebook gives the token/rate formula and explains repeated history, schemas and tool-result input.
  It does not total usage, so the call estimate cannot establish an exact bill. Check your configured destination's
  rates; loops and repeated cells can add work.

The tool section shows `@tool`, its schema, `bind_tools()` and manual `invoke()` steps before the loop. Its final
free-text answer has no structured quote checks. The troubleshooting table covers missing descriptions, absent tool
calls, missing tool results and the configured reasoning effort.

## Resources

- [LangChain: retrieval](https://docs.langchain.com/oss/python/deepagents/retrieval): retrieval pipelines,
  2-step RAG and retrieval through tools
- [LangChain: tools](https://docs.langchain.com/oss/python/langchain/tools): descriptions and argument schemas
- [LangChain: models, tool calling](https://docs.langchain.com/oss/python/langchain/models#tool-calling):
  binding tools, reading proposals and returning results
- [OpenAI: function calling](https://developers.openai.com/api/docs/guides/function-calling): tool interfaces and calls

Next: [Lab 10](../10_Long_Context_Or_Retrieval/README.md) (choose between long context and retrieval).
Back to [the lab list](../README.md).
