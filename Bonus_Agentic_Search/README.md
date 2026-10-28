# Bonus: How to let the model decide when to search

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatwan/rag_for_nlp/blob/main/Bonus_Agentic_Search/bonus_agentic_search.ipynb)

**CPU | API key: `OPENAI_API_KEY` (chat and embedding calls) | Practises: choosing between a fixed search-then-answer
pipeline and an agent that searches when it needs to, from measured answers and costs | Self-study**

## Goal

Give the model a search tool, let it decide whether and how often to search, and judge from answers, model calls and
tokens when that beats the pipeline that searches once with the question. The moment that teaches it: a question with
two parts gets one half wrong from the fixed pipeline, because the passage for that half ranks fourth and is never
read; the model splits it into two searches and answers both parts. On a simple question it costs an extra model
call, and on a greeting it skips a search the pipeline always makes.

The pipeline you have built searches once, with the question as typed or rewritten (Lab 09), and always reads
the top three passages. This baseline has no query-decomposition or small-talk branch; a fixed workflow could add them.
Here the model chooses whether and how to search, and we measure the complete design.

## What you will learn

- How a fixed search-then-answer pipeline handles a question with two parts, and a message that asks nothing.
- How to give the model a search tool and run its calls by hand, two calls in one reply included.
- How to separate a search budget from a hard model-call limit, and inspect what a tight budget loses.
- How to build the same loop with LangChain's `create_agent`, and read the graph it returns.
- How to compare two designs on the same questions in answers, model calls, searches and tokens, and decide which
  kind of question each one suits.

## Coming from

[Lab 05](../05_Rebuild_With_LangChain/README.md) built the fixed pipeline in `shared/lc_pipeline.py`: one search with
the question, one cited answer from the top three passages. [Lab 09](../09_Follow_Up_Questions/README.md) gave the model
a search tool so it could write a follow-up's query, and ran its calls in a loop. This lab hands the model the whole
decision (whether to search, how often and for what) and measures what that buys.

## What you will do

**Part A: Run the fixed pipeline on three kinds of question.** One fact, two parts, and a greeting; the two-part
question's wrong half passes the quote check, and the passage it needed ranks fourth.
**Part B: Give the model a search tool and run its calls by hand.** One reply with two tool calls, both run and sent
back, a complete call/ToolMessage pair with matching IDs, then a bounded loop.
**Part C: Add a stopping rule.** A question the documents do not answer, a search budget and a hard
four-call limit that returns a terminal notice if requests continue.
**Part D: Build the same loop with `create_agent`.** The graph it returns, the same run message by message, and the
separate limits as `ToolCallLimitMiddleware` and `ModelCallLimitMiddleware`.
**Part E: Compare the two designs on the same questions.** Eight messages of three kinds, three runs each through each
design: accepted phrases, model calls, attempted/executed searches and chat tokens.
**Challenge.** A question with three parts. Optional: a line that lets the agent answer small talk, tested where it
should change nothing.

## Checkpoints

| After | What to check |
|---|---|
| Part A | 277 documents; inspect the complete Normandy claims and evidence, including what the cited centuries describe; locate S02-P02-C00 |
| Part B | A complete tool-call record, its matching ToolMessage, the remaining tool results and final reply; at most four model calls in the loop |
| Part C | At most one executed query in the tight-budget runs; distinguish missing evidence from a terminal model-limit notice |
| Part D | A readable saved graph picture; requested, successful and blocked tool messages; at most four model calls |
| Part E | Phrase checks by kind, complete replies, attempted/executed searches and chat-token totals; read what each measure leaves open |

**About the saved outputs.** The examples use the configured chat and embedding models. Replies and rankings vary:
compare each checkpoint with your own results, and inspect support as well as accepted phrases.

## Before you start

- **Lab 00 first,** then Lab 05; Lab 09 helps but is not required. This lab uses `shared/lc_pipeline.py`,
  `shared/answer.py` and `shared/evals.py`.
- **Open the notebook:** on Colab with the badge above (add `OPENAI_API_KEY` under Secrets first); locally or in a
  codespace, open `bonus_agentic_search.ipynb` and choose the `.venv` kernel. Run Setup and Config first.
- **Data:** `data/squad-chunks.json` and `data/support-case.json`, read in place. The eight test messages are written
  in the notebook.
- **Packages:** `langchain`, `langchain-core`, `langchain-openai` and `langgraph` (installed with `langchain`), already
  installed by Lab 00's setup.
- **Network:** the configured chat and embedding APIs. Part D includes a bundled PNG graph and prints runtime
  Mermaid text. Displaying `agent` directly may use the mermaid.ink rendering service.
- **API work:** one corpus embedding, query embeddings for executed searches and short chat calls. Each agent
  invocation permits at most four model calls. Part E counts chat tokens; it omits embedding usage and latency,
  so it is not a total-cost or speed comparison. Charges depend on your configured provider and models.

The notebook ends with a troubleshooting table: a tool call sent back without its result, a recursion limit set too
low, the graph picture on a blocked network when `agent` is displayed, and tools on a reasoning model at the wrong effort.

## Resources

- [LangChain: retrieval](https://docs.langchain.com/oss/python/langchain/retrieval): 2-step RAG against agentic RAG,
  where the model decides when to retrieve
- [LangChain: agents](https://docs.langchain.com/oss/python/langchain/agents): `create_agent`, a model calling tools in
  a loop, and the middleware around it
- [LangChain: built-in middleware](https://docs.langchain.com/oss/python/langchain/middleware/built-in): the tool call
  limit and the model call limit
- [OpenAI: function calling](https://developers.openai.com/api/docs/guides/function-calling): tool calls, parallel calls
  in one reply, and the tool results the API expects back
- [Asai et al., *Self-RAG*, ICLR 2024](https://arxiv.org/abs/2310.11511): a model trained to decide when to retrieve

Back to [the lab list](../README.md).
