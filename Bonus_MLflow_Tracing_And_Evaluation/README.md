# Bonus: How to trace and evaluate RAG with MLflow

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatwan/rag_for_nlp/blob/main/Bonus_MLflow_Tracing_And_Evaluation/bonus_mlflow.ipynb)

**CPU | OpenAI API key | Bonus**

## Goal

Connect the search, evidence, tracing and evaluation concepts from the earlier labs in MLflow. Follow the same
Atlas question through retrieved policies, packed evidence and an answer, then open the scores beside that request.

## What you will learn

- Inspect an automatically recorded OpenAI call, including its messages, response, timing and tokens.
- Connect Lab 01's search, Lab 02's evidence and Lab 04's records through a tree of spans.
- Compare Lab 06's token F1 with an LLM judge on three constructed answers.
- Evaluate the saved RAG traces and read the evidence behind each assessment, using Lab 07's habit of checking
  the judge's reasons yourself.

## Before you start

Run [Lab 00](../00_Environment_Check/README.md) first. Labs [02](../02_Cited_Answers/README.md),
[04](../04_Trace_And_Diagnose/README.md) and [06](../06_Evaluation_Set/README.md) supply the pipeline and evaluation
concepts; [07](../07_LLM_Judge_Calibration/README.md) develops judge calibration further. The notebook can run on
its own with their shared code. It uses the five existing policies; it does not need a previous notebook's memory.

The vector store is the same NumPy matrix used in Lab 01. ChromaDB and other stores can supply retrieved documents
too; keeping our search unchanged lets us concentrate on what MLflow records and how its scores use those records.

## Install and start MLflow

From the repository root, with uv:

```bash
uv sync --group mlflow
mkdir -p .cache/mlflow-bonus
uv run --group mlflow mlflow server --host 127.0.0.1 --port 5057 --backend-store-uri sqlite:///.cache/mlflow-bonus/mlflow.db --artifacts-destination .cache/mlflow-bonus/artifacts
```

Keep that terminal running. Open `bonus_mlflow.ipynb` in VS Code and select the repository's `.venv` kernel. The
tracking server and browser UI use the same address, `http://127.0.0.1:5057`. Traces, scores and artifacts stay
under the gitignored `.cache/` folder. Stop the server with Ctrl+C when finished.

With pip, activate the environment from Lab 00, then use:

```bash
pip install -r Bonus_MLflow_Tracing_And_Evaluation/requirements.txt ipykernel
mkdir -p .cache/mlflow-bonus
python -m mlflow server --host 127.0.0.1 --port 5057 --backend-store-uri sqlite:///.cache/mlflow-bonus/mlflow.db --artifacts-destination .cache/mlflow-bonus/artifacts
```

On **Ubuntu**, use the same steps.

In **GitHub Codespaces**, use [`bonus_mlflow_codespaces.ipynb`](bonus_mlflow_codespaces.ipynb) instead: the same lab
without the Colab cells. Your browser reaches the server through a forwarded address such as
`https://<codespace-name>-5057.app.github.dev`, and MLflow refuses an address it does not know, so the server needs
two more settings. In a codespace terminal at the repository root:

```bash
uv sync --group mlflow
mkdir -p .cache/mlflow-bonus
UI_HOST="$CODESPACE_NAME-5057.$GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN"
uv run --group mlflow mlflow server --host 127.0.0.1 --port 5057 \
  --backend-store-uri sqlite:///.cache/mlflow-bonus/mlflow.db \
  --artifacts-destination .cache/mlflow-bonus/artifacts \
  --allowed-hosts "$UI_HOST,localhost:5057,127.0.0.1:5057" \
  --cors-allowed-origins "https://$UI_HOST"
```

The notebook's Set up section explains both settings. Keep port 5057 private in the **Ports** panel, and open the UI
from the link the notebook prints.

On **Colab**, use the badge and add `OPENAI_API_KEY` in Secrets. Setup installs the bonus dependencies and starts
the server inside the runtime. Part A opens Colab's port-proxy window; your laptop's localhost cannot reach that
runtime directly. The final cell stops the process. Trace objects and score tables are also displayed in Python.

On macOS, port 5000 can be owned by Control Center and return a 403 response. This lab uses 5057 to avoid that conflict.

If port 5057 is occupied, choose another server port and set `MLFLOW_TRACKING_URI` to the matching URL in your
kernel's environment before Config. Model names, API endpoints and reasoning options still come from `models.env`
and your overrides. A judge is a separate model call, configured with `JUDGE_MODEL`.

## What you will do

| Part | Purpose |
|---|---|
| A: Inspect an OpenAI call in MLflow | Enable autologging, make an ordinary SDK call and match its trace with its response and usage |
| B: Follow retrieval, packing and generation in one trace | Add Python spans around the familiar RAG stages and compare three requests |
| C: Compare word overlap with an LLM judge | Score `30 minutes`, `half an hour` and `60 minutes` against one reference |
| D: Score the RAG requests already traced | Attach expectations and judge those saved requests without generating answers again |
| Challenge | Change the paraphrase's k to 2 and explain its evidence, answer and assessments |

## Checkpoints

| After | Check |
|---|---|
| A | The trace holds the same answer and token counts as the printed SDK reply |
| B | Three `policy-rag` requests have connected retrieval, packing and generation spans; inspect whether H02 reaches each prompt |
| C | Token F1 is 1.0, 0.0 and 0.5; compare the judge's actual verdicts and reasons with your own decisions |
| D | Three evaluation rows connect back to the selected RAG traces; explain correctness and groundedness separately |

**About the saved outputs.** The answer and judge models are `gpt-4o-mini`, with `text-embedding-3-small` for search.
Replies, timings and judge rationales can change; inspect your evidence and answers before interpreting the scores.

## Data and API use

The policies in [support-case.json](../data/support-case.json) describe two made-up products. The three duration
answers in Part C are constructed examples. References are supplied to the evaluator, outside the answering
prompt. This small set demonstrates the workflow; it does not measure general judge accuracy.

The notebook makes four answer calls, embeds five policies on a cold cache, and embeds the four search queries
used in Part B. Part C adds four judge calls; Part D adds six judge evaluations. The challenge adds one answer,
one query embedding and two judge evaluations. Judge internals and retries can add calls. Check the configured
provider's prices; evaluating saved traces avoids generating their answers again but still pays for judging.

## Resources

- [MLflow: OpenAI tracing](https://mlflow.org/docs/latest/genai/tracing/integrations/listing/openai/)
- [MLflow: manual tracing](https://mlflow.org/docs/latest/genai/tracing/app-instrumentation/manual-tracing/)
- [MLflow: GenAI evaluation](https://mlflow.org/docs/latest/genai/eval-monitor/quickstart/)
- [MLflow: RAG judges](https://mlflow.org/docs/latest/genai/eval-monitor/scorers/llm-judge/rag/)
- [MLflow: evaluate existing traces](https://mlflow.org/docs/latest/genai/eval-monitor/running-evaluation/traces/)

Back to [the lab list](../README.md).
