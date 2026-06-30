# Domain-RAG-Agent-with-enhanced_sql_relation

A local-first Domain-RAG Agent prototype with controlled SQL-relation retrieval, evidence quality checks, answer reliability handling, and a human-review workflow.

This repository is a public, safety-staged version of a vertical-domain RAG engineering project. It uses a multi-UAV air-combat knowledge base as the original local case study, but the public repository is organized around reusable engineering patterns: ingestion boundaries, hybrid retrieval, controlled SQL-relation retrieval, evidence-aware answer generation, evaluation artifacts, and manual review loops.

This repository does not include local Chroma databases, SQLite indexes, raw imported documents, model weights, private logs, or full evaluation artifacts.

## Project Overview

The project explores how to build a practical RAG workflow for a specialized technical corpus. It is not a production-ready system and it is not a complete SAG implementation. The current public version focuses on reproducible code structure, safe configuration templates, documentation, and small illustrative examples.

The latest local milestone adds an `enhanced_sql_relation` retrieval mode that fuses a metadata/keyword retrieval channel with a read-only SQLite relation channel. The relation channel is used for auditable seed-entity lookup and one-hop expansion, then merged with conventional retrieval results through controlled scoring rules.

## Key Features

- Local-first RAG workflow with Ollama-compatible model configuration.
- Metadata-aware and keyword-aware retrieval utilities.
- `enhanced_sql_relation` fusion retrieval with relation paths and fusion reasons.
- Evidence gate before answer generation.
- Citation checker and evidence quality scoring.
- Local answer reliability handling, including empty-answer retry metadata.
- Human review task export and adjudication summary workflow.
- Streamlit inspection UI for retrieval, answer eval reports, failure triage, and review adjudication.
- LangGraph skeleton for controlled, traceable RAG workflow execution.

## Architecture

```text
query
-> retrieval
   -> enhanced_keyword
   -> sql_relation
   -> enhanced_sql_relation fusion
-> evidence gate
-> answer generation
-> citation checker
-> evidence quality
-> human review / adjudication
-> Streamlit inspection UI
```

See [docs/architecture.md](docs/architecture.md) for a longer overview.

## enhanced_sql_relation Retrieval

`enhanced_sql_relation` combines:

- an enhanced keyword/metadata retrieval channel,
- a SQLite relation retrieval channel,
- category intent adjustment,
- high-frequency entity dampening,
- relation score normalization,
- source diversity,
- auditable `relation_path`, `relation_reasons`, and `fusion_reasons`.

This is a controlled fusion method adapted for domain-specific RAG data, not a one-to-one reproduction of SAG.

See [docs/method_enhanced_sql_relation.md](docs/method_enhanced_sql_relation.md).

## Answer Reliability & Human Review Workflow

The project includes a rule-based answer evaluation loop:

1. Run retrieval and evidence gate.
2. Generate an answer only when evidence is sufficient.
3. Check citation grounding.
4. Score evidence quality.
5. Classify failures such as empty answer, unsupported claims, weak claims, and retrieval-insufficient refusal.
6. Export review tasks for human adjudication.
7. Import manual review labels and summarize fix categories.

See [docs/human_review_workflow.md](docs/human_review_workflow.md).

## Local Setup

Create and activate an isolated Python environment:

```powershell
conda create -n Lenginzed_RAG python=3.11
conda activate Lenginzed_RAG
python -m pip install -r requirements.txt
```

For command-by-command execution without activation:

```powershell
conda run -n Lenginzed_RAG python ...
```

Copy example configuration files before local use:

```powershell
Copy-Item config\ui_v4h.example.yaml config\ui_v4h.yaml
Copy-Item .env.example .env
```

The public repository does not include local indexes or private documents, so full retrieval/evaluation workflows require local data preparation first.

See [docs/local_setup.md](docs/local_setup.md).

## What Is Not Included

This public repository intentionally excludes:

- raw/private source documents,
- vector databases,
- SQLite relation indexes,
- private logs,
- full local evaluation outputs,
- model weights and local model caches,
- environment-specific configuration files.

Only sample artifacts are included under `examples/`.

## Data / Storage Policy

Local data and runtime artifacts must stay outside git. The included `.gitignore` blocks private document folders, runtime storage, logs, databases, local configs, and large binary artifacts.

See [docs/data_policy.md](docs/data_policy.md).

## Quick Test

These checks validate importability and lightweight logic only. They do not rebuild indexes or call an LLM:

```powershell
$env:PYTHONIOENCODING="utf-8"
conda run -n Lenginzed_RAG python -m pytest tests/test_answer_eval_v4g.py tests/test_v4i_answer_eval_triage.py tests/test_v4j_answer_eval_adjudication.py
```

Some tests in this research workspace depend on local-only artifacts. Public users should start from the documented examples and add their own local corpus/indexes.

## Current Status

The local development workspace has implemented:

- an 80-file expanded local collection,
- calibrated enhanced keyword retrieval,
- SQLite relation index and relation retriever,
- `enhanced_sql_relation` fusion retrieval,
- answer reliability handling,
- answer eval report browser,
- failure triage export,
- human adjudication summary.

The public repository contains the code and documentation needed to understand and reproduce the workflow with your own local data, but it does not contain the private runtime artifacts.

## Roadmap

- Safer local setup scripts for rebuilding sample indexes.
- More portable tests that do not assume local private artifacts.
- Optional UI refinements for human review and trace browsing.
- Optional broader evaluation once a public sample corpus is prepared.
- Later V5 work on a more complete controlled agent workflow.

## License

MIT License. See [LICENSE](LICENSE).
