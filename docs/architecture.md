# Architecture

This project is a local-first Domain-RAG research/engineering prototype organized around controlled retrieval, evidence verification, and human review.

It is designed to keep runtime artifacts local. The public repository contains code, documentation, example configs, and sanitized samples; it does not contain private corpora or generated indexes.

## End-to-End Flow

```text
query
-> retrieval mode selection
   -> dense
   -> enhanced
   -> keyword
   -> enhanced_keyword
   -> sql_relation
   -> enhanced_sql_relation
-> evidence gate
-> answer generation, optional local LLM
-> citation checker
-> evidence quality scoring
-> human review triage
-> adjudication summary
-> Streamlit inspection UI
```

## Retrieval Layer

The retrieval layer supports several modes:

- `dense`: vector similarity retrieval.
- `keyword`: lexical retrieval.
- `enhanced`: metadata-aware retrieval.
- `enhanced_keyword`: dense, keyword, metadata, and calibration signals.
- `sql_relation`: SQLite relation retrieval using event/entity joins.
- `enhanced_sql_relation`: controlled fusion of enhanced retrieval and SQL relation retrieval.

The SQL relation DB is generated locally from a user's own corpus and is not uploaded to the public repository.

## Evidence and Answer Layer

The answer path is intentionally gated:

1. retrieval returns candidate sources;
2. evidence gate decides whether available evidence is sufficient;
3. full answer generation may call a local Ollama-compatible LLM only after the gate passes;
4. citation checker evaluates source support;
5. evidence quality scoring summarizes retrieval, citation, coverage, diversity, and risk signals.

Retrieval-only workflows do not call an LLM.

## Review Layer

Answer eval outputs can be turned into review tasks. The review workflow records:

- failure types;
- evidence quality;
- citation checker results;
- relation paths and fusion reasons;
- local LLM attempts and retry metadata;
- manual review fields;
- adjudication status and suggested fix categories.

The workflow supports human review, but does not replace human judgment.

## LangGraph Trace Layer

LangGraph is used for controlled workflow orchestration and traceability. Each graph run can include:

- run ID;
- node-level trace summaries;
- retrieval mode and source count;
- evidence gate status;
- answer / refusal branch;
- evidence quality;
- human review decision.

The graph is a controlled workflow skeleton, not an open-ended tool-using agent.

## Streamlit Inspection UI

The Streamlit app is a local inspection surface for:

- manual questions;
- retrieval-only inspection;
- full local RAG runs;
- SQL relation fields;
- answer eval reports;
- failure reason browsing;
- review task export;
- adjudication summaries.

The UI reads local artifacts when available. The public repository includes the app code but not the private runtime artifacts.

## Main Modules

- `src/loaders/`: document loading and hardening utilities.
- `src/retrieval/`: dense, metadata, keyword, calibrated retrieval logic.
- `src/sql_sag/`: SQLite relation index, relation retrieval, and fusion utilities.
- `src/generation/`: evidence gate, answer generation helpers, citation checking.
- `src/evaluation/`: evidence quality scoring and evaluation helpers.
- `src/agent/`: controlled LangGraph workflow with trace output.
- `src/ui/`: Streamlit service layer, report browser, triage, and adjudication utilities.
- `apps/`: local Streamlit entrypoints.
- `scripts/`: command-line workflows.
- `tests/`: smoke and regression tests.

## Runtime Boundaries

- Chroma vector stores are local generated artifacts and are not included.
- SQLite relation indexes are local generated artifacts and are not included.
- Ollama-compatible local LLMs are optional and are not included.
- Raw/private documents are not included.
- Full logs and full evaluation outputs are not included.
