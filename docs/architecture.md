# Architecture

This project is a local-first Domain-RAG prototype organized around controlled retrieval, evidence verification, and human review.

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

## Main Modules

- `src/loaders/`: document loading and hardening utilities.
- `src/retrieval/`: dense, metadata, keyword, and calibrated retrieval logic.
- `src/sql_sag/`: SQLite-based relation index, relation retrieval, and fusion utilities.
- `src/generation/`: evidence gate, answer generation helpers, and citation checking.
- `src/evaluation/`: evidence quality scoring and evaluation helpers.
- `src/agent/`: controlled LangGraph workflow skeleton with trace output.
- `src/ui/`: Streamlit service layer, report browser, triage, and adjudication utilities.
- `apps/`: local Streamlit application entrypoints.
- `scripts/`: reproducible command-line workflows.
- `tests/`: smoke tests and regression tests.

## Runtime Boundaries

The public repository contains code and illustrative examples only. Local users must build their own document collection, vector store, and relation index.

The workflow intentionally keeps evidence checks and review decisions outside the LLM. The LLM is only used in full answer generation mode after evidence-gate checks pass.

