# Repository Map

```text
apps/      Streamlit app entrypoints
src/       core modules
scripts/   CLI workflows
tests/     lightweight and regression checks
config/    example configs only
docs/      documentation
examples/  sanitized sample artifacts
data/eval/ placeholder README only
```

## apps/

Contains the Streamlit local inspection app entrypoint.

## src/

Core project modules:

- `src/loaders/`: loaders and loader hardening.
- `src/chunkers/`: text chunking utilities.
- `src/metadata/`: metadata schemas and validation.
- `src/embeddings/`: local embedding interface.
- `src/indexes/`: vector index helpers.
- `src/retrieval/`: dense, keyword, metadata, query classification, calibration.
- `src/sql_sag/`: SQLite relation index, SQL relation retriever, fusion retriever.
- `src/generation/`: evidence gate, answer generation, citation checker.
- `src/evaluation/`: evidence quality scoring.
- `src/agent/`: controlled LangGraph workflow and trace utilities.
- `src/ui/`: service layer for Streamlit pages and report browsers.

## scripts/

Command-line workflows for ingestion, retrieval evaluation, SQL relation index building, answer eval, review export, adjudication import, and graph runs.

Some scripts require local artifacts that are not included in the public repository.

## tests/

Smoke and regression tests. Some tests are public-safe module checks; others assume local artifacts if run in the original research workspace.

## config/

Only `*.example.yaml` files should be tracked. Copy them to local config files before use.

## docs/

Public documentation for architecture, method design, local setup, data policy, project status, and FAQ.

## examples/

Sanitized samples that show output structure. These are not full local evaluation artifacts.

`examples/mini_corpus/` contains a synthetic public mini corpus for the runnable demo.

## data/eval/

Contains only a README placeholder in the public repository. Real evaluation outputs are ignored.
