# Project Status

## Current Public Status

- Safe public staging completed.
- Core code published.
- Documentation and example configs published.
- Local runtime artifacts excluded.
- Sanitized sample outputs included.
- V5.1 adds a fully synthetic public mini demo that builds a temporary relation index locally.
- V5.2 improves the public run walkthrough, expected output notes, troubleshooting, and documentation link checks.
- V5.3 prepared the repository for a `v0.1.0` release draft.
- V5.4 creates the `v0.1.0` release tag and GitHub Release from the safe public repository state.
- The public mini demo remains synthetic and retrieval-only.
- Local private data and generated artifacts are not included.

## Implemented in Code

- Document loaders and loader hardening.
- Metadata-aware chunking utilities.
- Dense, keyword, metadata, and calibrated retrieval modules.
- SQLite event/entity relation index utilities.
- SQL relation retrieval and `enhanced_sql_relation` fusion.
- Evidence gate and citation checker.
- Evidence quality scoring.
- Answer generation reliability metadata.
- Review task export and adjudication summary utilities.
- Streamlit inspection UI.
- Controlled LangGraph workflow trace skeleton.

## Local Artifacts Not Published

The public repository does not include:

- local vector stores;
- local relation DBs;
- private source documents;
- private logs;
- full local eval outputs;
- model weights or caches.

## What the Public Examples Show

The examples show schema and report shapes only:

- answer eval record shape;
- compare summary shape;
- review task shape;
- triage summary shape;
- calibration notes shape.

They should not be read as full benchmark results.

## Next Steps

- Add a tiny synthetic mini corpus.
- Extend the public mini demo with more synthetic relation cases.
- Add CI-safe public tests.
- Add reproducible mini-index build commands.
- Improve public demo instructions.
- Continue local evaluation before making broader claims.
