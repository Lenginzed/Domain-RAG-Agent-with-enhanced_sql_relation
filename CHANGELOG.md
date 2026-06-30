# Changelog

## [0.1.0] - 2026-06-30

### Added

- Local-first Domain-RAG Agent prototype structure.
- `enhanced_sql_relation` retrieval mode.
- SQLite relation index utilities.
- SQL relation retrieval with relation paths.
- Relation explanation fields: `relation_path`, `relation_reasons`, and `fusion_reasons`.
- Evidence gate before answer generation.
- Citation checker and evidence quality scoring.
- Answer reliability records for local answer generation.
- Failure taxonomy for retrieval and answer issues.
- Human review task export.
- Manual adjudication summary utilities.
- Streamlit inspection UI.
- Controlled LangGraph workflow skeleton with trace output.
- Synthetic public mini demo.

### Changed

- Public documentation now emphasizes local-first usage and explicit data boundaries.
- README now links to the public mini demo, walkthrough, expected output, troubleshooting, and release notes.
- Generated local artifacts are kept outside git.

### Documentation

- Architecture overview.
- `enhanced_sql_relation` method notes.
- Standard RAG comparison.
- Human review workflow.
- Data policy.
- Local setup.
- FAQ.
- Public walkthrough.
- Troubleshooting.
- Repository map.

### Public Demo

- Synthetic mini corpus.
- Temporary relation index build script.
- Public runnable demo script.
- Lightweight public verification.
- Expected output documentation.

### Safety / Data Policy

- This release does not include local Chroma databases, SQLite indexes, raw imported documents, model weights, private logs, or full evaluation artifacts.
- Local generated outputs remain ignored by git.
- Example files are short synthetic or sanitized samples.

### Not Included

- Private corpus.
- Prebuilt vector store.
- Prebuilt relation DB.
- Model weights or caches.
- Full local evaluation outputs.
- Private logs.

### Known Limitations

- The public mini demo is synthetic and retrieval-only.
- Full workflows require local data and generated artifacts.
- Rule-based relation extraction can miss synonyms.
- Local evaluation examples are not broad benchmark claims.
