# Public Release Notes

## V5 Public Documentation Polish

This public release packages the repository for GitHub presentation while preserving local-data boundaries.

## Included

- Source code for loaders, retrieval, SQL relation fusion, generation helpers, evaluation helpers, UI services, and controlled LangGraph workflow.
- Example configs.
- Sanitized examples.
- Public documentation.
- MIT license.

## Excluded

- Raw/private documents.
- Local vector stores.
- Local SQLite relation indexes.
- Private logs.
- Full local evaluation outputs.
- Model weights and caches.
- `.env` and local config files.

## Notes

The repository is intended as a local-first research/engineering prototype. Full workflows require local artifacts created from a user's own corpus.

`enhanced_sql_relation` is a controlled fusion retrieval mode for domain-specific corpora. It is not a one-to-one reproduction of SAG.

## v0.1.0 Release

The repository has a `v0.1.0` release prepared from the safe public repository state.

This release includes a changelog, release checklist, release notes, repository presentation check, and the synthetic public mini demo. The public mini demo remains retrieval-only. Local private data and generated artifacts are not included.
