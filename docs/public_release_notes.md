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
