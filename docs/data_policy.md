# Data and Storage Policy

This public repository intentionally excludes local runtime artifacts and private data.

## Not Included

- raw/private source documents,
- local vector stores,
- SQLite relation indexes,
- model weights,
- model caches,
- private logs,
- full evaluation outputs,
- environment-specific configuration files.

## Included

- source code,
- tests,
- documentation,
- example configuration templates,
- short sample evaluation snippets,
- public setup instructions.

## Local Rebuild Required

Users who want to run the full workflow must provide their own corpus and build local indexes. The public repository provides templates and code paths, but not the original private corpus or generated runtime artifacts.

## Safe Contribution Rule

Before committing, check that staged files do not contain:

- machine-specific absolute paths,
- private corpus paths,
- database files,
- vector-store files,
- long model responses,
- full local logs,
- large binary artifacts.

