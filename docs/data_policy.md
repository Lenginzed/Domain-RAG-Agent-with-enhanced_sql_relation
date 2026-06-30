# Data and Storage Policy

This public repository intentionally excludes private data and local runtime artifacts.

## Public Repository Exclusions

The repository does not include:

- raw/private source documents;
- local Chroma databases;
- local SQLite indexes;
- private logs;
- full local evaluation outputs;
- model weights;
- model caches;
- `.env` files;
- edited local configuration files;
- long raw LLM responses.

## Included Public Artifacts

The repository includes:

- source code;
- tests;
- documentation;
- example configuration templates;
- short sanitized sample outputs;
- public setup instructions.

## Local Build Requirement

Users must provide their own local corpus and build their own local indexes. The public code can be adapted to a user's corpus, but the original private corpus and generated runtime artifacts are not distributed.

## Configuration Policy

- Copy `*.example.yaml` files before local use.
- Edit copied local config files for local paths and model names.
- Do not commit edited local config files.
- Do not commit `.env`.

## Sample File Policy

Sample files under `examples/` are illustrative. They show expected shapes for reports, review tasks, and eval summaries. They are not full local outputs.

## Safe Contribution Checklist

Before committing, verify that staged files do not include:

- machine-specific absolute paths;
- private corpus snippets;
- database files;
- vector-store files;
- full logs;
- long raw model responses;
- model weights or caches;
- large binary artifacts.
