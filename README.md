# Domain-RAG-Agent-with-enhanced_sql_relation

![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.11%2B-3776AB)
![Local First](https://img.shields.io/badge/local--first-RAG-2E7D32)
![Status](https://img.shields.io/badge/status-research%20prototype-orange)

A local-first Domain-RAG Agent prototype with controlled SQL-relation retrieval, evidence quality checks, answer reliability handling, and human-review workflow.

This is a local-first research/engineering prototype, not a production-ready RAG platform.

This repository does not include local Chroma databases, SQLite indexes, raw imported documents, model weights, private logs, or full evaluation artifacts.

## Why this project?

Domain RAG systems often need more than semantic similarity. Technical corpora usually contain code files, configuration files, experiment notes, reports, and repeated domain terms. A query may refer to a filename, a code symbol, a config family, or a cross-file relationship that is hard to recover from vector similarity alone.

This project packages a local workflow for exploring those issues:

- safe document loading and metadata handling;
- hybrid retrieval and retrieval calibration;
- a lightweight SQLite relation index;
- controlled fusion through `enhanced_sql_relation`;
- evidence gate, citation checking, and evidence quality scoring;
- answer reliability records and human-review task export;
- a Streamlit inspection UI and traceable LangGraph workflow skeleton.

The goal is engineering clarity and reproducibility with local artifacts, not a claim of broad benchmark superiority.

## Key Features

- Local-first RAG workflow with Ollama-compatible model settings.
- Loaders for PDF, Markdown, text, Python, YAML, and CSV-style sources.
- CSV summary-aware loading and PDF quality filtering.
- Dense, keyword, metadata-aware, and calibrated retrieval utilities.
- `enhanced_sql_relation` retrieval mode with auditable relation paths.
- Evidence gate before answer generation.
- Citation checker and evidence quality scoring.
- Local answer reliability handling with retry metadata for empty answers.
- Failure taxonomy, review task export, and manual adjudication summary.
- Streamlit inspection UI for retrieval, answer eval reports, failure triage, and review adjudication.
- Controlled LangGraph workflow skeleton with run IDs and trace summaries.

## System Overview

```text
query
-> retrieval mode selection
   -> enhanced_keyword
   -> sql_relation
   -> enhanced_sql_relation
-> evidence gate
-> answer generation, optional local LLM
-> citation checker
-> evidence quality
-> human review triage
-> adjudication summary
-> Streamlit inspection UI
```

More detail: [docs/architecture.md](docs/architecture.md)

## What is enhanced_sql_relation?

`enhanced_sql_relation` combines an enhanced keyword/metadata retrieval channel with a lightweight SQLite relation channel. SQL relation retrieval provides auditable relation paths, while controlled fusion prevents high-frequency entities from dominating the final ranking.

It includes:

- enhanced keyword / metadata retrieval;
- SQL relation retrieval;
- query entity extraction;
- seed entity lookup;
- one-hop relation expansion;
- category intent gate;
- high-frequency entity dampening;
- relation score normalization;
- source diversity;
- `relation_path`, `relation_reasons`, and `fusion_reasons`.

enhanced_sql_relation is a controlled fusion retrieval mode adapted for domain-specific RAG data. It is not a one-to-one reproduction of SAG.

It is useful when technical corpora contain mixed file types, repeated high-frequency entities, and cross-file relationships such as code-config-report links.

More detail: [docs/method_enhanced_sql_relation.md](docs/method_enhanced_sql_relation.md)

## Why not just vector search?

Vector search is valuable, but domain corpora often contain signals that are not purely semantic:

- a filename or code symbol may be the strongest clue;
- a config file and a report may share entities without sharing much wording;
- broad terms such as `reward`, `risk`, or `model` may occur everywhere;
- reviewers need to understand why a source was retrieved.

The SQL relation channel adds explicit event/entity joins and relation paths. The fusion layer keeps this as a complementary signal rather than replacing vector or keyword retrieval.

## Compared with standard RAG and SAG-style retrieval

| Aspect | Standard vector/keyword RAG | enhanced_sql_relation |
| --- | --- | --- |
| Main signal | Similarity | Similarity + SQL relation path |
| Relation awareness | Weak | Explicit event/entity joins |
| Explainability | Score-based | `relation_path` / `fusion_reasons` |
| High-frequency entities | Can dominate | Dampened by controlled fusion |
| Domain file types | Often mixed | Category intent aware |
| LLM dependency | Optional | Optional; retrieval path remains local |

The method is designed as a complementary retrieval channel, not a universal replacement for vector search.

More detail: [docs/comparison_standard_rag_vs_enhanced_sql_relation.md](docs/comparison_standard_rag_vs_enhanced_sql_relation.md)

## Repository Contents

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

Repository map: [docs/repository_map.md](docs/repository_map.md)

## What is not included

This public repository intentionally excludes:

- local Chroma databases;
- local SQLite indexes;
- raw/private source documents;
- model weights and model caches;
- private logs;
- full local evaluation outputs;
- edited local config files;
- `.env` files.

Only short sanitized samples are included under [examples/](examples/).

Data policy: [docs/data_policy.md](docs/data_policy.md)

## Quick Start

Create an isolated Python environment:

```powershell
conda create -n Lenginzed_RAG python=3.11
conda activate Lenginzed_RAG
python -m pip install -r requirements.txt
```

Or run commands without activating the environment:

```powershell
conda run -n Lenginzed_RAG python ...
```

For readable Chinese terminal output:

```powershell
$env:PYTHONIOENCODING="utf-8"
```

## Local Configuration

Copy example files before local use:

```powershell
Copy-Item .env.example .env
Copy-Item config\ui_v4h.example.yaml config\ui_v4h.yaml
Copy-Item config\sql_sag_v4d.example.yaml config\sql_sag_v4d.yaml
Copy-Item config\langgraph_v4e.example.yaml config\langgraph_v4e.yaml
```

Then edit the copied files for your own corpus, local vector store, local relation DB, and local model names.

Do not commit edited local configs or `.env`.

Setup notes: [docs/local_setup.md](docs/local_setup.md)

## Lightweight Checks

These tests are lightweight checks for selected modules and public-safe workflows. They do not rebuild indexes or call an LLM:

```powershell
conda run -n Lenginzed_RAG python -m pytest tests/test_answer_eval_v4g.py tests/test_v4i_answer_eval_triage.py tests/test_v4j_answer_eval_adjudication.py
```

Some tests in the original research workspace depend on local-only artifacts. Public users should start from module-level tests and add their own local corpus/indexes before running full workflows.

## Public Mini Demo

The public mini demo uses a fully synthetic mini corpus under [examples/mini_corpus/](examples/mini_corpus/). It does not require Chroma, a prebuilt SQLite DB, Ollama, an LLM, or any private corpus.

It builds a temporary SQLite relation index in `tmp/mini_demo/`, then runs a minimal relation-style retrieval demonstration:

```powershell
conda run -n Lenginzed_RAG python scripts/build_mini_demo_index_v51.py
conda run -n Lenginzed_RAG python scripts/run_mini_demo_v51.py
conda run -n Lenginzed_RAG python -m pytest tests/test_v51_public_mini_demo.py
```

Expected retrieval targets:

- `EventDrivenReward` -> `event_driven_reward.py`
- `HierarchySelfplay` -> `HierarchySelfplay.yaml`
- `reward` / `risk` -> `stage13_risk_report.md`

The demo outputs candidate sources, `relation_path`, `relation_reasons`, and `fusion_reasons`. Generated files stay under `tmp/mini_demo/` and are ignored by git.

More detail: [docs/public_mini_demo.md](docs/public_mini_demo.md)

### Expected result

The three demo queries should retrieve:

- `EventDrivenReward` -> `event_driven_reward.py`
- `HierarchySelfplay` -> `HierarchySelfplay.yaml`
- `reward` / `risk` -> `stage13_risk_report.md`

More detail: [docs/public_walkthrough.md](docs/public_walkthrough.md)

Expected output: [examples/mini_corpus/EXPECTED_OUTPUT.md](examples/mini_corpus/EXPECTED_OUTPUT.md)

## Example Workflow

1. Prepare your own local technical corpus.
2. Create local configs from `config/*.example.yaml`.
3. Build local vector and relation indexes with your own data.
4. Run retrieval inspection through scripts or the Streamlit UI.
5. Inspect evidence, citations, quality scores, relation paths, and failure records.
6. Export review tasks for manual adjudication when needed.

The files under [examples/](examples/) show output shapes only. They are not full evaluation artifacts.

## Documentation

- [Architecture](docs/architecture.md)
- [enhanced_sql_relation method](docs/method_enhanced_sql_relation.md)
- [Comparison with standard RAG](docs/comparison_standard_rag_vs_enhanced_sql_relation.md)
- [Human review workflow](docs/human_review_workflow.md)
- [Data and storage policy](docs/data_policy.md)
- [Local setup](docs/local_setup.md)
- [Public mini demo](docs/public_mini_demo.md)
- [Public walkthrough](docs/public_walkthrough.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Repository map](docs/repository_map.md)
- [Project status](docs/project_status.md)
- [FAQ](docs/faq.md)
- [Public release notes](docs/public_release_notes.md)

## Project Status

Current public status:

- safe public staging completed;
- core code published;
- local artifacts excluded;
- `enhanced_sql_relation` implemented in code;
- answer eval and human-review workflow implemented for local use;
- public repo contains examples, not the private corpus or full local outputs.

More detail: [docs/project_status.md](docs/project_status.md)

## Roadmap

- Expand the synthetic mini corpus into a slightly richer public demonstration.
- Add CI-safe tests that do not require local Chroma, SQLite, Ollama, or private data.
- Improve public setup scripts for local index creation.
- Add clearer Streamlit demo instructions with sample artifacts.
- Continue refining retrieval and review workflows through local evaluation.

## Data and Safety Policy

Keep local data and runtime artifacts outside git. The repository is configured to ignore private data folders, runtime storage, logs, databases, local configs, and large binary artifacts.

Before contributing, verify that no local absolute paths, private corpus snippets, database files, vector-store files, long model responses, or private logs are staged.

More detail: [docs/data_policy.md](docs/data_policy.md)

## Citation and Acknowledgements

If you reference this project, please cite the repository URL and commit hash you used.

The implementation is built around common local RAG components and ideas: Python, SQLite, Streamlit, LangGraph, Chroma-compatible vector storage, and Ollama-compatible local model workflows.

## License

MIT License. See [LICENSE](LICENSE).
