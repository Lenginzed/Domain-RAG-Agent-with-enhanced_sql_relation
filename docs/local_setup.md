# Local Setup

This repository is a public code and documentation snapshot. It does not include local indexes or private documents, so full workflows require your own local corpus and generated artifacts.

## 1. Create an Environment

```powershell
conda create -n Lenginzed_RAG python=3.11
conda activate Lenginzed_RAG
python -m pip install -r requirements.txt
```

Or run commands through:

```powershell
conda run -n Lenginzed_RAG python ...
```

For readable Chinese output:

```powershell
$env:PYTHONIOENCODING="utf-8"
```

## 2. Copy Local Configs

Copy example configs before running workflows:

```powershell
Copy-Item .env.example .env
Copy-Item config\ui_v4h.example.yaml config\ui_v4h.yaml
Copy-Item config\sql_sag_v4d.example.yaml config\sql_sag_v4d.yaml
Copy-Item config\langgraph_v4e.example.yaml config\langgraph_v4e.yaml
```

Edit the copied files for your own local corpus, local vector store, local relation DB, and local model names.

Do not commit local configuration files after editing them.

## 3. Prepare a Local Corpus

The public repository does not ship a private corpus. To run full workflows, prepare your own local documents and adapt loader/ingest scripts for that corpus.

Supported loader areas include:

- Markdown/text;
- Python;
- YAML;
- CSV summary loading;
- PDF quality filtering.

## 4. Build Local Indexes

Full retrieval requires locally generated artifacts:

- a vector store for dense retrieval;
- a SQLite relation index for SQL relation retrieval;
- optional local eval outputs for report browsing.

These artifacts should remain outside git.

## 5. Optional Local Model

Full answer generation can use an Ollama-compatible local model. Retrieval-only workflows and report browsing do not require an LLM.

Model names should live in copied local configs, not hard-coded business logic.

## 6. Lightweight Tests

Some tests are module-level and public-safe:

```powershell
conda run -n Lenginzed_RAG python -m pytest tests/test_answer_eval_v4g.py tests/test_v4i_answer_eval_triage.py tests/test_v4j_answer_eval_adjudication.py
```

Some tests from the original workspace assume local artifacts. If a test expects a local index or private eval output, create your own local artifacts first or run a narrower test subset.

## 7. Public Mini Demo

The public mini demo uses only synthetic files and local temporary outputs:

```powershell
conda run -n Lenginzed_RAG python scripts/build_mini_demo_index_v51.py
conda run -n Lenginzed_RAG python scripts/run_mini_demo_v51.py
conda run -n Lenginzed_RAG python -m pytest tests/test_v51_public_mini_demo.py
```

It does not require Chroma, Ollama, a prebuilt SQLite DB, or private documents. Generated files are written under `tmp/mini_demo/` and should not be committed.

## 8. Streamlit UI

After local configs and artifacts are prepared:

```powershell
conda run -n Lenginzed_RAG streamlit run apps/streamlit_app_v25f.py
```

The UI can inspect retrieval, answer eval reports, failure triage, and adjudication summaries when the corresponding local files exist.
