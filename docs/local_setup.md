# Local Setup

This repository is a public code and documentation snapshot. It does not include local indexes or private documents, so full workflows require your own local corpus and generated artifacts.

## Environment

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

## Local Models

The project uses local Ollama-compatible settings in templates. You can change model names in local configuration files copied from `config/*.example.yaml`.

## Configuration

Copy example configs before running local workflows:

```powershell
Copy-Item config\ui_v4h.example.yaml config\ui_v4h.yaml
Copy-Item config\sql_sag_v4d.example.yaml config\sql_sag_v4d.yaml
Copy-Item .env.example .env
```

Do not commit local configuration files after editing them.

## Smoke Tests

```powershell
conda run -n Lenginzed_RAG python -m pytest
```

Some tests in the research workspace assume local-only artifacts. Public users may need to start with smaller module-level tests or create their own sample corpus.

