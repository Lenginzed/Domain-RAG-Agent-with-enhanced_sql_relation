# Troubleshooting

## `ModuleNotFoundError`

Run commands from the repository root and install dependencies first:

```powershell
python -m pip install -r requirements.txt
```

If you use conda, run commands through the environment shown in the README.

## `conda: command not found`

Use your current Python environment instead:

```powershell
python scripts/build_mini_demo_index_v51.py
python scripts/run_mini_demo_v51.py
python -m pytest tests/test_v51_public_mini_demo.py
```

Make sure the required packages are installed.

## `tmp/mini_demo/mini_demo_relation_index.db not found`

Build the mini relation index first:

```powershell
conda run -n Lenginzed_RAG python scripts/build_mini_demo_index_v51.py
```

The run script can also auto-build the DB if the configured file is missing.

## Git push fails or GitHub port 443 is unreachable

This is usually a network, proxy, credential, or remote-branch state issue.

Check:

- network access to GitHub;
- whether your environment requires a proxy;
- Git credential status;
- whether the remote branch has newer commits.

Do not commit credentials, proxy secrets, or local machine paths.

## Ollama is unavailable

The public mini demo does not require Ollama. Retrieval-only demo commands should run without any local LLM.

Full answer generation workflows may require a local model, but those are separate from the public mini demo.

## No Chroma or generated SQLite DB files are in the repo

This is intentional. The public repository excludes generated runtime artifacts. Build local indexes from your own corpus when running full workflows.
