# Public Mini Demo Walkthrough

## Purpose

The public mini demo is a small synthetic, retrieval-only path for visitors who want to run something without private data or local runtime artifacts.

It demonstrates a minimal relation-style retrieval loop with candidate sources, `relation_path`, `relation_reasons`, and `fusion_reasons`.

## Prerequisites

- Python environment with project dependencies installed.
- Repository root as the current working directory.
- No private corpus required.
- No Chroma required.
- No prebuilt SQLite DB required.
- No Ollama required.
- No LLM call.

## Step 1: Install dependencies

```powershell
conda create -n Lenginzed_RAG python=3.11
conda activate Lenginzed_RAG
python -m pip install -r requirements.txt
```

Or use your existing Python environment if it already has the required packages.

## Step 2: Build the mini relation index

```powershell
conda run -n Lenginzed_RAG python scripts/build_mini_demo_index_v51.py
```

This reads only `examples/mini_corpus/docs/` and writes a temporary DB under `tmp/mini_demo/`.

## Step 3: Run the mini demo

```powershell
conda run -n Lenginzed_RAG python scripts/run_mini_demo_v51.py
```

The run prints the top source for each configured query and writes local result files under `tmp/mini_demo/`.

## Step 4: Run the lightweight check

```powershell
conda run -n Lenginzed_RAG python -m pytest tests/test_v51_public_mini_demo.py
```

This verifies that the synthetic files exist, the temporary relation index can be built, the demo can run, and the three expected sources are retrieved.

## Expected output

The three demo queries should retrieve:

- `EventDrivenReward` -> `event_driven_reward.py`
- `HierarchySelfplay` -> `HierarchySelfplay.yaml`
- `reward` / `risk` -> `stage13_risk_report.md`

More details: [expected output](../examples/mini_corpus/EXPECTED_OUTPUT.md)

## Generated local files

The demo writes:

```text
tmp/mini_demo/mini_demo_relation_index.db
tmp/mini_demo/mini_demo_results.json
tmp/mini_demo/mini_demo_results.md
tmp/mini_demo/mini_demo_index_summary.json
```

These files are ignored by git.

## Cleanup

To remove generated demo outputs:

```powershell
Remove-Item -Recurse -Force tmp\mini_demo
```

This only removes the local demo output directory.

## What this demo proves

- The public synthetic corpus is readable.
- A temporary relation index can be built.
- The minimal relation-style retrieval path returns the expected files.
- Relation and fusion explanation fields are emitted.

## What this demo does not prove

- It does not evaluate a private corpus.
- It does not exercise Chroma vector retrieval.
- It does not generate answers.
- It does not check citations.
- It does not represent full local-system quality.
