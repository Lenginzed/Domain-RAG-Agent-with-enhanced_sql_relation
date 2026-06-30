# Public Mini Demo

## Purpose

The mini demo is a public, synthetic, retrieval-only demonstration.

It gives GitHub readers a small command-line path that does not require private data, Chroma, a prebuilt SQLite DB, Ollama, or any LLM call.

## Synthetic Corpus

The corpus lives in `examples/mini_corpus/docs/` and contains four short files:

- `event_driven_reward.py`: synthetic code file containing `EventDrivenReward`.
- `HierarchySelfplay.yaml`: synthetic config file referencing `EventDrivenReward` and `risk_penalty`.
- `stage13_risk_report.md`: synthetic report mentioning reward, risk, and evaluation notes.
- `missile_engine.md`: synthetic domain note with missile and engine terms.

## What it Demonstrates

The demo builds a temporary SQLite relation index and runs three retrieval questions:

- `EventDrivenReward` should retrieve `event_driven_reward.py`.
- `HierarchySelfplay` should retrieve `HierarchySelfplay.yaml`.
- `reward` / `risk` should retrieve `stage13_risk_report.md`.

The output includes:

- top candidate source;
- expected source hit;
- `relation_path`;
- `relation_reasons`;
- `fusion_reasons`.

## What it Does Not Demonstrate

The mini demo does not demonstrate:

- full private-corpus retrieval;
- Chroma vector search;
- local LLM answer generation;
- citation checking;
- full answer evaluation;
- production deployment.

It is only a minimal public runnable demo for relation-style retrieval fields.

## Commands

```powershell
conda run -n Lenginzed_RAG python scripts/build_mini_demo_index_v51.py
conda run -n Lenginzed_RAG python scripts/run_mini_demo_v51.py
conda run -n Lenginzed_RAG python -m pytest tests/test_v51_public_mini_demo.py
```

For a step-by-step guide, see [public walkthrough](public_walkthrough.md).

## Expected Outputs

The three configured queries should retrieve:

- `EventDrivenReward` -> `event_driven_reward.py`
- `HierarchySelfplay` -> `HierarchySelfplay.yaml`
- `reward` / `risk` -> `stage13_risk_report.md`

The run script writes:

```text
tmp/mini_demo/mini_demo_results.json
tmp/mini_demo/mini_demo_results.md
```

These files are generated locally and ignored by git.

For console examples, see [expected output](../examples/mini_corpus/EXPECTED_OUTPUT.md).

## Safety Boundaries

- The corpus is fully synthetic.
- The demo does not read private local corpus directories.
- The demo does not write Chroma.
- The demo does not call Ollama or an LLM.
- The demo does not commit the generated SQLite DB.
