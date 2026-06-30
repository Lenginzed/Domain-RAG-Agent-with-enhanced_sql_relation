# Mini Corpus

This directory contains a tiny synthetic corpus for the public runnable demo.

It is not derived from the private local corpus. The files are intentionally short and are only meant to demonstrate relation-style retrieval fields such as `relation_path`, `relation_reasons`, and `fusion_reasons`.

Run from the repository root:

```powershell
conda run -n Lenginzed_RAG python scripts/build_mini_demo_index_v51.py
conda run -n Lenginzed_RAG python scripts/run_mini_demo_v51.py
```

Generated artifacts are written to `tmp/mini_demo/` and are ignored by git.

Expected output: [EXPECTED_OUTPUT.md](EXPECTED_OUTPUT.md).
