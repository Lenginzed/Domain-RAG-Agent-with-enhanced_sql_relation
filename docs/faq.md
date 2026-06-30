# FAQ

## 1. Why are there no Chroma or SQLite files?

They are generated local runtime artifacts. The public repository excludes local vector stores and relation DBs to avoid uploading private data, indexes, and machine-specific state.

## 2. Can I run the full workflow after clone?

Not immediately. You can inspect the code, docs, example configs, and sample outputs. Full retrieval and answer workflows require your own corpus, local configs, and generated local indexes.

## 3. Is this ready for production use?

No. This is a local-first research/engineering prototype, not a production-ready RAG platform.

## 4. Is this a direct SAG reproduction?

No. `enhanced_sql_relation` is a controlled fusion method adapted for domain-specific RAG data. It is not a one-to-one reproduction of SAG.

## 5. Why use SQL relation retrieval?

SQL relation retrieval gives an auditable path from query entities to candidate chunks. This can help when a domain corpus has code-config-report relationships, filename references, and repeated technical terms.

## 6. What is enhanced_sql_relation?

It is a retrieval mode that combines enhanced keyword/metadata retrieval with a lightweight SQLite relation channel. The fusion layer keeps relation paths and reasons while controlling high-frequency entity noise.

## 7. Does it require a local LLM?

Retrieval-only workflows do not require an LLM. Full answer generation can use a local Ollama-compatible model if configured.

## 8. Can I use my own corpus?

Yes. The code is organized so you can adapt loaders, configs, and index-building scripts to your own local corpus. Keep private data and generated indexes outside git.

## 9. How do I run the public mini demo?

Run these commands from the repository root:

```powershell
conda run -n Lenginzed_RAG python scripts/build_mini_demo_index_v51.py
conda run -n Lenginzed_RAG python scripts/run_mini_demo_v51.py
conda run -n Lenginzed_RAG python -m pytest tests/test_v51_public_mini_demo.py
```

The demo uses only synthetic files under `examples/mini_corpus/`.

## 10. Why does the repo not include generated SQLite DB files?

Generated DB files are runtime artifacts. They may contain local corpus-derived data, so the public repo excludes them. The public mini demo builds a temporary DB locally under `tmp/mini_demo/`.

## 11. Does the mini demo require a local LLM?

No. The public mini demo is retrieval-only. It does not require Ollama or any LLM call.

## 12. Why are generated files under `tmp/mini_demo/` ignored?

They are local outputs. Ignoring them keeps generated DBs and result files out of git while still letting users rerun the demo.

## 13. What should I check if GitHub push fails?

Check general network access to GitHub, proxy settings, credential status, and whether the remote branch has newer commits. Avoid adding credentials, proxy details, or local machine paths to committed files.
