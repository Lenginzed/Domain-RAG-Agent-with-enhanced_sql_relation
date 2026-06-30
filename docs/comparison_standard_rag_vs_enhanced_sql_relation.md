# Comparison: Standard RAG vs enhanced_sql_relation

`enhanced_sql_relation` is a complementary retrieval channel for domain-specific RAG. It is designed to expose explicit relation paths while still preserving conventional retrieval signals.

| Aspect | Standard vector/keyword RAG | enhanced_sql_relation |
| --- | --- | --- |
| Main signal | Similarity | Similarity + SQL relation path |
| Relation awareness | Weak | Explicit event/entity joins |
| Explainability | Score-based | `relation_path` / `fusion_reasons` |
| High-frequency entities | Can dominate | Dampened by controlled fusion |
| Domain file types | Often mixed | Category intent aware |
| LLM dependency | Optional | Optional; retrieval path remains local |

## Standard RAG Strengths

Standard vector/keyword RAG is simple, general, and often effective when the query and target text have strong semantic or lexical overlap.

It is still part of this project. `enhanced_sql_relation` does not remove dense, keyword, or metadata retrieval.

## Why Add SQL Relation Retrieval?

Domain corpora often contain relationships that are easier to inspect through structured joins:

- code file to config file;
- config file to experiment report;
- filename entity to note;
- method name to result summary;
- repeated domain term to multiple categories.

SQL relation retrieval creates an auditable path from query entities to candidate chunks.

## Controlled Fusion

The fusion stage combines relation candidates with enhanced keyword/metadata candidates. It includes:

- category intent adjustment;
- high-frequency entity dampening;
- relation score normalization;
- source diversity;
- SQL-only candidate handling.

The method is designed as a complementary retrieval channel, not a universal replacement for vector search.

## When It Helps

It can help when:

- filenames or code symbols matter;
- evidence is spread across mixed file types;
- reviewers need an explicit relation path;
- high-frequency entities create noisy retrieval results.

## When It May Not Help

It may add noise when:

- entity extraction is too broad;
- the relation index misses synonyms;
- the local corpus has weak metadata;
- the query is already handled well by direct similarity.

Small local evals should be treated as engineering diagnostics rather than broad performance claims.
