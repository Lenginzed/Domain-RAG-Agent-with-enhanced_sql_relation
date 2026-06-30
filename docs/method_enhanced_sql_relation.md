# Method: enhanced_sql_relation

`enhanced_sql_relation` is a controlled fusion retrieval mode for domain-specific RAG. It combines conventional retrieval signals with a lightweight SQLite relation channel.

enhanced_sql_relation is a controlled fusion retrieval mode adapted for domain-specific RAG data. It is not a one-to-one reproduction of SAG.

## 1. Motivation

Domain RAG corpora often include mixed technical artifacts:

- source code;
- configuration files;
- experiment notes;
- reports;
- papers or thesis-style summaries;
- CSV and structured data summaries.

Queries over this kind of corpus may refer to filenames, config families, method names, metrics, stages, or cross-file relationships. A similarity-only retriever can miss these links or over-rank broad repeated terms.

## 2. Problem with Domain-Specific RAG

Standard vector or keyword retrieval can struggle when:

- many files reuse the same terms;
- important signals live in filenames or metadata;
- a query asks for a relation between code, configs, and reports;
- broad entities such as `reward`, `risk`, or `model` appear in many places;
- reviewers need an auditable explanation for why a source was retrieved.

The SQL relation channel provides a second local signal based on explicit event/entity links.

## 3. Event/Entity Relation Index

The local relation index is stored in SQLite and generated from the local corpus. It contains:

- documents;
- chunks;
- events;
- entities;
- event-entity links.

Each chunk can become a lightweight event. Rule-based extraction identifies filename entities, code symbols, YAML keys, stage markers, and domain terms. The relation DB is generated locally and is not included in the public repository.

## 4. SQL Relation Channel

The SQL relation channel performs:

1. query entity extraction;
2. seed entity lookup;
3. seed event retrieval;
4. one-hop shared-entity expansion;
5. mapping expanded events back to chunks and documents;
6. relation scoring with auditable reasons.

The output includes `relation_path` values such as:

```text
query_entity:reward
-> seed_entity:reward
-> seed_event:example_report chunk_2
-> shared_entity:risk
-> candidate_event:example_note chunk_1
```

## 5. Controlled Fusion

The fusion stage merges enhanced keyword/metadata candidates with SQL relation candidates.

It preserves:

- `retrieval_channels`;
- `enhanced_keyword_score`;
- `relation_score`;
- `relation_score_normalized`;
- `relation_path`;
- `relation_reasons`;
- `fusion_reasons`.

SQL-only candidates may be retained, but their scores are normalized and controlled by category and frequency rules.

## 6. Category Intent Gate

The category intent gate uses simple rules to infer likely target categories from the query. For example:

- "config" or "yaml" suggests config files;
- "code", "class", or "function" suggests code;
- "experiment" or "report" suggests experiments or reports;
- "notes" or route/planning language suggests notes or reports.

The gate can add a bonus to preferred categories and a penalty to less relevant categories. It does not delete evidence; it only adjusts ranking.

## 7. High-Frequency Entity Dampening

Some entities appear across many files. If a candidate is mainly supported by broad entities and lacks a more specific filename, config, code symbol, or stage entity, the fusion stage can apply a penalty.

This helps reduce ranking noise from repeated terms while still preserving relation paths for inspection.

## 8. Relation Score Normalization

Raw SQL relation scores can be much larger than retrieval scores from other channels. The method clips and scales relation scores before fusion so that a high relation score does not dominate ranking by itself.

## 9. Explainability Fields

The method exposes fields intended for inspection:

- `relation_path`: SQL join path from query entity to candidate event.
- `relation_reasons`: relation scoring reasons.
- `fusion_reasons`: final merge and adjustment reasons.
- `retrieval_channels`: whether a source came from enhanced retrieval, SQL relation retrieval, or both.
- `category_intent_adjustment`: category gate contribution.
- `high_frequency_entity_penalty`: broad-entity penalty.

These fields are useful in the Streamlit UI and LangGraph trace output.

## 10. Limitations

- Requires a local relation index.
- Rule-based entity extraction may miss synonyms.
- High-frequency entity dampening is heuristic.
- Small local evals are not enough for broad claims.
- Full results depend on local corpus quality.
- SQL relation retrieval can add noise when extracted entities are too broad.
- Category intent rules need corpus-specific review.

## 11. What this method is not

- It is not a one-to-one reproduction of SAG.
- It is not a replacement for vector search.
- It is not a graph database.
- It is not an LLM-based reranker.
- It is not a production platform.
- It is not a claim that relation fusion improves every query.
