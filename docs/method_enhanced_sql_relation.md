# Method: enhanced_sql_relation

`enhanced_sql_relation` is a controlled fusion retrieval mode for domain-specific RAG. It combines conventional retrieval signals with a lightweight SQLite relation channel.

This is a controlled fusion method adapted for domain-specific RAG data, not a one-to-one reproduction of SAG.

## Retrieval Channels

### enhanced_keyword

The enhanced keyword channel combines:

- dense retrieval,
- metadata matching,
- keyword matching,
- source diversity,
- optional filename/title calibration.

### sql_relation

The SQL relation channel performs:

- query entity extraction,
- seed entity lookup in SQLite,
- seed event retrieval,
- one-hop shared-entity expansion,
- mapping events back to chunks/documents,
- relation scoring with auditable reasons.

## Fusion

The fusion stage merges candidates from both channels. It preserves:

- `retrieval_channels`,
- `relation_score`,
- `relation_score_normalized`,
- `relation_path`,
- `relation_reasons`,
- `fusion_reasons`.

## Controls

- Category intent gate: promotes categories that match query intent and penalizes mismatched high-frequency hits.
- High-frequency entity dampening: prevents broad entities from dominating ranking.
- Relation score normalization: clips and scales raw relation scores before fusion.
- Source diversity: limits repeated chunks from a single source.

## Intended Use

The method is useful when filename/entity references matter and when relation paths help reviewers understand why a source was retrieved. It should be evaluated with local data and should not be treated as a production-grade retriever without further validation.

