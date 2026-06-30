from __future__ import annotations

from pathlib import Path
from typing import Any

from src.sql_sag.fusion_retriever import load_v4d_config, retrieve_enhanced_sql_relation
from src.sql_sag.relation_retriever import retrieve_sql_relation


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_sql_sag_retrieval(
    *,
    question: str,
    retrieval_mode: str,
    config_path: str | Path,
    final_top_k: int | None = None,
) -> dict[str, Any]:
    config = load_v4d_config(config_path)
    if final_top_k is not None:
        config.setdefault("retrieval", {})["final_top_k"] = int(final_top_k)
        config.setdefault("sql_relation_config", {}).setdefault("retriever", {})["final_top_k"] = int(final_top_k)

    if retrieval_mode == "enhanced_sql_relation":
        raw = retrieve_enhanced_sql_relation(question, config)
        final_sources = [normalize_source(row) for row in raw.get("final_sources", [])]
        return {
            "question": question,
            "retrieval_mode": retrieval_mode,
            "final_sources": final_sources,
            "category_intent": raw.get("category_intent", {}),
            "enhanced_keyword_candidates": raw.get("enhanced_keyword_candidates", []),
            "sql_relation_result": raw.get("sql_relation_result", {}),
            "sql_relation_debug": summarize_sql_relation(raw.get("sql_relation_result", {})),
            "fusion_debug": raw.get("debug", {}),
            "debug": raw,
        }

    if retrieval_mode == "sql_relation":
        raw = retrieve_sql_relation(question, config["sql_relation_config"])
        final_sources = [normalize_source(row, sql_only=True) for row in raw.get("candidates", [])]
        return {
            "question": question,
            "retrieval_mode": retrieval_mode,
            "final_sources": final_sources,
            "category_intent": {},
            "enhanced_keyword_candidates": [],
            "sql_relation_result": raw,
            "sql_relation_debug": summarize_sql_relation(raw),
            "fusion_debug": {},
            "debug": raw,
        }

    raise ValueError(f"Unsupported SQL-SAG retrieval_mode={retrieval_mode!r}")


def normalize_source(row: dict[str, Any], *, sql_only: bool = False) -> dict[str, Any]:
    channels = row.get("retrieval_channels", [])
    if isinstance(channels, str):
        channels = [channels]
    if sql_only and "sql_relation" not in channels:
        channels = [*channels, "sql_relation"]
    source = {
        "rank": row.get("rank"),
        "source": row.get("source", ""),
        "imported_source": row.get("imported_source") or row.get("source", ""),
        "original_source": row.get("original_source", ""),
        "category_dir": row.get("category_dir", ""),
        "doc_type": row.get("doc_type", ""),
        "chunk_id": row.get("chunk_id", ""),
        "section": row.get("section", ""),
        "final_score": row.get("final_score", row.get("relation_score", 0.0)),
        "content_preview": row.get("content_preview", ""),
        "retrieval_channels": channels,
        "enhanced_keyword_score": row.get("enhanced_keyword_score", 0.0),
        "relation_score": row.get("relation_score", 0.0),
        "relation_score_raw": row.get("relation_score_raw", row.get("relation_score", 0.0)),
        "relation_score_normalized": row.get("relation_score_normalized", 0.0),
        "relation_reasons": row.get("relation_reasons", []),
        "relation_path": row.get("relation_path", []),
        "fusion_reasons": row.get("fusion_reasons", []),
        "category_intent_adjustment": row.get("category_intent_adjustment", 0.0),
        "high_frequency_entity_penalty": row.get("high_frequency_entity_penalty", 0.0),
        "sql_only_candidate": "sql_relation" in channels and "enhanced_keyword" not in channels,
        "calibration_score": row.get("calibration_score", 0.0),
        "matched_entities": row.get("matched_entities", []),
        "matched_terms": row.get("matched_terms", []),
        "matched_fields": row.get("matched_fields", []),
        "why_selected": row.get("why_selected", ""),
    }
    source["page_content"] = source["content_preview"]
    return source


def summarize_sql_relation(result: dict[str, Any]) -> dict[str, Any]:
    candidates = result.get("candidates", []) if isinstance(result, dict) else []
    return {
        "seed_entity_count": (result.get("debug", {}) or {}).get("seed_entity_count", 0) if isinstance(result, dict) else 0,
        "seed_event_count": (result.get("debug", {}) or {}).get("seed_event_count", 0) if isinstance(result, dict) else 0,
        "expanded_event_count": (result.get("debug", {}) or {}).get("expanded_event_count", 0) if isinstance(result, dict) else 0,
        "candidate_count": len(candidates),
        "relation_path_found_count": sum(1 for row in candidates if row.get("relation_path")),
        "top_sources": [
            {
                "rank": row.get("rank"),
                "source": row.get("source", ""),
                "category_dir": row.get("category_dir", ""),
                "relation_score": row.get("relation_score", 0.0),
            }
            for row in candidates[:5]
        ],
    }
