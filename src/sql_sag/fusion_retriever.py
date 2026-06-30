from __future__ import annotations

import csv
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from src.retrieval.real_smoke_retriever import retrieve_real_smoke, summarize_source_diversity
from src.sql_sag.relation_retriever import load_sql_sag_config, retrieve_sql_relation


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_v4d_config(path: str | Path) -> dict[str, Any]:
    config_path = resolve_project_path(path)
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Invalid YAML config: {config_path}")
    if config.get("base_sql_config"):
        sql_config = load_sql_sag_config(config["base_sql_config"])
        config["sql_relation_config"] = deep_merge(sql_config, {"sqlite": config.get("sqlite", {})})
    else:
        config["sql_relation_config"] = {"sqlite": config.get("sqlite", {})}
    if config.get("calibration_config"):
        calibration_file = load_yaml(resolve_project_path(config["calibration_config"]))
        config["calibration"] = calibration_file.get("calibration", calibration_file)
    return config


def infer_category_intent(question: str, config: dict[str, Any]) -> dict[str, Any]:
    intent_config = config.get("category_intent", {})
    if not intent_config.get("enabled", True):
        return {"intent": "general", "preferred_categories": [], "penalized_categories": [], "matched_keywords": []}
    question_lower = str(question).lower()
    matches: list[dict[str, Any]] = []
    for name, rule in (intent_config.get("rules", {}) or {}).items():
        matched = [kw for kw in rule.get("keywords", []) if str(kw).lower() in question_lower]
        if matched:
            matches.append(
                {
                    "intent": name,
                    "preferred_categories": [str(item) for item in rule.get("preferred_categories", [])],
                    "penalized_categories": [str(item) for item in rule.get("penalized_categories", [])],
                    "matched_keywords": matched,
                    "match_count": len(matched),
                }
            )
    if not matches:
        return {"intent": "general", "preferred_categories": [], "penalized_categories": [], "matched_keywords": []}
    matches.sort(key=lambda row: (-int(row["match_count"]), row["intent"]))
    best = matches[0]
    return {
        "intent": best["intent"],
        "preferred_categories": best["preferred_categories"],
        "penalized_categories": best["penalized_categories"],
        "matched_keywords": best["matched_keywords"],
    }


def normalize_relation_score(score: float, config: dict[str, Any]) -> float:
    fusion_config = config.get("fusion", {})
    if not fusion_config.get("normalize_relation_score", True):
        return float(score)
    clip_max = float(fusion_config.get("relation_score_clip_max", 120.0))
    if clip_max <= 0:
        return 0.0
    return round(min(max(float(score), 0.0), clip_max) / clip_max * 100.0, 6)


def apply_category_intent_adjustment(candidate: dict[str, Any], intent: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    category = str(candidate.get("category_dir", ""))
    fusion_config = config.get("fusion", {})
    adjustment = 0.0
    reasons: list[str] = []
    if category in set(intent.get("preferred_categories", [])):
        adjustment += float(fusion_config.get("category_intent_bonus", 25.0))
        reasons.append(f"category_intent_bonus:{category}")
    if category in set(intent.get("penalized_categories", [])):
        adjustment += float(fusion_config.get("category_intent_penalty", -40.0))
        reasons.append(f"category_intent_penalty:{category}")
    candidate["category_intent_adjustment"] = adjustment
    if reasons:
        candidate.setdefault("fusion_reasons", []).extend(reasons)
    return candidate


def apply_high_frequency_entity_penalty(candidate: dict[str, Any], high_frequency_entities: set[str], config: dict[str, Any]) -> dict[str, Any]:
    fusion_config = config.get("fusion", {})
    if not fusion_config.get("high_frequency_entity_penalty_enabled", True):
        candidate["high_frequency_entity_penalty"] = 0.0
        return candidate
    matched = {str(item) for item in candidate.get("matched_entities", [])}
    generic_hits = matched & high_frequency_entities
    if not generic_hits:
        candidate["high_frequency_entity_penalty"] = 0.0
        return candidate
    specific_hits = {
        item
        for item in matched
        if item not in high_frequency_entities
        and ("_" in item or item.startswith("stage") or len(item) >= 14)
    }
    if specific_hits:
        candidate["high_frequency_entity_penalty"] = 0.0
        candidate.setdefault("fusion_reasons", []).append(f"high_frequency_penalty_skipped:specific={','.join(sorted(specific_hits)[:5])}")
        return candidate
    penalty = float(fusion_config.get("high_frequency_penalty", -20.0))
    candidate["high_frequency_entity_penalty"] = penalty
    candidate.setdefault("fusion_reasons", []).append(f"high_frequency_entity_penalty:{','.join(sorted(generic_hits)[:5])}:{penalty:g}")
    return candidate


def merge_retrieval_candidates(
    enhanced_candidates: list[dict[str, Any]],
    sql_candidates: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    category_intent: dict[str, Any],
    high_frequency_entities: set[str],
) -> list[dict[str, Any]]:
    fusion_config = config.get("fusion", {})
    merged: dict[str, dict[str, Any]] = {}
    for candidate in enhanced_candidates:
        key = candidate_key(candidate)
        row = dict(candidate)
        channels = list(dict.fromkeys([*to_list(row.get("retrieval_channels")), "enhanced_keyword"]))
        row["retrieval_channels"] = channels
        row["enhanced_keyword_score"] = float(
            row.get("enhanced_keyword_base_score", row.get("final_score", row.get("score", 0))) or 0
        )
        row["relation_score"] = 0.0
        row["relation_score_raw"] = 0.0
        row["relation_score_normalized"] = 0.0
        row["relation_reasons"] = []
        row["relation_path"] = []
        row["fusion_reasons"] = ["from_enhanced_keyword"]
        merged[key] = row

    for candidate in sql_candidates:
        key = candidate_key(candidate)
        row = merged.get(key, {})
        if not row:
            row = {
                "source": candidate.get("source", ""),
                "imported_source": candidate.get("imported_source") or candidate.get("source", ""),
                "original_source": candidate.get("original_source", ""),
                "chunk_id": candidate.get("chunk_id", ""),
                "category_dir": candidate.get("category_dir", ""),
                "doc_type": candidate.get("doc_type", ""),
                "section": candidate.get("section", ""),
                "content_preview": candidate.get("content_preview", ""),
                "enhanced_keyword_score": 0.0,
                "calibration_score": 0.0,
                "retrieval_channels": [],
                "fusion_reasons": ["sql_only_candidate"],
            }
        row["retrieval_channels"] = list(dict.fromkeys([*to_list(row.get("retrieval_channels")), "sql_relation"]))
        row["relation_score_raw"] = float(candidate.get("relation_score", 0) or 0)
        row["relation_score"] = row["relation_score_raw"]
        row["relation_score_normalized"] = normalize_relation_score(row["relation_score_raw"], config)
        row["relation_reasons"] = candidate.get("relation_reasons", [])
        row["relation_path"] = candidate.get("relation_path", [])
        row["matched_entities"] = candidate.get("matched_entities", [])
        merged[key] = row

    for row in merged.values():
        row.setdefault("matched_entities", [])
        row.setdefault("calibration_score", row.get("calibration_score", 0.0) or 0.0)
        row.setdefault("category_intent_adjustment", 0.0)
        row.setdefault("high_frequency_entity_penalty", 0.0)
        apply_category_intent_adjustment(row, category_intent, config)
        apply_high_frequency_entity_penalty(row, high_frequency_entities, config)
        enhanced_score = float(row.get("enhanced_keyword_score", 0) or 0)
        if "sql_relation" in row.get("retrieval_channels", []) and not enhanced_score:
            enhanced_score = float(fusion_config.get("sql_only_base_score", 80.0))
            row["fusion_reasons"].append(f"sql_only_base_score:{enhanced_score:g}")
        relation_component = float(row.get("relation_score_normalized", 0) or 0) * float(fusion_config.get("relation_score_weight", 0.35))
        enhanced_component = enhanced_score * float(fusion_config.get("enhanced_keyword_score_weight", 1.0))
        calibration_component = float(row.get("calibration_score", 0) or 0) * float(fusion_config.get("calibration_score_weight", 1.0))
        final_score = (
            enhanced_component
            + relation_component
            + calibration_component
            + float(row.get("category_intent_adjustment", 0) or 0)
            + float(row.get("high_frequency_entity_penalty", 0) or 0)
        )
        row["final_score"] = round(final_score, 6)
        row["fusion_reasons"].extend(
            [
                f"enhanced_component:{enhanced_component:.3f}",
                f"relation_component:{relation_component:.3f}",
                f"calibration_component:{calibration_component:.3f}",
            ]
        )

    ranked = sorted(
        merged.values(),
        key=lambda row: (-float(row.get("final_score", 0) or 0), str(row.get("source", "")), str(row.get("chunk_id", ""))),
    )
    return apply_source_diversity(ranked, config)


def retrieve_enhanced_sql_relation(question: str, config: dict[str, Any]) -> dict[str, Any]:
    collection = config["collection"]
    retrieval_config = config.get("retrieval", {})
    category_intent = infer_category_intent(question, config)
    enhanced_candidates: list[dict[str, Any]] = []
    enhanced_debug: dict[str, Any] = {}
    if retrieval_config.get("include_enhanced_keyword", True):
        enhanced_result = retrieve_real_smoke(
            question=question,
            persist_directory=resolve_project_path(collection["persist_directory"]),
            collection_name=collection["collection_name"],
            retrieval_mode="enhanced_keyword",
            dense_top_k=int(retrieval_config.get("dense_top_k", 5)),
            metadata_top_k=int(retrieval_config.get("metadata_top_k", 8)),
            keyword_top_k=int(retrieval_config.get("keyword_top_k", 8)),
            final_top_k=int(retrieval_config.get("final_top_k", 8)),
            max_chunks_per_source=int(retrieval_config.get("max_chunks_per_source", 2)),
            enable_source_diversity=bool(retrieval_config.get("enable_source_diversity", True)),
            source_key=str(retrieval_config.get("source_key", "imported_source")),
            calibration_config=config.get("calibration") if retrieval_config.get("enable_calibration", True) else None,
        )
        enhanced_debug = enhanced_result.debug
        enhanced_candidates = [document_to_candidate(document, rank) for rank, document in enumerate(enhanced_result.documents, start=1)]

    sql_result: dict[str, Any] = {}
    sql_candidates: list[dict[str, Any]] = []
    if retrieval_config.get("include_sql_relation", True):
        sql_config = dict(config["sql_relation_config"])
        sql_config["sqlite"] = config["sqlite"]
        sql_result = retrieve_sql_relation(question, sql_config)
        sql_candidates = sql_result.get("candidates", [])

    high_frequency_entities = load_high_frequency_entities(config)
    final_sources = merge_retrieval_candidates(
        enhanced_candidates,
        sql_candidates,
        config,
        category_intent=category_intent,
        high_frequency_entities=high_frequency_entities,
    )
    return {
        "question": question,
        "category_intent": category_intent,
        "enhanced_keyword_candidates": enhanced_candidates,
        "enhanced_keyword_debug": enhanced_debug,
        "sql_relation_result": sql_result,
        "final_sources": final_sources,
        "debug": {
            "enhanced_candidate_count": len(enhanced_candidates),
            "sql_candidate_count": len(sql_candidates),
            "final_source_count": len(final_sources),
            "high_frequency_entities": sorted(high_frequency_entities),
        },
    }


def document_to_candidate(document: Any, rank: int) -> dict[str, Any]:
    metadata = dict(document.metadata or {})
    final_score = float(metadata.get("final_score", 0) or 0)
    calibration_score = float(metadata.get("calibration_score", 0) or 0)
    return {
        "rank": rank,
        "source": metadata.get("source", ""),
        "imported_source": metadata.get("imported_source") or metadata.get("source", ""),
        "original_source": metadata.get("original_source", ""),
        "chunk_id": metadata.get("chunk_id", ""),
        "category_dir": metadata.get("category_dir", ""),
        "doc_type": metadata.get("doc_type", ""),
        "section": metadata.get("section", ""),
        "content_preview": document.page_content[:500],
        "final_score": final_score,
        "score": metadata.get("score"),
        "dense_score": metadata.get("dense_score"),
        "metadata_score": metadata.get("metadata_score"),
        "keyword_score": metadata.get("keyword_score"),
        "calibration_score": calibration_score,
        "enhanced_keyword_base_score": max(0.0, final_score - calibration_score),
        "calibration_reasons": metadata.get("calibration_reasons", []),
        "retrieval_channels": metadata.get("retrieval_channels", []),
        "matched_terms": metadata.get("matched_terms", []),
        "matched_fields": metadata.get("matched_fields", []),
        "why_selected": metadata.get("why_selected", ""),
    }


def apply_source_diversity(candidates: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    retrieval_config = config.get("retrieval", {})
    final_top_k = int(retrieval_config.get("final_top_k", 8))
    max_chunks = int(retrieval_config.get("max_chunks_per_source", 2))
    source_key = str(retrieval_config.get("source_key", "imported_source"))
    if not retrieval_config.get("enable_source_diversity", True):
        selected = candidates[:final_top_k]
    else:
        selected = []
        counts: Counter[str] = Counter()
        for candidate in candidates:
            source = str(candidate.get(source_key) or candidate.get("source") or "")
            if counts[source] >= max_chunks:
                continue
            selected.append(candidate)
            counts[source] += 1
            if len(selected) >= final_top_k:
                break
    source_counts: Counter[str] = Counter()
    for rank, candidate in enumerate(selected, start=1):
        source = str(candidate.get(source_key) or candidate.get("source") or "")
        source_counts[source] += 1
        candidate["rank"] = rank
        candidate["source_rank_within_file"] = source_counts[source]
        candidate["source_diversity_key"] = source
    return selected


def load_high_frequency_entities(config: dict[str, Any]) -> set[str]:
    threshold = int(config.get("fusion", {}).get("high_frequency_threshold", 25))
    with connect_readonly(config["sqlite"]["db_path"]) as conn:
        rows = conn.execute(
            """
            SELECT e.normalized_name, COUNT(ee.event_id) AS frequency
            FROM entities e
            JOIN event_entities ee ON e.entity_id = ee.entity_id
            GROUP BY e.entity_id
            HAVING frequency >= ?
            """,
            (threshold,),
        ).fetchall()
    return {str(row["normalized_name"]) for row in rows}


def connect_readonly(db_path: str | Path) -> sqlite3.Connection:
    path = resolve_project_path(db_path)
    uri = f"file:{path.as_posix()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def candidate_key(candidate: dict[str, Any]) -> str:
    source = str(candidate.get("imported_source") or candidate.get("source") or "")
    return f"{source}|{candidate.get('chunk_id', '')}"


def to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str) and value:
        return [value]
    return []


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    resolved = resolve_project_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    resolved = resolve_project_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    resolved = resolve_project_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        resolved.write_text("", encoding="utf-8-sig")
        return
    with resolved.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML config: {path}")
    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def resolve_project_path(path: str | Path) -> Path:
    value = Path(str(path))
    if value.is_absolute():
        return value
    return PROJECT_ROOT / value
