from __future__ import annotations

from collections import Counter
from typing import Any


DEFAULT_CONFIG: dict[str, Any] = {
    "weights": {
        "retrieval_confidence": 0.30,
        "citation_support": 0.30,
        "coverage": 0.20,
        "source_diversity": 0.10,
        "risk_penalty": 0.10,
    },
    "levels": {
        "high": 0.80,
        "medium": 0.60,
        "low": 0.40,
    },
    "risk_penalty": {
        "needs_human_review": 0.25,
        "unsupported_claim": 0.10,
        "weak_claim": 0.05,
        "evidence_insufficient": 1.0,
        "citation_check_failed": 0.25,
    },
    "source_diversity": {
        "pass_score": 1.0,
        "fail_score": 0.5,
        "max_chunks_per_source": 2,
    },
}


def score_answer_record(record: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
    scoring_config = merge_config(DEFAULT_CONFIG, config or {})
    reasons: list[str] = []
    warnings: list[str] = []

    retrieval_confidence_score = score_retrieval_confidence(record, reasons, warnings)
    citation_support_score = score_citation_support(record, reasons, warnings)
    coverage_score = score_coverage(record, reasons, warnings)
    source_diversity_score = score_source_diversity(record, scoring_config, reasons, warnings)
    risk_penalty = score_risk_penalty(record, scoring_config, reasons, warnings)

    weights = scoring_config["weights"]
    raw_score = (
        float(weights.get("retrieval_confidence", 0.30)) * retrieval_confidence_score
        + float(weights.get("citation_support", 0.30)) * citation_support_score
        + float(weights.get("coverage", 0.20)) * coverage_score
        + float(weights.get("source_diversity", 0.10)) * source_diversity_score
        - float(weights.get("risk_penalty", 0.10)) * risk_penalty
    )
    evidence_quality_score = round(clamp(raw_score), 6)
    level = quality_level(
        score=evidence_quality_score,
        evidence_sufficient=as_bool(record.get("evidence_sufficient", True)),
        insufficient_answer=as_bool(record.get("insufficient_answer", False)),
        levels=scoring_config["levels"],
    )
    if level == "insufficient":
        reasons.append("negative_or_insufficient_answer_handled")

    return {
        "id": str(record.get("id") or record.get("question_id") or ""),
        "question": str(record.get("question") or ""),
        "query_type": str(record.get("query_type") or ""),
        "evidence_quality_score": evidence_quality_score,
        "evidence_quality_level": level,
        "retrieval_confidence_score": retrieval_confidence_score,
        "citation_support_score": citation_support_score,
        "coverage_score": coverage_score,
        "source_diversity_score": source_diversity_score,
        "risk_penalty": risk_penalty,
        "evidence_quality_reasons": unique_values(reasons),
        "evidence_quality_warnings": unique_values(warnings),
        "needs_human_review": as_bool(record.get("needs_human_review", False)),
        "insufficient_answer": as_bool(record.get("insufficient_answer", False)),
    }


def score_retrieval_confidence(record: dict[str, Any], reasons: list[str], warnings: list[str]) -> float:
    final_sources = as_list(record.get("final_sources"))
    score = 0.0
    if final_sources:
        score += 0.30
        reasons.append("final_sources_available")
    else:
        warnings.append("no_final_sources")
        return 0.0

    if any(int_or_none(source.get("rank")) is not None and int_or_none(source.get("rank")) <= 3 for source in final_sources):
        score += 0.10
        reasons.append("top_rank_sources_available")

    channels = collect_channels(final_sources)
    has_matched_terms = any(as_list(source.get("matched_terms")) for source in final_sources)
    matched_fields = collect_matched_fields(final_sources)

    if not channels and not has_matched_terms and not matched_fields:
        warnings.append("missing_retrieval_debug_fields")
    if "dense" in channels:
        score += 0.15
        reasons.append("dense_channel_available")
    if "keyword" in channels:
        score += 0.15
        reasons.append("keyword_channel_available")
    if "metadata" in channels:
        score += 0.15
        reasons.append("metadata_channel_available")
    if has_matched_terms:
        score += 0.10
        reasons.append("matched_terms_available")
    if {"content", "filename"}.intersection(matched_fields):
        score += 0.05
        reasons.append("content_or_filename_match_available")
    return round(clamp(score), 6)


def score_citation_support(record: dict[str, Any], reasons: list[str], warnings: list[str]) -> float:
    if as_bool(record.get("insufficient_answer", False)):
        reasons.append("insufficient_answer_no_claim_support_required")
        return 1.0

    supported_claim_ratio = float_or_default(record.get("supported_claim_ratio"), 0.0)
    unsupported_count = int_or_default(record.get("unsupported_claim_count"), 0)
    weak_count = int_or_default(record.get("weak_claim_count"), 0)
    score = supported_claim_ratio - 0.08 * unsupported_count - 0.04 * weak_count
    if supported_claim_ratio >= 0.8:
        reasons.append("citation_support_ratio_high")
    if unsupported_count:
        warnings.append("unsupported_claims_present")
    if weak_count:
        warnings.append("weak_claims_present")
    if not as_bool(record.get("citation_check_passed", True)):
        warnings.append("citation_check_failed")
    return round(clamp(score), 6)


def score_coverage(record: dict[str, Any], reasons: list[str], warnings: list[str]) -> float:
    signals = record.get("evidence_gate_signals")
    signals = signals if isinstance(signals, dict) else {}

    required_all_categories = as_list(signals.get("required_all_categories"))
    missing_required = as_list(signals.get("missing_required_categories"))
    source_hits = as_list(signals.get("source_hits"))
    category_hits = as_list(signals.get("category_hits"))

    has_explicit_coverage_fields = any(
        key in record
        for key in ["required_all_categories_hit", "expected_source_hit", "expected_category_hit"]
    ) or bool(signals)

    if as_bool(record.get("required_all_categories_hit", False)) or (required_all_categories and not missing_required):
        reasons.append("required_all_categories_covered")
        return 1.0
    if as_bool(record.get("expected_source_hit", False)) or bool(source_hits):
        reasons.append("expected_source_hit")
        return 1.0
    if as_bool(record.get("expected_category_hit", False)) or bool(category_hits):
        reasons.append("expected_category_hit")
        return 0.85

    if not has_explicit_coverage_fields:
        warnings.append("missing_eval_coverage_fields")
    if as_list(record.get("final_sources")):
        reasons.append("coverage_estimated_from_final_sources")
        return 0.6
    return 0.0


def score_source_diversity(
    record: dict[str, Any],
    config: dict[str, Any],
    reasons: list[str],
    warnings: list[str],
) -> float:
    final_sources = as_list(record.get("final_sources"))
    if not final_sources:
        warnings.append("source_diversity_no_sources")
        return 0.0

    diversity_config = config.get("source_diversity", {})
    pass_score = float(diversity_config.get("pass_score", 1.0))
    fail_score = float(diversity_config.get("fail_score", 0.5))
    max_chunks = int(diversity_config.get("max_chunks_per_source", 2))

    explicit_diversity = record.get("source_diversity")
    if isinstance(explicit_diversity, dict):
        violations = explicit_diversity.get("violations") or {}
        if violations:
            warnings.append("source_diversity_violations")
            return round(clamp(fail_score), 6)
        reasons.append("source_diversity_passed")
        return round(clamp(pass_score), 6)

    counts = Counter(source_identifier(source) for source in final_sources)
    violations = {source: count for source, count in counts.items() if count > max_chunks}
    if violations:
        warnings.append("source_diversity_violations")
        return round(clamp(fail_score), 6)
    reasons.append("source_diversity_passed")
    return round(clamp(pass_score), 6)


def score_risk_penalty(record: dict[str, Any], config: dict[str, Any], reasons: list[str], warnings: list[str]) -> float:
    penalty_config = config.get("risk_penalty", {})
    if not as_bool(record.get("evidence_sufficient", True)):
        warnings.append("evidence_insufficient")
        return round(clamp(float(penalty_config.get("evidence_insufficient", 1.0))), 6)

    penalty = 0.0
    if as_bool(record.get("needs_human_review", False)):
        penalty += float(penalty_config.get("needs_human_review", 0.25))
        warnings.append("needs_human_review")
    unsupported_count = int_or_default(record.get("unsupported_claim_count"), 0)
    weak_count = int_or_default(record.get("weak_claim_count"), 0)
    if unsupported_count:
        penalty += float(penalty_config.get("unsupported_claim", 0.10)) * unsupported_count
        warnings.append("unsupported_claims_present")
    if weak_count:
        penalty += float(penalty_config.get("weak_claim", 0.05)) * weak_count
        warnings.append("weak_claims_present")
    if not as_bool(record.get("citation_check_passed", True)):
        penalty += float(penalty_config.get("citation_check_failed", 0.25))
        warnings.append("citation_check_failed")
    if penalty == 0:
        reasons.append("no_risk_penalty")
    return round(clamp(penalty), 6)


def quality_level(
    *,
    score: float,
    evidence_sufficient: bool,
    insufficient_answer: bool,
    levels: dict[str, Any],
) -> str:
    if not evidence_sufficient or insufficient_answer:
        return "insufficient"
    if score >= float(levels.get("high", 0.80)):
        return "high"
    if score >= float(levels.get("medium", 0.60)):
        return "medium"
    if score >= float(levels.get("low", 0.40)):
        return "low"
    return "insufficient"


def collect_channels(final_sources: list[Any]) -> set[str]:
    channels: set[str] = set()
    for source in final_sources:
        if not isinstance(source, dict):
            continue
        for channel in as_list(source.get("retrieval_channels")):
            channels.add(str(channel))
        retrieval_source = source.get("retrieval_source")
        if isinstance(retrieval_source, str):
            channels.update(item for item in retrieval_source.split("+") if item)
    return channels


def collect_matched_fields(final_sources: list[Any]) -> set[str]:
    fields: set[str] = set()
    for source in final_sources:
        if isinstance(source, dict):
            fields.update(str(item) for item in as_list(source.get("matched_fields")))
    return fields


def source_identifier(source: Any) -> str:
    if not isinstance(source, dict):
        return "unknown_source"
    for key in ["imported_source", "source", "original_source", "chunk_id"]:
        value = str(source.get(key) or "").strip()
        if value:
            return value
    return "unknown_source"


def merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = {key: value.copy() if isinstance(value, dict) else value for key, value in base.items()}
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    return [value]


def int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def int_or_default(value: Any, default: int) -> int:
    parsed = int_or_none(value)
    return default if parsed is None else parsed


def float_or_default(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def unique_values(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result
