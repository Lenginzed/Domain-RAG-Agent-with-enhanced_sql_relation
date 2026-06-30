from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any


DEFAULT_CONFIG = {
    "min_final_sources": 1,
    "min_source_hit_for_non_negative": 1,
    "min_category_hit_for_non_negative": 1,
    "require_expected_source_for_eval": True,
    "negative_force_insufficient": True,
    "weak_evidence_keywords": ["没有", "是否有", "有没有", "完整", "真实飞行试验", "视频数据", "3v3"],
}


@dataclass
class EvidenceGateResult:
    evidence_sufficient: bool
    reason: str
    signals: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_evidence(
    *,
    question: str,
    retrieval_result: Any,
    eval_item: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Rule-based evidence sufficiency check for V2c.0.

    The gate intentionally uses only retrieval metadata and optional eval labels.
    It does not call an LLM and should remain auditable.
    """

    gate_config = {**DEFAULT_CONFIG, **(config or {})}
    final_sources = get_final_sources(retrieval_result)
    final_categories = [str(item.get("category_dir", "")) for item in final_sources if item.get("category_dir")]
    source_blob = build_source_blob(final_sources)
    expected_terms = [str(item).lower() for item in (eval_item or {}).get("expected_sources_contains", [])]
    expected_categories = [str(item) for item in (eval_item or {}).get("expected_categories", [])]
    required_all_categories = [str(item) for item in (eval_item or {}).get("required_all_categories", [])]
    source_hits = [term for term in expected_terms if term and term in source_blob]
    category_hits = sorted(set(final_categories).intersection(expected_categories))
    missing_required_categories = [category for category in required_all_categories if category not in final_categories]
    weak_hits = weak_evidence_hits(question, gate_config)
    is_negative = bool(eval_item and str(eval_item.get("query_type", "")) == "negative")

    signals = {
        "final_source_count": len(final_sources),
        "final_categories": sorted(set(final_categories)),
        "category_counts": dict(Counter(final_categories)),
        "expected_sources_contains": expected_terms,
        "source_hits": source_hits,
        "expected_categories": expected_categories,
        "category_hits": category_hits,
        "required_all_categories": required_all_categories,
        "missing_required_categories": missing_required_categories,
        "weak_evidence_keyword_hits": weak_hits,
        "is_eval_negative": is_negative,
    }

    if is_negative and bool(gate_config.get("negative_force_insufficient", True)):
        return EvidenceGateResult(
            evidence_sufficient=False,
            reason="eval_item.query_type is negative; force insufficient evidence for answer-stage refusal.",
            signals=signals,
        ).to_dict()

    min_final_sources = int(gate_config.get("min_final_sources", 1))
    if len(final_sources) < min_final_sources:
        return EvidenceGateResult(
            evidence_sufficient=False,
            reason=f"final_sources count {len(final_sources)} is below min_final_sources={min_final_sources}.",
            signals=signals,
        ).to_dict()

    require_expected_source = bool(gate_config.get("require_expected_source_for_eval", True))
    min_source_hits = int(gate_config.get("min_source_hit_for_non_negative", 1))
    if eval_item and expected_terms and require_expected_source and len(source_hits) < min_source_hits:
        return EvidenceGateResult(
            evidence_sufficient=False,
            reason="expected_sources_contains was provided but no retrieved source matched it.",
            signals=signals,
        ).to_dict()

    if eval_item and required_all_categories and missing_required_categories:
        return EvidenceGateResult(
            evidence_sufficient=False,
            reason=f"required_all_categories missing: {missing_required_categories}.",
            signals=signals,
        ).to_dict()

    min_category_hits = int(gate_config.get("min_category_hit_for_non_negative", 1))
    if eval_item and expected_categories and len(category_hits) < min_category_hits:
        return EvidenceGateResult(
            evidence_sufficient=False,
            reason="expected_categories was provided but retrieved categories did not match.",
            signals=signals,
        ).to_dict()

    if not eval_item and looks_like_unverifiable_existence_question(weak_hits):
        return EvidenceGateResult(
            evidence_sufficient=False,
            reason="question contains weak/negative existence cues and no eval label can verify a positive answer.",
            signals=signals,
        ).to_dict()

    return EvidenceGateResult(
        evidence_sufficient=True,
        reason="retrieved sources satisfy the V2c.0 rule-based evidence gate.",
        signals=signals,
    ).to_dict()


def get_final_sources(retrieval_result: Any) -> list[dict[str, Any]]:
    if isinstance(retrieval_result, dict):
        if isinstance(retrieval_result.get("final_sources"), list):
            return [dict(item) for item in retrieval_result["final_sources"]]
        debug = retrieval_result.get("debug")
        if isinstance(debug, dict) and isinstance(debug.get("final_sources"), list):
            return [dict(item) for item in debug["final_sources"]]

    debug = getattr(retrieval_result, "debug", None)
    if isinstance(debug, dict) and isinstance(debug.get("final_sources"), list):
        return [dict(item) for item in debug["final_sources"]]

    documents = getattr(retrieval_result, "documents", None)
    if isinstance(documents, list):
        rows: list[dict[str, Any]] = []
        for document in documents:
            metadata = getattr(document, "metadata", {}) or {}
            rows.append(dict(metadata))
        return rows
    return []


def build_source_blob(final_sources: list[dict[str, Any]]) -> str:
    fields = [
        "source",
        "imported_source",
        "original_source",
        "title",
        "section",
        "doc_type",
        "category_dir",
        "chunk_id",
    ]
    values: list[str] = []
    for source in final_sources:
        values.extend(str(source.get(field, "")) for field in fields)
    return " ".join(values).lower()


def weak_evidence_hits(question: str, config: dict[str, Any]) -> list[str]:
    lowered = question.lower()
    hits: list[str] = []
    for keyword in config.get("weak_evidence_keywords", []):
        text = str(keyword)
        if text and text.lower() in lowered:
            hits.append(text)
    return hits


def looks_like_unverifiable_existence_question(weak_hits: list[str]) -> bool:
    if not weak_hits:
        return False
    high_risk = {"完整", "真实飞行试验", "视频数据", "3v3"}
    return len(weak_hits) >= 2 or bool(high_risk.intersection(weak_hits))
