from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from langchain_core.documents import Document


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_./\\-]*|\d+")


def apply_retrieval_calibration(
    document: Document,
    *,
    question: str,
    classification: dict[str, Any] | None,
    config: dict[str, Any] | None,
) -> Document:
    result = calibrate_document_score(
        question=question,
        metadata=dict(document.metadata or {}),
        classification=classification,
        config=config,
    )
    metadata = dict(document.metadata or {})
    calibration_score = float(result["calibration_score"])
    metadata["calibration_score"] = calibration_score
    metadata["calibration_reasons"] = result["calibration_reasons"]
    metadata["calibration_query_tokens"] = result["query_tokens"]
    metadata["calibration_source_tokens"] = result["source_tokens"]
    metadata["calibration_breakdown"] = result["calibration_breakdown"]
    if calibration_score:
        metadata["final_score"] = round(float(metadata.get("final_score", 0) or 0) + calibration_score, 4)
        metadata["why_selected"] = ";".join(
            filter(
                None,
                [
                    str(metadata.get("why_selected", "")),
                    f"calibration_score:{calibration_score:.2f}",
                    *result["calibration_reasons"],
                ],
            )
        )
    document.metadata = metadata
    return document


def calibrate_document_score(
    *,
    question: str,
    metadata: dict[str, Any],
    classification: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = config or {}
    if not bool(config.get("enabled", True)):
        return empty_result()

    min_len = int(config.get("matched_source_token_min_len", 3))
    ignored_tokens = {str(item).lower() for item in config.get("ignored_source_tokens", [])}
    query_tokens = extract_source_like_tokens(question, min_len=min_len)
    scoring_query_tokens = query_tokens - ignored_tokens
    filename = metadata_filename(metadata)
    filename_tokens = extract_source_like_tokens(Path(filename).stem, min_len=min_len)
    metadata_tokens = extract_metadata_tokens(metadata, min_len=min_len)
    title_section_tokens = extract_source_like_tokens(
        " ".join([str(metadata.get("title", "")), str(metadata.get("section", ""))]),
        min_len=min_len,
    )

    score = 0.0
    reasons: list[str] = []
    breakdown: list[dict[str, Any]] = []
    normalized_question = normalize_for_match(question)
    normalized_filename = normalize_for_match(filename)
    normalized_stem = normalize_for_match(Path(filename).stem)

    if filename and (normalized_filename in normalized_question or normalized_stem in normalized_question):
        boost = float(config.get("exact_filename_boost", 8.0))
        score += boost
        reasons.append(f"exact_filename_match:{filename}")
        breakdown.append({"reason": "exact_filename_match", "value": filename, "score": boost})

    stem_overlap = sorted(scoring_query_tokens.intersection(filename_tokens - ignored_tokens))
    if stem_overlap:
        boost = float(config.get("stem_token_boost", 3.0)) * len(stem_overlap)
        score += boost
        reasons.append(f"source_stem_token_overlap:{','.join(stem_overlap)}")
        breakdown.append({"reason": "source_stem_token_overlap", "value": stem_overlap, "score": boost})

    title_overlap = sorted(scoring_query_tokens.intersection(title_section_tokens - ignored_tokens))
    if title_overlap:
        boost = float(config.get("metadata_title_boost", 3.0)) * len(title_overlap)
        score += boost
        reasons.append(f"title_section_token_overlap:{','.join(title_overlap)}")
        breakdown.append({"reason": "title_section_token_overlap", "value": title_overlap, "score": boost})

    target_categories = set(str(item) for item in (classification or {}).get("target_categories", []))
    category = str(metadata.get("category_dir", ""))
    if category and category in target_categories:
        boost = float(config.get("category_match_boost", 1.5))
        score += boost
        reasons.append(f"category_match:{category}")
        breakdown.append({"reason": "category_match", "value": category, "score": boost})

    metadata_overlap = sorted(scoring_query_tokens.intersection(metadata_tokens - filename_tokens - title_section_tokens))
    if metadata_overlap:
        # Diagnostic only. The weighted retrieval channels already account for path-level matches.
        reasons.append(f"metadata_token_overlap:{','.join(metadata_overlap[:8])}")
        breakdown.append({"reason": "metadata_token_overlap", "value": metadata_overlap[:8], "score": 0.0})

    return {
        "calibration_score": round(score, 4),
        "calibration_reasons": reasons,
        "calibration_breakdown": breakdown,
        "query_tokens": sorted(query_tokens),
        "source_tokens": sorted(metadata_tokens),
    }


def empty_result() -> dict[str, Any]:
    return {
        "calibration_score": 0.0,
        "calibration_reasons": [],
        "calibration_breakdown": [],
        "query_tokens": [],
        "source_tokens": [],
    }


def metadata_filename(metadata: dict[str, Any]) -> str:
    for key in ["imported_source", "source", "original_source"]:
        value = str(metadata.get(key, "") or "").strip()
        if value:
            return Path(value.replace("\\", "/")).name
    return ""


def extract_metadata_tokens(metadata: dict[str, Any], *, min_len: int = 3) -> set[str]:
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
    tokens: set[str] = set()
    for field in fields:
        tokens.update(extract_source_like_tokens(str(metadata.get(field, "")), min_len=min_len))
    return tokens


def extract_source_like_tokens(text: str, *, min_len: int = 3) -> set[str]:
    tokens: set[str] = set()
    for match in TOKEN_RE.findall(str(text)):
        normalized = normalize_token(match)
        for part in split_token(normalized):
            if len(part) >= min_len:
                tokens.add(part)
    return tokens


def split_token(token: str) -> list[str]:
    pieces: list[str] = []
    for raw in re.split(r"[/\\_.\-]+", token):
        if not raw:
            continue
        pieces.append(raw)
        pieces.extend(split_camel_case(raw))
    return [piece for piece in pieces if piece]


def split_camel_case(text: str) -> list[str]:
    if text.islower() or text.isupper():
        return []
    parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+", text)
    return [part.lower() for part in parts if part and part.lower() != text.lower()]


def normalize_token(text: str) -> str:
    return str(text).strip().lower()


def normalize_for_match(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())
