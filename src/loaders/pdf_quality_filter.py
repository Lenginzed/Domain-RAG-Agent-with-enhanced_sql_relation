from __future__ import annotations

import re
from typing import Any


DEFAULT_PDF_QUALITY_CONFIG: dict[str, Any] = {
    "min_text_chars": 200,
    "max_uni_marker_count": 20,
    "max_uni_marker_ratio": 0.05,
    "low_text_page_ratio_threshold": 0.6,
    "classify_garbled_pdf": True,
    "classify_figure_pdf": True,
}


def analyze_pdf_quality(
    page_texts: list[str] | None = None,
    *,
    error: str | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify extracted PDF text quality with transparent heuristics."""

    merged_config = {**DEFAULT_PDF_QUALITY_CONFIG, **(config or {})}
    if error:
        return {
            "pdf_quality": "unreadable",
            "text_char_count": 0,
            "page_count": 0,
            "uni_marker_count": 0,
            "uni_marker_ratio": 0.0,
            "low_text_page_ratio": 1.0,
            "should_index": False,
            "reason": f"pdf_read_error: {error}",
        }

    pages = page_texts or []
    text = "\n".join(pages)
    text_char_count = len(text.strip())
    page_count = len(pages)
    uni_marker_count = count_uni_markers(text)
    uni_marker_ratio = round(uni_marker_count / max(text_char_count, 1), 6)
    min_text_chars = int(merged_config.get("min_text_chars", 200))
    low_text_pages = sum(len((page or "").strip()) < min_text_chars for page in pages)
    low_text_page_ratio = round(low_text_pages / max(page_count, 1), 6)

    quality = "ok"
    reason = "text_quality_ok"
    if (
        bool(merged_config.get("classify_garbled_pdf", True))
        and (
            uni_marker_count > int(merged_config.get("max_uni_marker_count", 20))
            or uni_marker_ratio > float(merged_config.get("max_uni_marker_ratio", 0.05))
        )
    ):
        quality = "garbled"
        reason = (
            f"/uni markers exceed threshold: count={uni_marker_count}, "
            f"ratio={uni_marker_ratio}"
        )
    elif (
        bool(merged_config.get("classify_figure_pdf", True))
        and page_count > 1
        and low_text_page_ratio >= float(merged_config.get("low_text_page_ratio_threshold", 0.6))
    ):
        quality = "figure_like"
        reason = f"many pages have low extractable text: ratio={low_text_page_ratio}"
    elif text_char_count < min_text_chars:
        quality = "low_text"
        reason = f"text_char_count {text_char_count} below min_text_chars {min_text_chars}"

    return {
        "pdf_quality": quality,
        "text_char_count": text_char_count,
        "page_count": page_count,
        "uni_marker_count": uni_marker_count,
        "uni_marker_ratio": uni_marker_ratio,
        "low_text_page_ratio": low_text_page_ratio,
        "should_index": quality == "ok",
        "reason": reason,
    }


def count_uni_markers(text: str) -> int:
    return len(re.findall(r"/uni(?:[0-9a-fA-F]{4})?", text or "", flags=re.IGNORECASE))
