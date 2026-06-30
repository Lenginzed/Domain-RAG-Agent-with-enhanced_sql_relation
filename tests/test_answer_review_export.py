from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.export_answer_review_template import (
    INPUT_JSON,
    OUTPUT_CSV,
    OUTPUT_JSON,
    OUTPUT_MD,
    export_review_template,
)


def test_export_answer_review_template_outputs_files() -> None:
    summary = export_review_template()

    assert OUTPUT_MD.exists()
    assert OUTPUT_CSV.exists()
    assert OUTPUT_JSON.exists()
    assert summary["output_markdown"] == str(OUTPUT_MD)
    assert summary["output_csv_template"] == str(OUTPUT_CSV)


def test_markdown_contains_manual_review_fields() -> None:
    export_review_template()
    markdown = OUTPUT_MD.read_text(encoding="utf-8")
    assert "manual_judgment" in markdown
    assert "manual_citation_judgment" in markdown
    assert "逐题审阅卡片" in markdown


def test_csv_template_contains_manual_fields() -> None:
    export_review_template()
    with OUTPUT_CSV.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = set(reader.fieldnames or [])
        rows = list(reader)
    assert rows
    assert {"manual_judgment", "manual_citation_judgment", "manual_notes"}.issubset(fieldnames)


def test_summary_needs_human_review_count_matches_input() -> None:
    summary = export_review_template()
    input_result = json.loads(INPUT_JSON.read_text(encoding="utf-8"))
    expected = input_result["metrics"]["needs_human_review_count"]
    output_summary = json.loads(OUTPUT_JSON.read_text(encoding="utf-8"))
    assert summary["needs_human_review_count"] == expected
    assert output_summary["needs_human_review_count"] == expected
