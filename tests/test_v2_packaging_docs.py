from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_readme_exists() -> None:
    assert (PROJECT_ROOT / "README.md").exists()


def test_v2_packaging_docs_exist() -> None:
    required = [
        "docs/project_structure.md",
        "docs/v2_demo_commands.md",
        "docs/v2_metrics_summary.md",
        "docs/resume_project_description.md",
        "docs/interview_talking_points.md",
        "docs/dev_report_v2_packaging.md",
    ]
    missing = [path for path in required if not (PROJECT_ROOT / path).exists()]
    assert missing == []


def test_readme_contains_v2_keywords() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    required_keywords = [
        "enhanced_keyword",
        "evidence_quality_score",
        "negative_refusal_rate",
        "citation",
        "manual review",
    ]
    missing = [keyword for keyword in required_keywords if keyword not in readme]
    assert missing == []
