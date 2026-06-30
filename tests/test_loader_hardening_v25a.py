from __future__ import annotations

from pathlib import Path

import yaml

from src.loaders.csv_summary_loader import load_csv_summary_file
from src.loaders.pdf_quality_filter import analyze_pdf_quality


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "loader_hardening.yaml"


def test_csv_summary_loader_outputs_compact_document(tmp_path: Path) -> None:
    csv_path = tmp_path / "demo_results.csv"
    csv_path.write_text(
        "episode,reward,policy\n"
        "1,10.5,baseline\n"
        "2,12.0,baseline\n"
        "3,18.25,enhanced\n",
        encoding="utf-8",
    )

    documents = load_csv_summary_file(csv_path)

    assert 1 <= len(documents) <= 2
    assert "row_count: 3" in documents[0].page_content
    assert "Numeric stats" in documents[0].page_content


def test_csv_summary_metadata_contains_profile_fields(tmp_path: Path) -> None:
    csv_path = tmp_path / "demo_results.csv"
    csv_path.write_text("a,b\n1,x\n2,y\n", encoding="utf-8")

    document = load_csv_summary_file(csv_path)[0]
    metadata = document.metadata

    assert metadata["loader_type"] == "csv_summary"
    assert metadata["row_count"] == 2
    assert metadata["column_count"] == 2
    assert metadata["file_ext"] == ".csv"


def test_pdf_quality_filter_detects_garbled_uni_markers() -> None:
    page_texts = ["/uni4E00 /uni4E01 /uni4E02 " * 20]

    quality = analyze_pdf_quality(page_texts)

    assert quality["pdf_quality"] == "garbled"
    assert quality["should_index"] is False
    assert quality["uni_marker_count"] > 20


def test_loader_hardening_config_paths_exist() -> None:
    assert CONFIG_PATH.exists()
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    assert config["audit"]["source_dir"] == "data/raw_imported"
    assert config["audit"]["output_dir"] == "data/processed/loader_hardening_v25a"
    assert config["csv"]["mode"] == "summary_aware"
