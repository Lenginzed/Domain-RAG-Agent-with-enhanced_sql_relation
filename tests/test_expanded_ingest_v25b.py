from __future__ import annotations

import csv
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "expanded_ingest_v25b.yaml"
PLAN_PATH = PROJECT_ROOT / "data" / "processed" / "loader_hardening_v25a" / "candidate_expansion_plan_v25b.csv"


def test_expanded_ingest_config_exists() -> None:
    assert CONFIG_PATH.exists()


def test_candidate_plan_has_80_recommended_files() -> None:
    rows = read_plan_rows()
    recommended = [row for row in rows if as_bool(row.get("recommended_for_v25b", ""))]

    assert len(recommended) == 80


def test_recommended_files_exclude_garbled_pdf_and_train_episode() -> None:
    recommended = [row for row in read_plan_rows() if as_bool(row.get("recommended_for_v25b", ""))]

    assert not [
        row
        for row in recommended
        if row.get("pdf_quality") == "garbled" and not as_bool(row.get("should_index", ""))
    ]
    assert not [row for row in recommended if "train_env0_episodes" in row.get("source", "").lower()]


def test_expanded_collection_target_does_not_reuse_v2b1_directory() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    persist_directory = str(config["chroma"]["persist_directory"]).replace("\\", "/")
    collection_name = str(config["chroma"]["collection_name"])

    assert persist_directory != "storage/chroma_real_smoke_v2b1"
    assert collection_name != "domain_rag_real_smoke_v2b1"


def read_plan_rows() -> list[dict[str, str]]:
    with PLAN_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}
