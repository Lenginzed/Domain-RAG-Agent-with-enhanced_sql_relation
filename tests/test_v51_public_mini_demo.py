from __future__ import annotations

import subprocess
from pathlib import Path

from scripts.build_mini_demo_index_v51 import PROJECT_ROOT, build_index, load_config, resolve_project_path
from scripts.run_mini_demo_v51 import run_demo


def test_public_mini_demo_files_exist() -> None:
    docs_dir = PROJECT_ROOT / "examples" / "mini_corpus" / "docs"
    assert docs_dir.exists()
    assert (docs_dir / "event_driven_reward.py").exists()
    assert (docs_dir / "HierarchySelfplay.yaml").exists()
    assert (docs_dir / "stage13_risk_report.md").exists()
    assert (docs_dir / "missile_engine.md").exists()
    assert (PROJECT_ROOT / "config" / "mini_demo_v51.example.yaml").exists()


def test_public_mini_demo_end_to_end() -> None:
    config = load_config()
    private_marker = Path("data") / "raw_imported"
    corpus_dir = Path(config["mini_demo"]["corpus_dir"])
    assert private_marker.as_posix() not in corpus_dir.as_posix()
    assert "chroma" not in config["mini_demo"]["output_dir"].lower()

    summary = build_index(config)
    assert summary["documents"] == 4
    assert summary["chunks"] == 4
    assert summary["events"] == 4
    assert summary["entities"] >= 10
    assert summary["event_entities"] >= 10

    payload = run_demo(config)
    assert len(payload["results"]) == 3
    for result in payload["results"]:
        assert result["expected_source_hit"], result
        assert result["final_sources"]
        top_source = result["final_sources"][0]
        assert top_source.get("relation_path") or top_source.get("fusion_reasons")

    db_path = resolve_project_path(config["mini_demo"]["sqlite_db_path"])
    assert db_path.exists()
    db_rel = db_path.relative_to(PROJECT_ROOT).as_posix()
    check = subprocess.run(["git", "check-ignore", "-q", db_rel], cwd=PROJECT_ROOT)
    assert check.returncode == 0
