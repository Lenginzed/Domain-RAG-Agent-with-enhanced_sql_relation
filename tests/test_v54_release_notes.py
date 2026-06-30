from __future__ import annotations

import re
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_v010_release_notes_are_finalized() -> None:
    release_notes_path = PROJECT_ROOT / "docs" / "releases" / "v0.1.0.md"
    assert release_notes_path.exists()
    text = release_notes_path.read_text(encoding="utf-8")

    assert "# v0.1.0 Release Notes" in text
    assert "Release tag:" in text
    assert "v0.1.0" in text
    assert "Public Mini Demo" in text
    assert "What is included" in text
    assert "What is not included" in text
    assert "Known limitations" in text
    assert "Release" + " " + "Draft" not in text
    assert "No GitHub release or tag has been created yet" not in text
    assert "Prepared after commit:" not in text


def test_readme_points_to_release_notes() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    assert "## Release" in readme
    assert "v0.1.0 release notes" in readme
    assert "(docs/releases/v0.1.0.md)" in readme
    assert "Release" + " " + "Draft" not in readme
    assert "No GitHub release or tag has been created in this pass" not in readme


def test_public_docs_avoid_restricted_wording_and_local_paths() -> None:
    restricted = [
        "state-" + "of-" + "the-art",
        "production-" + "ready",
        "fully" + " " + "automated",
        "complete" + " " + "SAG" + " " + "implementation",
        "guaran" + "teed",
        "best",
    ]
    paths = [
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "CHANGELOG.md",
        PROJECT_ROOT / "docs" / "project_status.md",
        PROJECT_ROOT / "docs" / "public_release_notes.md",
        PROJECT_ROOT / "docs" / "releases" / "v0.1.0.md",
        PROJECT_ROOT / "docs" / "release_checklist.md",
        PROJECT_ROOT / "docs" / "repository_presentation_check.md",
    ]
    failures: list[str] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        lower = text.lower()
        for phrase in restricted:
            if phrase.lower() in lower:
                failures.append(f"{path.relative_to(PROJECT_ROOT)} contains restricted wording: {phrase}")
        if "F:" + "\\" in text or "E:" + "\\" in text:
            failures.append(f"{path.relative_to(PROJECT_ROOT)} contains a local absolute path")
    assert not failures, "\n".join(failures)


def test_tracked_files_exclude_generated_runtime_artifacts() -> None:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    forbidden_patterns = [
        re.compile(r"\.db$", re.IGNORECASE),
        re.compile(r"\.sqlite3?$", re.IGNORECASE),
        re.compile(r"(^|/)storage/chroma", re.IGNORECASE),
        re.compile(r"(^|/)storage/sqlite", re.IGNORECASE),
        re.compile(r"(^|/)storage/logs", re.IGNORECASE),
    ]
    bad_paths = [
        path
        for path in result.stdout.splitlines()
        if any(pattern.search(path.replace("\\", "/")) for pattern in forbidden_patterns)
    ]
    assert not bad_paths, "\n".join(bad_paths)
