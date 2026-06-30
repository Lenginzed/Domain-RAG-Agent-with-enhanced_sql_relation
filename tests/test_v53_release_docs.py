from __future__ import annotations

import re
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_release_documents_exist() -> None:
    assert (PROJECT_ROOT / "CHANGELOG.md").exists()
    assert (PROJECT_ROOT / "docs" / "releases" / "v0.1.0.md").exists()
    assert (PROJECT_ROOT / "docs" / "release_checklist.md").exists()
    assert (PROJECT_ROOT / "docs" / "repository_presentation_check.md").exists()


def test_readme_release_links_exist() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    expected_links = [
        "CHANGELOG.md",
        "docs/releases/v0.1.0.md",
        "docs/release_checklist.md",
        "docs/repository_presentation_check.md",
    ]
    for link in expected_links:
        assert f"({link})" in readme
        assert (PROJECT_ROOT / link).exists()


def test_release_docs_avoid_banned_wording_and_local_paths() -> None:
    banned = [
        "smoke" + " " + "test",
        "state-" + "of-" + "the-art",
        "production-" + "ready",
        "fully" + " " + "automated",
        "complete" + " " + "SAG" + " " + "implementation",
        "guaran" + "teed",
        "best",
    ]
    paths = [
        PROJECT_ROOT / "CHANGELOG.md",
        PROJECT_ROOT / "docs" / "releases" / "v0.1.0.md",
        PROJECT_ROOT / "docs" / "release_checklist.md",
        PROJECT_ROOT / "docs" / "repository_presentation_check.md",
    ]
    failures: list[str] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        lower = text.lower()
        for phrase in banned:
            if phrase.lower() in lower:
                failures.append(f"{path.relative_to(PROJECT_ROOT)} contains banned wording: {phrase}")
        if "F:" + "\\" in text or "E:" + "\\" in text:
            failures.append(f"{path.relative_to(PROJECT_ROOT)} contains local absolute path")
    assert not failures, "\n".join(failures)


def test_tracked_files_exclude_runtime_artifacts() -> None:
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
