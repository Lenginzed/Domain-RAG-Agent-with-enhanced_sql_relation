from __future__ import annotations

import re
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LINK_RE = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")


def tracked_public_markdown_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "README.md", "docs/*.md", "examples/*.md", "examples/**/*.md"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return [PROJECT_ROOT / line.strip() for line in result.stdout.splitlines() if line.strip()]


def test_readme_exists() -> None:
    assert (PROJECT_ROOT / "README.md").exists()


def test_relative_markdown_links_resolve() -> None:
    missing: list[str] = []
    for path in tracked_public_markdown_files():
        text = path.read_text(encoding="utf-8")
        for raw_target in LINK_RE.findall(text):
            target = raw_target.strip()
            if not target or target.startswith(("#", "http://", "https://", "mailto:")):
                continue
            target_path_text = target.split("#", 1)[0]
            if not target_path_text:
                continue
            resolved = (path.parent / target_path_text).resolve()
            if not resolved.exists():
                missing.append(f"{path.relative_to(PROJECT_ROOT)} -> {target}")
    assert not missing, "\n".join(missing)


def test_public_docs_have_no_local_absolute_paths_or_forbidden_phrase() -> None:
    forbidden_phrase = "smoke" + " " + "test"
    failures: list[str] = []
    for path in tracked_public_markdown_files():
        text = path.read_text(encoding="utf-8")
        if "F:" + "\\" in text or "E:" + "\\" in text:
            failures.append(f"local absolute path in {path.relative_to(PROJECT_ROOT)}")
        if forbidden_phrase.lower() in text.lower():
            failures.append(f"forbidden phrase in {path.relative_to(PROJECT_ROOT)}")
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
