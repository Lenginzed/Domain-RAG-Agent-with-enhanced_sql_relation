from __future__ import annotations

import json
import re
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "mini_demo_v51.example.yaml"


SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE documents (
  doc_id TEXT PRIMARY KEY,
  source TEXT NOT NULL,
  category_dir TEXT,
  doc_type TEXT,
  file_ext TEXT,
  title TEXT
);

CREATE TABLE chunks (
  chunk_id TEXT PRIMARY KEY,
  doc_id TEXT NOT NULL,
  chunk_index INTEGER,
  source TEXT,
  text TEXT,
  content_preview TEXT,
  FOREIGN KEY(doc_id) REFERENCES documents(doc_id)
);

CREATE TABLE events (
  event_id TEXT PRIMARY KEY,
  chunk_id TEXT NOT NULL,
  doc_id TEXT NOT NULL,
  event_text TEXT,
  event_type TEXT,
  source TEXT,
  confidence REAL,
  FOREIGN KEY(chunk_id) REFERENCES chunks(chunk_id),
  FOREIGN KEY(doc_id) REFERENCES documents(doc_id)
);

CREATE TABLE entities (
  entity_id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  normalized_name TEXT NOT NULL,
  entity_type TEXT,
  UNIQUE(normalized_name, entity_type)
);

CREATE TABLE event_entities (
  event_id TEXT NOT NULL,
  entity_id TEXT NOT NULL,
  role TEXT,
  confidence REAL,
  PRIMARY KEY(event_id, entity_id, role),
  FOREIGN KEY(event_id) REFERENCES events(event_id),
  FOREIGN KEY(entity_id) REFERENCES entities(entity_id)
);

CREATE INDEX idx_documents_source ON documents(source);
CREATE INDEX idx_entities_name ON entities(normalized_name);
CREATE INDEX idx_event_entities_event ON event_entities(event_id);
CREATE INDEX idx_event_entities_entity ON event_entities(entity_id);
"""


DOMAIN_TERMS = {
    "config",
    "control",
    "engine",
    "evaluation",
    "event",
    "event_driven_reward",
    "guidance",
    "hierarchy",
    "hierarchy_selfplay",
    "missile",
    "missile_engine",
    "report",
    "reward",
    "risk",
    "risk_penalty",
    "selfplay",
    "stage13",
}

STOP_TERMS = {
    "demo",
    "example",
    "note",
    "small",
    "such",
    "synthetic",
    "terms",
    "that",
    "this",
    "true",
    "used",
}


@dataclass(frozen=True)
class MiniEntity:
    name: str
    normalized_name: str
    entity_type: str
    role: str


def load_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    resolved = resolve_project_path(path)
    with resolved.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid mini demo config: {resolved}")
    return data


def resolve_project_path(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return PROJECT_ROOT / value


def normalize_name(value: str) -> str:
    text = str(value).strip()
    text = re.sub(r"\.(py|yaml|yml|md|txt)$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    text = text.lower()
    text = re.sub(r"[\s./\\:-]+", "_", text)
    text = re.sub(r"[^a-z0-9_]+", "", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def split_entity_parts(value: str) -> list[str]:
    normalized = normalize_name(value)
    parts = [part for part in normalized.split("_") if len(part) >= 3]
    return [normalized, *parts] if normalized else parts


def iter_camel_case_names(text: str) -> list[str]:
    return re.findall(r"\b[A-Z][A-Za-z0-9]+(?:[A-Z][A-Za-z0-9]+)*\b", text)


def extract_entities(source: Path, text: str) -> list[MiniEntity]:
    entities: list[MiniEntity] = []
    stem = source.stem
    suffix = source.suffix.lower()
    file_entity_type = "code_file" if suffix == ".py" else "config" if suffix in {".yaml", ".yml"} else "file_stem"

    for item in split_entity_parts(stem):
        if item in STOP_TERMS:
            continue
        entities.append(MiniEntity(name=item, normalized_name=normalize_name(item), entity_type=file_entity_type, role="filename"))

    for item in iter_camel_case_names(text):
        if normalize_name(item) in STOP_TERMS:
            continue
        entities.append(MiniEntity(name=item, normalized_name=normalize_name(item), entity_type="code_symbol", role="content"))
        for part in split_entity_parts(item):
            if part in STOP_TERMS:
                continue
            entities.append(MiniEntity(name=part, normalized_name=normalize_name(part), entity_type="domain_term", role="content"))

    token_blob = " ".join([text, stem])
    for token in re.findall(r"\b[A-Za-z][A-Za-z0-9_]*\b", token_blob):
        normalized = normalize_name(token)
        if normalized in STOP_TERMS:
            continue
        if normalized in DOMAIN_TERMS or normalized.startswith("stage"):
            entities.append(MiniEntity(name=token, normalized_name=normalized, entity_type="domain_term", role="content"))
        if "_" in normalized:
            for part in split_entity_parts(normalized):
                if part in DOMAIN_TERMS and part not in STOP_TERMS:
                    entities.append(MiniEntity(name=part, normalized_name=part, entity_type="domain_term", role="content"))

    deduped: dict[tuple[str, str, str], MiniEntity] = {}
    for entity in entities:
        if len(entity.normalized_name) < 3:
            continue
        deduped.setdefault((entity.normalized_name, entity.entity_type, entity.role), entity)
    return list(deduped.values())


def infer_category(path: Path) -> tuple[str, str]:
    suffix = path.suffix.lower()
    stem = path.stem.lower()
    if suffix == ".py":
        return "code", "code"
    if suffix in {".yaml", ".yml"}:
        return "configs", "config"
    if "report" in stem:
        return "experiments", "report"
    return "notes", "note"


def safe_corpus_files(corpus_dir: Path) -> list[Path]:
    resolved = corpus_dir.resolve()
    expected_root = (PROJECT_ROOT / "examples" / "mini_corpus" / "docs").resolve()
    if resolved != expected_root:
        raise ValueError(f"Mini demo corpus must be under examples/mini_corpus/docs: {resolved}")
    if "raw_imported" in {part.lower() for part in resolved.parts}:
        raise ValueError("Mini demo refuses private corpus-like paths.")
    files = sorted(path for path in resolved.iterdir() if path.is_file())
    if not files:
        raise FileNotFoundError(resolved)
    return files


def reset_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    connection.executescript(SCHEMA_SQL)
    connection.commit()
    return connection


def get_or_create_entity(connection: sqlite3.Connection, entity: MiniEntity) -> str:
    existing = connection.execute(
        "SELECT entity_id FROM entities WHERE normalized_name = ? AND entity_type = ?",
        (entity.normalized_name, entity.entity_type),
    ).fetchone()
    if existing:
        return str(existing["entity_id"])
    count = int(connection.execute("SELECT COUNT(*) FROM entities").fetchone()[0]) + 1
    entity_id = f"ent_{count:04d}"
    connection.execute(
        "INSERT INTO entities(entity_id, name, normalized_name, entity_type) VALUES (?, ?, ?, ?)",
        (entity_id, entity.name, entity.normalized_name, entity.entity_type),
    )
    return entity_id


def build_index(config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = load_config() if config is None else config
    corpus_dir = resolve_project_path(config["mini_demo"]["corpus_dir"])
    output_dir = resolve_project_path(config["mini_demo"]["output_dir"])
    db_path = resolve_project_path(config["mini_demo"]["sqlite_db_path"])
    files = safe_corpus_files(corpus_dir)
    connection = reset_db(db_path)

    for doc_index, path in enumerate(files, start=1):
        text = path.read_text(encoding="utf-8")
        category_dir, doc_type = infer_category(path)
        rel_source = path.relative_to(PROJECT_ROOT).as_posix()
        doc_id = f"doc_{doc_index:03d}_{normalize_name(path.stem)}"
        chunk_id = f"{doc_id}_chunk_000"
        event_id = f"{doc_id}_event_000"
        preview = " ".join(text.split())[:240]
        event_text = f"{path.name} | category={category_dir} | preview={preview}"

        connection.execute(
            "INSERT INTO documents(doc_id, source, category_dir, doc_type, file_ext, title) VALUES (?, ?, ?, ?, ?, ?)",
            (doc_id, rel_source, category_dir, doc_type, path.suffix.lower(), path.stem),
        )
        connection.execute(
            "INSERT INTO chunks(chunk_id, doc_id, chunk_index, source, text, content_preview) VALUES (?, ?, ?, ?, ?, ?)",
            (chunk_id, doc_id, 0, rel_source, text, preview),
        )
        connection.execute(
            "INSERT INTO events(event_id, chunk_id, doc_id, event_text, event_type, source, confidence) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (event_id, chunk_id, doc_id, event_text, "mini_chunk_event", rel_source, 1.0),
        )

        for entity in extract_entities(path, text):
            entity_id = get_or_create_entity(connection, entity)
            connection.execute(
                "INSERT OR IGNORE INTO event_entities(event_id, entity_id, role, confidence) VALUES (?, ?, ?, ?)",
                (event_id, entity_id, entity.role, 1.0),
            )

    connection.commit()
    summary = table_counts(connection)
    summary["db_path"] = db_path.relative_to(PROJECT_ROOT).as_posix()
    summary["corpus_files"] = [path.name for path in files]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "mini_demo_index_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    connection.close()
    return summary


def table_counts(connection: sqlite3.Connection) -> dict[str, int]:
    return {
        "documents": int(connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]),
        "chunks": int(connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]),
        "events": int(connection.execute("SELECT COUNT(*) FROM events").fetchone()[0]),
        "entities": int(connection.execute("SELECT COUNT(*) FROM entities").fetchone()[0]),
        "event_entities": int(connection.execute("SELECT COUNT(*) FROM event_entities").fetchone()[0]),
    }


def main() -> int:
    summary = build_index()
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
