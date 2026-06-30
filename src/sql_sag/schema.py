from __future__ import annotations

import sqlite3
from pathlib import Path


SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS documents (
  doc_id TEXT PRIMARY KEY,
  source TEXT NOT NULL,
  imported_source TEXT,
  original_source TEXT,
  category_dir TEXT,
  doc_type TEXT,
  file_ext TEXT,
  title TEXT,
  created_at TEXT
);

CREATE TABLE IF NOT EXISTS chunks (
  chunk_id TEXT PRIMARY KEY,
  doc_id TEXT NOT NULL,
  chunk_index INTEGER,
  source TEXT,
  section TEXT,
  text TEXT,
  content_preview TEXT,
  chroma_chunk_id TEXT,
  FOREIGN KEY(doc_id) REFERENCES documents(doc_id)
);

CREATE TABLE IF NOT EXISTS events (
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

CREATE TABLE IF NOT EXISTS entities (
  entity_id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  normalized_name TEXT NOT NULL,
  entity_type TEXT,
  source_hint TEXT,
  UNIQUE(normalized_name, entity_type)
);

CREATE TABLE IF NOT EXISTS event_entities (
  event_id TEXT NOT NULL,
  entity_id TEXT NOT NULL,
  role TEXT,
  confidence REAL,
  PRIMARY KEY(event_id, entity_id, role),
  FOREIGN KEY(event_id) REFERENCES events(event_id),
  FOREIGN KEY(entity_id) REFERENCES entities(entity_id)
);

CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source);
CREATE INDEX IF NOT EXISTS idx_documents_category ON documents(category_dir);
CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_events_chunk_id ON events(chunk_id);
CREATE INDEX IF NOT EXISTS idx_events_doc_id ON events(doc_id);
CREATE INDEX IF NOT EXISTS idx_entities_normalized_name ON entities(normalized_name);
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_event_entities_event ON event_entities(event_id);
CREATE INDEX IF NOT EXISTS idx_event_entities_entity ON event_entities(entity_id);
"""


TABLES = ["documents", "chunks", "events", "entities", "event_entities"]


def connect(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(SCHEMA_SQL)
    connection.commit()


def reset_database(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    if path.exists():
        path.unlink()
    connection = connect(path)
    create_schema(connection)
    return connection


def table_counts(connection: sqlite3.Connection) -> dict[str, int]:
    counts: dict[str, int] = {}
    for table in TABLES:
        counts[table] = int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    return counts
