from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chunkers.text_chunker import chunk_documents  # noqa: E402
from src.indexes.vector_index import DEFAULT_COLLECTION_NAME, build_index, collection_count  # noqa: E402
from src.loaders.base import SUPPORTED_EXTENSIONS, load_file  # noqa: E402


RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
LOG_PATH = PROJECT_ROOT / "storage" / "logs" / "ingest_v1.json"


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )


def ingest(
    *,
    raw_data_dir: Path = RAW_DATA_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    reset: bool = True,
    write_log: bool = True,
) -> dict[str, Any]:
    scanned_files = [path for path in raw_data_dir.rglob("*") if path.is_file()]
    supported_files = [path for path in scanned_files if path.suffix.lower() in SUPPORTED_EXTENSIONS]
    skipped_files = [
        _relative(path)
        for path in scanned_files
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS
    ]
    errors: list[str] = []

    documents = []
    for path in tqdm(supported_files, desc="Loading files"):
        before = len(documents)
        try:
            documents.extend(load_file(path))
        except Exception as exc:  # noqa: BLE001 - keep ingest moving.
            errors.append(f"{_relative(path)}: {exc}")
        if len(documents) == before:
            skipped_files.append(_relative(path))

    chunks = chunk_documents(documents)
    vector_store = build_index(chunks, collection_name=collection_name, reset=reset)
    indexed_count = collection_count(vector_store)

    stats = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "collection_name": collection_name,
        "scanned_files": len(scanned_files),
        "supported_files": len(supported_files),
        "loaded_documents": len(documents),
        "generated_chunks": len(chunks),
        "indexed_chunks": indexed_count,
        "skipped_files": skipped_files,
        "errors": errors,
    }
    if write_log:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        LOG_PATH.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def main() -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description="Ingest raw documents into the V1 Chroma vector index.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--collection-name", default=DEFAULT_COLLECTION_NAME)
    parser.add_argument("--no-reset", action="store_true", help="Append to existing collection instead of rebuilding.")
    args = parser.parse_args()

    try:
        stats = ingest(
            raw_data_dir=args.raw_dir,
            collection_name=args.collection_name,
            reset=not args.no_reset,
        )
    except Exception as exc:  # noqa: BLE001 - script should report clear failures.
        print("Ingestion failed.")
        print(f"reason: {exc}")
        return 1

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"Saved ingest log to: {LOG_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
