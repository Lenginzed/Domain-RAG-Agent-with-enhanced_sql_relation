from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "real_ingest_smoke.yaml"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document

from src.chunkers.text_chunker import chunk_documents
from src.indexes.vector_index import build_index, collection_count
from src.loaders.base import load_file
from src.metadata.validators import normalize_metadata


DOC_TYPE_BY_CATEGORY = {
    "code": "code",
    "configs": "config",
    "experiments": "experiment",
    "notes": "note",
    "papers": "paper",
    "thesis_or_reports": "report",
}


def main() -> int:
    try:
        config = load_config()
        summary = ingest_real_smoke(config)
    except Exception as exc:  # noqa: BLE001 - smoke ingest failures should be explicit.
        print("Real smoke ingest failed.")
        print(f"reason: {exc}")
        return 1

    print("Real smoke ingest completed.")
    print(f"environment: {summary['used_environment']}")
    print(f"selected_files: {summary['selected_files']}")
    print(f"loaded_documents: {summary['loaded_documents']}")
    print(f"indexed_chunks: {summary['indexed_chunks']}")
    print(f"collection_name: {summary['collection_name']}")
    print(f"persist_directory: {summary['persist_directory']}")
    print(f"collection_count: {summary['collection_count']}")
    print(f"category_dir_coverage: {summary['category_dir_coverage']}")
    print(f"imported_source_coverage: {summary['imported_source_coverage']}")
    print(f"original_source_coverage: {summary['original_source_coverage']}")
    print(f"doc_type_counts: {summary['doc_type_counts']}")
    print(f"output_log: {summary['output_log']}")
    return 0 if summary["ingest_passed"] else 2


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Invalid config: {path}")
    return config


def ingest_real_smoke(config: dict[str, Any]) -> dict[str, Any]:
    selected_path = PROJECT_ROOT / str(config["output_selected_files"])
    selected_rows = read_csv(selected_path)
    max_chunks_total = int(config.get("max_chunks_total", 1000))
    collection_name = str(config["collection_name"])
    persist_directory = PROJECT_ROOT / str(config["persist_directory"])
    output_log = PROJECT_ROOT / str(config["output_ingest_log"])

    documents: list[Document] = []
    load_errors: list[dict[str, str]] = []
    for row in selected_rows:
        relative_path = str(row.get("relative_path") or "")
        path = PROJECT_ROOT / relative_path
        category_dir = str(row.get("category_dir") or category_for_imported_file(path))
        imported_source = relative_to_project(path)
        original_source = str(row.get("original_source") or "")
        doc_type = DOC_TYPE_BY_CATEGORY.get(category_dir, "unknown")
        try:
            loaded = load_file(path)
        except Exception as exc:  # noqa: BLE001
            load_errors.append({"relative_path": relative_path, "error": str(exc)})
            continue

        for document in loaded:
            metadata = dict(document.metadata or {})
            metadata.update(
                {
                    "source": imported_source,
                    "doc_type": doc_type,
                    "category_dir": category_dir,
                    "imported_source": imported_source,
                    "original_source": original_source,
                }
            )
            document.metadata = normalize_metadata(metadata)
            documents.append(document)

    chunks = chunk_documents(documents)
    if len(chunks) > max_chunks_total:
        raise ValueError(f"Refusing to ingest {len(chunks)} chunks; max_chunks_total={max_chunks_total}")

    for chunk in chunks:
        metadata = dict(chunk.metadata or {})
        metadata["source"] = str(metadata.get("imported_source") or metadata.get("source") or "")
        chunk.metadata = normalize_metadata(metadata)

    vector_store = build_index(
        chunks,
        persist_directory=persist_directory,
        collection_name=collection_name,
        reset=True,
    )
    stored_count = collection_count(vector_store)

    summary = build_summary(
        config=config,
        selected_rows=selected_rows,
        documents=documents,
        chunks=chunks,
        load_errors=load_errors,
        collection_name=collection_name,
        persist_directory=persist_directory,
        collection_count_value=stored_count,
        output_log=output_log,
    )
    output_log.parent.mkdir(parents=True, exist_ok=True)
    output_log.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def build_summary(
    *,
    config: dict[str, Any],
    selected_rows: list[dict[str, str]],
    documents: list[Document],
    chunks: list[Document],
    load_errors: list[dict[str, str]],
    collection_name: str,
    persist_directory: Path,
    collection_count_value: int,
    output_log: Path,
) -> dict[str, Any]:
    total_chunks = len(chunks)
    category_dir_count = sum(bool(chunk.metadata.get("category_dir")) for chunk in chunks)
    imported_source_count = sum(bool(chunk.metadata.get("imported_source")) for chunk in chunks)
    original_source_count = sum(bool(chunk.metadata.get("original_source")) for chunk in chunks)
    doc_type_counts = Counter(str(chunk.metadata.get("doc_type") or "unknown") for chunk in chunks)
    source_counts = Counter(str(chunk.metadata.get("source") or "unknown") for chunk in chunks)
    category_counts = Counter(str(chunk.metadata.get("category_dir") or "unknown") for chunk in chunks)
    selected_filenames = [Path(str(row.get("relative_path") or "")).name for row in selected_rows]
    train_episode_selected = [name for name in selected_filenames if "train_env0_episodes" in name.lower()]
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "used_environment": "Lenginzed_RAG",
        "selected_files": len(selected_rows),
        "selected_file_paths": [row.get("relative_path", "") for row in selected_rows],
        "loaded_documents": len(documents),
        "indexed_chunks": total_chunks,
        "max_chunks_total": int(config.get("max_chunks_total", 1000)),
        "collection_name": collection_name,
        "persist_directory": str(persist_directory),
        "collection_count": collection_count_value,
        "category_dir_coverage": ratio(category_dir_count, total_chunks),
        "imported_source_coverage": ratio(imported_source_count, total_chunks),
        "original_source_coverage": ratio(original_source_count, total_chunks),
        "category_dir_count": category_dir_count,
        "imported_source_count": imported_source_count,
        "original_source_count": original_source_count,
        "missing_original_source_count": total_chunks - original_source_count,
        "doc_type_counts": dict(doc_type_counts),
        "chunk_counts_by_category_dir": dict(category_counts),
        "chunk_counts_by_source": dict(source_counts),
        "load_errors": load_errors,
        "train_env0_episodes_selected": train_episode_selected,
        "embedding_run": True,
        "chroma_written": True,
        "v1_collection_used": collection_name == "domain_rag_v1",
        "external_source_modified": False,
        "raw_imported_modified": False,
        "ingest_passed": all(
            [
                len(selected_rows) <= int(config.get("max_files_total", 30)),
                total_chunks <= int(config.get("max_chunks_total", 1000)),
                collection_name != "domain_rag_v1",
                "storage/chroma_real_smoke_v2b1" in str(persist_directory).replace("\\", "/"),
                collection_count_value == total_chunks,
                category_dir_count == total_chunks,
                imported_source_count == total_chunks,
                original_source_count == total_chunks,
                not train_episode_selected,
                not load_errors,
            ]
        ),
        "output_log": str(output_log),
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV file: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def ratio(count: int, total: int) -> float:
    return round(count / total, 6) if total else 0.0


def category_for_imported_file(path: Path) -> str:
    import_dir = PROJECT_ROOT / "data" / "raw_imported"
    try:
        return path.resolve().relative_to(import_dir).parts[0]
    except Exception:  # noqa: BLE001
        return "unknown"


def relative_to_project(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


if __name__ == "__main__":
    sys.exit(main())
