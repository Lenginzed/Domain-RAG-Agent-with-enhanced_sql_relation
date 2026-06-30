from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from langchain_core.documents import Document


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_loader_hardening_v25a import audit_one_file
from src.indexes.vector_index import build_index, collection_count


CONFIG_PATH = PROJECT_ROOT / "config" / "expanded_ingest_v25b.yaml"
V2B1_PERSIST_DIR = (PROJECT_ROOT / "storage" / "chroma_real_smoke_v2b1").resolve()
V2B1_COLLECTION = "domain_rag_real_smoke_v2b1"

INGEST_FILE_FIELDS = [
    "source",
    "category_dir",
    "file_ext",
    "doc_type",
    "loader_type",
    "document_count",
    "chunk_count",
    "pdf_quality",
    "csv_large",
    "should_index",
    "ingested",
    "reason",
]


def main() -> int:
    configure_stdout()
    try:
        summary = ingest_expanded_collection()
    except Exception as exc:  # noqa: BLE001 - ingest should fail explicitly.
        print("V2.5b expanded collection ingest failed.")
        print(f"reason: {exc}")
        return 1
    print_summary(summary)
    return 0 if summary.get("ingest_passed") else 2


def configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def ingest_expanded_collection(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = load_yaml(config_path)
    hardening_config = load_yaml(resolve_project_path(config["loader_hardening_config"]))
    plan_path = resolve_project_path(config["input_plan"])
    source_dir = resolve_project_path(config["source_dir"]).resolve()
    selected_rows = select_plan_rows(plan_path, config)
    validate_selection(selected_rows, source_dir, config)

    persist_directory = resolve_project_path(config["chroma"]["persist_directory"]).resolve()
    collection_name = str(config["chroma"]["collection_name"])
    validate_chroma_target(persist_directory, collection_name)

    chunk_config = config.get("chunking", {})
    chunk_size = int(chunk_config.get("chunk_size", 800))
    chunk_overlap = int(chunk_config.get("chunk_overlap", 120))
    too_many_threshold = int(hardening_config.get("audit", {}).get("too_many_chunks_threshold", 100))

    all_chunks: list[Document] = []
    file_rows: list[dict[str, Any]] = []
    load_errors: list[dict[str, str]] = []
    for row in selected_rows:
        source = str(row["source"])
        path = resolve_project_path(source)
        audit = audit_one_file(
            path=path,
            config=hardening_config,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            too_many_threshold=too_many_threshold,
        )
        ingested = audit.load_status == "loaded" and audit.should_index and bool(audit.chunks)
        if not ingested:
            load_errors.append({"source": source, "error": audit.error_message or "not_indexable"})
        all_chunks.extend(audit.chunks if ingested else [])
        file_rows.append(
            {
                "source": source,
                "category_dir": audit.category_dir,
                "file_ext": audit.file_ext,
                "doc_type": audit.documents[0].metadata.get("doc_type", "") if audit.documents else row.get("doc_type", ""),
                "loader_type": audit.loader_type,
                "document_count": len(audit.documents),
                "chunk_count": len(audit.chunks),
                "pdf_quality": audit.pdf_quality.get("pdf_quality", ""),
                "csv_large": bool(audit.documents[0].metadata.get("large_csv", False)) if audit.documents else False,
                "should_index": audit.should_index,
                "ingested": ingested,
                "reason": row.get("reason", ""),
            }
        )

    if load_errors:
        raise ValueError(f"Selected files include non-indexable files: {load_errors[:5]}")

    vector_store = build_index(
        all_chunks,
        persist_directory=persist_directory,
        collection_name=collection_name,
        reset=True,
    )
    stored_count = collection_count(vector_store)

    summary = build_summary(
        config=config,
        config_path=config_path,
        selected_rows=selected_rows,
        file_rows=file_rows,
        chunks=all_chunks,
        stored_count=stored_count,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )
    write_outputs(summary, file_rows, config)
    return summary


def select_plan_rows(plan_path: Path, config: dict[str, Any]) -> list[dict[str, str]]:
    rows = read_csv(plan_path)
    recommended_only = bool(config.get("selection", {}).get("recommended_only", True))
    if recommended_only:
        rows = [row for row in rows if as_bool(row.get("recommended_for_v25b", ""))]
    return rows


def validate_selection(rows: list[dict[str, str]], source_dir: Path, config: dict[str, Any]) -> None:
    target_count = int(config.get("selection", {}).get("target_file_count", 80))
    if len(rows) != target_count:
        raise ValueError(f"Expected {target_count} recommended files, found {len(rows)}")
    missing = [row["source"] for row in rows if not resolve_project_path(row["source"]).exists()]
    if missing:
        raise FileNotFoundError(f"Recommended files are missing: {missing[:10]}")
    outside = []
    for row in rows:
        path = resolve_project_path(row["source"]).resolve()
        try:
            path.relative_to(source_dir)
        except ValueError:
            outside.append(str(path))
    if outside:
        raise ValueError(f"Recommended files outside source_dir: {outside[:10]}")
    garbled_pdfs = [
        row["source"]
        for row in rows
        if str(row.get("pdf_quality", "")).lower() == "garbled" and not as_bool(row.get("should_index", ""))
    ]
    if garbled_pdfs:
        raise ValueError(f"Recommended plan contains garbled PDFs: {garbled_pdfs[:10]}")
    train_episode = [row["source"] for row in rows if "train_env0_episodes" in row["source"].lower()]
    if train_episode:
        raise ValueError(f"Recommended plan contains train_env0_episodes files: {train_episode[:10]}")


def validate_chroma_target(persist_directory: Path, collection_name: str) -> None:
    if persist_directory == V2B1_PERSIST_DIR:
        raise ValueError(f"Refusing to write V2b.1 persist directory: {persist_directory}")
    if collection_name == V2B1_COLLECTION:
        raise ValueError(f"Refusing to use V2b.1 collection name: {collection_name}")


def build_summary(
    *,
    config: dict[str, Any],
    config_path: Path,
    selected_rows: list[dict[str, str]],
    file_rows: list[dict[str, Any]],
    chunks: list[Document],
    stored_count: int,
    persist_directory: Path,
    collection_name: str,
) -> dict[str, Any]:
    chunks_by_category = Counter(str(chunk.metadata.get("category_dir", "unknown")) for chunk in chunks)
    chunks_by_extension = Counter(str(chunk.metadata.get("file_ext", "unknown")) for chunk in chunks)
    file_counts_by_category = Counter(str(row["category_dir"]) for row in file_rows)
    file_counts_by_extension = Counter(str(row["file_ext"]) for row in file_rows)
    csv_summary_files = sum(row["loader_type"] == "csv_summary" for row in file_rows)
    pdf_files = sum(row["file_ext"] == ".pdf" for row in file_rows)
    garbled_pdf_files = sum(row.get("pdf_quality") == "garbled" for row in file_rows)
    train_episode_files = sum("train_env0_episodes" in str(row["source"]).lower() for row in file_rows)
    selected_file_sources = [row["source"] for row in selected_rows]
    output_summary = resolve_project_path(config["outputs"]["ingest_summary_json"])
    embedding_model = str(config.get("embedding", {}).get("model", ""))
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "used_environment": "Lenginzed_RAG",
        "config_path": str(config_path),
        "selected_files": len(selected_rows),
        "selected_file_sources": selected_file_sources,
        "loaded_files": len(file_rows),
        "failed_files": 0,
        "total_documents": sum(int(row["document_count"]) for row in file_rows),
        "total_chunks": len(chunks),
        "indexed_chunks": len(chunks),
        "collection_count": stored_count,
        "chunks_by_category": dict(chunks_by_category),
        "chunks_by_extension": dict(chunks_by_extension),
        "files_by_category": dict(file_counts_by_category),
        "files_by_extension": dict(file_counts_by_extension),
        "csv_summary_files": csv_summary_files,
        "pdf_files": pdf_files,
        "garbled_pdf_files": garbled_pdf_files,
        "train_env0_episodes_files": train_episode_files,
        "persist_directory": str(persist_directory),
        "collection_name": collection_name,
        "embedding_provider": str(config.get("embedding", {}).get("provider", "ollama")),
        "embedding_model": embedding_model,
        "embedding_run": True,
        "chroma_written": True,
        "llm_called": False,
        "external_source_modified": False,
        "raw_imported_modified": False,
        "v2b1_collection_overwritten": False,
        "ingest_passed": all(
            [
                len(selected_rows) == int(config.get("selection", {}).get("target_file_count", 80)),
                len(file_rows) == len(selected_rows),
                stored_count == len(chunks),
                len(chunks) > 0,
                pdf_files == 0,
                garbled_pdf_files == 0,
                train_episode_files == 0,
                persist_directory != V2B1_PERSIST_DIR,
                collection_name != V2B1_COLLECTION,
            ]
        ),
        "output_log": str(output_summary),
    }


def write_outputs(summary: dict[str, Any], file_rows: list[dict[str, Any]], config: dict[str, Any]) -> None:
    summary_path = resolve_project_path(config["outputs"]["ingest_summary_json"])
    files_path = resolve_project_path(config["outputs"]["ingest_files_csv"])
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    files_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with files_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=INGEST_FILE_FIELDS)
        writer.writeheader()
        for row in file_rows:
            writer.writerow(row)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML config: {path}")
    return data


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV file: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def resolve_project_path(value: str | Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def print_summary(summary: dict[str, Any]) -> None:
    print("V2.5b expanded collection ingest completed.")
    print(json.dumps({
        "selected_files": summary["selected_files"],
        "loaded_files": summary["loaded_files"],
        "failed_files": summary["failed_files"],
        "total_documents": summary["total_documents"],
        "total_chunks": summary["total_chunks"],
        "collection_count": summary["collection_count"],
        "chunks_by_category": summary["chunks_by_category"],
        "chunks_by_extension": summary["chunks_by_extension"],
        "csv_summary_files": summary["csv_summary_files"],
        "pdf_files": summary["pdf_files"],
        "garbled_pdf_files": summary["garbled_pdf_files"],
        "train_env0_episodes_files": summary["train_env0_episodes_files"],
        "persist_directory": summary["persist_directory"],
        "collection_name": summary["collection_name"],
        "embedding_model": summary["embedding_model"],
        "ingest_passed": summary["ingest_passed"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.exit(main())
