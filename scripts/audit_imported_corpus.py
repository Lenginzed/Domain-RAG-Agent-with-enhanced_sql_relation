from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chunkers.text_chunker import chunk_documents, load_chunk_config
from src.loaders.base import SUPPORTED_EXTENSIONS, load_file
from src.metadata.schema import CORE_METADATA_FIELDS
from src.metadata.validators import normalize_metadata

IMPORT_DIR = PROJECT_ROOT / "data" / "raw_imported"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "imported_audit"

DOCUMENT_STATS_PATH = OUTPUT_DIR / "document_load_stats.csv"
CHUNK_STATS_PATH = OUTPUT_DIR / "chunk_stats.csv"
METADATA_AUDIT_PATH = OUTPUT_DIR / "metadata_audit.csv"
CHUNK_PREVIEW_PATH = OUTPUT_DIR / "chunk_preview.csv"
LOAD_ERRORS_PATH = OUTPUT_DIR / "load_errors.csv"
SUMMARY_PATH = OUTPUT_DIR / "audit_summary.json"

EMPTY_TEXT_FILE_THRESHOLD = 0
TINY_TEXT_FILE_THRESHOLD = 20
TOO_MANY_CHUNKS_FILE_THRESHOLD = 100
SUSPICIOUS_PDF_TEXT_THRESHOLD = 100
SUSPICIOUS_CSV_CHUNK_THRESHOLD = 100
GARBLED_PDF_UNI_MARKER_THRESHOLD = 20
PREVIEW_PER_CATEGORY = 5
PREVIEW_CHARS = 220

DOCUMENT_STATS_FIELDS = [
    "relative_path",
    "category_dir",
    "extension",
    "file_size_bytes",
    "load_status",
    "document_count",
    "total_text_chars",
    "min_doc_chars",
    "max_doc_chars",
    "error_message",
]

CHUNK_STATS_FIELDS = [
    "relative_path",
    "category_dir",
    "extension",
    "document_count",
    "chunk_count",
    "min_chunk_chars",
    "max_chunk_chars",
    "mean_chunk_chars",
    "empty_chunk_count",
    "oversized_chunk_count",
]

METADATA_AUDIT_FIELDS = [
    "relative_path",
    "chunk_id",
    "doc_type",
    "domain",
    "title",
    "section",
    "source",
    "language",
    "is_citable",
    "missing_required_fields",
    "warning_count",
]

CHUNK_PREVIEW_FIELDS = [
    "relative_path",
    "chunk_id",
    "category_dir",
    "extension",
    "chunk_chars",
    "preview_text",
]

LOAD_ERROR_FIELDS = [
    "relative_path",
    "category_dir",
    "extension",
    "file_size_bytes",
    "issue_type",
    "message",
]


@dataclass
class FileAudit:
    path: Path
    relative_path: str
    category_dir: str
    extension: str
    file_size_bytes: int
    load_status: str
    documents: list[Any]
    chunks: list[Any]
    error_message: str = ""


def main() -> int:
    try:
        summary = audit_imported_corpus()
    except Exception as exc:  # noqa: BLE001 - audit should surface exact failures.
        print("Imported corpus audit failed.")
        print(f"reason: {exc}")
        return 1

    print("Imported corpus audit completed.")
    print(f"environment: {summary['used_environment']}")
    print(f"import_dir: {summary['import_dir']}")
    print(f"total_files: {summary['total_files']}")
    print(f"supported_files: {summary['supported_files']}")
    print(f"loaded_files: {summary['loaded_files']}")
    print(f"failed_files: {summary['failed_files']}")
    print(f"total_documents: {summary['total_documents']}")
    print(f"total_chunks: {summary['total_chunks']}")
    print(f"empty_text_files: {summary['empty_text_files']}")
    print(f"suspicious_pdf_files: {summary['suspicious_pdf_files']}")
    print(f"suspicious_csv_files: {summary['suspicious_csv_files']}")
    print(f"empty_chunks: {summary['empty_chunks']}")
    print(f"oversized_chunks: {summary['oversized_chunks']}")
    print(f"metadata_missing_required_count: {summary['metadata_missing_required_count']}")
    print(f"audit_passed: {summary['audit_passed']}")
    print(f"recommend_v2b1_small_ingest: {summary['recommend_v2b1_small_ingest']}")
    print(f"summary_path: {summary['summary_path']}")
    return 0 if summary["audit_completed"] else 2


def audit_imported_corpus() -> dict[str, Any]:
    if not IMPORT_DIR.exists():
        raise FileNotFoundError(f"Missing imported corpus directory: {IMPORT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    chunk_config = load_chunk_config()
    oversized_chunk_threshold = int(chunk_config["chunk_size"] * 1.5)

    all_files = sorted(path for path in IMPORT_DIR.rglob("*") if path.is_file())
    supported_files = [path for path in all_files if path.suffix.lower() in SUPPORTED_EXTENSIONS]

    document_rows: list[dict[str, Any]] = []
    chunk_rows: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    preview_rows: list[dict[str, Any]] = []
    load_error_rows: list[dict[str, Any]] = []
    audits: list[FileAudit] = []
    preview_counts: Counter[str] = Counter()

    for path in supported_files:
        audit = audit_one_file(path, oversized_chunk_threshold)
        audits.append(audit)

        document_row = build_document_stats_row(audit)
        document_rows.append(document_row)
        chunk_row = build_chunk_stats_row(audit, oversized_chunk_threshold)
        chunk_rows.append(chunk_row)

        collect_file_issues(
            audit=audit,
            document_row=document_row,
            chunk_row=chunk_row,
            load_error_rows=load_error_rows,
        )

        for chunk in audit.chunks:
            metadata = normalize_metadata(chunk.metadata)
            missing_fields = missing_required_fields(metadata)
            warning_count = len(missing_fields)
            if metadata.get("parse_warning"):
                warning_count += 1
            metadata_rows.append(
                {
                    "relative_path": audit.relative_path,
                    "chunk_id": metadata.get("chunk_id", ""),
                    "doc_type": metadata.get("doc_type", ""),
                    "domain": metadata.get("domain", ""),
                    "title": metadata.get("title", ""),
                    "section": metadata.get("section", ""),
                    "source": metadata.get("source", ""),
                    "language": metadata.get("language", ""),
                    "is_citable": metadata.get("is_citable", ""),
                    "missing_required_fields": "|".join(missing_fields),
                    "warning_count": warning_count,
                }
            )

            if preview_counts[audit.category_dir] < PREVIEW_PER_CATEGORY:
                preview_rows.append(
                    {
                        "relative_path": audit.relative_path,
                        "chunk_id": metadata.get("chunk_id", ""),
                        "category_dir": audit.category_dir,
                        "extension": audit.extension,
                        "chunk_chars": len(chunk.page_content or ""),
                        "preview_text": preview_text(chunk.page_content),
                    }
                )
                preview_counts[audit.category_dir] += 1

    write_csv(DOCUMENT_STATS_PATH, document_rows, DOCUMENT_STATS_FIELDS)
    write_csv(CHUNK_STATS_PATH, chunk_rows, CHUNK_STATS_FIELDS)
    write_csv(METADATA_AUDIT_PATH, metadata_rows, METADATA_AUDIT_FIELDS)
    write_csv(CHUNK_PREVIEW_PATH, preview_rows, CHUNK_PREVIEW_FIELDS)
    write_csv(LOAD_ERRORS_PATH, load_error_rows, LOAD_ERROR_FIELDS)

    summary = build_summary(
        all_files=all_files,
        supported_files=supported_files,
        audits=audits,
        document_rows=document_rows,
        chunk_rows=chunk_rows,
        metadata_rows=metadata_rows,
        load_error_rows=load_error_rows,
        oversized_chunk_threshold=oversized_chunk_threshold,
        chunk_config=chunk_config,
    )
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def audit_one_file(path: Path, oversized_chunk_threshold: int) -> FileAudit:
    relative_path = relative_to_project(path)
    category_dir = category_for_imported_file(path)
    extension = path.suffix.lower()
    file_size_bytes = path.stat().st_size
    documents: list[Any] = []
    chunks: list[Any] = []
    error_message = ""
    load_status = "loaded"

    try:
        documents = load_file(path)
        for document in documents:
            document.metadata = normalize_metadata(document.metadata)
        if documents:
            chunks = chunk_documents(documents)
        else:
            load_status = "failed"
            error_message = "loader_returned_no_documents"
    except Exception as exc:  # noqa: BLE001 - one file must not stop the audit.
        load_status = "failed"
        error_message = str(exc)

    total_text_chars = sum(len(document.page_content or "") for document in documents)
    if load_status == "loaded" and total_text_chars == EMPTY_TEXT_FILE_THRESHOLD:
        load_status = "loaded_empty_text"
        error_message = "loaded_documents_have_zero_text"
    elif load_status == "loaded" and total_text_chars < TINY_TEXT_FILE_THRESHOLD:
        load_status = "loaded_tiny_text"

    # Keep the threshold argument visible in this function signature for audit readability.
    _ = oversized_chunk_threshold
    return FileAudit(
        path=path,
        relative_path=relative_path,
        category_dir=category_dir,
        extension=extension,
        file_size_bytes=file_size_bytes,
        load_status=load_status,
        documents=documents,
        chunks=chunks,
        error_message=error_message,
    )


def build_document_stats_row(audit: FileAudit) -> dict[str, Any]:
    doc_lengths = [len(document.page_content or "") for document in audit.documents]
    return {
        "relative_path": audit.relative_path,
        "category_dir": audit.category_dir,
        "extension": audit.extension,
        "file_size_bytes": audit.file_size_bytes,
        "load_status": audit.load_status,
        "document_count": len(audit.documents),
        "total_text_chars": sum(doc_lengths),
        "min_doc_chars": min(doc_lengths) if doc_lengths else 0,
        "max_doc_chars": max(doc_lengths) if doc_lengths else 0,
        "error_message": audit.error_message,
    }


def build_chunk_stats_row(audit: FileAudit, oversized_chunk_threshold: int) -> dict[str, Any]:
    chunk_lengths = [len(chunk.page_content or "") for chunk in audit.chunks]
    return {
        "relative_path": audit.relative_path,
        "category_dir": audit.category_dir,
        "extension": audit.extension,
        "document_count": len(audit.documents),
        "chunk_count": len(audit.chunks),
        "min_chunk_chars": min(chunk_lengths) if chunk_lengths else 0,
        "max_chunk_chars": max(chunk_lengths) if chunk_lengths else 0,
        "mean_chunk_chars": round(mean(chunk_lengths), 2) if chunk_lengths else 0,
        "empty_chunk_count": sum(length == 0 for length in chunk_lengths),
        "oversized_chunk_count": sum(length > oversized_chunk_threshold for length in chunk_lengths),
    }


def collect_file_issues(
    *,
    audit: FileAudit,
    document_row: dict[str, Any],
    chunk_row: dict[str, Any],
    load_error_rows: list[dict[str, Any]],
) -> None:
    if audit.load_status == "failed":
        add_issue(load_error_rows, audit, "load_failed", audit.error_message or "load failed")
        return

    total_text_chars = int(document_row["total_text_chars"])
    chunk_count = int(chunk_row["chunk_count"])
    empty_chunk_count = int(chunk_row["empty_chunk_count"])
    oversized_chunk_count = int(chunk_row["oversized_chunk_count"])

    if total_text_chars == EMPTY_TEXT_FILE_THRESHOLD:
        add_issue(load_error_rows, audit, "empty_text_file", "loaded text length is 0")
    elif total_text_chars < TINY_TEXT_FILE_THRESHOLD:
        add_issue(
            load_error_rows,
            audit,
            "tiny_text_file",
            f"loaded text length {total_text_chars} is below {TINY_TEXT_FILE_THRESHOLD}",
        )
    if audit.extension == ".pdf" and total_text_chars < SUSPICIOUS_PDF_TEXT_THRESHOLD:
        add_issue(
            load_error_rows,
            audit,
            "suspicious_pdf",
            f"PDF extracted text length {total_text_chars} is below {SUSPICIOUS_PDF_TEXT_THRESHOLD}",
        )
    if audit.extension == ".pdf" and pdf_uni_marker_count(audit.documents) >= GARBLED_PDF_UNI_MARKER_THRESHOLD:
        add_issue(
            load_error_rows,
            audit,
            "suspicious_pdf_garbled_text",
            f"PDF text contains at least {GARBLED_PDF_UNI_MARKER_THRESHOLD} /uni markers",
        )
    if audit.extension == ".csv" and chunk_count > SUSPICIOUS_CSV_CHUNK_THRESHOLD:
        add_issue(
            load_error_rows,
            audit,
            "suspicious_csv",
            f"CSV produced {chunk_count} chunks, above {SUSPICIOUS_CSV_CHUNK_THRESHOLD}",
        )
    if chunk_count > TOO_MANY_CHUNKS_FILE_THRESHOLD:
        add_issue(
            load_error_rows,
            audit,
            "too_many_chunks_file",
            f"file produced {chunk_count} chunks, above {TOO_MANY_CHUNKS_FILE_THRESHOLD}",
        )
    if empty_chunk_count:
        add_issue(load_error_rows, audit, "empty_chunks", f"{empty_chunk_count} empty chunks")
    if oversized_chunk_count:
        add_issue(load_error_rows, audit, "oversized_chunks", f"{oversized_chunk_count} oversized chunks")


def add_issue(rows: list[dict[str, Any]], audit: FileAudit, issue_type: str, message: str) -> None:
    rows.append(
        {
            "relative_path": audit.relative_path,
            "category_dir": audit.category_dir,
            "extension": audit.extension,
            "file_size_bytes": audit.file_size_bytes,
            "issue_type": issue_type,
            "message": message,
        }
    )


def build_summary(
    *,
    all_files: list[Path],
    supported_files: list[Path],
    audits: list[FileAudit],
    document_rows: list[dict[str, Any]],
    chunk_rows: list[dict[str, Any]],
    metadata_rows: list[dict[str, Any]],
    load_error_rows: list[dict[str, Any]],
    oversized_chunk_threshold: int,
    chunk_config: dict[str, int],
) -> dict[str, Any]:
    total_documents = sum(int(row["document_count"]) for row in document_rows)
    total_chunks = sum(int(row["chunk_count"]) for row in chunk_rows)
    failed_files = sum(row["load_status"] == "failed" for row in document_rows)
    loaded_files = sum(row["load_status"] != "failed" for row in document_rows)
    metadata_missing_required_count = sum(bool(row["missing_required_fields"]) for row in metadata_rows)
    empty_text_files = sum(int(row["total_text_chars"]) == 0 for row in document_rows)
    tiny_text_files = sum(0 < int(row["total_text_chars"]) < TINY_TEXT_FILE_THRESHOLD for row in document_rows)
    suspicious_pdf_files = sum(row["issue_type"] == "suspicious_pdf" for row in load_error_rows)
    suspicious_pdf_garbled_files = sum(
        row["issue_type"] == "suspicious_pdf_garbled_text" for row in load_error_rows
    )
    suspicious_csv_files = sum(row["issue_type"] == "suspicious_csv" for row in load_error_rows)
    too_many_chunks_files = sum(row["issue_type"] == "too_many_chunks_file" for row in load_error_rows)
    empty_chunks = sum(int(row["empty_chunk_count"]) for row in chunk_rows)
    oversized_chunks = sum(int(row["oversized_chunk_count"]) for row in chunk_rows)
    source_raw_imported_count = sum(
        str(row["source"]).replace("\\", "/").startswith("data/raw_imported/") for row in metadata_rows
    )
    original_source_metadata_count = sum("original_source" in chunk.metadata for audit in audits for chunk in audit.chunks)

    chunks_by_category = Counter()
    chunks_by_extension = Counter()
    for row in chunk_rows:
        chunks_by_category[str(row["category_dir"])] += int(row["chunk_count"])
        chunks_by_extension[str(row["extension"]) or "(none)"] += int(row["chunk_count"])

    load_error_summary = Counter(str(row["issue_type"]) for row in load_error_rows)
    audit_passed = all(
        [
            len(all_files) > 0,
            len(all_files) == len(supported_files),
            failed_files == 0,
            empty_text_files == 0,
            empty_chunks == 0,
            oversized_chunks == 0,
            metadata_missing_required_count == 0,
            source_raw_imported_count == len(metadata_rows),
        ]
    )
    recommend_v2b1_small_ingest = all(
        [
            failed_files == 0,
            empty_text_files == 0,
            metadata_missing_required_count == 0,
            total_chunks > 0,
        ]
    )

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "used_environment": "Lenginzed_RAG",
        "import_dir": str(IMPORT_DIR),
        "output_dir": str(OUTPUT_DIR),
        "audit_completed": True,
        "external_source_modified": False,
        "raw_imported_modified": False,
        "embedding_run": False,
        "chroma_written": False,
        "thresholds": {
            "empty_text_file_chars": EMPTY_TEXT_FILE_THRESHOLD,
            "tiny_text_file_chars": TINY_TEXT_FILE_THRESHOLD,
            "oversized_chunk_chars": oversized_chunk_threshold,
            "too_many_chunks_file": TOO_MANY_CHUNKS_FILE_THRESHOLD,
            "suspicious_pdf_text_chars": SUSPICIOUS_PDF_TEXT_THRESHOLD,
            "garbled_pdf_uni_marker_count": GARBLED_PDF_UNI_MARKER_THRESHOLD,
            "suspicious_csv_chunks": SUSPICIOUS_CSV_CHUNK_THRESHOLD,
            "chunk_size": chunk_config["chunk_size"],
            "chunk_overlap": chunk_config["chunk_overlap"],
        },
        "total_files": len(all_files),
        "supported_files": len(supported_files),
        "unsupported_files": len(all_files) - len(supported_files),
        "loaded_files": loaded_files,
        "failed_files": failed_files,
        "total_documents": total_documents,
        "total_chunks": total_chunks,
        "chunks_by_category": dict(chunks_by_category),
        "chunks_by_extension": dict(chunks_by_extension),
        "empty_text_files": empty_text_files,
        "tiny_text_files": tiny_text_files,
        "empty_chunks": empty_chunks,
        "oversized_chunks": oversized_chunks,
        "suspicious_pdf_files": suspicious_pdf_files,
        "suspicious_pdf_garbled_files": suspicious_pdf_garbled_files,
        "suspicious_csv_files": suspicious_csv_files,
        "too_many_chunks_files": too_many_chunks_files,
        "metadata_missing_required_count": metadata_missing_required_count,
        "metadata_rows_count": len(metadata_rows),
        "source_points_to_raw_imported_count": source_raw_imported_count,
        "original_source_metadata_count": original_source_metadata_count,
        "original_source_metadata_missing_count": max(0, len(metadata_rows) - original_source_metadata_count),
        "load_error_issue_counts": dict(load_error_summary),
        "top_20_files_by_chunk_count": top_files(chunk_rows, key="chunk_count"),
        "top_20_files_by_text_chars": top_files(document_rows, key="total_text_chars"),
        "audit_passed": audit_passed,
        "recommend_v2b1_small_ingest": recommend_v2b1_small_ingest,
        "document_load_stats_path": str(DOCUMENT_STATS_PATH),
        "chunk_stats_path": str(CHUNK_STATS_PATH),
        "metadata_audit_path": str(METADATA_AUDIT_PATH),
        "chunk_preview_path": str(CHUNK_PREVIEW_PATH),
        "load_errors_path": str(LOAD_ERRORS_PATH),
        "summary_path": str(SUMMARY_PATH),
    }


def top_files(rows: list[dict[str, Any]], *, key: str) -> list[dict[str, Any]]:
    sorted_rows = sorted(rows, key=lambda row: int(float(row.get(key, 0) or 0)), reverse=True)
    return [
        {
            "relative_path": row["relative_path"],
            "category_dir": row["category_dir"],
            "extension": row["extension"],
            key: int(float(row.get(key, 0) or 0)),
        }
        for row in sorted_rows[:20]
    ]


def missing_required_fields(metadata: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    for field in CORE_METADATA_FIELDS:
        value = metadata.get(field)
        if value is None or value == "":
            missing.append(field)
    return missing


def pdf_uni_marker_count(documents: list[Any]) -> int:
    text = "\n".join(str(document.page_content or "") for document in documents)
    return text.count("/uni")


def relative_to_project(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def category_for_imported_file(path: Path) -> str:
    try:
        relative = path.resolve().relative_to(IMPORT_DIR)
    except ValueError:
        return "outside_import_dir"
    return relative.parts[0] if relative.parts else "root"


def preview_text(text: str) -> str:
    collapsed = " ".join((text or "").split())
    return collapsed[:PREVIEW_CHARS]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    sys.exit(main())
