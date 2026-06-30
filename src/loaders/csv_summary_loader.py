from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_core.documents import Document

from src.loaders.utils import base_metadata, detect_language, relative_source


logger = logging.getLogger(__name__)

DEFAULT_CSV_CONFIG: dict[str, Any] = {
    "max_preview_rows": 5,
    "max_profile_numeric_columns": 30,
    "max_profile_categorical_columns": 30,
    "max_unique_values_preview": 20,
    "large_csv_row_threshold": 1000,
    "large_csv_size_mb_threshold": 5,
    "text_length_limit": 12000,
    "include_columns": True,
    "include_dtypes": True,
    "include_numeric_stats": True,
    "include_categorical_preview": True,
    "include_missing_values": True,
}

CSV_ENCODINGS = ["utf-8-sig", "utf-8", "gbk", "gb18030"]


def load_csv_summary_file(path: Path, config: dict[str, Any] | None = None) -> list[Document]:
    """Load a CSV as a compact profile document instead of one document per row."""

    merged_config = {**DEFAULT_CSV_CONFIG, **(config or {})}
    dataframe, encoding = read_csv_with_fallback(path)
    file_size_bytes = path.stat().st_size
    row_count = int(len(dataframe))
    column_count = int(len(dataframe.columns))
    large_csv = is_large_csv(path, row_count, merged_config)
    text, truncated = build_csv_summary_text(
        path=path,
        dataframe=dataframe,
        encoding=encoding,
        file_size_bytes=file_size_bytes,
        large_csv=large_csv,
        config=merged_config,
    )
    category_dir = category_for_imported_file(path)
    metadata = base_metadata(
        path,
        doc_type="experiment" if category_dir == "experiments" else "table",
        title=path.stem,
        section="csv_summary",
        language=detect_language(text),
        is_citable=True,
        imported_source=relative_source(path),
        file_ext=".csv",
        category_dir=category_dir,
        loader_type="csv_summary",
        row_count=row_count,
        column_count=column_count,
        large_csv=large_csv,
        csv_encoding=encoding,
        file_size_bytes=file_size_bytes,
        truncated=truncated,
    )
    return [Document(page_content=text, metadata=metadata)]


def read_csv_with_fallback(path: Path) -> tuple[pd.DataFrame, str]:
    errors: list[str] = []
    for encoding in CSV_ENCODINGS:
        try:
            dataframe = pd.read_csv(path, encoding=encoding, low_memory=False)
            return dataframe, encoding
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
        except Exception as exc:  # noqa: BLE001 - keep trying encodings/parsers.
            errors.append(f"{encoding}: {type(exc).__name__}: {exc}")

    # A final Python-engine pass handles some irregular CSV dialects.
    for encoding in CSV_ENCODINGS:
        try:
            dataframe = pd.read_csv(path, encoding=encoding, engine="python", on_bad_lines="skip")
            logger.warning("Read CSV with python engine and skipped bad lines: %s", path)
            return dataframe, f"{encoding}:python_engine_skip_bad_lines"
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{encoding}:python_engine: {type(exc).__name__}: {exc}")

    raise ValueError(f"Unable to read CSV {path}; attempts: {' | '.join(errors)}")


def is_large_csv(path: Path, row_count: int, config: dict[str, Any]) -> bool:
    size_mb = path.stat().st_size / (1024 * 1024)
    return (
        row_count >= int(config.get("large_csv_row_threshold", 1000))
        or size_mb >= float(config.get("large_csv_size_mb_threshold", 5))
    )


def build_csv_summary_text(
    *,
    path: Path,
    dataframe: pd.DataFrame,
    encoding: str,
    file_size_bytes: int,
    large_csv: bool,
    config: dict[str, Any],
) -> tuple[str, bool]:
    row_count = int(len(dataframe))
    column_count = int(len(dataframe.columns))
    lines: list[str] = [
        "CSV summary-aware loader output",
        f"source: {relative_source(path)}",
        f"filename: {path.name}",
        f"encoding: {encoding}",
        f"file_size_bytes: {file_size_bytes}",
        f"row_count: {row_count}",
        f"column_count: {column_count}",
        f"large_csv: {str(large_csv).lower()}",
    ]

    if config.get("include_columns", True):
        lines.extend(["", "Columns:", format_list([str(column) for column in dataframe.columns])])

    if config.get("include_dtypes", True):
        lines.extend(["", "Dtypes:"])
        for column, dtype in dataframe.dtypes.items():
            lines.append(f"- {column}: {dtype}")

    if config.get("include_missing_values", True):
        lines.extend(["", "Missing values:"])
        missing = dataframe.isna().sum()
        for column, value in missing.items():
            if int(value) > 0:
                lines.append(f"- {column}: {int(value)}")
        if not any(int(value) > 0 for value in missing.values):
            lines.append("- none")

    if config.get("include_numeric_stats", True):
        append_numeric_stats(lines, dataframe, int(config.get("max_profile_numeric_columns", 30)))

    if config.get("include_categorical_preview", True):
        append_categorical_preview(
            lines,
            dataframe,
            max_columns=int(config.get("max_profile_categorical_columns", 30)),
            max_values=int(config.get("max_unique_values_preview", 20)),
        )

    append_preview_rows(lines, dataframe, int(config.get("max_preview_rows", 5)))

    text = "\n".join(lines).strip() + "\n"
    limit = int(config.get("text_length_limit", 12000))
    if len(text) <= limit:
        return text, False
    marker = "\n[TRUNCATED: CSV summary exceeded text_length_limit]\n"
    return text[: max(0, limit - len(marker))].rstrip() + marker, True


def append_numeric_stats(lines: list[str], dataframe: pd.DataFrame, max_columns: int) -> None:
    numeric_columns = [
        column for column in dataframe.columns if pd.api.types.is_numeric_dtype(dataframe[column])
    ][:max_columns]
    lines.extend(["", "Numeric stats:"])
    if not numeric_columns:
        lines.append("- none")
        return
    for column in numeric_columns:
        series = pd.to_numeric(dataframe[column], errors="coerce")
        lines.append(
            "- "
            f"{column}: count={int(series.count())}, "
            f"mean={format_number(series.mean())}, "
            f"min={format_number(series.min())}, "
            f"max={format_number(series.max())}"
        )


def append_categorical_preview(
    lines: list[str],
    dataframe: pd.DataFrame,
    *,
    max_columns: int,
    max_values: int,
) -> None:
    categorical_columns = [
        column
        for column in dataframe.columns
        if not pd.api.types.is_numeric_dtype(dataframe[column])
    ][:max_columns]
    lines.extend(["", "Categorical preview:"])
    if not categorical_columns:
        lines.append("- none")
        return
    for column in categorical_columns:
        series = dataframe[column]
        unique_count = int(series.nunique(dropna=False))
        top_values = series.astype("string").fillna("<NA>").value_counts(dropna=False).head(max_values)
        preview = "; ".join(f"{safe_inline(value)}={int(count)}" for value, count in top_values.items())
        lines.append(f"- {column}: unique_count={unique_count}; top_values={preview}")


def append_preview_rows(lines: list[str], dataframe: pd.DataFrame, max_rows: int) -> None:
    lines.extend(["", f"Preview rows (first {max_rows}):"])
    if dataframe.empty:
        lines.append("- empty dataframe")
        return
    preview = dataframe.head(max_rows).to_csv(index=False).strip()
    lines.append(preview)


def format_list(values: list[str]) -> str:
    if not values:
        return "- none"
    return "\n".join(f"- {value}" for value in values)


def format_number(value: Any) -> str:
    if pd.isna(value):
        return "nan"
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return str(value)


def safe_inline(value: Any, limit: int = 80) -> str:
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def category_for_imported_file(path: Path) -> str:
    parts = path.as_posix().split("/")
    if "raw_imported" in parts:
        index = parts.index("raw_imported")
        if index + 1 < len(parts):
            return parts[index + 1]
    return "unknown"
