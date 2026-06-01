"""Centralized dataset loading, validation, and reporting for Phase 1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd

from .logger import get_logger

LOGGER = get_logger("data")

RAW_DATA_DIR: Final[Path] = Path("data/raw")
TRAIN_FILENAME: Final[str] = "train.csv"
TEST_FILENAME: Final[str] = "test.csv"
TIMESTAMP_COLUMN: Final[str] = "timestamp"
PARSED_TIMESTAMP_COLUMN: Final[str] = "timestamp_parsed"
KEY_COLUMNS: Final[tuple[str, ...]] = ("geohash", "day", TIMESTAMP_COLUMN)
SORT_COLUMNS: Final[tuple[str, ...]] = ("geohash", PARSED_TIMESTAMP_COLUMN)
SUSPICIOUS_DUPLICATE_RATIO: Final[float] = 0.005
SUSPICIOUS_DUPLICATE_ROWS: Final[int] = 25
EXPECTED_TRAIN_COLUMNS: Final[tuple[str, ...]] = (
    "Index",
    "geohash",
    "day",
    TIMESTAMP_COLUMN,
    "demand",
    "RoadType",
    "NumberofLanes",
    "LargeVehicles",
    "Landmarks",
    "Temperature",
    "Weather",
)
EXPECTED_TEST_COLUMNS: Final[tuple[str, ...]] = (
    "Index",
    "geohash",
    "day",
    TIMESTAMP_COLUMN,
    "RoadType",
    "NumberofLanes",
    "LargeVehicles",
    "Landmarks",
    "Temperature",
    "Weather",
)


class DataValidationError(ValueError):
    """Raised when the raw dataset fails a hard validation rule."""


@dataclass(frozen=True, slots=True)
class DatasetAudit:
    """Validated dataset metadata used for reports and debugging."""

    name: str
    path: Path
    row_count: int
    column_count: int
    columns: tuple[str, ...]
    missing_columns: tuple[str, ...]
    extra_columns: tuple[str, ...]
    exact_duplicate_rows: int
    key_duplicate_rows: int
    duplicate_ratio: float
    null_counts: dict[str, int]
    timestamp_min: str | None
    timestamp_max: str | None
    geohash_count: int
    day_count: int
    target_null_count: int | None


@dataclass(frozen=True, slots=True)
class LoadedData:
    """Container for the validated train and test datasets."""

    train: pd.DataFrame
    test: pd.DataFrame
    train_audit: DatasetAudit
    test_audit: DatasetAudit


def load_data(data_dir: Path | str = RAW_DATA_DIR) -> LoadedData:
    """Load, validate, and sort the competition train/test datasets."""

    raw_data_dir = Path(data_dir).expanduser().resolve()
    train = _load_single_dataset(
        raw_data_dir / TRAIN_FILENAME,
        expected_columns=EXPECTED_TRAIN_COLUMNS,
        dataset_name="train",
        target_column="demand",
    )
    test = _load_single_dataset(
        raw_data_dir / TEST_FILENAME,
        expected_columns=EXPECTED_TEST_COLUMNS,
        dataset_name="test",
        target_column=None,
    )
    return LoadedData(
        train=train[0],
        test=test[0],
        train_audit=train[1],
        test_audit=test[1],
    )


def write_dataset_reports(bundle: LoadedData, reports_dir: Path | str) -> dict[str, Path]:
    """Write the Phase 1 markdown reports to disk."""

    output_dir = Path(reports_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    report_paths = {
        "schema": output_dir / "schema_report.md",
        "nulls": output_dir / "null_report.md",
        "summary": output_dir / "dataset_summary.md",
    }

    report_paths["schema"].write_text(_render_schema_report(bundle), encoding="utf-8")
    report_paths["nulls"].write_text(_render_null_report(bundle), encoding="utf-8")
    report_paths["summary"].write_text(_render_summary_report(bundle), encoding="utf-8")

    LOGGER.info("Wrote dataset reports to %s.", output_dir)
    return report_paths


def format_dataset_summary(bundle: LoadedData) -> str:
    """Return a compact human-readable summary for stdout or logs."""

    train = bundle.train_audit
    test = bundle.test_audit
    return (
        f"Train rows: {train.row_count:,} | Test rows: {test.row_count:,} | "
        f"Train geohashes: {train.geohash_count:,} | Test geohashes: {test.geohash_count:,} | "
        f"Train duplicates: {train.key_duplicate_rows:,} | Test duplicates: {test.key_duplicate_rows:,} | "
        f"Train null cells: {sum(train.null_counts.values()):,} | Test null cells: {sum(test.null_counts.values()):,}"
    )


def _load_single_dataset(
    path: Path,
    *,
    expected_columns: tuple[str, ...],
    dataset_name: str,
    target_column: str | None,
) -> tuple[pd.DataFrame, DatasetAudit]:
    if not path.exists():
        raise DataValidationError(f"{dataset_name.title()} dataset not found: {path}")

    frame = pd.read_csv(path)
    _validate_schema(frame, expected_columns=expected_columns, dataset_name=dataset_name, path=path)

    parsed_timestamp = _parse_timestamp_column(frame[TIMESTAMP_COLUMN], dataset_name=dataset_name, path=path)
    working = frame.copy()
    working[PARSED_TIMESTAMP_COLUMN] = parsed_timestamp
    working = working.sort_values(list(SORT_COLUMNS), kind="mergesort").reset_index(drop=True)

    _validate_timestamp_ordering(working, dataset_name=dataset_name, path=path)

    exact_duplicate_rows = int(working.duplicated().sum())
    key_duplicate_rows = int(working.duplicated(subset=list(KEY_COLUMNS)).sum())
    duplicate_ratio = key_duplicate_rows / len(working) if len(working) else 0.0
    _validate_duplicates(
        dataset_name=dataset_name,
        path=path,
        duplicate_rows=key_duplicate_rows,
        duplicate_ratio=duplicate_ratio,
    )

    audit = DatasetAudit(
        name=dataset_name,
        path=path,
        row_count=int(len(working)),
        column_count=int(len(frame.columns)),
        columns=tuple(frame.columns),
        missing_columns=tuple(column for column in expected_columns if column not in frame.columns),
        extra_columns=tuple(column for column in frame.columns if column not in expected_columns),
        exact_duplicate_rows=exact_duplicate_rows,
        key_duplicate_rows=key_duplicate_rows,
        duplicate_ratio=duplicate_ratio,
        null_counts={column: int(frame[column].isna().sum()) for column in frame.columns},
        timestamp_min=_format_timestamp(working[PARSED_TIMESTAMP_COLUMN].min()),
        timestamp_max=_format_timestamp(working[PARSED_TIMESTAMP_COLUMN].max()),
        geohash_count=int(working["geohash"].nunique(dropna=True)),
        day_count=int(working["day"].nunique(dropna=True)),
        target_null_count=int(working[target_column].isna().sum()) if target_column else None,
    )
    return working, audit


def _validate_schema(
    frame: pd.DataFrame,
    *,
    expected_columns: tuple[str, ...],
    dataset_name: str,
    path: Path,
) -> None:
    missing_columns = [column for column in expected_columns if column not in frame.columns]
    if missing_columns:
        raise DataValidationError(
            f"{dataset_name.title()} schema missing required columns in {path}: {missing_columns}."
        )


def _parse_timestamp_column(series: pd.Series, *, dataset_name: str, path: Path) -> pd.Series:
    values = series.astype("string").str.strip()
    extracted = values.str.extract(r"^(?P<hour>\d{1,2}):(?P<minute>\d{1,2})$")
    invalid_rows = extracted.isna().any(axis=1)
    if invalid_rows.any():
        samples = values[invalid_rows].head(5).tolist()
        raise DataValidationError(
            f"{dataset_name.title()} timestamp parsing failed in {path}. Sample invalid values: {samples}."
        )

    hours = extracted["hour"].astype(int)
    minutes = extracted["minute"].astype(int)
    invalid_range = (hours > 23) | (minutes > 59)
    if invalid_range.any():
        samples = values[invalid_range].head(5).tolist()
        raise DataValidationError(
            f"{dataset_name.title()} timestamp values out of range in {path}. Sample invalid values: {samples}."
        )

    normalized = hours.astype(str).str.zfill(2) + ":" + minutes.astype(str).str.zfill(2)
    parsed = pd.to_datetime(normalized, format="%H:%M", errors="raise")
    if parsed.isna().any():
        raise DataValidationError(f"{dataset_name.title()} timestamp parsing produced missing values in {path}.")
    return parsed


def _validate_timestamp_ordering(frame: pd.DataFrame, *, dataset_name: str, path: Path) -> None:
    grouped = frame.groupby(["geohash", "day"], sort=False)[PARSED_TIMESTAMP_COLUMN]
    invalid_groups = [
        str(index)
        for index, values in grouped
        if not values.is_monotonic_increasing
    ]
    if invalid_groups:
        raise DataValidationError(
            f"{dataset_name.title()} timestamp ordering is invalid in {path}. Problem groups: {invalid_groups[:5]}."
        )


def _validate_duplicates(
    *,
    dataset_name: str,
    path: Path,
    duplicate_rows: int,
    duplicate_ratio: float,
) -> None:
    if duplicate_rows == 0:
        return
    if duplicate_rows >= SUSPICIOUS_DUPLICATE_ROWS and duplicate_ratio >= SUSPICIOUS_DUPLICATE_RATIO:
        raise DataValidationError(
            f"{dataset_name.title()} duplicate count is suspiciously high in {path}: "
            f"{duplicate_rows} key duplicates ({duplicate_ratio:.2%})."
        )


def _format_timestamp(value: pd.Timestamp | pd.NaT) -> str | None:
    if pd.isna(value):
        return None
    return value.strftime("%Y-%m-%d %H:%M:%S")


def _render_schema_report(bundle: LoadedData) -> str:
    lines = ["# Schema Report", ""]
    for audit, expected_columns in (
        (bundle.train_audit, EXPECTED_TRAIN_COLUMNS),
        (bundle.test_audit, EXPECTED_TEST_COLUMNS),
    ):
        lines.extend(
            [
                f"## {audit.name.title()}",
                "",
                "- Status: PASS",
                f"- Rows: {audit.row_count:,}",
                f"- Columns: {audit.column_count:,}",
                f"- Missing columns: {_format_list(audit.missing_columns)}",
                f"- Extra columns: {_format_list(audit.extra_columns)}",
                f"- Expected schema: {_format_list(expected_columns)}",
                f"- Timestamp helper column: {PARSED_TIMESTAMP_COLUMN}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _render_null_report(bundle: LoadedData) -> str:
    lines = ["# Null Report", ""]
    for audit in (bundle.train_audit, bundle.test_audit):
        lines.extend([f"## {audit.name.title()}", "", "| Column | Nulls | Null % |", "| --- | ---: | ---: |"])
        for column, null_count in audit.null_counts.items():
            null_ratio = (null_count / audit.row_count) if audit.row_count else 0.0
            lines.append(f"| {column} | {null_count:,} | {null_ratio:.2%} |")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _render_summary_report(bundle: LoadedData) -> str:
    train = bundle.train_audit
    test = bundle.test_audit
    lines = ["# Dataset Summary", ""]
    lines.extend(
        [
            "| Metric | Train | Test |",
            "| --- | ---: | ---: |",
            f"| Rows | {train.row_count:,} | {test.row_count:,} |",
            f"| Columns | {train.column_count:,} | {test.column_count:,} |",
            f"| Geohashes | {train.geohash_count:,} | {test.geohash_count:,} |",
            f"| Days | {train.day_count:,} | {test.day_count:,} |",
            f"| Exact duplicates | {train.exact_duplicate_rows:,} | {test.exact_duplicate_rows:,} |",
            f"| Key duplicates | {train.key_duplicate_rows:,} | {test.key_duplicate_rows:,} |",
            f"| Null cells | {sum(train.null_counts.values()):,} | {sum(test.null_counts.values()):,} |",
            f"| Timestamp min | {_safe_text(train.timestamp_min)} | {_safe_text(test.timestamp_min)} |",
            f"| Timestamp max | {_safe_text(train.timestamp_max)} | {_safe_text(test.timestamp_max)} |",
        ]
    )
    if train.target_null_count is not None:
        lines.extend([
            "",
            f"- Train target nulls: {train.target_null_count:,}",
            "- Sort order: geohash -> timestamp",
        ])
    return "\n".join(lines).rstrip() + "\n"


def _format_list(values: tuple[str, ...]) -> str:
    if not values:
        return "None"
    return ", ".join(values)


def _safe_text(value: str | None) -> str:
    return value if value is not None else "N/A"