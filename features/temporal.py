"""Baseline feature engineering for the traffic demand prediction project."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence

import numpy as np
import pandas as pd

from utils.logger import get_logger

LOGGER = get_logger("features")

TIMESTAMP_COLUMN: Final[str] = "timestamp"
DAY_COLUMN: Final[str] = "day"
SOURCE_LANES_COLUMN: Final[str] = "NumberofLanes"
OUTPUT_LANES_COLUMN: Final[str] = "NumberOfLanes"
OUTPUT_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "hour",
    "minute",
    "day_index",
    "hour_sin",
    "hour_cos",
    "minute_sin",
    "minute_cos",
    "RoadType",
    "Weather",
    "Temperature",
    OUTPUT_LANES_COLUMN,
    "LargeVehicles",
)
TEMPORAL_COLUMNS: Final[tuple[str, ...]] = ("hour", "minute", "day_index")
CYCLE_COLUMNS: Final[tuple[str, ...]] = ("hour_sin", "hour_cos", "minute_sin", "minute_cos")
CATEGORICAL_COLUMNS: Final[tuple[str, ...]] = ("RoadType", "Weather")
NUMERIC_COLUMNS: Final[tuple[str, ...]] = (
    "hour",
    "minute",
    "day_index",
    "hour_sin",
    "hour_cos",
    "minute_sin",
    "minute_cos",
    "Temperature",
    OUTPUT_LANES_COLUMN,
    "LargeVehicles",
)
DEFAULT_ARTIFACT_DIR: Final[Path] = Path("artifacts/features")
DEFAULT_REPORT_PATH: Final[Path] = Path("reports/feature_reports/baseline_features.md")
CYCLICAL_EPSILON: Final[float] = 1e-6
NULL_EXPLOSION_FACTOR: Final[float] = 1.0


class FeatureValidationError(ValueError):
    """Raised when the baseline feature set fails a hard validation rule."""


@dataclass(frozen=True, slots=True)
class FeatureMetadata:
    """Compact metadata for a generated feature table."""

    dataset_name: str
    row_count: int
    feature_count: int
    feature_names: tuple[str, ...]
    dtypes: dict[str, str]
    null_counts: dict[str, int]
    categorical_columns: tuple[str, ...]
    numeric_columns: tuple[str, ...]
    memory_usage_bytes: int


@dataclass(frozen=True, slots=True)
class FeatureBundle:
    """Train/test baseline feature frames with generated metadata."""

    train: pd.DataFrame
    test: pd.DataFrame
    train_metadata: FeatureMetadata
    test_metadata: FeatureMetadata
    report_path: Path
    artifact_dir: Path


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate the safe baseline feature set from the original raw columns."""

    working = df.copy()
    _validate_source_columns(working)

    timestamp_parts = _extract_timestamp_parts(working[TIMESTAMP_COLUMN])
    day_index = _build_day_index(working[DAY_COLUMN])

    features = pd.DataFrame(index=working.index)
    features["hour"] = timestamp_parts["hour"].astype("Int64")
    features["minute"] = timestamp_parts["minute"].astype("Int64")
    features["day_index"] = day_index.astype("Int64")

    hour_angle = 2.0 * np.pi * (features["hour"].astype(float) / 24.0)
    minute_angle = 2.0 * np.pi * (features["minute"].astype(float) / 60.0)
    features["hour_sin"] = np.sin(hour_angle)
    features["hour_cos"] = np.cos(hour_angle)
    features["minute_sin"] = np.sin(minute_angle)
    features["minute_cos"] = np.cos(minute_angle)

    features["RoadType"] = working["RoadType"].astype("string")
    features["Weather"] = working["Weather"].astype("string")
    features["Temperature"] = pd.to_numeric(working["Temperature"], errors="coerce")
    features[OUTPUT_LANES_COLUMN] = pd.to_numeric(working[SOURCE_LANES_COLUMN], errors="coerce")
    features["LargeVehicles"] = _encode_large_vehicles(working["LargeVehicles"])

    features = features.loc[:, OUTPUT_FEATURE_COLUMNS]
    _validate_feature_columns(features)
    return features


def validate_feature_pair(
    train_source: pd.DataFrame,
    train_features: pd.DataFrame,
    test_source: pd.DataFrame,
    test_features: pd.DataFrame,
) -> None:
    """Validate the train/test feature pair for leakage-safe parity."""

    _validate_feature_frame(train_source, train_features, dataset_name="train")
    _validate_feature_frame(test_source, test_features, dataset_name="test")
    if list(train_features.columns) != list(test_features.columns):
        raise FeatureValidationError("Train/test feature mismatch: generated feature columns differ.")
    if train_features.dtypes.astype(str).to_dict() != test_features.dtypes.astype(str).to_dict():
        raise FeatureValidationError("Train/test feature mismatch: generated feature dtypes differ.")


def build_feature_bundle(train_source: pd.DataFrame, test_source: pd.DataFrame) -> FeatureBundle:
    """Generate, validate, and package the baseline feature set."""

    train_features = generate_features(train_source)
    test_features = generate_features(test_source)
    validate_feature_pair(train_source, train_features, test_source, test_features)

    artifact_dir = DEFAULT_ARTIFACT_DIR.expanduser().resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    train_metadata = build_feature_metadata("train", train_source, train_features)
    test_metadata = build_feature_metadata("test", test_source, test_features)
    save_feature_metadata(train_metadata, test_metadata, artifact_dir)

    report_path = DEFAULT_REPORT_PATH.expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_render_feature_report(train_metadata, test_metadata), encoding="utf-8")

    LOGGER.info("Saved baseline feature metadata to %s.", artifact_dir)
    LOGGER.info("Wrote baseline feature report to %s.", report_path)

    return FeatureBundle(
        train=train_features,
        test=test_features,
        train_metadata=train_metadata,
        test_metadata=test_metadata,
        report_path=report_path,
        artifact_dir=artifact_dir,
    )


def build_feature_metadata(dataset_name: str, source_df: pd.DataFrame, feature_df: pd.DataFrame) -> FeatureMetadata:
    """Collect feature dtypes, null counts, and column groups for reporting."""

    feature_names = tuple(feature_df.columns)
    dtypes = {column: str(feature_df[column].dtype) for column in feature_names}
    null_counts = {column: int(feature_df[column].isna().sum()) for column in feature_names}
    categorical_columns = tuple(column for column in CATEGORICAL_COLUMNS if column in feature_df.columns)
    numeric_columns = tuple(column for column in feature_names if column not in categorical_columns)

    return FeatureMetadata(
        dataset_name=dataset_name,
        row_count=int(len(feature_df)),
        feature_count=int(len(feature_names)),
        feature_names=feature_names,
        dtypes=dtypes,
        null_counts=null_counts,
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
        memory_usage_bytes=int(feature_df.memory_usage(deep=True).sum()),
    )


def save_feature_metadata(train_metadata: FeatureMetadata, test_metadata: FeatureMetadata, artifact_dir: Path) -> Path:
    """Persist baseline feature metadata to disk."""

    metadata_path = artifact_dir / "baseline_feature_metadata.json"
    metadata = {
        "train": _metadata_to_dict(train_metadata),
        "test": _metadata_to_dict(test_metadata),
        "feature_names": list(train_metadata.feature_names),
        "categorical_columns": list(train_metadata.categorical_columns),
        "numeric_columns": list(train_metadata.numeric_columns),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata_path


def format_feature_summary(bundle: FeatureBundle) -> str:
    """Return a compact feature engineering summary for stdout and logs."""

    train = bundle.train_metadata
    test = bundle.test_metadata
    return (
        f"Features: {train.feature_count} | "
        f"Categorical: {len(train.categorical_columns)} | "
        f"Numeric: {len(train.numeric_columns)} | "
        f"Train rows: {train.row_count:,} | Test rows: {test.row_count:,} | "
        f"Report: {bundle.report_path}"
    )


def _validate_source_columns(df: pd.DataFrame) -> None:
    required_columns = {TIMESTAMP_COLUMN, DAY_COLUMN, "RoadType", "Weather", "Temperature", SOURCE_LANES_COLUMN, "LargeVehicles"}
    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        raise FeatureValidationError(f"Missing required source columns for feature generation: {missing_columns}.")


def _extract_timestamp_parts(series: pd.Series) -> pd.DataFrame:
    values = series.astype("string").str.strip()
    extracted = values.str.extract(r"^(?P<hour>\d{1,2}):(?P<minute>\d{1,2})$")
    if extracted.isna().any(axis=None):
        samples = values[extracted.isna().any(axis=1)].head(5).tolist()
        raise FeatureValidationError(f"Invalid timestamp values encountered while generating features: {samples}.")

    parts = extracted.astype(int)
    if ((parts["hour"] < 0) | (parts["hour"] > 23) | (parts["minute"] < 0) | (parts["minute"] > 59)).any():
        raise FeatureValidationError("Timestamp values out of range during feature generation.")
    return parts


def _build_day_index(series: pd.Series) -> pd.Series:
    """Build a deterministic zero-based day index without calendar semantics."""

    day_values = pd.to_numeric(series, errors="coerce")
    if day_values.isna().any():
        raise FeatureValidationError("Day values contain nulls or invalid entries required for temporal features.")
    return (day_values.astype(int) - int(day_values.min())).astype(int)


def _encode_large_vehicles(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").str.strip().str.lower()
    mapping = {"allowed": 1.0, "not allowed": 0.0}
    encoded = normalized.map(mapping)
    invalid_mask = normalized.notna() & encoded.isna() & (normalized != "<na>")
    if invalid_mask.any():
        samples = normalized[invalid_mask].head(5).tolist()
        raise FeatureValidationError(f"Unexpected LargeVehicles values encountered: {samples}.")
    return encoded


def _validate_feature_columns(feature_df: pd.DataFrame) -> None:
    duplicated = feature_df.columns[feature_df.columns.duplicated()].tolist()
    if duplicated:
        raise FeatureValidationError(f"Duplicate feature names detected: {duplicated}.")

    cyclical_ranges = {
        "hour_sin": feature_df["hour_sin"],
        "hour_cos": feature_df["hour_cos"],
        "minute_sin": feature_df["minute_sin"],
        "minute_cos": feature_df["minute_cos"],
    }
    for column, values in cyclical_ranges.items():
        finite_values = values.dropna()
        if ((finite_values < -1.0 - CYCLICAL_EPSILON) | (finite_values > 1.0 + CYCLICAL_EPSILON)).any():
            raise FeatureValidationError(f"Cyclical feature {column} is outside the expected [-1, 1] range.")


def _validate_feature_frame(source_df: pd.DataFrame, feature_df: pd.DataFrame, *, dataset_name: str) -> None:
    _validate_feature_columns(feature_df)
    _validate_null_profile(source_df, feature_df, dataset_name=dataset_name)


def _validate_null_profile(source_df: pd.DataFrame, feature_df: pd.DataFrame, *, dataset_name: str) -> None:
    source_nulls = {
        TIMESTAMP_COLUMN: int(source_df[TIMESTAMP_COLUMN].isna().sum()),
        DAY_COLUMN: int(source_df[DAY_COLUMN].isna().sum()),
        "RoadType": int(source_df["RoadType"].isna().sum()),
        "Weather": int(source_df["Weather"].isna().sum()),
        "Temperature": int(source_df["Temperature"].isna().sum()),
        OUTPUT_LANES_COLUMN: int(source_df[SOURCE_LANES_COLUMN].isna().sum()),
        "LargeVehicles": int(source_df["LargeVehicles"].isna().sum()),
    }
    expected_null_limits = {
        "hour": source_nulls[TIMESTAMP_COLUMN],
        "minute": source_nulls[TIMESTAMP_COLUMN],
        "day_index": source_nulls[DAY_COLUMN],
        "hour_sin": source_nulls[TIMESTAMP_COLUMN],
        "hour_cos": source_nulls[TIMESTAMP_COLUMN],
        "minute_sin": source_nulls[TIMESTAMP_COLUMN],
        "minute_cos": source_nulls[TIMESTAMP_COLUMN],
        "RoadType": source_nulls["RoadType"],
        "Weather": source_nulls["Weather"],
        "Temperature": source_nulls["Temperature"],
        OUTPUT_LANES_COLUMN: source_nulls[OUTPUT_LANES_COLUMN],
        "LargeVehicles": source_nulls["LargeVehicles"],
    }

    for column, max_nulls in expected_null_limits.items():
        feature_nulls = int(feature_df[column].isna().sum())
        if feature_nulls > max_nulls * NULL_EXPLOSION_FACTOR:
            raise FeatureValidationError(
                f"{dataset_name.title()} feature null explosion detected for {column}: {feature_nulls} nulls (expected at most {max_nulls})."
            )


def _render_feature_report(train_metadata: FeatureMetadata, test_metadata: FeatureMetadata) -> str:
    lines = [
        "# Baseline Features",
        "",
        "## Feature Groups",
        "",
        f"- Categorical columns: {', '.join(train_metadata.categorical_columns)}",
        f"- Numeric columns: {', '.join(train_metadata.numeric_columns)}",
        "- Temporal columns: hour, minute, day_index",
        "",
        "## Feature Table",
        "",
        "| Feature | Train dtype | Train nulls | Test dtype | Test nulls |",
        "| --- | --- | ---: | --- | ---: |",
    ]
    for feature_name in train_metadata.feature_names:
        lines.append(
            f"| {feature_name} | {train_metadata.dtypes[feature_name]} | {train_metadata.null_counts[feature_name]:,} | {test_metadata.dtypes[feature_name]} | {test_metadata.null_counts[feature_name]:,} |"
        )

    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- Train rows: {train_metadata.row_count:,}",
            f"- Test rows: {test_metadata.row_count:,}",
            f"- Train feature memory: { _format_bytes(train_metadata.memory_usage_bytes) }",
            f"- Test feature memory: { _format_bytes(test_metadata.memory_usage_bytes) }",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _metadata_to_dict(metadata: FeatureMetadata) -> dict[str, object]:
    return {
        "dataset_name": metadata.dataset_name,
        "row_count": metadata.row_count,
        "feature_count": metadata.feature_count,
        "feature_names": list(metadata.feature_names),
        "dtypes": metadata.dtypes,
        "null_counts": metadata.null_counts,
        "categorical_columns": list(metadata.categorical_columns),
        "numeric_columns": list(metadata.numeric_columns),
        "memory_usage_bytes": metadata.memory_usage_bytes,
    }


def _format_bytes(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"

    units = ("KB", "MB", "GB", "TB")
    value = float(num_bytes)
    for unit in units:
        value /= 1024.0
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
    return f"{value:.2f} TB"