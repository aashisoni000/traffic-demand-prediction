"""Baseline feature engineering for the traffic demand prediction project."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence

import numpy as np
import pandas as pd

from utils.logger import get_logger
from validation import ValidationRun

LOGGER = get_logger("features")

TIMESTAMP_COLUMN: Final[str] = "timestamp"
DAY_COLUMN: Final[str] = "day"
GEOHASH_COLUMN: Final[str] = "geohash"
TARGET_COLUMN: Final[str] = "demand"
SOURCE_LANES_COLUMN: Final[str] = "NumberofLanes"
OUTPUT_LANES_COLUMN: Final[str] = "NumberOfLanes"
GEOHASH_HOUR_MEAN_COLUMN: Final[str] = "geohash_hour_mean"
ROADTYPE_HOUR_MEAN_COLUMN: Final[str] = "RoadType_hour_mean"
GROUPED_MEAN_COLUMNS: Final[tuple[str, ...]] = (GEOHASH_HOUR_MEAN_COLUMN, ROADTYPE_HOUR_MEAN_COLUMN)
LAG1_DEMAND_COLUMN: Final[str] = "lag1_demand"
LAG2_DEMAND_COLUMN: Final[str] = "lag2_demand"
LAG3_DEMAND_COLUMN: Final[str] = "lag3_demand"
LAG6_DEMAND_COLUMN: Final[str] = "lag6_demand"
ROLLING_MEAN_3_COLUMN: Final[str] = "rolling_mean_3"
LAG_FEATURE_COLUMNS: Final[tuple[str, ...]] = (LAG1_DEMAND_COLUMN, LAG2_DEMAND_COLUMN, LAG3_DEMAND_COLUMN, LAG6_DEMAND_COLUMN, ROLLING_MEAN_3_COLUMN)
BASE_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
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
GROUPED_OUTPUT_FEATURE_COLUMNS: Final[tuple[str, ...]] = BASE_FEATURE_COLUMNS + GROUPED_MEAN_COLUMNS
OUTPUT_FEATURE_COLUMNS: Final[tuple[str, ...]] = BASE_FEATURE_COLUMNS + GROUPED_MEAN_COLUMNS + LAG_FEATURE_COLUMNS
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
    GEOHASH_HOUR_MEAN_COLUMN,
    ROADTYPE_HOUR_MEAN_COLUMN,
    LAG1_DEMAND_COLUMN,
    LAG2_DEMAND_COLUMN,
    LAG3_DEMAND_COLUMN,
    LAG6_DEMAND_COLUMN,
    ROLLING_MEAN_3_COLUMN,
)
DEFAULT_ARTIFACT_DIR: Final[Path] = Path("artifacts/features")
DEFAULT_REPORT_PATH: Final[Path] = Path("reports/feature_reports/baseline_features.md")
GROUPED_FEATURE_REPORT_PATH: Final[Path] = Path("reports/feature_reports/grouped_mean_features.md")
LAG_FEATURE_REPORT_PATH: Final[Path] = Path("reports/feature_reports/lag_features.md")
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


@dataclass(frozen=True, slots=True)
class MeanEncodingMapping:
    """Learned fallback-aware mean encoding state."""

    group_means: pd.Series
    hour_means: pd.Series
    global_mean: float


@dataclass(frozen=True, slots=True)
class GroupedMeanSpec:
    """Definition for one fold-safe grouped target mean feature."""

    output_column: str
    group_columns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GroupedMeanDiagnostics:
    """Validation diagnostics for one grouped target mean feature."""

    feature_name: str
    group_columns: tuple[str, ...]
    train_bucket_count: int
    test_bucket_count: int
    train_nulls: int
    test_nulls: int
    fold_rows: tuple[dict[str, object], ...]


@dataclass(frozen=True, slots=True)
class LagFeatureDiagnostics:
    """Validation diagnostics for one fold-safe lag feature."""

    feature_name: str
    group_column: str
    lag_order: int
    train_nulls: int
    test_nulls: int
    train_rows: int
    test_rows: int
    test_coverage_rows: int
    fold_rows: tuple[dict[str, object], ...]


GROUPED_MEAN_SPECS: Final[tuple[GroupedMeanSpec, ...]] = (
    GroupedMeanSpec(GEOHASH_HOUR_MEAN_COLUMN, (GEOHASH_COLUMN, "hour")),
    GroupedMeanSpec(ROADTYPE_HOUR_MEAN_COLUMN, ("RoadType", "hour")),
)


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

    features = features.loc[:, BASE_FEATURE_COLUMNS]
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


def build_feature_bundle(
    train_source: pd.DataFrame,
    test_source: pd.DataFrame,
    validation_run: ValidationRun,
    enabled_features: Sequence[str] | None = None,
) -> FeatureBundle:
    """Generate, validate, and package the baseline feature set."""

    enabled_set = set(enabled_features) if enabled_features is not None else None
    LOGGER.info("Building features with enabled set: %s", enabled_set)

    train_features = generate_features(train_source)
    test_features = generate_features(test_source)

    active_specs = [
        spec for spec in GROUPED_MEAN_SPECS 
        if enabled_set is None or spec.output_column in enabled_set
    ]
    train_features, test_features, grouped_diagnostics = add_fold_safe_grouped_means(
        train_source=train_source,
        train_features=train_features,
        test_source=test_source,
        test_features=test_features,
        validation_run=validation_run,
        specs=tuple(active_specs),
    )

    active_lags = [col for col in LAG_FEATURE_COLUMNS if enabled_set is None or col in enabled_set]
    train_features, test_features, lag_diagnostics = add_fold_safe_lag_features(
        train_source=train_source,
        train_features=train_features,
        test_source=test_source,
        test_features=test_features,
        validation_run=validation_run,
        columns=tuple(active_lags),
    )
    validate_feature_pair(train_source, train_features, test_source, test_features)
    LOGGER.info("Final feature columns: %s", list(train_features.columns))

    artifact_dir = DEFAULT_ARTIFACT_DIR.expanduser().resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    train_metadata = build_feature_metadata("train", train_source, train_features)
    test_metadata = build_feature_metadata("test", test_source, test_features)
    save_feature_metadata(train_metadata, test_metadata, artifact_dir)

    report_path = DEFAULT_REPORT_PATH.expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_render_feature_report(train_metadata, test_metadata), encoding="utf-8")
    grouped_report_path = GROUPED_FEATURE_REPORT_PATH.expanduser().resolve()
    grouped_report_path.parent.mkdir(parents=True, exist_ok=True)
    grouped_report_path.write_text(_render_grouped_feature_report(grouped_diagnostics), encoding="utf-8")
    lag_report_path = LAG_FEATURE_REPORT_PATH.expanduser().resolve()
    lag_report_path.parent.mkdir(parents=True, exist_ok=True)
    
    if lag_diagnostics:
        lag_report_path.write_text(_render_lag_feature_report(lag_diagnostics), encoding="utf-8")

    LOGGER.info("Saved baseline feature metadata to %s.", artifact_dir)
    LOGGER.info("Wrote baseline feature report to %s.", report_path)
    LOGGER.info("Wrote grouped feature diagnostics report to %s.", grouped_report_path)
    LOGGER.info("Wrote lag feature diagnostics report to %s.", lag_report_path)

    return FeatureBundle(
        train=train_features,
        test=test_features,
        train_metadata=train_metadata,
        test_metadata=test_metadata,
        report_path=report_path,
        artifact_dir=artifact_dir,
    )


def add_fold_safe_grouped_means(
    *,
    train_source: pd.DataFrame,
    train_features: pd.DataFrame,
    test_source: pd.DataFrame,
    test_features: pd.DataFrame,
    validation_run: ValidationRun,
    specs: tuple[GroupedMeanSpec, ...] = GROUPED_MEAN_SPECS,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[GroupedMeanDiagnostics, ...]]:
    """Add fold-safe grouped target mean features to train/test frames."""

    _validate_mean_encoding_sources(train_source, test_source)
    train_output = train_features.copy()
    test_output = test_features.copy()
    diagnostics: list[GroupedMeanDiagnostics] = []

    for spec in specs:
        train_output, test_output, feature_diagnostics = _add_fold_safe_grouped_mean(
            spec=spec,
            train_source=train_source,
            train_features=train_output,
            test_source=test_source,
            test_features=test_output,
            validation_run=validation_run,
        )
        diagnostics.append(feature_diagnostics)

    active_cols = list(train_features.columns) + [spec.output_column for spec in specs]
    return train_output.loc[:, active_cols], test_output.loc[:, active_cols], tuple(diagnostics)


def add_fold_safe_geohash_hour_mean(
    *,
    train_source: pd.DataFrame,
    train_features: pd.DataFrame,
    test_source: pd.DataFrame,
    test_features: pd.DataFrame,
    validation_run: ValidationRun,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add only the fold-safe geohash-hour target mean feature."""

    spec = GROUPED_MEAN_SPECS[0]
    train_output, test_output, _ = _add_fold_safe_grouped_mean(
        spec=spec,
        train_source=train_source,
        train_features=train_features,
        test_source=test_source,
        test_features=test_features,
        validation_run=validation_run,
    )
    return train_output.loc[:, BASE_FEATURE_COLUMNS + (GEOHASH_HOUR_MEAN_COLUMN,)], test_output.loc[
        :, BASE_FEATURE_COLUMNS + (GEOHASH_HOUR_MEAN_COLUMN,)
    ]


def _add_fold_safe_grouped_mean(
    *,
    spec: GroupedMeanSpec,
    train_source: pd.DataFrame,
    train_features: pd.DataFrame,
    test_source: pd.DataFrame,
    test_features: pd.DataFrame,
    validation_run: ValidationRun,
) -> tuple[pd.DataFrame, pd.DataFrame, GroupedMeanDiagnostics]:
    train_output = train_features.copy()
    test_output = test_features.copy()
    full_train_mapping = _fit_grouped_hour_mean(train_source, train_output, train_source.index, spec=spec)
    train_output[spec.output_column] = np.nan
    fold_rows: list[dict[str, object]] = []

    for fold in validation_run.folds:
        train_index = pd.Index(fold.train_row_ids)
        validation_index = pd.Index(fold.validation_row_ids)
        _validate_fold_indices(fold.fold_id, train_index, validation_index, len(train_source), spec.output_column)

        fold_mapping = _fit_grouped_hour_mean(train_source, train_output, train_index, spec=spec)
        unfilled_train_index = train_index[train_output.iloc[train_index][spec.output_column].isna().to_numpy()]
        if len(unfilled_train_index) > 0:
            train_output.iloc[
                unfilled_train_index,
                train_output.columns.get_loc(spec.output_column),
            ] = _transform_grouped_hour_mean(
                train_source.iloc[unfilled_train_index],
                train_output.iloc[unfilled_train_index],
                fold_mapping,
                spec=spec,
            ).to_numpy()

        validation_values, fallback_usage = _transform_grouped_hour_mean(
            train_source.iloc[validation_index],
            train_output.iloc[validation_index],
            fold_mapping,
            spec=spec,
            return_fallback_usage=True,
        )
        train_output.iloc[
            validation_index,
            train_output.columns.get_loc(spec.output_column),
        ] = validation_values.to_numpy()
        fold_rows.append(
            _build_grouped_mean_fold_diagnostics(
                spec=spec,
                fold_id=fold.fold_id,
                train_source=train_source.iloc[train_index],
                train_features=train_output.iloc[train_index],
                validation_source=train_source.iloc[validation_index],
                validation_features=train_output.iloc[validation_index],
                fallback_usage=fallback_usage,
            )
        )

    if train_output[spec.output_column].isna().any():
        missing_rows = int(train_output[spec.output_column].isna().sum())
        raise FeatureValidationError(
            f"{spec.output_column} produced {missing_rows:,} missing train values after fold-safe encoding."
        )
    _validate_fold_safe_grouped_mean(train_source, train_output, validation_run, spec=spec)

    test_output[spec.output_column] = _transform_grouped_hour_mean(test_source, test_output, full_train_mapping, spec=spec)
    train_output[spec.output_column] = train_output[spec.output_column].astype(float)
    test_output[spec.output_column] = test_output[spec.output_column].astype(float)
    diagnostics = GroupedMeanDiagnostics(
        feature_name=spec.output_column,
        group_columns=spec.group_columns,
        train_bucket_count=_count_buckets(train_source, train_output, spec),
        test_bucket_count=_count_buckets(test_source, test_output, spec),
        train_nulls=int(train_output[spec.output_column].isna().sum()),
        test_nulls=int(test_output[spec.output_column].isna().sum()),
        fold_rows=tuple(fold_rows),
    )
    return train_output, test_output, diagnostics


def add_fold_safe_lag_features(
    *,
    train_source: pd.DataFrame,
    train_features: pd.DataFrame,
    test_source: pd.DataFrame,
    test_features: pd.DataFrame,
    validation_run: ValidationRun,
    columns: tuple[str, ...] = LAG_FEATURE_COLUMNS,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[LagFeatureDiagnostics, ...]]:
    """Add fold-safe lag and rolling demand features within geohash."""

    _validate_lag_sources(train_source, test_source)
    train_output = train_features.copy()
    test_output = test_features.copy()
    for col in columns:
        train_output[col] = np.nan
    assigned_rows = pd.Series(False, index=train_output.index)
    fold_diagnostics_map: dict[str, list[dict[str, object]]] = {col: [] for col in columns}

    for fold in validation_run.folds:
        train_index = pd.Index(fold.train_row_ids)
        validation_index = pd.Index(fold.validation_row_ids)
        _validate_fold_indices(fold.fold_id, train_index, validation_index, len(train_source), "lag_features")

        unassigned_train_index = train_index[~assigned_rows.iloc[train_index].to_numpy()]
        if len(unassigned_train_index) > 0:
            train_results = _compute_lag_features_from_history(
                history_source=train_source.iloc[train_index],
                history_features=train_output.iloc[train_index],
                query_source=train_source.iloc[unassigned_train_index],
                query_features=train_output.iloc[unassigned_train_index],
            )
            for col in columns:
                train_output.iloc[
                    unassigned_train_index,
                    train_output.columns.get_loc(col),
                ] = train_results[col].to_numpy()
            assigned_rows.iloc[unassigned_train_index] = True

        validation_results = _compute_lag_features_from_history(
            history_source=train_source.iloc[train_index],
            history_features=train_output.iloc[train_index],
            query_source=train_source.iloc[validation_index],
            query_features=train_output.iloc[validation_index],
        )
        for col in columns:
            train_output.iloc[
                validation_index,
                train_output.columns.get_loc(col),
            ] = validation_results[col].to_numpy()
        assigned_rows.iloc[validation_index] = True

        for col in columns:
            fold_diagnostics_map[col].append(
                _build_lag_fold_diagnostics(
                    fold_id=fold.fold_id,
                    train_source=train_source.iloc[train_index],
                    validation_source=train_source.iloc[validation_index],
                    validation_features=train_output.iloc[validation_index],
                    validation_lag_values=validation_results[col],
                    validation_source_times=validation_results["_lag1_source_times"],
                )
            )

    _validate_fold_safe_lag_features(train_source, train_output, validation_run, columns=columns)
    test_results = _compute_lag_features_from_history(
        history_source=train_source,
        history_features=train_output,
        query_source=test_source,
        query_features=test_output,
    )

    for col in columns:
        test_output[col] = test_results[col].to_numpy()
        train_output[col] = train_output[col].astype(float)
        test_output[col] = test_output[col].astype(float)

    diagnostics = []
    for col in columns:
        if col == LAG1_DEMAND_COLUMN:
            lag_order = 1
        elif col == LAG2_DEMAND_COLUMN:
            lag_order = 2
        elif col == LAG3_DEMAND_COLUMN:
            lag_order = 3
        elif col == LAG6_DEMAND_COLUMN:
            lag_order = 6
        else:  # ROLLING_MEAN_3_COLUMN
            lag_order = 3

        diagnostics.append(
            LagFeatureDiagnostics(
                feature_name=col,
                group_column=GEOHASH_COLUMN,
                lag_order=lag_order,
                train_nulls=int(train_output[col].isna().sum()),
                test_nulls=int(test_output[col].isna().sum()),
                train_rows=int(len(train_output)),
                test_rows=int(len(test_output)),
                test_coverage_rows=int(test_output[col].notna().sum()),
                fold_rows=tuple(fold_diagnostics_map[col]),
            )
        )
    active_cols = list(train_features.columns) + list(columns)
    return train_output.loc[:, active_cols], test_output.loc[:, active_cols], tuple(diagnostics)


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
        "fold_safe_target_encoded_columns": list(GROUPED_MEAN_COLUMNS),
        "fold_safe_lag_columns": list(LAG_FEATURE_COLUMNS),
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
    required_columns = {
        TIMESTAMP_COLUMN,
        DAY_COLUMN,
        GEOHASH_COLUMN,
        "RoadType",
        "Weather",
        "Temperature",
        SOURCE_LANES_COLUMN,
        "LargeVehicles",
    }
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


def _validate_mean_encoding_sources(train_source: pd.DataFrame, test_source: pd.DataFrame) -> None:
    missing_train = sorted({GEOHASH_COLUMN, TARGET_COLUMN, "RoadType"}.difference(train_source.columns))
    if missing_train:
        raise FeatureValidationError(f"Missing train columns for fold-safe grouped means: {missing_train}.")
    missing_test = sorted({GEOHASH_COLUMN, "RoadType"}.difference(test_source.columns))
    if missing_test:
        raise FeatureValidationError(f"Missing test columns for fold-safe grouped means: {missing_test}.")
    if pd.to_numeric(train_source[TARGET_COLUMN], errors="coerce").isna().any():
        raise FeatureValidationError(f"{TARGET_COLUMN} contains null or invalid values required for fold-safe grouped means.")


def _validate_lag_sources(train_source: pd.DataFrame, test_source: pd.DataFrame) -> None:
    missing_train = sorted({GEOHASH_COLUMN, DAY_COLUMN, TIMESTAMP_COLUMN, TARGET_COLUMN}.difference(train_source.columns))
    if missing_train:
        raise FeatureValidationError(f"Missing train columns for {LAG1_DEMAND_COLUMN}: {missing_train}.")
    missing_test = sorted({GEOHASH_COLUMN, DAY_COLUMN, TIMESTAMP_COLUMN}.difference(test_source.columns))
    if missing_test:
        raise FeatureValidationError(f"Missing test columns for {LAG1_DEMAND_COLUMN}: {missing_test}.")
    if pd.to_numeric(train_source[TARGET_COLUMN], errors="coerce").isna().any():
        raise FeatureValidationError(f"{TARGET_COLUMN} contains null or invalid values required for {LAG1_DEMAND_COLUMN}.")


def _fit_grouped_hour_mean(
    source_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    row_index: pd.Index,
    *,
    spec: GroupedMeanSpec,
) -> MeanEncodingMapping:
    if len(row_index) == 0:
        raise FeatureValidationError(f"Cannot fit {spec.output_column} from an empty row set.")

    fit_frame = _build_group_key_frame(source_df.iloc[row_index], feature_df.iloc[row_index], spec).reset_index(drop=True)
    fit_frame[TARGET_COLUMN] = pd.to_numeric(source_df.iloc[row_index][TARGET_COLUMN], errors="raise").astype(float).reset_index(drop=True)
    if fit_frame[TARGET_COLUMN].isna().any():
        raise FeatureValidationError(f"{TARGET_COLUMN} contains missing values while fitting {spec.output_column}.")

    return MeanEncodingMapping(
        group_means=fit_frame.groupby(list(spec.group_columns), sort=True, dropna=False)[TARGET_COLUMN].mean(),
        hour_means=fit_frame.groupby("hour", sort=True, dropna=False)[TARGET_COLUMN].mean(),
        global_mean=float(fit_frame[TARGET_COLUMN].mean()),
    )


def _transform_grouped_hour_mean(
    source_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    mapping: MeanEncodingMapping,
    *,
    spec: GroupedMeanSpec,
    return_fallback_usage: bool = False,
) -> pd.Series | tuple[pd.Series, dict[str, int]]:
    key_frame = _build_group_key_frame(source_df, feature_df, spec)
    row_keys = pd.MultiIndex.from_frame(key_frame.reset_index(drop=True))
    group_values = pd.Series(row_keys.map(mapping.group_means), index=source_df.index, dtype="float64")
    hour_fallback = feature_df["hour"].map(mapping.hour_means).astype("float64")
    encoded = group_values.fillna(pd.Series(hour_fallback.to_numpy(), index=source_df.index))
    encoded = encoded.fillna(mapping.global_mean)
    if encoded.isna().any():
        raise FeatureValidationError(f"{spec.output_column} transform produced missing values.")

    if not return_fallback_usage:
        return encoded.astype(float)

    exact_rows = int(group_values.notna().sum())
    hour_rows = int(group_values.isna().sum() - hour_fallback[group_values.isna()].isna().sum())
    global_rows = int(len(encoded) - exact_rows - hour_rows)
    return encoded.astype(float), {
        "exact_rows": exact_rows,
        "hour_fallback_rows": hour_rows,
        "global_fallback_rows": global_rows,
    }


def _build_group_key_frame(source_df: pd.DataFrame, feature_df: pd.DataFrame, spec: GroupedMeanSpec) -> pd.DataFrame:
    key_data: dict[str, pd.Series] = {}
    for column in spec.group_columns:
        if column == "hour":
            key_data[column] = feature_df[column].astype("Int64").reset_index(drop=True)
        elif column in feature_df.columns:
            key_data[column] = feature_df[column].astype("string").reset_index(drop=True)
        elif column in source_df.columns:
            key_data[column] = source_df[column].astype("string").reset_index(drop=True)
        else:
            raise FeatureValidationError(f"Missing key column {column} for {spec.output_column}.")
    return pd.DataFrame(key_data)


def _count_buckets(source_df: pd.DataFrame, feature_df: pd.DataFrame, spec: GroupedMeanSpec) -> int:
    return int(_build_group_key_frame(source_df, feature_df, spec).drop_duplicates().shape[0])


def _validate_fold_indices(
    fold_id: int,
    train_index: pd.Index,
    validation_index: pd.Index,
    total_rows: int,
    feature_name: str,
) -> None:
    train_rows = set(int(row_id) for row_id in train_index)
    validation_rows = set(int(row_id) for row_id in validation_index)
    if train_rows.intersection(validation_rows):
        raise FeatureValidationError(f"Fold {fold_id} has overlapping train/validation rows for {feature_name}.")
    if any(row_id < 0 or row_id >= total_rows for row_id in train_rows.union(validation_rows)):
        raise FeatureValidationError(f"Fold {fold_id} has out-of-range row ids for {feature_name}.")


def _validate_fold_safe_grouped_mean(
    train_source: pd.DataFrame,
    train_features: pd.DataFrame,
    validation_run: ValidationRun,
    *,
    spec: GroupedMeanSpec,
) -> None:
    for fold in validation_run.folds:
        train_index = pd.Index(fold.train_row_ids)
        validation_index = pd.Index(fold.validation_row_ids)
        fold_mapping = _fit_grouped_hour_mean(train_source, train_features, train_index, spec=spec)
        expected = _transform_grouped_hour_mean(
            train_source.iloc[validation_index],
            train_features.iloc[validation_index],
            fold_mapping,
            spec=spec,
        ).reset_index(drop=True)
        actual = train_features.iloc[validation_index][spec.output_column].reset_index(drop=True)
        if not np.allclose(actual.to_numpy(dtype=float), expected.to_numpy(dtype=float), equal_nan=False):
            raise FeatureValidationError(
                f"Fold {fold.fold_id} {spec.output_column} values do not match the training-fold-only mapping."
            )


def _build_grouped_mean_fold_diagnostics(
    *,
    spec: GroupedMeanSpec,
    fold_id: int,
    train_source: pd.DataFrame,
    train_features: pd.DataFrame,
    validation_source: pd.DataFrame,
    validation_features: pd.DataFrame,
    fallback_usage: dict[str, int],
) -> dict[str, object]:
    train_keys = _build_group_key_frame(train_source, train_features, spec)
    validation_keys = _build_group_key_frame(validation_source, validation_features, spec)
    train_bucket_frame = train_keys.drop_duplicates()
    validation_bucket_frame = validation_keys.drop_duplicates()
    train_bucket_index = pd.MultiIndex.from_frame(train_bucket_frame)
    validation_bucket_index = pd.MultiIndex.from_frame(validation_bucket_frame)
    matched_validation_buckets = int(validation_bucket_index.isin(train_bucket_index).sum())
    validation_buckets = int(len(validation_bucket_index))
    validation_rows = int(len(validation_source))
    exact_rows = int(fallback_usage["exact_rows"])
    hour_fallback_rows = int(fallback_usage["hour_fallback_rows"])
    global_fallback_rows = int(fallback_usage["global_fallback_rows"])
    return {
        "fold": int(fold_id),
        "train_rows": int(len(train_source)),
        "validation_rows": validation_rows,
        "train_buckets": int(len(train_bucket_index)),
        "validation_buckets": validation_buckets,
        "matched_validation_buckets": matched_validation_buckets,
        "unseen_validation_buckets": int(validation_buckets - matched_validation_buckets),
        "validation_bucket_coverage": float(matched_validation_buckets / validation_buckets) if validation_buckets else 0.0,
        "exact_rows": exact_rows,
        "exact_row_rate": float(exact_rows / validation_rows) if validation_rows else 0.0,
        "hour_fallback_rows": hour_fallback_rows,
        "hour_fallback_rate": float(hour_fallback_rows / validation_rows) if validation_rows else 0.0,
        "global_fallback_rows": global_fallback_rows,
        "global_fallback_rate": float(global_fallback_rows / validation_rows) if validation_rows else 0.0,
    }


def _compute_lag_features_from_history(
    *,
    history_source: pd.DataFrame,
    history_features: pd.DataFrame,
    query_source: pd.DataFrame,
    query_features: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    history = pd.DataFrame(
        {
            GEOHASH_COLUMN: history_source[GEOHASH_COLUMN].astype("string"),
            "_time_key": _build_absolute_time_key(history_source, history_features),
            TARGET_COLUMN: pd.to_numeric(history_source[TARGET_COLUMN], errors="raise").astype(float),
        },
        index=history_source.index,
    ).sort_values([GEOHASH_COLUMN, "_time_key"], kind="mergesort")
    query = pd.DataFrame(
        {
            GEOHASH_COLUMN: query_source[GEOHASH_COLUMN].astype("string"),
            "_time_key": _build_absolute_time_key(query_source, query_features),
        },
        index=query_source.index,
    )

    results = {
        LAG1_DEMAND_COLUMN: pd.Series(np.nan, index=query_source.index, dtype="float64"),
        LAG2_DEMAND_COLUMN: pd.Series(np.nan, index=query_source.index, dtype="float64"),
        LAG3_DEMAND_COLUMN: pd.Series(np.nan, index=query_source.index, dtype="float64"),
        LAG6_DEMAND_COLUMN: pd.Series(np.nan, index=query_source.index, dtype="float64"),
        ROLLING_MEAN_3_COLUMN: pd.Series(np.nan, index=query_source.index, dtype="float64"),
        "_lag1_source_times": pd.Series(np.nan, index=query_source.index, dtype="float64"),
    }

    for geohash, query_group in query.groupby(GEOHASH_COLUMN, sort=False, dropna=False):
        history_group = history[history[GEOHASH_COLUMN] == geohash]
        if history_group.empty:
            continue
        history_times = history_group["_time_key"].to_numpy(dtype=np.int64)
        history_targets = history_group[TARGET_COLUMN].to_numpy(dtype=float)
        query_times = query_group["_time_key"].to_numpy(dtype=np.int64)
        positions = np.searchsorted(history_times, query_times, side="left") - 1

        # Lag 1
        valid_lag1 = positions >= 0
        if valid_lag1.any():
            idx = query_group.index[valid_lag1]
            pos = positions[valid_lag1]
            results[LAG1_DEMAND_COLUMN].loc[idx] = history_targets[pos]
            results["_lag1_source_times"].loc[idx] = history_times[pos].astype(float)

        # Lag 2
        valid_lag2 = positions >= 1
        if valid_lag2.any():
            results[LAG2_DEMAND_COLUMN].loc[query_group.index[valid_lag2]] = history_targets[positions[valid_lag2] - 1]

        # Lag 3
        valid_lag3 = positions >= 2
        if valid_lag3.any():
            results[LAG3_DEMAND_COLUMN].loc[query_group.index[valid_lag3]] = history_targets[positions[valid_lag3] - 2]

        # Lag 6
        valid_lag6 = positions >= 5
        if valid_lag6.any():
            results[LAG6_DEMAND_COLUMN].loc[query_group.index[valid_lag6]] = history_targets[positions[valid_lag6] - 5]

        # Rolling Mean 3
        valid_rolling3 = positions >= 2
        if valid_rolling3.any():
            idx = query_group.index[valid_rolling3]
            pos = positions[valid_rolling3]
            results[ROLLING_MEAN_3_COLUMN].loc[idx] = (history_targets[pos] + history_targets[pos-1] + history_targets[pos-2]) / 3.0

    return results


def _build_absolute_time_key(source_df: pd.DataFrame, feature_df: pd.DataFrame) -> pd.Series:
    days = pd.to_numeric(source_df[DAY_COLUMN], errors="raise").astype(int).reset_index(drop=True)
    hours = feature_df["hour"].astype(int).reset_index(drop=True)
    minutes = feature_df["minute"].astype(int).reset_index(drop=True)
    return pd.Series((days * 24 * 60 + hours * 60 + minutes).to_numpy(dtype=np.int64), index=source_df.index)


def _validate_fold_safe_lag_features(
    train_source: pd.DataFrame,
    train_features: pd.DataFrame,
    validation_run: ValidationRun,
    columns: tuple[str, ...] = LAG_FEATURE_COLUMNS,
) -> None:
    for fold in validation_run.folds:
        train_index = pd.Index(fold.train_row_ids)
        validation_index = pd.Index(fold.validation_row_ids)
        expected_results = _compute_lag_features_from_history(
            history_source=train_source.iloc[train_index],
            history_features=train_features.iloc[train_index],
            query_source=train_source.iloc[validation_index],
            query_features=train_features.iloc[validation_index],
        )

        for col in columns:
            actual = train_features.iloc[validation_index][col]
            actual_values = actual.reset_index(drop=True).to_numpy(dtype=float)
            expected_values = expected_results[col].reset_index(drop=True).to_numpy(dtype=float)
            matches = (np.isnan(actual_values) & np.isnan(expected_values)) | np.isclose(
                actual_values,
                expected_values,
                equal_nan=False,
            )
            if not bool(matches.all()):
                raise FeatureValidationError(
                    f"Fold {fold.fold_id} {col} values do not match the training-fold-only history."
                )

        query_times = _build_absolute_time_key(train_source.iloc[validation_index], train_features.iloc[validation_index])
        valid_source_times = expected_results["_lag1_source_times"].dropna()
        if not valid_source_times.empty:
            aligned_query_times = query_times.loc[valid_source_times.index]
            if (valid_source_times.astype(int) >= aligned_query_times.astype(int)).any():
                raise FeatureValidationError(f"Fold {fold.fold_id} has non-past lag source rows.")


def _build_lag_fold_diagnostics(
    *,
    fold_id: int,
    train_source: pd.DataFrame,
    validation_source: pd.DataFrame,
    validation_features: pd.DataFrame,
    validation_lag_values: pd.Series,
    validation_source_times: pd.Series,
) -> dict[str, object]:
    validation_rows = int(len(validation_source))
    coverage_rows = int(validation_lag_values.notna().sum())
    query_times = _build_absolute_time_key(validation_source, validation_features)
    valid_source_times = validation_source_times.dropna()
    chronological_pass = True
    if not valid_source_times.empty:
        chronological_pass = bool((valid_source_times.astype(int) < query_times.loc[valid_source_times.index].astype(int)).all())
    train_geohashes = set(train_source[GEOHASH_COLUMN].astype("string").dropna().unique().tolist())
    validation_geohashes = set(validation_source[GEOHASH_COLUMN].astype("string").dropna().unique().tolist())
    matched_validation_geohashes = len(validation_geohashes.intersection(train_geohashes))
    return {
        "fold": int(fold_id),
        "train_rows": int(len(train_source)),
        "validation_rows": validation_rows,
        "train_geohashes": int(len(train_geohashes)),
        "validation_geohashes": int(len(validation_geohashes)),
        "matched_validation_geohashes": int(matched_validation_geohashes),
        "validation_geohash_coverage": float(matched_validation_geohashes / len(validation_geohashes)) if validation_geohashes else 0.0,
        "coverage_rows": coverage_rows,
        "coverage_rate": float(coverage_rows / validation_rows) if validation_rows else 0.0,
        "nan_rows": int(validation_rows - coverage_rows),
        "nan_rate": float((validation_rows - coverage_rows) / validation_rows) if validation_rows else 0.0,
        "chronological_pass": chronological_pass,
    }


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
    for column in GROUPED_MEAN_COLUMNS:
        if column in feature_df.columns:
            expected_null_limits[column] = 0

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
        f"- Fold-safe target mean columns: {', '.join(GROUPED_MEAN_COLUMNS)}",
        f"- Fold-safe lag columns: {', '.join(LAG_FEATURE_COLUMNS)}",
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


def _render_lag_feature_report(diagnostics: Sequence[LagFeatureDiagnostics]) -> str:
    lines = [
        "# Fold-Safe Lag Features",
        "",
        "## Summary",
        "",
    ]

    for diagnostic in diagnostics:
        validation_rows = sum(int(row["validation_rows"]) for row in diagnostic.fold_rows)
        validation_coverage_rows = sum(int(row["coverage_rows"]) for row in diagnostic.fold_rows)
        validation_nan_rows = sum(int(row["nan_rows"]) for row in diagnostic.fold_rows)
        lines.extend([
            f"### {diagnostic.feature_name}",
            "",
            f"- Target: {TARGET_COLUMN}",
            f"- Group column: {diagnostic.group_column}",
            f"- Lag order: {diagnostic.lag_order}",
            "- Ordering: strictly chronological by day, hour, minute within geohash",
            "- Validation lag source: training fold rows only",
            "- Test lag source: full training rows only",
            f"- Train coverage rows: {diagnostic.train_rows - diagnostic.train_nulls:,} ({_format_percent(diagnostic.train_rows - diagnostic.train_nulls, diagnostic.train_rows)})",
            f"- Train NaN rows: {diagnostic.train_nulls:,} ({_format_percent(diagnostic.train_nulls, diagnostic.train_rows)})",
            f"- Test coverage rows: {diagnostic.test_coverage_rows:,} ({_format_percent(diagnostic.test_coverage_rows, diagnostic.test_rows)})",
            f"- Test NaN rows: {diagnostic.test_nulls:,} ({_format_percent(diagnostic.test_nulls, diagnostic.test_rows)})",
            f"- Validation coverage rows: {validation_coverage_rows:,} ({_format_percent(validation_coverage_rows, validation_rows)})",
            f"- Validation NaN rows: {validation_nan_rows:,} ({_format_percent(validation_nan_rows, validation_rows)})",
            "",
        ])

    lines.extend([
        "## Fold Details",
        "",
        "| Feature | Fold | Train rows | Validation rows | Train geohashes | Validation geohashes | Matched validation geohashes | Geohash coverage | Lag coverage rows | NaN rows | Chronological check |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ])
    for diagnostic in diagnostics:
        for row in diagnostic.fold_rows:
            lines.append(
                "| "
                f"{diagnostic.feature_name} | "
                f"{int(row['fold'])} | "
                f"{int(row['train_rows']):,} | "
                f"{int(row['validation_rows']):,} | "
                f"{int(row['train_geohashes']):,} | "
                f"{int(row['validation_geohashes']):,} | "
                f"{int(row['matched_validation_geohashes']):,} | "
                f"{float(row['validation_geohash_coverage']):.2%} | "
                f"{int(row['coverage_rows']):,} ({float(row['coverage_rate']):.2%}) | "
                f"{int(row['nan_rows']):,} ({float(row['nan_rate']):.2%}) | "
                f"{'PASS' if bool(row['chronological_pass']) else 'FAIL'} |"
            )

    chronological_pass = all(all(bool(row["chronological_pass"]) for row in diag.fold_rows) for diag in diagnostics)
    lines.extend(
        [
            "",
            "## Verification",
            "",
            "- No future leakage: PASS" if chronological_pass else "- No future leakage: FAIL",
            "- Chronological correctness: PASS" if chronological_pass else "- Chronological correctness: FAIL",
            "- Validation rows use only fold training history.",
            "- Lag values are missing only when no prior row exists in the permitted history for that geohash.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _render_grouped_feature_report(diagnostics: Sequence[GroupedMeanDiagnostics]) -> str:
    lines = [
        "# Fold-Safe Grouped Mean Features",
        "",
        "## Summary",
        "",
    ]
    for diagnostic in diagnostics:
        total_validation_rows = sum(int(row["validation_rows"]) for row in diagnostic.fold_rows)
        exact_rows = sum(int(row["exact_rows"]) for row in diagnostic.fold_rows)
        hour_fallback_rows = sum(int(row["hour_fallback_rows"]) for row in diagnostic.fold_rows)
        global_fallback_rows = sum(int(row["global_fallback_rows"]) for row in diagnostic.fold_rows)
        avg_validation_bucket_coverage = (
            sum(float(row["validation_bucket_coverage"]) for row in diagnostic.fold_rows) / len(diagnostic.fold_rows)
            if diagnostic.fold_rows
            else 0.0
        )
        lines.extend(
            [
                f"### {diagnostic.feature_name}",
                "",
                f"- Group columns: {', '.join(diagnostic.group_columns)}",
                f"- Train buckets: {diagnostic.train_bucket_count:,}",
                f"- Test buckets: {diagnostic.test_bucket_count:,}",
                f"- Train nulls: {diagnostic.train_nulls:,}",
                f"- Test nulls: {diagnostic.test_nulls:,}",
                f"- Validation exact rows: {exact_rows:,} ({_format_percent(exact_rows, total_validation_rows)})",
                f"- Validation hour fallback rows: {hour_fallback_rows:,} ({_format_percent(hour_fallback_rows, total_validation_rows)})",
                f"- Validation global fallback rows: {global_fallback_rows:,} ({_format_percent(global_fallback_rows, total_validation_rows)})",
                f"- Average validation bucket coverage: {avg_validation_bucket_coverage:.2%}",
                "",
            ]
        )

    lines.extend(
        [
            "## Fold Details",
            "",
            "| Feature | Fold | Train buckets | Validation buckets | Matched validation buckets | Validation bucket coverage | Exact rows | Hour fallback rows | Global fallback rows |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for diagnostic in diagnostics:
        for row in diagnostic.fold_rows:
            lines.append(
                "| "
                f"{diagnostic.feature_name} | "
                f"{int(row['fold'])} | "
                f"{int(row['train_buckets']):,} | "
                f"{int(row['validation_buckets']):,} | "
                f"{int(row['matched_validation_buckets']):,} | "
                f"{float(row['validation_bucket_coverage']):.2%} | "
                f"{int(row['exact_rows']):,} ({float(row['exact_row_rate']):.2%}) | "
                f"{int(row['hour_fallback_rows']):,} ({float(row['hour_fallback_rate']):.2%}) | "
                f"{int(row['global_fallback_rows']):,} ({float(row['global_fallback_rate']):.2%}) |"
            )

    lines.extend(
        [
            "",
            "## Verification",
            "",
            "- Fold mappings are fit on training fold rows only.",
            "- Validation rows are transformed from their fold-specific mapping.",
            "- Test rows are transformed from a full-training mapping for train/test parity.",
            "- NaN checks: PASS",
            "- Train/test feature parity checks run in the baseline feature bundle.",
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


def _format_percent(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "0.00%"
    return f"{(numerator / denominator):.2%}"
