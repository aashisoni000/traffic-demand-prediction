"""Deterministic purged time-series cross-validation utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence

import pandas as pd

from utils.data import DataValidationError, LoadedData
from utils.logger import get_logger

LOGGER = get_logger("validation")

DAY_COLUMN: Final[str] = "day"
GEOHASH_COLUMN: Final[str] = "geohash"
TIMESTAMP_COLUMN: Final[str] = "timestamp"

ROW_ID_COLUMN: Final[str] = "_validation_row_id"
PARSED_TIMESTAMP_COLUMN: Final[str] = "_validation_timestamp_parsed"
GROUP_POSITION_COLUMN: Final[str] = "_validation_group_position"
GROUP_LABEL_COLUMN: Final[str] = "_validation_group_label"

DEFAULT_N_SPLITS: Final[int] = 5
DEFAULT_EMBARGO_GROUPS: Final[int] = 1
DEFAULT_ARTIFACT_DIR: Final[Path] = Path("artifacts/folds")
DEFAULT_REPORT_PATH: Final[Path] = Path("reports/validation_reports/fold_summary.md")


@dataclass(frozen=True, slots=True)
class PurgedFold:
    """Single deterministic purged time-series fold."""

    fold_id: int
    train_row_ids: tuple[int, ...]
    validation_row_ids: tuple[int, ...]
    train_group_positions: tuple[int, ...]
    validation_group_positions: tuple[int, ...]
    train_rows: int
    validation_rows: int
    train_groups: int
    validation_groups: int
    train_geohashes: int
    validation_geohashes: int
    train_time_start: str
    train_time_end: str
    validation_time_start: str
    validation_time_end: str
    embargo_groups: int
    embargo_time_start: str | None
    embargo_time_end: str | None
    train_group_min_position: int
    train_group_max_position: int
    validation_group_min_position: int
    validation_group_max_position: int
    overlap_rows: int
    leakage_safe: bool


@dataclass(frozen=True, slots=True)
class ValidationRun:
    """Complete result of the validation pipeline."""

    folds: tuple[PurgedFold, ...]
    artifact_dir: Path
    report_path: Path
    n_splits: int
    embargo_groups: int
    total_rows: int
    total_time_groups: int
    min_validation_rows: int
    max_validation_rows: int
    validation_size_ratio: float
    imbalance_warning: str | None


def run_validation_pipeline(
    data: LoadedData,
    *,
    n_splits: int = DEFAULT_N_SPLITS,
    embargo_groups: int = DEFAULT_EMBARGO_GROUPS,
    artifact_dir: Path | str = DEFAULT_ARTIFACT_DIR,
    report_path: Path | str = DEFAULT_REPORT_PATH,
) -> ValidationRun:
    """Generate, verify, save, and report deterministic purged folds."""

    validation_frame = _prepare_validation_frame(data.train)
    folds = generate_purged_folds(validation_frame, n_splits=n_splits, embargo_groups=embargo_groups)
    verify_purged_folds(validation_frame, folds, embargo_groups=embargo_groups)

    output_dir = Path(artifact_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_fold_artifacts(validation_frame, folds, output_dir)

    output_report_path = Path(report_path).expanduser().resolve()
    output_report_path.parent.mkdir(parents=True, exist_ok=True)
    output_report_path.write_text(_render_fold_summary(validation_frame, folds, embargo_groups), encoding="utf-8")

    LOGGER.info("Saved %d fold artifacts to %s.", len(folds), output_dir)
    LOGGER.info("Wrote fold summary report to %s.", output_report_path)

    return ValidationRun(
        folds=tuple(folds),
        artifact_dir=output_dir,
        report_path=output_report_path,
        n_splits=n_splits,
        embargo_groups=embargo_groups,
        total_rows=int(len(validation_frame)),
        total_time_groups=int(validation_frame[GROUP_POSITION_COLUMN].nunique()),
        min_validation_rows=min(fold.validation_rows for fold in folds),
        max_validation_rows=max(fold.validation_rows for fold in folds),
        validation_size_ratio=(max(fold.validation_rows for fold in folds) / min(fold.validation_rows for fold in folds)) if folds and min(fold.validation_rows for fold in folds) > 0 else 0.0,
        imbalance_warning=_fold_imbalance_warning(folds),
    )


def generate_purged_folds(
    frame: pd.DataFrame,
    *,
    n_splits: int = DEFAULT_N_SPLITS,
    embargo_groups: int = DEFAULT_EMBARGO_GROUPS,
) -> list[PurgedFold]:
    """Create deterministic expanding-window purged folds."""

    if n_splits <= 0:
        raise DataValidationError("Number of splits must be positive.")
    if embargo_groups < 0:
        raise DataValidationError("Embargo groups must be zero or positive.")
    if frame.empty:
        raise DataValidationError("Validation cannot be built from an empty dataset.")

    sorted_frame = _ensure_validation_frame(frame)
    groups = _unique_time_groups(sorted_frame)

    block_count = n_splits + 1
    if len(groups) < block_count:
        raise DataValidationError(
            f"Not enough chronological time groups for {n_splits} folds: found {len(groups)} groups."
        )

    blocks = _contiguous_blocks(groups, block_count)
    if any(not block for block in blocks):
        raise DataValidationError("Fold generation produced an empty time block.")

    folds: list[PurgedFold] = []
    for fold_id in range(n_splits):
        history_groups = [group for block in blocks[: fold_id + 1] for group in block]
        validation_groups = blocks[fold_id + 1]

        if len(history_groups) <= embargo_groups:
            raise DataValidationError(
                f"Fold {fold_id + 1} has no usable training groups after applying the embargo."
            )

        embargo_slice = history_groups[-embargo_groups:] if embargo_groups else []
        train_groups = history_groups[:-embargo_groups] if embargo_groups else history_groups

        train_positions = tuple(group.position for group in train_groups)
        validation_positions = tuple(group.position for group in validation_groups)
        train_mask = sorted_frame[GROUP_POSITION_COLUMN].isin(train_positions)
        validation_mask = sorted_frame[GROUP_POSITION_COLUMN].isin(validation_positions)

        train_rows = sorted_frame.loc[train_mask, ROW_ID_COLUMN].astype(int).tolist()
        validation_rows = sorted_frame.loc[validation_mask, ROW_ID_COLUMN].astype(int).tolist()

        if not train_rows or not validation_rows:
            raise DataValidationError(f"Fold {fold_id + 1} contains an empty train or validation partition.")

        overlap_rows = len(set(train_rows).intersection(validation_rows))
        if overlap_rows > 0:
            raise DataValidationError(f"Fold {fold_id + 1} has overlapping train and validation rows.")

        train_min_position = min(train_positions)
        train_max_position = max(train_positions)
        validation_min_position = min(validation_positions)
        validation_max_position = max(validation_positions)
        leakage_safe = train_max_position < validation_min_position

        embargo_start_label, embargo_end_label = _embargo_range(embargo_slice)

        folds.append(
            PurgedFold(
                fold_id=fold_id + 1,
                train_row_ids=tuple(train_rows),
                validation_row_ids=tuple(validation_rows),
                train_group_positions=train_positions,
                validation_group_positions=validation_positions,
                train_rows=len(train_rows),
                validation_rows=len(validation_rows),
                train_groups=len(train_positions),
                validation_groups=len(validation_positions),
                train_geohashes=int(sorted_frame.loc[train_mask, GEOHASH_COLUMN].nunique()),
                validation_geohashes=int(sorted_frame.loc[validation_mask, GEOHASH_COLUMN].nunique()),
                train_time_start=train_groups[0].label,
                train_time_end=train_groups[-1].label,
                validation_time_start=validation_groups[0].label,
                validation_time_end=validation_groups[-1].label,
                embargo_groups=embargo_groups,
                embargo_time_start=embargo_start_label,
                embargo_time_end=embargo_end_label,
                train_group_min_position=train_min_position,
                train_group_max_position=train_max_position,
                validation_group_min_position=validation_min_position,
                validation_group_max_position=validation_max_position,
                overlap_rows=overlap_rows,
                leakage_safe=leakage_safe,
            )
        )

    return folds


def verify_purged_folds(
    frame: pd.DataFrame,
    folds: Sequence[PurgedFold],
    *,
    embargo_groups: int,
) -> None:
    """Hard-fail if any fold violates chronology, overlap, or leakage rules."""

    if not folds:
        raise DataValidationError("No validation folds were generated.")

    _ensure_validation_frame(frame)
    seen_validation_positions: set[int] = set()
    previous_validation_max: int | None = None

    for fold in folds:
        if fold.train_rows == 0 or fold.validation_rows == 0:
            raise DataValidationError(f"Fold {fold.fold_id} is empty.")
        if fold.overlap_rows > 0:
            raise DataValidationError(f"Fold {fold.fold_id} has overlapping train and validation rows.")
        if not fold.leakage_safe:
            raise DataValidationError(f"Fold {fold.fold_id} has a future leakage risk.")

        validation_positions = set(fold.validation_group_positions)
        if seen_validation_positions.intersection(validation_positions):
            raise DataValidationError(f"Validation folds overlap on fold {fold.fold_id}.")
        seen_validation_positions.update(validation_positions)

        if previous_validation_max is not None and fold.validation_group_min_position <= previous_validation_max:
            raise DataValidationError(f"Fold {fold.fold_id} breaks chronological validation ordering.")
        previous_validation_max = fold.validation_group_max_position

        gap_groups = fold.validation_group_min_position - fold.train_group_max_position - 1
        if gap_groups < embargo_groups:
            raise DataValidationError(
                f"Fold {fold.fold_id} embargo gap is too small: expected at least {embargo_groups} embargo groups, found {gap_groups}."
            )


def save_fold_artifacts(frame: pd.DataFrame, folds: Sequence[PurgedFold], artifact_dir: Path) -> None:
    """Persist fold-level train and validation indices for reproducibility."""

    sorted_frame = _ensure_validation_frame(frame)
    for fold in folds:
        fold_dir = artifact_dir / f"fold_{fold.fold_id:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_frame = sorted_frame.loc[
            sorted_frame[ROW_ID_COLUMN].isin(fold.train_row_ids),
            [ROW_ID_COLUMN, GROUP_POSITION_COLUMN, GEOHASH_COLUMN, DAY_COLUMN, TIMESTAMP_COLUMN],
        ]
        validation_frame = sorted_frame.loc[
            sorted_frame[ROW_ID_COLUMN].isin(fold.validation_row_ids),
            [ROW_ID_COLUMN, GROUP_POSITION_COLUMN, GEOHASH_COLUMN, DAY_COLUMN, TIMESTAMP_COLUMN],
        ]

        train_frame.to_csv(fold_dir / "train_rows.csv", index=False)
        validation_frame.to_csv(fold_dir / "validation_rows.csv", index=False)

    manifest = {
        "n_splits": len(folds),
        "embargo_groups": folds[0].embargo_groups if folds else 0,
        "folds": [_fold_to_dict(fold) for fold in folds],
    }
    (artifact_dir / "fold_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def format_validation_summary(run: ValidationRun) -> str:
    """Return a compact summary for logs and stdout."""

    return (
        f"Purged CV folds: {len(run.folds)} | "
        f"Embargo groups: {run.embargo_groups} | "
        f"Fold size ratio: {run.validation_size_ratio:.2f} | "
        f"Rows: {run.total_rows:,} | "
        f"Time groups: {run.total_time_groups:,} | "
        f"Artifacts: {run.artifact_dir} | Report: {run.report_path}"
    )


def _prepare_validation_frame(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy().reset_index(names=ROW_ID_COLUMN)
    return _ensure_validation_frame(working)


def _ensure_validation_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing_columns = [column for column in (GEOHASH_COLUMN, DAY_COLUMN, TIMESTAMP_COLUMN) if column not in frame.columns]
    if missing_columns:
        raise DataValidationError(f"Validation requires columns: {', '.join(missing_columns)}.")

    working = frame.copy()
    if ROW_ID_COLUMN not in working.columns:
        working = working.reset_index(names=ROW_ID_COLUMN)

    working[DAY_COLUMN] = pd.to_numeric(working[DAY_COLUMN], errors="raise").astype(int)
    working[PARSED_TIMESTAMP_COLUMN] = _parse_timestamp_column(working[TIMESTAMP_COLUMN])
    working[GROUP_POSITION_COLUMN] = _build_group_positions(working[DAY_COLUMN], working[PARSED_TIMESTAMP_COLUMN])
    working[GROUP_LABEL_COLUMN] = working[DAY_COLUMN].astype(str) + " " + working[PARSED_TIMESTAMP_COLUMN].dt.strftime("%H:%M")

    working = working.sort_values([GROUP_POSITION_COLUMN, GEOHASH_COLUMN, ROW_ID_COLUMN], kind="mergesort").reset_index(drop=True)
    _validate_timestamp_integrity(working)
    return working


@dataclass(frozen=True, slots=True)
class TimeGroup:
    position: int
    label: str


def _parse_timestamp_column(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.strip()
    extracted = values.str.extract(r"^(?P<hour>\d{1,2}):(?P<minute>\d{1,2})$")
    invalid_rows = extracted.isna().any(axis=1)
    if invalid_rows.any():
        samples = values[invalid_rows].head(5).tolist()
        raise DataValidationError(f"Invalid timestamp values encountered: {samples}.")

    hours = extracted["hour"].astype(int)
    minutes = extracted["minute"].astype(int)
    invalid_range = (hours > 23) | (minutes > 59)
    if invalid_range.any():
        samples = values[invalid_range].head(5).tolist()
        raise DataValidationError(f"Timestamp values out of range: {samples}.")

    normalized = hours.astype(str).str.zfill(2) + ":" + minutes.astype(str).str.zfill(2)
    parsed = pd.to_datetime(normalized, format="%H:%M", errors="raise")
    if parsed.isna().any():
        raise DataValidationError("Timestamp parsing produced missing values.")
    return parsed


def _build_group_positions(days: pd.Series, parsed_timestamps: pd.Series) -> pd.Series:
    time_key = days.astype(int) * 24 * 60 + parsed_timestamps.dt.hour * 60 + parsed_timestamps.dt.minute
    unique_keys = pd.Index(time_key.drop_duplicates().sort_values(kind="mergesort"))
    key_to_position = {int(key): position for position, key in enumerate(unique_keys.tolist())}
    return time_key.map(key_to_position).astype(int)


def _validate_timestamp_integrity(frame: pd.DataFrame) -> None:
    if frame[PARSED_TIMESTAMP_COLUMN].isna().any():
        raise DataValidationError("Invalid timestamps detected after parsing.")
    if not frame[GROUP_POSITION_COLUMN].is_monotonic_increasing:
        raise DataValidationError("Chronological ordering is invalid after fold preparation.")
    if frame[[GROUP_POSITION_COLUMN, GROUP_LABEL_COLUMN]].drop_duplicates().shape[0] != frame[GROUP_POSITION_COLUMN].nunique():
        raise DataValidationError("Time group integrity failed during fold preparation.")


def _unique_time_groups(frame: pd.DataFrame) -> list[TimeGroup]:
    unique_groups = (
        frame[[GROUP_POSITION_COLUMN, GROUP_LABEL_COLUMN]]
        .drop_duplicates(subset=[GROUP_POSITION_COLUMN])
        .sort_values(GROUP_POSITION_COLUMN, kind="mergesort")
        .itertuples(index=False, name=None)
    )
    return [TimeGroup(position=int(position), label=str(label)) for position, label in unique_groups]


def _contiguous_blocks(groups: Sequence[TimeGroup], block_count: int) -> list[list[TimeGroup]]:
    if block_count <= 0:
        raise DataValidationError("Number of fold blocks must be positive.")

    total = len(groups)
    base_size, remainder = divmod(total, block_count)
    blocks: list[list[TimeGroup]] = []
    start = 0
    for block_index in range(block_count):
        block_size = base_size + (1 if block_index < remainder else 0)
        end = start + block_size
        blocks.append(list(groups[start:end]))
        start = end
    return blocks


def _embargo_range(embargo_slice: Sequence[TimeGroup]) -> tuple[str | None, str | None]:
    if not embargo_slice:
        return None, None
    return embargo_slice[0].label, embargo_slice[-1].label


def _render_fold_summary(frame: pd.DataFrame, folds: Sequence[PurgedFold], embargo_groups: int) -> str:
    fold_sizes = [fold.validation_rows for fold in folds]
    size_ratio = (max(fold_sizes) / min(fold_sizes)) if fold_sizes and min(fold_sizes) > 0 else 0.0
    imbalance_warning = _fold_imbalance_warning(folds)

    lines = [
        "# Validation Fold Summary",
        "",
        f"- Total rows: {len(frame):,}",
        f"- Total time groups: {frame[GROUP_POSITION_COLUMN].nunique():,}",
        f"- Folds: {len(folds)}",
        f"- Embargo groups: {embargo_groups}",
        f"- Validation fold size ratio: {size_ratio:.2f}",
        f"- Smallest validation fold: {min(fold_sizes):,}" if fold_sizes else "- Smallest validation fold: N/A",
        f"- Largest validation fold: {max(fold_sizes):,}" if fold_sizes else "- Largest validation fold: N/A",
        "",
        "## Fold Details",
        "",
        "| Fold | Train rows | Validation rows | Validation share | Train groups | Validation groups | Train range | Validation range | Embargo range | Overlap | Leakage |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | --- |",
    ]
    for fold in folds:
        embargo_range = f"{fold.embargo_time_start} -> {fold.embargo_time_end}" if fold.embargo_time_start and fold.embargo_time_end else "None"
        validation_share = (fold.validation_rows / len(frame)) if len(frame) else 0.0
        lines.append(
            f"| {fold.fold_id} | {fold.train_rows:,} | {fold.validation_rows:,} | {validation_share:.2%} | {fold.train_groups:,} | {fold.validation_groups:,} | from {fold.train_time_start} to {fold.train_time_end} | from {fold.validation_time_start} to {fold.validation_time_end} | {embargo_range} | {fold.overlap_rows:,} | {'PASS' if fold.leakage_safe else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## Verification",
            "",
            "- Chronological ordering: PASS",
            "- Embargo support: PASS",
            "- Overlap checks: PASS",
            "- Timestamp integrity: PASS",
            "- Future leakage risk: PASS",
            f"- Fold imbalance warning: {'PASS' if imbalance_warning is None else 'WARN'}",
        ]
    )
    if imbalance_warning is not None:
        lines.extend([
            "",
            "## Imbalance Warning",
            "",
            imbalance_warning,
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _fold_imbalance_warning(folds: Sequence[PurgedFold]) -> str | None:
    validation_rows = [fold.validation_rows for fold in folds if fold.validation_rows > 0]
    if not validation_rows:
        return None

    smallest = min(validation_rows)
    largest = max(validation_rows)
    size_ratio = largest / smallest if smallest > 0 else 0.0

    if size_ratio > 2.0 or smallest < 0.5 * largest:
        return (
            f"Validation fold sizes are imbalanced: smallest fold = {smallest:,}, largest fold = {largest:,}, "
            f"ratio = {size_ratio:.2f}."
        )
    return None


def _fold_to_dict(fold: PurgedFold) -> dict[str, object]:
    return {
        "fold_id": fold.fold_id,
        "train_rows": fold.train_rows,
        "validation_rows": fold.validation_rows,
        "train_groups": fold.train_groups,
        "validation_groups": fold.validation_groups,
        "train_geohashes": fold.train_geohashes,
        "validation_geohashes": fold.validation_geohashes,
        "train_time_start": fold.train_time_start,
        "train_time_end": fold.train_time_end,
        "validation_time_start": fold.validation_time_start,
        "validation_time_end": fold.validation_time_end,
        "embargo_time_start": fold.embargo_time_start,
        "embargo_time_end": fold.embargo_time_end,
        "embargo_groups": fold.embargo_groups,
        "overlap_rows": fold.overlap_rows,
        "leakage_safe": fold.leakage_safe,
    }