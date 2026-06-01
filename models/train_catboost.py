"""Deterministic CatBoost baseline training for the traffic demand prediction project."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from features import FeatureBundle
from utils.data import LoadedData
from utils.logger import get_logger
from validation import PurgedFold, ValidationRun

LOGGER = get_logger("models.catboost")

TARGET_COLUMN: Final[str] = "demand"
MODEL_DIR: Final[Path] = Path("artifacts/models")
OOF_DIR: Final[Path] = Path("oof/catboost")
REPORT_PATH: Final[Path] = Path("reports/baseline_reports/catboost_baseline.md")
FOLD_METRICS_FILENAME: Final[str] = "catboost_fold_metrics.csv"
OOF_FILENAME: Final[str] = "oof_predictions.csv"
PARAMS_FILENAME: Final[str] = "catboost_params.json"
MODEL_PREFIX: Final[str] = "catboost_fold_"
DEFAULT_PARAMS: Final[dict[str, object]] = {
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "iterations": 500,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "random_seed": 42,
    "bootstrap_type": "No",
    "random_strength": 0.0,
    "od_type": "Iter",
    "od_wait": 50,
    "allow_writing_files": False,
    "verbose": False,
    "thread_count": 1,
}


class CatBoostTrainingError(ValueError):
    """Raised when the CatBoost baseline fails a hard validation rule."""


@dataclass(frozen=True, slots=True)
class FoldMetric:
    """Per-fold CatBoost metrics and artifact locations."""

    fold_id: int
    rmse: float
    train_rows: int
    validation_rows: int
    train_runtime_seconds: float
    best_iteration: int
    model_path: Path


@dataclass(frozen=True, slots=True)
class CatBoostBaselineRun:
    """Complete result bundle for the CatBoost baseline."""

    fold_metrics: tuple[FoldMetric, ...]
    mean_cv: float
    fold_std: float
    oof_rmse: float
    oof_coverage_rows: int
    oof_coverage_ratio: float
    runtime_seconds: float
    feature_count: int
    params: dict[str, object]
    model_dir: Path
    oof_dir: Path
    report_path: Path
    fold_metrics_path: Path
    oof_predictions_path: Path


def train_catboost_baseline(
    data: LoadedData,
    feature_bundle: FeatureBundle,
    validation_run: ValidationRun,
    *,
    seed: int,
    model_dir: Path | str = MODEL_DIR,
    oof_dir: Path | str = OOF_DIR,
    report_path: Path | str = REPORT_PATH,
    params: dict[str, object] | None = None,
) -> CatBoostBaselineRun:
    """Train fold-safe CatBoost baseline and generate OOF predictions."""

    runtime_start = time.perf_counter()
    train_features = feature_bundle.train.reset_index(drop=True)
    test_features = feature_bundle.test.reset_index(drop=True)
    target = pd.to_numeric(data.train[TARGET_COLUMN], errors="raise").reset_index(drop=True)
    params = _build_params(seed=seed, params=params)

    _validate_feature_parity(train_features, test_features)
    _validate_folds(validation_run.folds, len(train_features))

    model_root = Path(model_dir).expanduser().resolve()
    oof_root = Path(oof_dir).expanduser().resolve()
    report_file = Path(report_path).expanduser().resolve()
    model_root.mkdir(parents=True, exist_ok=True)
    oof_root.mkdir(parents=True, exist_ok=True)
    report_file.parent.mkdir(parents=True, exist_ok=True)

    categorical_features = list(feature_bundle.train_metadata.categorical_columns)
    oof_predictions = pd.DataFrame(
        {
            "row_id": np.arange(len(train_features), dtype=int),
            "target": target.astype(float),
            "prediction": np.nan,
            "fold_id": -1,
        }
    )

    fold_metrics: list[FoldMetric] = []
    for fold in validation_run.folds:
        fold_metric, fold_predictions = _train_single_fold(
            fold=fold,
            train_features=train_features,
            target=target,
            categorical_features=categorical_features,
            params=params,
            model_root=model_root,
        )
        fold_metrics.append(fold_metric)
        validation_index = list(fold.validation_row_ids)
        oof_predictions.loc[validation_index, "prediction"] = fold_predictions
        oof_predictions.loc[validation_index, "fold_id"] = fold.fold_id

    validate_oof_alignment(oof_predictions, len(train_features))

    covered_oof = oof_predictions[oof_predictions["prediction"].notna()].copy()
    validate_no_nan_predictions(covered_oof["prediction"])

    fold_rmse_values = [metric.rmse for metric in fold_metrics]
    mean_cv = float(np.mean(fold_rmse_values))
    fold_std = float(np.std(fold_rmse_values, ddof=1)) if len(fold_rmse_values) > 1 else 0.0
    oof_rmse = evaluate_rmse(covered_oof["target"], covered_oof["prediction"])
    oof_coverage_rows = int(len(covered_oof))
    oof_coverage_ratio = float(oof_coverage_rows / len(train_features)) if len(train_features) else 0.0
    runtime_seconds = float(time.perf_counter() - runtime_start)

    fold_metrics_path = model_root / FOLD_METRICS_FILENAME
    pd.DataFrame(
        [
            {
                "fold_id": metric.fold_id,
                "rmse": metric.rmse,
                "train_rows": metric.train_rows,
                "validation_rows": metric.validation_rows,
                "train_runtime_seconds": metric.train_runtime_seconds,
                "best_iteration": metric.best_iteration,
                "model_path": str(metric.model_path),
            }
            for metric in fold_metrics
        ]
    ).to_csv(fold_metrics_path, index=False)

    oof_predictions_path = oof_root / OOF_FILENAME
    oof_predictions.to_csv(oof_predictions_path, index=False)

    params_path = model_root / PARAMS_FILENAME
    params_path.write_text(json.dumps(params, indent=2), encoding="utf-8")

    report_file.write_text(
        _render_report(
            fold_metrics=fold_metrics,
            mean_cv=mean_cv,
            fold_std=fold_std,
            oof_rmse=oof_rmse,
            oof_coverage_rows=oof_coverage_rows,
            oof_coverage_ratio=oof_coverage_ratio,
            runtime_seconds=runtime_seconds,
            feature_count=len(train_features.columns),
            feature_names=list(train_features.columns),
            params=params,
            fold_metrics_path=fold_metrics_path,
            oof_predictions_path=oof_predictions_path,
            params_path=params_path,
        ),
        encoding="utf-8",
    )

    LOGGER.info("Saved CatBoost fold metrics to %s.", fold_metrics_path)
    LOGGER.info("Saved CatBoost OOF predictions to %s.", oof_predictions_path)
    LOGGER.info("Wrote CatBoost baseline report to %s.", report_file)

    return CatBoostBaselineRun(
        fold_metrics=tuple(fold_metrics),
        mean_cv=mean_cv,
        fold_std=fold_std,
        oof_rmse=oof_rmse,
        oof_coverage_rows=oof_coverage_rows,
        oof_coverage_ratio=oof_coverage_ratio,
        runtime_seconds=runtime_seconds,
        feature_count=len(train_features.columns),
        feature_names=list(train_features.columns),
        params=params,
        model_dir=model_root,
        oof_dir=oof_root,
        report_path=report_file,
        fold_metrics_path=fold_metrics_path,
        oof_predictions_path=oof_predictions_path,
    )


def train_single_fold(
    *,
    fold: PurgedFold,
    train_features: pd.DataFrame,
    target: pd.Series,
    categorical_features: Sequence[str],
    params: dict[str, object],
    model_root: Path,
) -> FoldMetric:
    """Train one CatBoost fold and save its model artifact."""

    fold_metric, _ = _train_single_fold(
        fold=fold,
        train_features=train_features,
        target=target,
        categorical_features=categorical_features,
        params=params,
        model_root=model_root,
    )
    return fold_metric


def predict_fold_oof(model_path: Path, validation_features: pd.DataFrame, categorical_features: Sequence[str]) -> np.ndarray:
    """Load a saved fold model and generate validation predictions."""

    model = CatBoostRegressor()
    model.load_model(str(model_path))
    validation_pool = Pool(_prepare_frame(validation_features), cat_features=list(categorical_features))
    predictions = np.asarray(model.predict(validation_pool), dtype=float)
    validate_no_nan_predictions(pd.Series(predictions))
    return predictions


def evaluate_rmse(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    """Compute RMSE with alignment checks."""

    truth = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if truth.shape != pred.shape:
        raise CatBoostTrainingError("RMSE evaluation mismatch: prediction and target lengths differ.")
    return float(np.sqrt(np.mean((truth - pred) ** 2)))


def validate_train_test_features(train_features: pd.DataFrame, test_features: pd.DataFrame) -> None:
    """Validate train/test feature parity before model training."""

    _validate_feature_parity(train_features, test_features)


def validate_oof_alignment(oof_predictions: pd.DataFrame, expected_rows: int) -> None:
    """Validate that OOF rows are aligned and complete."""

    if len(oof_predictions) != expected_rows:
        raise CatBoostTrainingError("OOF alignment mismatch: row count does not match training rows.")
    if oof_predictions["row_id"].duplicated().any():
        raise CatBoostTrainingError("OOF alignment mismatch: duplicate row ids were detected.")
    if oof_predictions["prediction"].notna().sum() == 0:
        raise CatBoostTrainingError("OOF alignment mismatch: no OOF predictions were generated.")
    if (oof_predictions.loc[oof_predictions["prediction"].notna(), "fold_id"] < 0).any():
        raise CatBoostTrainingError("OOF alignment mismatch: predicted rows are missing fold ids.")


def validate_no_nan_predictions(predictions: pd.Series | np.ndarray) -> None:
    """Validate that prediction vectors do not contain NaN values."""

    values = pd.Series(predictions)
    if values.isna().any():
        raise CatBoostTrainingError("NaN predictions detected in CatBoost output.")


def _train_single_fold(
    *,
    fold: PurgedFold,
    train_features: pd.DataFrame,
    target: pd.Series,
    categorical_features: Sequence[str],
    params: dict[str, object],
    model_root: Path,
) -> tuple[FoldMetric, np.ndarray]:
    train_index = list(fold.train_row_ids)
    validation_index = list(fold.validation_row_ids)
    _validate_fold_leakage(fold, train_index, validation_index)

    train_frame = _prepare_frame(train_features.iloc[train_index])
    validation_frame = _prepare_frame(train_features.iloc[validation_index])
    train_target = target.iloc[train_index].astype(float)
    validation_target = target.iloc[validation_index].astype(float)

    train_pool = Pool(train_frame, label=train_target, cat_features=list(categorical_features))
    validation_pool = Pool(validation_frame, label=validation_target, cat_features=list(categorical_features))

    model = CatBoostRegressor(**params)
    fold_dir = model_root / f"fold_{fold.fold_id:02d}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    model_path = fold_dir / "model.cbm"

    start_time = time.perf_counter()
    model.fit(train_pool, eval_set=validation_pool, use_best_model=True)
    train_runtime_seconds = float(time.perf_counter() - start_time)

    predictions = np.asarray(model.predict(validation_pool), dtype=float)
    validate_no_nan_predictions(pd.Series(predictions))
    rmse = evaluate_rmse(validation_target, predictions)

    model.save_model(str(model_path))

    return (
        FoldMetric(
            fold_id=fold.fold_id,
            rmse=rmse,
            train_rows=len(train_index),
            validation_rows=len(validation_index),
            train_runtime_seconds=train_runtime_seconds,
            best_iteration=int(model.get_best_iteration() or model.tree_count_),
            model_path=model_path,
        ),
        predictions,
    )


def _build_params(*, seed: int, params: dict[str, object] | None) -> dict[str, object]:
    merged = dict(DEFAULT_PARAMS)
    if params:
        merged.update(params)
    merged["random_seed"] = seed
    return merged


def _prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    for column in ["RoadType", "Weather"]:
        prepared[column] = prepared[column].astype("string").fillna("__MISSING__")
    return prepared


def _validate_feature_parity(train_features: pd.DataFrame, test_features: pd.DataFrame) -> None:
    if list(train_features.columns) != list(test_features.columns):
        raise CatBoostTrainingError("Train/test feature mismatch: feature columns differ.")
    if train_features.dtypes.astype(str).to_dict() != test_features.dtypes.astype(str).to_dict():
        raise CatBoostTrainingError("Train/test feature mismatch: feature dtypes differ.")
    missing_features = [column for column in train_features.columns if train_features[column].isna().all()]
    if missing_features:
        raise CatBoostTrainingError(f"Missing features detected in training data: {missing_features}.")


def _validate_folds(folds: Sequence[PurgedFold], total_rows: int) -> None:
    if not folds:
        raise CatBoostTrainingError("No validation folds were provided.")

    seen_validation_rows: set[int] = set()
    for fold in folds:
        if not fold.leakage_safe:
            raise CatBoostTrainingError(f"Fold {fold.fold_id} failed leakage safety checks.")
        train_rows = set(fold.train_row_ids)
        validation_rows = set(fold.validation_row_ids)
        if train_rows.intersection(validation_rows):
            raise CatBoostTrainingError(f"Fold {fold.fold_id} has overlapping train and validation rows.")
        if any(row_id < 0 or row_id >= total_rows for row_id in train_rows.union(validation_rows)):
            raise CatBoostTrainingError(f"Fold {fold.fold_id} contains row ids outside the training frame.")
        if seen_validation_rows.intersection(validation_rows):
            raise CatBoostTrainingError(f"Validation folds overlap across folds at fold {fold.fold_id}.")
        seen_validation_rows.update(validation_rows)


def _validate_fold_leakage(fold: PurgedFold, train_index: Sequence[int], validation_index: Sequence[int]) -> None:
    if set(train_index).intersection(validation_index):
        raise CatBoostTrainingError(f"Fold {fold.fold_id} has overlapping train and validation rows.")
    if not fold.leakage_safe:
        raise CatBoostTrainingError(f"Fold {fold.fold_id} is not leakage safe.")


def _render_report(
    *,
    fold_metrics: Sequence[FoldMetric],
    mean_cv: float,
    fold_std: float,
    oof_rmse: float,
    oof_coverage_rows: int,
    oof_coverage_ratio: float,
    runtime_seconds: float,
    feature_count: int,
    feature_names: Sequence[str],
    params: dict[str, object],
    fold_metrics_path: Path,
    oof_predictions_path: Path,
    params_path: Path,
) -> str:
    lines = [
        "# CatBoost Baseline",
        "",
        "## Summary",
        "",
        f"- Fold RMSE: {', '.join(f'{metric.rmse:.6f}' for metric in fold_metrics)}",
        f"- Mean CV: {mean_cv:.6f}",
        f"- Fold std: {fold_std:.6f}",
        f"- OOF RMSE: {oof_rmse:.6f}",
        f"- OOF coverage rows: {oof_coverage_rows:,}",
        f"- OOF coverage ratio: {oof_coverage_ratio:.2%}",
        f"- Training runtime: {runtime_seconds:.2f} seconds",
        f"- Feature count: {feature_count}",
        f"- Enabled features: {', '.join(feature_names)}",
        f"- Fold metrics: {fold_metrics_path}",
        f"- OOF predictions: {oof_predictions_path}",
        f"- CatBoost parameters: {params_path}",
        "",
        "## Fold Scores",
        "",
        "| Fold | RMSE | Train rows | Validation rows | Train runtime (s) | Best iteration |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for metric in fold_metrics:
        lines.append(
            f"| {metric.fold_id} | {metric.rmse:.6f} | {metric.train_rows:,} | {metric.validation_rows:,} | {metric.train_runtime_seconds:.2f} | {metric.best_iteration:,} |"
        )

    lines.extend([
        "",
        "## CatBoost Parameters",
        "",
        "```json",
        json.dumps(params, indent=2),
        "```",
        "",
    ])
    return "\n".join(lines).rstrip() + "\n"