"""Single entrypoint for the competition skeleton.

Phase 0 is intentionally thin: load config, configure logging, seed all
random sources, and confirm the runtime started successfully.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from features import FeatureValidationError, build_feature_bundle, format_feature_summary
from models import CatBoostTrainingError, train_catboost_baseline
from utils.data import DataValidationError, load_data
from utils.io import ensure_output_directories, load_config
from utils.logger import get_logger, setup_logging
from utils.seed import set_deterministic_seed
from validation import format_validation_summary, run_validation_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the competition baseline skeleton.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/baseline.yaml"),
        help="Path to a YAML config file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    logger = setup_logging(config.log_level)
    set_deterministic_seed(config.seed)
    ensure_output_directories(config)

    runtime_logger = get_logger("train")
    runtime_logger.info("Loaded config for project '%s'.", config.project_name)
    runtime_logger.info("Deterministic seed initialized to %d.", config.seed)
    logger.info("Runtime directories are ready.")

    try:
        bundle = load_data()
    except DataValidationError:
        runtime_logger.exception("Phase 1 dataset validation failed.")
        raise

    try:
        validation_run = run_validation_pipeline(
            bundle,
            artifact_dir=config.paths.artifacts_dir / "folds",
            report_path=config.paths.reports_dir / "validation_reports" / "fold_summary.md",
        )
    except DataValidationError:
        runtime_logger.exception("Validation fold generation failed.")
        raise

    runtime_logger.info("Validation summary: %s", format_validation_summary(validation_run))

    try:
        feature_bundle = build_feature_bundle(
            bundle.train, bundle.test, validation_run, enabled_features=config.features.enabled
        )
    except FeatureValidationError:
        runtime_logger.exception("Baseline feature generation failed.")
        raise

    summary = format_feature_summary(feature_bundle)
    runtime_logger.info("Baseline feature summary: %s", summary)

    try:
        catboost_run = train_catboost_baseline(
            bundle,
            feature_bundle,
            validation_run,
            seed=config.seed,
            model_dir=config.paths.artifacts_dir / "models",
            oof_dir=config.paths.oof_dir / "catboost",
            report_path=config.paths.reports_dir / "baseline_reports" / "catboost_baseline.md",
        )
    except CatBoostTrainingError:
        runtime_logger.exception("CatBoost baseline training failed.")
        raise

    runtime_logger.info(
        "CatBoost baseline CV: mean=%.6f std=%.6f oof=%.6f runtime=%.2fs",
        catboost_run.mean_cv,
        catboost_run.fold_std,
        catboost_run.oof_rmse,
        catboost_run.runtime_seconds,
    )

    print(f"CatBoost baseline ready for {config.project_name}.")
    print(
        f"Mean CV: {catboost_run.mean_cv:.6f} | Fold std: {catboost_run.fold_std:.6f} | "
        f"OOF RMSE: {catboost_run.oof_rmse:.6f} | Runtime: {catboost_run.runtime_seconds:.2f}s | "
        f"Features: {catboost_run.feature_count}"
    )


if __name__ == "__main__":
    main()
