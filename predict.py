"""Inference and submission generation script for traffic demand prediction."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from features import build_feature_bundle, FeatureBundle
from utils.data import load_data, LoadedData
from utils.io import load_config
from utils.logger import get_logger, setup_logging
from validation import run_validation_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate submission by averaging fold models.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/exp04_lag1_plus_rolling.yaml"),
        help="Path to the experiment YAML config file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    setup_logging(config.log_level)
    logger = get_logger("inference")

    logger.info("Loading data and generating features for inference...")
    
    # 1. Load Data
    bundle: LoadedData = load_data()
    
    # 2. Run Validation Pipeline to establish fold structure for feature engineering parity
    validation_run = run_validation_pipeline(
        bundle,
        artifact_dir=config.paths.artifacts_dir / "folds",
        report_path=config.paths.reports_dir / "validation_reports" / "fold_summary_inference.md",
    )

    # 3. Build Features (Test set uses full-train mapping for parity)
    feature_bundle: FeatureBundle = build_feature_bundle(
        bundle.train, 
        bundle.test, 
        validation_run, 
        enabled_features=config.features.enabled
    )
    
    # 4. Prepare Test Data
    # Categorical features must be handled identically to training (_prepare_frame in train_catboost.py)
    test_features = feature_bundle.test.copy()
    categorical_columns = list(feature_bundle.train_metadata.categorical_columns)
    
    for column in ["RoadType", "Weather"]:
        if column in test_features.columns:
            test_features[column] = test_features[column].astype("string").fillna("__MISSING__")
    
    test_pool = Pool(test_features, cat_features=categorical_columns)

    # 5. Aggregate Fold Predictions
    model_dir = config.paths.artifacts_dir / "models"
    fold_models = sorted(list(model_dir.glob("fold_*/model.cbm")))
    
    if not fold_models:
        logger.error("No fold models found in %s", model_dir)
        return

    logger.info("Found %d fold models. Starting inference...", len(fold_models))
    fold_predictions = []
    
    for model_path in fold_models:
        model = CatBoostRegressor()
        model.load_model(str(model_path))
        preds = model.predict(test_pool)
        fold_predictions.append(preds)

    # Average predictions across folds
    final_predictions = np.mean(fold_predictions, axis=0)
    
    # 6. Validation and Post-processing
    if len(final_predictions) != len(bundle.test):
        raise ValueError(f"Prediction mismatch: got {len(final_predictions)}, expected {len(bundle.test)}")
    
    if np.isnan(final_predictions).any():
        raise ValueError("Detected NaNs in predictions.")
        
    # Traffic demand cannot be negative
    if (final_predictions < 0).any():
        logger.warning("Clipping %d negative predictions to 0.0", (final_predictions < 0).sum())
        final_predictions = np.maximum(final_predictions, 0.0)

    # 7. Create Submission (Index and demand columns)
    submission_df = pd.DataFrame({
        "Index": bundle.test["Index"],
        "demand": final_predictions
    })

    output_path = config.paths.submission_dir / f"{args.config.stem}_submission.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(output_path, index=False)

    print(f"\nInference complete for {args.config.name}")
    print(f"Prediction stats: Mean={final_predictions.mean():.6f}, Min={final_predictions.min():.6f}, Max={final_predictions.max():.6f}")
    print(f"Submission rows: {len(submission_df)}")
    print(f"Output file: {output_path}")


if __name__ == "__main__":
    main()