"""Feature engineering utilities for the competition skeleton."""

from .temporal import (
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_REPORT_PATH,
    FeatureBundle,
    FeatureMetadata,
    FeatureValidationError,
    add_fold_safe_grouped_means,
    add_fold_safe_lag_features,
    build_feature_bundle,
    build_feature_metadata,
    format_feature_summary,
    generate_features,
    save_feature_metadata,
    validate_feature_pair,
)

__all__ = [
    "DEFAULT_ARTIFACT_DIR",
    "DEFAULT_REPORT_PATH",
    "FeatureBundle",
    "FeatureMetadata",
    "FeatureValidationError",
    "add_fold_safe_grouped_means",
    "add_fold_safe_lag_features",
    "build_feature_bundle",
    "build_feature_metadata",
    "format_feature_summary",
    "generate_features",
    "save_feature_metadata",
    "validate_feature_pair",
]
