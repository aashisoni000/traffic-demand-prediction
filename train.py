"""Single entrypoint for the competition skeleton.

Phase 0 is intentionally thin: load config, configure logging, seed all
random sources, and confirm the runtime started successfully.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from features import FeatureValidationError, build_feature_bundle, format_feature_summary
from utils.data import DataValidationError, load_data
from utils.io import ensure_output_directories, load_config
from utils.logger import get_logger, setup_logging
from utils.seed import set_deterministic_seed


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
        feature_bundle = build_feature_bundle(bundle.train, bundle.test)
    except FeatureValidationError:
        runtime_logger.exception("Baseline feature generation failed.")
        raise

    summary = format_feature_summary(feature_bundle)
    runtime_logger.info("Baseline feature summary: %s", summary)

    print(f"Feature pipeline ready for {config.project_name}.")
    print(summary)


if __name__ == "__main__":
    main()
