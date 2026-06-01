"""Single entrypoint for the competition skeleton.

Phase 0 is intentionally thin: load config, configure logging, seed all
random sources, and confirm the runtime started successfully.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from utils.data import DataValidationError, format_dataset_summary, load_data, write_dataset_reports
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

    write_dataset_reports(bundle, config.paths.reports_dir)
    summary = format_dataset_summary(bundle)
    runtime_logger.info("Phase 1 dataset summary: %s", summary)

    validation_run = run_validation_pipeline(
        bundle,
        artifact_dir=config.paths.artifacts_dir / "folds",
        report_path=config.paths.reports_dir / "validation_reports" / "fold_summary.md",
    )
    validation_summary = format_validation_summary(validation_run)
    runtime_logger.info("Phase 2 validation summary: %s", validation_summary)

    print(f"Validation pipeline ready for {config.project_name}.")
    print(validation_summary)


if __name__ == "__main__":
    main()
