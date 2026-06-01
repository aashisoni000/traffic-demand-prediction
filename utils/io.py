"""Config and filesystem helpers for the competition skeleton."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from .schema import AppConfig, ConfigError, load_app_config


def read_yaml(path: str | Path) -> Mapping[str, Any]:
    """Read a YAML file and return the raw mapping."""

    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise ConfigError(f"Config file does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise ConfigError(f"Config file must contain a mapping at the top level: {config_path}")
    return data


def load_config(path: str | Path) -> AppConfig:
    """Load and validate the baseline config.

    Config parsing stays isolated here so the entrypoint remains a thin
    orchestration layer.
    """

    config_path = Path(path).expanduser().resolve()
    raw_config = read_yaml(config_path)
    # The config file lives under configs/, so the project root is its parent.
    project_root = config_path.parent.parent
    return load_app_config(raw_config, base_dir=project_root)


def ensure_output_directories(config: AppConfig) -> None:
    """Create the standard artifact folders if they do not exist."""

    for directory in (
        config.paths.artifacts_dir,
        config.paths.oof_dir,
        config.paths.reports_dir,
        config.paths.logs_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
