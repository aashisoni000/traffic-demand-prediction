"""Typed configuration schema for the competition skeleton.

The schema is intentionally small for Phase 0: it validates the startup
configuration without pulling in any model or data logic yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


class ConfigError(ValueError):
    """Raised when the configuration file is missing or invalid."""


@dataclass(frozen=True, slots=True)
class PathConfig:
    """Resolved runtime paths used by the pipeline."""

    artifacts_dir: Path
    oof_dir: Path
    reports_dir: Path
    logs_dir: Path


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Top-level application config for the baseline pipeline."""

    project_name: str
    seed: int
    log_level: str
    paths: PathConfig


def _require_mapping(data: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(data, Mapping):
        raise ConfigError(f"Expected {context} to be a mapping, got {type(data).__name__}.")
    return data


def _resolve_path(base_dir: Path, raw_value: Any, *, field_name: str) -> Path:
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise ConfigError(f"Field '{field_name}' must be a non-empty string path.")
    path = Path(raw_value).expanduser()
    return path if path.is_absolute() else (base_dir / path).resolve()


def load_app_config(data: Mapping[str, Any], *, base_dir: Path) -> AppConfig:
    """Convert a raw config mapping into a validated dataclass tree."""

    project_name = str(data.get("project_name", "traffic_demand_prediction")).strip()
    if not project_name:
        raise ConfigError("Field 'project_name' must not be empty.")

    try:
        seed = int(data.get("seed", 42))
    except (TypeError, ValueError) as exc:
        raise ConfigError("Field 'seed' must be an integer.") from exc

    log_level = str(data.get("log_level", "INFO")).strip().upper()
    if not log_level:
        raise ConfigError("Field 'log_level' must not be empty.")

    paths_data = _require_mapping(data.get("paths", {}), context="'paths'")
    paths = PathConfig(
        artifacts_dir=_resolve_path(base_dir, paths_data.get("artifacts_dir", "artifacts"), field_name="paths.artifacts_dir"),
        oof_dir=_resolve_path(base_dir, paths_data.get("oof_dir", "oof"), field_name="paths.oof_dir"),
        reports_dir=_resolve_path(base_dir, paths_data.get("reports_dir", "reports"), field_name="paths.reports_dir"),
        logs_dir=_resolve_path(base_dir, paths_data.get("logs_dir", "artifacts/logs"), field_name="paths.logs_dir"),
    )

    return AppConfig(
        project_name=project_name,
        seed=seed,
        log_level=log_level,
        paths=paths,
    )
