"""Shared configuration and artifact paths."""

import json
import warnings
from pathlib import Path

import yaml

from .features.feature_list import FEATURE_LIST


def load_config(config_path=None):
    root = Path(__file__).resolve().parents[1]
    path = Path(config_path) if config_path else root / "config/config.yaml"
    with path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if cfg["features"]["fixed_list"] != FEATURE_LIST:
        raise ValueError("features.fixed_list must match the canonical feature order")
    return cfg, root


def run_metadata_dir(cfg, root, run_name):
    return root / cfg["artifacts"]["metadata_dir"] / run_name


def resolve_split_path(cfg, root, run_name):
    path = run_metadata_dir(cfg, root, run_name) / "splits.json"
    if path.exists():
        return path
    warnings.warn(
        f"Legacy run {run_name!r} has no run-specific split. Using local shared "
        "splits.json; its correspondence to the original run is unverified.",
        RuntimeWarning,
        stacklevel=2,
    )
    return root / cfg["data"]["processed_dir"] / "splits.json"


def resolve_run_paths(cfg, root, run_name):
    return (
        *resolve_model_paths(cfg, root, run_name),
        resolve_split_path(cfg, root, run_name),
    )


def resolve_model_paths(cfg, root, run_name):
    return (
        root / cfg["artifacts"]["models_dir"] / run_name / "model.joblib",
        root / cfg["artifacts"]["scalers_dir"] / run_name / "preprocessor.json",
        run_metadata_dir(cfg, root, run_name) / "metadata.json",
    )


def resolve_artifacts(cfg, root, run_name=None):
    model_dir = root / cfg["artifacts"]["models_dir"]
    scaler_dir = root / cfg["artifacts"]["scalers_dir"]
    if run_name:
        model_dir, scaler_dir = model_dir / run_name, scaler_dir / run_name
    return model_dir / "model.joblib", scaler_dir / "preprocessor.json"


def load_run_config(cfg, root, run_name):
    path = run_metadata_dir(cfg, root, run_name) / "config.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    warnings.warn(
        f"Legacy run {run_name!r} has no config snapshot; using current config.",
        RuntimeWarning,
        stacklevel=2,
    )
    return cfg
