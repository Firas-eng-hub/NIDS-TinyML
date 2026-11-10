"""Log1p + MinMax scaling utilities."""

import json
from typing import Dict, List

import numpy as np
import pandas as pd


def fit_preprocessor(
    df: pd.DataFrame, feature_order: List[str], skew_threshold: float
) -> Dict[str, object]:
    """Fit preprocessing parameters on training data."""
    log1p_features: List[str] = []
    for feat in feature_order:
        series = pd.to_numeric(df[feat], errors="coerce").astype(float)
        if series.min() < 0:
            continue
        skew_val = float(series.skew())
        if np.isnan(skew_val):
            skew_val = 0.0
        if skew_val > skew_threshold:
            log1p_features.append(feat)

    transformed = df[feature_order].copy()
    for feat in log1p_features:
        transformed[feat] = np.log1p(transformed[feat])

    min_vals = {k: float(v) for k, v in transformed.min().to_dict().items()}
    max_vals = {k: float(v) for k, v in transformed.max().to_dict().items()}

    return {
        "feature_order": list(feature_order),
        "log1p_features": log1p_features,
        "min_vals": min_vals,
        "max_vals": max_vals,
    }


def transform(df: pd.DataFrame, preprocessor: Dict[str, object]) -> np.ndarray:
    feature_order = preprocessor["feature_order"]
    log1p_features = set(preprocessor["log1p_features"])
    min_vals = preprocessor["min_vals"]
    max_vals = preprocessor["max_vals"]

    data = df[feature_order].copy()
    for feat in feature_order:
        if feat in log1p_features:
            data[feat] = np.log1p(data[feat])

        denom = max_vals[feat] - min_vals[feat]
        if denom == 0:
            denom = 1.0
        data[feat] = (data[feat] - min_vals[feat]) / denom

    return data.values.astype(np.float32)


def save_preprocessor(path: str, preprocessor: Dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(preprocessor, f, indent=2)


def load_preprocessor(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
