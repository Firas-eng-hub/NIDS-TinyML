"""Model training utilities."""

from typing import Dict, Tuple

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from .architecture import build_model


def compute_weights(y: np.ndarray, class_names: list) -> Dict[str, float]:
    class_names_arr = np.asarray(class_names, dtype=object)
    present = np.unique(y).astype(object)
    missing = [c for c in class_names_arr if c not in present]
    if missing:
        weights = compute_class_weight(class_weight="balanced", classes=present, y=y)
        weight_map = {cls: float(w) for cls, w in zip(present, weights)}
        for cls in missing:
            weight_map[cls] = 1.0
        return weight_map

    weights = compute_class_weight(class_weight="balanced", classes=class_names_arr, y=y)
    return {cls: float(w) for cls, w in zip(class_names_arr, weights)}


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: Dict[str, object],
    class_names: list,
) -> Tuple[object, Dict[str, float]]:
    model_type = str(config["train"].get("model_type", "xgboost")).strip().lower()
    class_weights = compute_weights(y_train, class_names)
    model = build_model(config, class_names)

    if model_type == "random_forest":
        model.set_params(class_weight=class_weights)
        model.fit(X_train, y_train)
        return model, class_weights

    if model_type == "xgboost":
        sample_weight = np.asarray([float(class_weights[str(label)]) for label in y_train], dtype=np.float32)
        model.fit(X_train, y_train, sample_weight=sample_weight)
        return model, class_weights

    if model_type == "logistic_regression":
        model.set_params(class_weight=class_weights)
        model.fit(X_train, y_train)
        return model, class_weights

    raise ValueError(
        "Unsupported train.model_type. Expected 'xgboost', 'random_forest', or 'logistic_regression'."
    )
