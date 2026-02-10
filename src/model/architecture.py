"""Model factory for configurable training backends."""

import inspect
import re
from typing import Dict, Iterable, List, Sequence

import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


class XGBoostStringLabelClassifier:
    """XGBoost wrapper that keeps string labels across predict/predict_proba APIs."""

    def __init__(self, class_names: Sequence[str], **xgb_params):
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "XGBoost backend is selected but 'xgboost' is not installed. "
                "Install dependencies with: pip install -r requirements.txt"
            ) from exc

        self.class_names = [str(c) for c in class_names]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.classes_ = np.asarray(self.class_names, dtype=object)
        self._model = XGBClassifier(**xgb_params)

    def _encode_labels(self, y: Iterable[object]) -> np.ndarray:
        encoded: List[int] = []
        missing = set()

        for label in y:
            key = str(label)
            idx = self.class_to_idx.get(key)
            if idx is None:
                missing.add(key)
                continue
            encoded.append(idx)

        if missing:
            missing_labels = ", ".join(sorted(missing))
            raise ValueError(f"Found unknown labels for XGBoost model: {missing_labels}")

        return np.asarray(encoded, dtype=np.int32)

    def fit(self, X, y, sample_weight=None):
        y_encoded = self._encode_labels(y)
        self._model.fit(X, y_encoded, sample_weight=sample_weight)
        return self

    def predict_proba(self, X) -> np.ndarray:
        probs = self._model.predict_proba(X)
        if probs.ndim != 2:
            raise ValueError("Expected 2D probability output from XGBoost.")

        if probs.shape[1] == len(self.class_names):
            return probs

        base_classes = np.asarray(getattr(self._model, "classes_", []), dtype=np.int32)
        if base_classes.size != probs.shape[1]:
            raise ValueError("XGBoost classes/probabilities shape mismatch.")

        aligned = np.zeros((probs.shape[0], len(self.class_names)), dtype=probs.dtype)
        for col_idx, cls_idx in enumerate(base_classes):
            aligned[:, int(cls_idx)] = probs[:, col_idx]
        return aligned

    def predict(self, X) -> np.ndarray:
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]

    def get_params(self, deep: bool = True) -> Dict[str, object]:
        params = dict(self._model.get_params(deep=deep))
        params["class_names"] = list(self.class_names)
        return params

    def set_params(self, **params):
        params = dict(params)
        class_names = params.pop("class_names", None)
        if class_names is not None:
            self.class_names = [str(c) for c in class_names]
            self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
            self.classes_ = np.asarray(self.class_names, dtype=object)
        self._model.set_params(**params)
        return self


def _build_random_forest(train_cfg: Dict[str, object]) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=int(train_cfg["rf_n_estimators"]),
        max_depth=None if train_cfg["rf_max_depth"] in (None, 0) else int(train_cfg["rf_max_depth"]),
        min_samples_leaf=int(train_cfg["rf_min_samples_leaf"]),
        n_jobs=int(train_cfg["rf_n_jobs"]),
        random_state=int(train_cfg["random_state"]),
        class_weight=None,
    )


def _build_xgboost(train_cfg: Dict[str, object], class_names: Sequence[str]) -> XGBoostStringLabelClassifier:
    return XGBoostStringLabelClassifier(
        class_names=class_names,
        objective="multi:softprob",
        num_class=len(class_names),
        eval_metric="mlogloss",
        n_estimators=int(train_cfg.get("xgb_n_estimators", 300)),
        max_depth=int(train_cfg.get("xgb_max_depth", 8)),
        learning_rate=float(train_cfg.get("xgb_learning_rate", 0.1)),
        subsample=float(train_cfg.get("xgb_subsample", 1.0)),
        colsample_bytree=float(train_cfg.get("xgb_colsample_bytree", 1.0)),
        min_child_weight=float(train_cfg.get("xgb_min_child_weight", 1.0)),
        reg_lambda=float(train_cfg.get("xgb_reg_lambda", 1.0)),
        n_jobs=int(train_cfg.get("xgb_n_jobs", -1)),
        random_state=int(train_cfg["random_state"]),
        tree_method=str(train_cfg.get("xgb_tree_method", "hist")),
        verbosity=int(train_cfg.get("xgb_verbosity", 1)),
    )


def _sklearn_major_minor() -> tuple:
    match = re.match(r"^(\d+)\.(\d+)", str(sklearn.__version__))
    if not match:
        return (0, 0)
    return (int(match.group(1)), int(match.group(2)))


def _build_logistic_regression(train_cfg: Dict[str, object]) -> LogisticRegression:
    penalty_cfg = str(train_cfg.get("lr_penalty", "l2")).strip().lower()
    penalty = None if penalty_cfg == "none" else penalty_cfg

    # Keep kwargs version-safe: sklearn has changed LogisticRegression args across releases.
    requested_kwargs = {
        "C": float(train_cfg.get("lr_c", 1.0)),
        "max_iter": int(train_cfg.get("lr_max_iter", 1000)),
        "solver": str(train_cfg.get("lr_solver", "lbfgs")),
        "multi_class": str(train_cfg.get("lr_multi_class", "auto")),
        "class_weight": None,
        "random_state": int(train_cfg["random_state"]),
    }
    skl_mm = _sklearn_major_minor()
    if skl_mm < (1, 8):
        requested_kwargs["penalty"] = penalty
        requested_kwargs["n_jobs"] = int(train_cfg.get("lr_n_jobs", -1))
    else:
        # On sklearn>=1.8, 'penalty' is deprecated and 'n_jobs' is ignored.
        # Keep defaults to avoid warnings while preserving default L2 behavior.
        if penalty is None:
            requested_kwargs["C"] = float("inf")

    signature = inspect.signature(LogisticRegression)
    supported = set(signature.parameters.keys())
    safe_kwargs = {k: v for k, v in requested_kwargs.items() if k in supported}
    return LogisticRegression(**safe_kwargs)


def build_model(config: Dict[str, object], class_names: Sequence[str]):
    train_cfg = config["train"]
    model_type = str(train_cfg.get("model_type", "xgboost")).strip().lower()

    if model_type == "random_forest":
        return _build_random_forest(train_cfg)

    if model_type == "xgboost":
        return _build_xgboost(train_cfg, class_names)

    if model_type == "logistic_regression":
        return _build_logistic_regression(train_cfg)

    raise ValueError(
        "Unsupported train.model_type. Expected 'xgboost', 'random_forest', or 'logistic_regression'."
    )
