"""Inference utilities."""

from typing import Dict

import joblib
import numpy as np

from ..features.preprocess import load_preprocessor, transform


class SklearnPredictor:
    """Wrapper for scikit-learn models with preprocessing."""

    def __init__(self, model, preprocessor: Dict[str, object]):
        self.model = model
        self.preprocessor = preprocessor

    @classmethod
    def from_artifacts(cls, model_path: str, preprocessor_path: str) -> "SklearnPredictor":
        model = joblib.load(model_path)
        preprocessor = load_preprocessor(preprocessor_path)
        return cls(model, preprocessor)

    def predict_proba(self, df) -> np.ndarray:
        X = transform(df, self.preprocessor)
        return self.model.predict_proba(X)

    def predict(self, df) -> np.ndarray:
        X = transform(df, self.preprocessor)
        return self.model.predict(X)
