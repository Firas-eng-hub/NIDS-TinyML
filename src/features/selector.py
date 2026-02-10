"""Feature selector using the fixed list."""

from typing import List

import pandas as pd


def select_features(df: pd.DataFrame, feature_list: List[str]) -> pd.DataFrame:
    missing = [f for f in feature_list if f not in df.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    return df[feature_list].copy()
