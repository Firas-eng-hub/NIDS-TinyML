"""Clean and map IDS2018 CSV chunks."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .label_map import map_label


def _find_column_case_insensitive(columns: List[str], target: str) -> Optional[str]:
    for col in columns:
        if col.strip().lower() == target.lower():
            return col
    return None


def clean_chunk(df: pd.DataFrame, feature_list: List[str]) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Clean a raw IDS2018 chunk and return a filtered dataframe + stats."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    label_col = _find_column_case_insensitive(list(df.columns), "Label")
    if label_col is None:
        raise ValueError("Label column not found in CSV chunk")

    timestamp_col = _find_column_case_insensitive(list(df.columns), "Timestamp")

    stats = {
        "rows_in": int(len(df)),
        "dropped_label": 0,
        "dropped_nan": 0,
        "rows_out": 0,
    }

    df["Label_mapped"] = df[label_col].apply(map_label)
    stats["dropped_label"] = int(df["Label_mapped"].isna().sum())
    df = df[df["Label_mapped"].notna()].copy()

    missing = [f for f in feature_list if f not in df.columns]
    if missing:
        raise ValueError(f"Missing required features in CSV: {missing}")

    for feat in feature_list:
        df[feat] = pd.to_numeric(df[feat], errors="coerce")

    finite_mask = np.isfinite(df[feature_list].values).all(axis=1)
    stats["dropped_nan"] = int((~finite_mask).sum())
    df = df[finite_mask].copy()

    df[feature_list] = df[feature_list].astype(np.float32)

    keep_cols = list(feature_list) + ["Label_mapped"]
    if timestamp_col:
        keep_cols.append(timestamp_col)

    df = df[keep_cols].copy()
    stats["rows_out"] = int(len(df))

    return df, stats
