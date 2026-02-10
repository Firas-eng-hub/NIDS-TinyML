"""Train/val/test splitting utilities."""

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def stratified_split(
    df: pd.DataFrame,
    label_col: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, list]]:
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Split ratios must sum to 1.0")

    df = df.reset_index(drop=True)
    labels = df[label_col].values
    indices = np.arange(len(df))

    temp_ratio = val_ratio + test_ratio
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=temp_ratio,
        stratify=labels,
        random_state=random_state,
    )

    val_share = val_ratio / temp_ratio
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=1 - val_share,
        stratify=labels[temp_idx],
        random_state=random_state,
    )

    split_indices = {
        "train": train_idx.tolist(),
        "val": val_idx.tolist(),
        "test": test_idx.tolist(),
    }

    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()
    test_df = df.iloc[test_idx].copy()

    return train_df, val_df, test_df, split_indices
