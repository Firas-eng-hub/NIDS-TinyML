"""CSV loading utilities."""

from typing import Dict, Generator, List, Tuple

import pandas as pd

from .cleaner import clean_chunk


def list_csv_files(raw_dir: str) -> List[str]:
    from pathlib import Path

    directory = Path(raw_dir)
    return sorted(
        str(p)
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() == ".csv"
    )


def iter_csv_chunks(
    paths: List[str], chunksize: int
) -> Generator[pd.DataFrame, None, None]:
    for path in paths:
        for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
            yield chunk


def load_and_clean_data(
    raw_dir: str, chunksize: int, feature_list: List[str]
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    paths = list_csv_files(raw_dir)
    if not paths:
        raise FileNotFoundError(f"No CSV files found in: {raw_dir}")

    all_chunks = []
    stats_total = {"rows_in": 0, "dropped_label": 0, "dropped_nan": 0, "rows_out": 0}

    for chunk in iter_csv_chunks(paths, chunksize):
        cleaned, stats = clean_chunk(chunk, feature_list)
        if not cleaned.empty:
            all_chunks.append(cleaned)

        for key in stats_total:
            stats_total[key] += stats.get(key, 0)

    if not all_chunks:
        raise ValueError("No valid rows after cleaning")

    df = pd.concat(all_chunks, ignore_index=True)
    return df, stats_total
