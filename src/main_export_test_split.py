"""Export the saved test split to CSV."""

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from .data.loader import load_and_clean_data
from .features.preprocess import load_preprocessor, transform


def load_config(config_path: Optional[str] = None) -> Tuple[dict, Path]:
    root = Path(__file__).resolve().parents[1]
    if config_path is None:
        config_path = root / "config" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    return cfg, root


def rebuild_training_dataframe(cfg: dict, root: Path) -> pd.DataFrame:
    feature_order = list(cfg["features"]["fixed_list"])
    raw_dir = root / cfg["data"]["raw_dir"]
    random_state = int(cfg["train"]["random_state"])

    df, _ = load_and_clean_data(str(raw_dir), int(cfg["data"]["chunksize"]), feature_order)

    sample_frac = float(cfg["data"]["sample_frac"])
    max_rows = int(cfg["data"]["max_rows"])

    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)
    if max_rows > 0 and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=random_state).reset_index(drop=True)

    return df.reset_index(drop=True)


def load_test_indices(splits_path: Path) -> list[int]:
    with open(splits_path, "r", encoding="utf-8") as handle:
        split_indices = json.load(handle)
    return list(split_indices["test"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--run_name", required=True, help="Run name used for preprocessing metadata")
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path. Defaults to data/processed/test_split_<run_name>.csv",
    )
    parser.add_argument(
        "--preprocessed",
        action="store_true",
        help="Export scaled/log1p-transformed features instead of raw cleaned feature values",
    )
    parser.add_argument(
        "--cubemx",
        action="store_true",
        help="Export only the 15 model input columns, ready for X-CUBE-AI desktop validation",
    )
    args = parser.parse_args()

    cfg, root = load_config(args.config)
    processed_dir = root / cfg["data"]["processed_dir"]
    splits_path = processed_dir / "splits.json"
    preprocessor_path = root / cfg["artifacts"]["scalers_dir"] / args.run_name / "preprocessor.json"

    if not splits_path.exists():
        raise FileNotFoundError(f"Split index file not found: {splits_path}")
    if args.preprocessed and not preprocessor_path.exists():
        raise FileNotFoundError(f"Preprocessor not found for run '{args.run_name}': {preprocessor_path}")

    print("Rebuilding cleaned dataset...")
    df = rebuild_training_dataframe(cfg, root)
    test_indices = load_test_indices(splits_path)
    test_df = df.iloc[test_indices].reset_index(drop=True)

    feature_order = list(cfg["features"]["fixed_list"])
    timestamp_cols = [c for c in test_df.columns if c.strip().lower() == "timestamp"]

    if args.preprocessed:
        preprocessor = load_preprocessor(str(preprocessor_path))
        scaled = transform(test_df, preprocessor)
        export_df = pd.DataFrame(scaled, columns=feature_order)
    else:
        export_df = test_df[feature_order].copy()

    finite_mask = np.isfinite(export_df.to_numpy()).all(axis=1)
    dropped_non_finite = int((~finite_mask).sum())
    if dropped_non_finite:
        export_df = export_df.loc[finite_mask].reset_index(drop=True)
        test_df = test_df.loc[finite_mask].reset_index(drop=True)
        print(f"Dropped rows with NaN/Inf after export transform: {dropped_non_finite}")

    if not args.cubemx:
        export_df["Label_mapped"] = test_df["Label_mapped"].values
        if timestamp_cols:
            export_df[timestamp_cols[0]] = test_df[timestamp_cols[0]].values

    if args.output:
        output_path = Path(args.output).expanduser()
        if not output_path.is_absolute():
            output_path = root / output_path
    else:
        if args.cubemx:
            suffix = "cubemx_preprocessed" if args.preprocessed else "cubemx_raw"
        else:
            suffix = "preprocessed" if args.preprocessed else "raw"
        output_path = processed_dir / f"test_split_{args.run_name}_{suffix}.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.cubemx:
        export_df.to_csv(output_path, index=False, header=False, float_format="%.8f")
    else:
        export_df.to_csv(output_path, index=False)

    print(f"Rows exported: {len(export_df)}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
