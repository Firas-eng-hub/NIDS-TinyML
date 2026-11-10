"""Rebuild a run's dataset using its saved configuration and input fingerprints."""

import hashlib
import json

from ..config import load_run_config, run_metadata_dir
from .loader import list_csv_files, load_and_clean_data


def dataset_manifest(raw_dir):
    entries = []
    for filename in list_csv_files(str(raw_dir)):
        from pathlib import Path

        path = Path(filename)
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        entries.append({"name": path.name, "sha256": digest.hexdigest()})
    return entries


def rebuild_training_dataframe(cfg, root, run_name=None):
    if run_name:
        cfg = load_run_config(cfg, root, run_name)
        manifest_path = run_metadata_dir(cfg, root, run_name) / "dataset_manifest.json"
        if manifest_path.exists():
            expected = json.loads(manifest_path.read_text(encoding="utf-8"))
            if dataset_manifest(root / cfg["data"]["raw_dir"]) != expected:
                raise ValueError(
                    "Raw dataset differs from the training run; refusing to reuse split indices"
                )
    df, _ = load_and_clean_data(
        str(root / cfg["data"]["raw_dir"]),
        int(cfg["data"]["chunksize"]),
        cfg["features"]["fixed_list"],
    )
    random_state = int(cfg["train"]["random_state"])
    if float(cfg["data"]["sample_frac"]) < 1.0:
        df = df.sample(
            frac=float(cfg["data"]["sample_frac"]), random_state=random_state
        )
    if int(cfg["data"]["max_rows"]) > 0 and len(df) > int(cfg["data"]["max_rows"]):
        df = df.sample(n=int(cfg["data"]["max_rows"]), random_state=random_state)
    return df.reset_index(drop=True)


def load_split(df, splits_path, split_name):
    indices = json.loads(splits_path.read_text(encoding="utf-8"))[split_name]
    return df.iloc[indices].reset_index(drop=True)
