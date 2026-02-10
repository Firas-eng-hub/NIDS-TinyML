"""End-to-end training pipeline."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shutil
import warnings

import numpy as np
import yaml

from .data.loader import load_and_clean_data
from .data.split import stratified_split
from .data.label_map import TARGET_CLASSES, LABEL_MAPPING_INFO
from .features.preprocess import fit_preprocessor, transform, save_preprocessor
from .model.train import train_model
from .model.evaluate import evaluate_model
from .model.export import export_model, export_metadata


warnings.filterwarnings(
    "ignore",
    message=(
        r"`sklearn\.utils\.parallel\.delayed` should be used with "
        r"`sklearn\.utils\.parallel\.Parallel`.*"
    ),
    category=UserWarning,
    module=r"sklearn\.utils\.parallel",
)


def load_config(config_path: str = None):
    root = Path(__file__).resolve().parents[1]
    if config_path is None:
        config_path = root / "config" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg, root


def _to_json_safe(value):
    if isinstance(value, dict):
        return {str(k): _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_to_json_safe(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _extract_model_params(model) -> dict:
    if not hasattr(model, "get_params"):
        return {}
    return _to_json_safe(model.get_params(deep=False))


def _normalize_run_name(run_name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", run_name).strip("._")
    if not cleaned:
        raise ValueError("Run name is empty after normalization. Use letters, digits, '.', '_' or '-'.")
    return cleaned


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--run_name", default=None, help="Optional run name for versioned artifacts")
    args = parser.parse_args()

    cfg, root = load_config(args.config)

    feature_order = cfg["features"]["fixed_list"]
    skew_threshold = float(cfg["features"]["log1p_skew_threshold"])
    model_type = str(cfg["train"].get("model_type", "xgboost")).strip().lower()
    random_state = int(cfg["train"]["random_state"])

    raw_dir = root / cfg["data"]["raw_dir"]
    processed_dir = root / cfg["data"]["processed_dir"]
    models_dir = root / cfg["artifacts"]["models_dir"]
    reports_dir = root / cfg["artifacts"]["reports_dir"]
    scaler_dir = root / cfg["artifacts"]["scalers_dir"]
    metadata_dir = root / cfg["artifacts"]["metadata_dir"]

    run_name = args.run_name
    if run_name is None:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        run_name = f"{model_type}_{timestamp}"
    run_name = _normalize_run_name(run_name)

    run_models_dir = models_dir / run_name
    run_reports_dir = reports_dir / run_name
    run_scaler_dir = scaler_dir / run_name
    run_metadata_dir = metadata_dir / run_name

    for run_dir in (run_models_dir, run_reports_dir, run_scaler_dir, run_metadata_dir):
        if run_dir.exists():
            raise FileExistsError(f"Run artifact directory already exists: {run_dir}")

    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(scaler_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)
    os.makedirs(run_models_dir, exist_ok=True)
    os.makedirs(run_reports_dir, exist_ok=True)
    os.makedirs(run_scaler_dir, exist_ok=True)
    os.makedirs(run_metadata_dir, exist_ok=True)

    print(f"Run name: {run_name}")

    print("Loading and cleaning CSVs...")
    df, stats = load_and_clean_data(str(raw_dir), int(cfg["data"]["chunksize"]), feature_order)
    print(f"Rows in: {stats['rows_in']}")
    print(f"Dropped (label): {stats['dropped_label']}")
    print(f"Dropped (nan/inf): {stats['dropped_nan']}")
    print(f"Rows out: {stats['rows_out']}")

    sample_frac = float(cfg["data"]["sample_frac"])
    max_rows = int(cfg["data"]["max_rows"])

    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)
        print(f"Sampled fraction: {sample_frac}, rows: {len(df)}")

    if max_rows > 0 and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=random_state).reset_index(drop=True)
        print(f"Capped rows: {len(df)}")

    df = df.reset_index(drop=True)

    train_df, val_df, test_df, split_indices = stratified_split(
        df,
        label_col="Label_mapped",
        train_ratio=float(cfg["split"]["train_ratio"]),
        val_ratio=float(cfg["split"]["val_ratio"]),
        test_ratio=float(cfg["split"]["test_ratio"]),
        random_state=random_state,
    )

    splits_path = Path(processed_dir) / "splits.json"
    with open(splits_path, "w", encoding="utf-8") as f:
        json.dump(split_indices, f, indent=2)

    print("Fitting preprocessor...")
    preprocessor = fit_preprocessor(train_df[feature_order], feature_order, skew_threshold)

    preproc_run_path = run_scaler_dir / "preprocessor.json"
    preproc_latest_path = scaler_dir / "preprocessor.json"
    save_preprocessor(str(preproc_run_path), preprocessor)
    shutil.copy2(preproc_run_path, preproc_latest_path)

    print("Transforming features...")
    X_train = transform(train_df, preprocessor)
    X_val = transform(val_df, preprocessor)
    X_test = transform(test_df, preprocessor)

    y_train = train_df["Label_mapped"].values
    y_val = val_df["Label_mapped"].values
    y_test = test_df["Label_mapped"].values

    print(f"Training model backend: {model_type}")
    model, class_weights = train_model(X_train, y_train, cfg, TARGET_CLASSES)

    model_run_path = run_models_dir / "model.joblib"
    model_latest_path = models_dir / "model.joblib"
    export_model(model, str(model_run_path))
    shutil.copy2(model_run_path, model_latest_path)

    print("Evaluating on validation split...")
    val_report = evaluate_model(model, X_val, y_val, TARGET_CLASSES, str(run_reports_dir), prefix="val")

    print("Evaluating on test split...")
    test_report = evaluate_model(model, X_test, y_test, TARGET_CLASSES, str(run_reports_dir), prefix="test")

    report_artifact_keys = (
        "report_path",
        "confusion_matrix_path",
        "roc_curve_path",
        "pr_curve_path",
    )
    for report_obj in (val_report, test_report):
        for key in report_artifact_keys:
            src_str = report_obj.get(key)
            if not src_str:
                continue
            src = Path(src_str)
            if src.exists():
                shutil.copy2(src, reports_dir / src.name)

    label_map_run_path = run_metadata_dir / "label_map.json"
    with open(label_map_run_path, "w", encoding="utf-8") as f:
        json.dump(LABEL_MAPPING_INFO, f, indent=2)
    shutil.copy2(label_map_run_path, metadata_dir / "label_map.json")

    feature_order_run_path = run_metadata_dir / "feature_order.json"
    with open(feature_order_run_path, "w", encoding="utf-8") as f:
        json.dump(feature_order, f, indent=2)
    shutil.copy2(feature_order_run_path, metadata_dir / "feature_order.json")

    metadata = {
        "run_name": run_name,
        "feature_order": feature_order,
        "log1p_features": preprocessor["log1p_features"],
        "min_vals": preprocessor["min_vals"],
        "max_vals": preprocessor["max_vals"],
        "class_names": TARGET_CLASSES,
        "model_type": model_type,
        "train_config": _to_json_safe(cfg["train"]),
        "model_params": _extract_model_params(model),
        "class_weights": class_weights,
        "label_map": LABEL_MAPPING_INFO,
        "model_path": str(model_run_path),
        "preprocessor_path": str(preproc_run_path),
        "run_reports_dir": str(run_reports_dir),
        "latest_model_path": str(model_latest_path),
        "latest_preprocessor_path": str(preproc_latest_path),
        "test_report": test_report,
    }

    run_metadata_path = run_metadata_dir / "metadata.json"
    export_metadata(metadata, str(run_metadata_path))
    shutil.copy2(run_metadata_path, metadata_dir / "metadata.json")

    print("Saving processed dataset for streaming...")
    timestamp_cols = [c for c in df.columns if c.strip().lower() == "timestamp"]
    keep_cols = list(feature_order) + ["Label_mapped"]
    if timestamp_cols:
        keep_cols.append(timestamp_cols[0])

    processed_path = Path(processed_dir) / "processed.csv"
    df[keep_cols].to_csv(processed_path, index=False)

    print(f"Done. Versioned artifacts saved under: {run_name}")


if __name__ == "__main__":
    main()
