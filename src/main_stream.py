"""Stream test set rows into the dashboard backend."""

import argparse
import time
import warnings
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import requests

from .config import load_config
from .features.preprocess import load_preprocessor, transform

warnings.filterwarnings(
    "ignore",
    message=(
        r"`sklearn\.utils\.parallel\.delayed` should be used with "
        r"`sklearn\.utils\.parallel\.Parallel`.*"
    ),
    category=UserWarning,
    module=r"sklearn\.utils\.parallel",
)


def find_timestamp_column(columns):
    for c in columns:
        if c.strip().lower() == "timestamp":
            return c
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument(
        "--run_name",
        default=None,
        help="Use artifacts from a specific run (e.g., rf_run_01) instead of latest files",
    )
    parser.add_argument("--speed", type=float, default=None, help="Events per second")
    parser.add_argument("--min_confidence", type=float, default=None)
    parser.add_argument("--only_attacks", action="store_true")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument(
        "--quiet", action="store_true", help="Disable per-flow console output"
    )
    args = parser.parse_args()

    cfg, root = load_config(args.config)

    processed_dir = root / cfg["data"]["processed_dir"]
    processed_path = processed_dir / "processed.csv"
    splits_path = processed_dir / "splits.json"

    if not args.run_name and (not processed_path.exists() or not splits_path.exists()):
        raise FileNotFoundError(
            "Run main_train first to generate processed.csv and splits.json"
        )

    from .config import resolve_split_path
    from .data.replay import load_split, rebuild_training_dataframe

    if args.run_name:
        df = rebuild_training_dataframe(cfg, root, args.run_name)
        test_df = load_split(df, resolve_split_path(cfg, root, args.run_name), "test")
    else:
        df = pd.read_csv(processed_path)
        test_df = load_split(df, splits_path, "test")

    feature_order = cfg["features"]["fixed_list"]
    timestamp_col = find_timestamp_column(test_df.columns)

    models_dir = root / cfg["artifacts"]["models_dir"]
    scalers_dir = root / cfg["artifacts"]["scalers_dir"]
    if args.run_name:
        model_path = models_dir / args.run_name / "model.joblib"
        preproc_path = scalers_dir / args.run_name / "preprocessor.json"
    else:
        model_path = models_dir / "model.joblib"
        preproc_path = scalers_dir / "preprocessor.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    if not preproc_path.exists():
        raise FileNotFoundError(f"Preprocessor artifact not found: {preproc_path}")

    if args.run_name:
        print(f"Using run artifacts: {args.run_name}")
    else:
        print("Using latest artifacts (no --run_name provided)")

    model = joblib.load(model_path)
    preprocessor = load_preprocessor(preproc_path)

    speed = cfg["stream"]["speed"] if args.speed is None else args.speed
    min_conf = (
        cfg["stream"]["min_confidence"]
        if args.min_confidence is None
        else args.min_confidence
    )
    speed = float(speed)
    min_conf = float(min_conf)
    only_attacks = cfg["stream"]["only_attacks"] or args.only_attacks
    batch_size = (
        cfg["stream"]["batch_size"] if args.batch_size is None else args.batch_size
    )
    batch_size = max(1, int(batch_size))

    host = cfg["dashboard"]["host"]
    port = int(cfg["dashboard"]["port"])
    url = f"http://{host}:{port}/api/events"

    total = len(test_df)
    print(f"Streaming {total} events to {url}")
    printed = 0
    compared = 0
    correct = 0

    # Fail fast if the dashboard backend is not running.
    try:
        requests.get(f"http://{host}:{port}/api/stats", timeout=3)
    except requests.RequestException as exc:
        raise ConnectionError(
            f"Dashboard backend is unreachable at http://{host}:{port}. "
            "Start it first with: python -m src.main_serve"
        ) from exc

    for start in range(0, total, batch_size):
        batch = test_df.iloc[start : start + batch_size].copy()
        X = transform(batch, preprocessor)
        probs = model.predict_proba(X)
        preds = model.classes_[np.argmax(probs, axis=1)]

        events = []
        for offset, (_, row) in enumerate(batch.iterrows()):
            confidence = float(probs[offset].max())
            pred_class = preds[offset]
            true_label = str(row.get("Label_mapped", "Unknown"))

            if only_attacks and pred_class == "Normal":
                continue
            if confidence < float(min_conf):
                continue

            ts = (
                row[timestamp_col]
                if timestamp_col
                else datetime.now(timezone.utc).isoformat()
            )
            values = {f: float(row[f]) for f in feature_order}
            top_features = dict(
                sorted(values.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
            )

            events.append(
                {
                    "ts": ts,
                    "predicted_class": pred_class,
                    "confidence": confidence,
                    "flow_id": f"flow_{start + offset}",
                    "top_feature_values": top_features,
                    "scaled_feature_preview": X[offset][:5].tolist(),
                }
            )

            if true_label != "Unknown":
                compared += 1
                is_correct = pred_class == true_label
                if is_correct:
                    correct += 1
            else:
                is_correct = None

            if not args.quiet:
                correctness = "NA" if is_correct is None else str(is_correct)
                print(
                    f"flow_id=flow_{start + offset} true={true_label} "
                    f"pred={pred_class} conf={confidence:.4f} correct={correctness}"
                )
                printed += 1

        if events:
            try:
                requests.post(url, json=events, timeout=5)
            except requests.RequestException as exc:
                print(f"Failed to post events: {exc}")

        if speed and speed > 0:
            time.sleep(len(events) / speed)

    print("Streaming complete.")
    if compared > 0:
        acc = correct / compared
        print(f"Compared: {compared}, Correct: {correct}, Accuracy: {acc:.4f}")
    if not args.quiet:
        print(f"Printed predictions: {printed}")


if __name__ == "__main__":
    main()
