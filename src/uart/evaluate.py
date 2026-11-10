"""Shared device evaluation; probability curves use a separately labeled PC reference."""

import argparse
import json
import time

import joblib
import numpy as np

from ..config import load_config, resolve_run_paths
from ..data.replay import load_split, rebuild_training_dataframe
from ..features.preprocess import load_preprocessor, transform
from ..model.evaluate import evaluate_predictions
from .protocol import encode_row, parse_response


def uart_infer_split(
    ser, feature_X, labels, class_names, split_name, limit, start_index, delay_ms
):
    """Return truths, predictions, and exact row indices of valid replies."""
    if start_index < 0 or limit <= 0 or delay_ms < 0:
        raise ValueError(
            "start_index/delay_ms must be nonnegative and limit must be positive"
        )
    end = min(len(feature_X), start_index + limit)
    if start_index >= end:
        raise ValueError(f"[{split_name}] Requested row range is empty")
    truth, predicted, accepted = [], [], []
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    for row_id in range(start_index, end):
        ser.write(encode_row(row_id, feature_X[row_id].tolist()))
        raw = ser.readline().decode("utf-8", errors="replace").strip()
        try:
            echoed_id, pred_id, confidence = parse_response(raw)
            if echoed_id is not None and echoed_id != row_id:
                raise ValueError(f"row_id mismatch: expected {row_id}, got {echoed_id}")
            if not 0 <= pred_id < len(class_names):
                raise ValueError(f"invalid pred_id={pred_id}")
        except (ValueError, TypeError, KeyError) as exc:
            print(f"[{split_name}] row={row_id} skipped: {exc}")
            continue
        truth.append(str(labels[row_id]))
        predicted.append(class_names[pred_id])
        accepted.append(row_id)
        print(
            f"[{split_name}] row={row_id} truth={truth[-1]} pred={predicted[-1]} conf={confidence:.4f}"
        )
        if delay_ms:
            time.sleep(delay_ms / 1000.0)
    return np.asarray(truth), np.asarray(predicted), accepted


def main(backend="plain_c_rf"):
    parser = argparse.ArgumentParser(
        description=f"Evaluate {backend} STM32 predictions over UART"
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--train_run_name", required=True)
    parser.add_argument("--report_run_name", default=None)
    parser.add_argument("--port", required=True)
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--delay_ms", type=float, default=0.0)
    parser.add_argument("--skip_val", action="store_true")
    parser.add_argument(
        "--device_preprocess",
        action="store_true",
        help="Send raw features; firmware must apply this run's preprocessing",
    )
    args = parser.parse_args()
    if (
        args.limit <= 0
        or args.start_index < 0
        or args.delay_ms < 0
        or args.timeout <= 0
    ):
        parser.error(
            "limit/timeout must be positive; start_index/delay_ms must be nonnegative"
        )
    import serial

    cfg, root = load_config(args.config)
    model_path, scaler_path, metadata_path, splits_path = resolve_run_paths(
        cfg, root, args.train_run_name
    )
    for path in (model_path, scaler_path, metadata_path, splits_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required run artifact not found: {path}")
    model = joblib.load(model_path)
    preprocessor = load_preprocessor(str(scaler_path))
    class_names = [str(c) for c in model.classes_]
    df = rebuild_training_dataframe(cfg, root, args.train_run_name)
    prefix = "rfc" if backend == "plain_c_rf" else "m2c"
    report_name = args.report_run_name or f"{prefix}_{args.train_run_name}"
    reports_dir = root / cfg["artifacts"]["reports_dir"] / report_name
    reports_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}
    splits = (["val"] if not args.skip_val else []) + ["test"]
    with serial.Serial(args.port, args.baudrate, timeout=args.timeout) as ser:
        time.sleep(2.0)
        for name in splits:
            split_df = load_split(df, splits_path, name)
            scaled = transform(split_df, preprocessor)
            features = (
                split_df[preprocessor["feature_order"]].to_numpy(dtype=np.float32)
                if args.device_preprocess
                else scaled
            )
            y_true, y_pred, accepted = uart_infer_split(
                ser,
                features,
                split_df["Label_mapped"].to_numpy(),
                class_names,
                name,
                args.limit,
                args.start_index,
                args.delay_ms,
            )
            attempted = min(args.limit, len(features) - args.start_index)
            summaries[name] = {
                "attempted": attempted,
                "accepted": len(accepted),
                "skipped": attempted - len(accepted),
                "accepted_rows": accepted,
            }
            if not accepted:
                continue
            # Select actual accepted rows, including gaps caused by invalid replies.
            probabilities = model.predict_proba(scaled[accepted])
            evaluate_predictions(
                y_true,
                y_pred,
                probabilities,
                class_names,
                str(reports_dir),
                prefix=name,
                prediction_source="stm32_uart",
                probability_source="pc_joblib_reference",
                extra_metadata=summaries[name],
            )
    metadata = {
        "train_run_name": args.train_run_name,
        "report_run_name": report_name,
        "export_backend": backend,
        "feature_order": preprocessor["feature_order"],
        "feature_mode": "raw" if args.device_preprocess else "scaled",
        "class_names": class_names,
        "uart_port": args.port,
        "baudrate": args.baudrate,
        "prediction_source": "stm32_uart",
        "probability_source": "pc_joblib_reference",
        "splits": summaries,
        "note": "Accuracy/F1 use valid device replies. ROC/PR use PC probabilities; export equivalence is not assumed.",
    }
    filename = (
        "rfc_eval_metadata.json"
        if backend == "plain_c_rf"
        else "m2cgen_eval_metadata.json"
    )
    (reports_dir / filename).write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    if not any(summary["accepted"] for summary in summaries.values()):
        raise RuntimeError("No valid device replies received; evaluation failed")
