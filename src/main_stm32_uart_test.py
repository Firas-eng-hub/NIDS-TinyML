"""Replay preprocessed test-set rows to an STM32 over UART."""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import pandas as pd
import requests
import yaml

from .data.loader import load_and_clean_data
from .features.preprocess import load_preprocessor, transform

try:
    import serial
except ImportError as exc:  # pragma: no cover - import guard for optional dependency
    raise ImportError("pyserial is required for UART replay. Install it with: pip install pyserial") from exc


def load_config(config_path: Optional[str] = None) -> Tuple[dict, Path]:
    root = Path(__file__).resolve().parents[1]
    if config_path is None:
        config_path = root / "config" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    return cfg, root


def resolve_run_paths(cfg: dict, root: Path, run_name: str) -> Tuple[Path, Path, Path, Path]:
    model_path = root / cfg["artifacts"]["models_dir"] / run_name / "model.joblib"
    scaler_path = root / cfg["artifacts"]["scalers_dir"] / run_name / "preprocessor.json"
    metadata_path = root / cfg["artifacts"]["metadata_dir"] / run_name / "metadata.json"
    splits_path = root / cfg["data"]["processed_dir"] / "splits.json"
    return model_path, scaler_path, metadata_path, splits_path


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


def load_test_split(df: pd.DataFrame, splits_path: Path) -> pd.DataFrame:
    with open(splits_path, "r", encoding="utf-8") as handle:
        split_indices = json.load(handle)
    test_indices = split_indices["test"]
    return df.iloc[test_indices].reset_index(drop=True)


def load_run_info(model_path: Path, metadata_path: Path, scaler_path: Path) -> Tuple[List[str], Dict[str, object]]:
    model = joblib.load(model_path)
    preprocessor = load_preprocessor(str(scaler_path))
    if hasattr(model, "classes_"):
        class_names = [str(name) for name in model.classes_]
    else:
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        class_names = list(metadata["class_names"])
    return class_names, preprocessor


def encode_row(row_id: int, feature_values: List[float]) -> bytes:
    payload = [str(row_id)] + [f"{value:.6f}" for value in feature_values]
    return (",".join(payload) + "\n").encode("ascii")


def parse_response(line: str) -> Tuple[Optional[int], int, float]:
    line = line.strip()
    if not line:
        raise ValueError("Received an empty UART line")

    if line.startswith("{"):
        payload = json.loads(line)
        row_id = payload.get("row_id")
        pred_id = int(payload["pred_id"])
        confidence = float(payload.get("confidence", 0.0))
        return row_id, pred_id, confidence

    parts = [part.strip() for part in line.split(",")]
    if len(parts) == 2:
        confidence = float(parts[1])
        if confidence > 1.0:
            confidence /= 1_000_000.0
        return None, int(parts[0]), confidence
    if len(parts) == 3:
        confidence = float(parts[2])
        if confidence > 1.0:
            confidence /= 1_000_000.0
        return int(parts[0]), int(parts[1]), confidence
    raise ValueError(f"Unsupported STM32 response format: {line}")


def build_event(
    row_id: int,
    truth_label: str,
    predicted_label: str,
    confidence: float,
    scaled_row: List[float],
) -> Dict[str, object]:
    return {
        "flow_id": f"test_row_{row_id}",
        "predicted_class": predicted_label,
        "confidence": confidence,
        "ground_truth": truth_label,
        "source": "stm32_uart",
        "scaled_feature_preview": scaled_row[:5],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--run_name", required=True, help="Trained run name, e.g. rf_run_04")
    parser.add_argument("--port", required=True, help="UART port, e.g. COM6")
    parser.add_argument("--baudrate", type=int, default=115200, help="UART baudrate")
    parser.add_argument("--timeout", type=float, default=3.0, help="Serial read timeout in seconds")
    parser.add_argument("--limit", type=int, default=100, help="How many test rows to send")
    parser.add_argument("--start_index", type=int, default=0, help="Start offset inside the test split")
    parser.add_argument("--delay_ms", type=float, default=0.0, help="Delay between rows in milliseconds")
    parser.add_argument("--dashboard", action="store_true", help="Post STM32 predictions to the dashboard")
    parser.add_argument("--dashboard_url", default=None, help="Override dashboard events URL")
    args = parser.parse_args()

    cfg, root = load_config(args.config)
    model_path, scaler_path, metadata_path, splits_path = resolve_run_paths(cfg, root, args.run_name)

    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Preprocessor artifact not found: {scaler_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Run metadata not found: {metadata_path}")
    if not splits_path.exists():
        raise FileNotFoundError(f"Split index file not found: {splits_path}")

    class_names, preprocessor = load_run_info(model_path, metadata_path, scaler_path)
    feature_order = list(preprocessor["feature_order"])

    print("Rebuilding cleaned dataset...")
    df = rebuild_training_dataframe(cfg, root)
    test_df = load_test_split(df, splits_path)
    scaled_test = transform(test_df, preprocessor)

    total_available = len(test_df)
    start_index = max(0, int(args.start_index))
    end_index = min(total_available, start_index + max(0, int(args.limit)))

    if start_index >= end_index:
        raise ValueError(
            f"Requested range is empty. test_rows={total_available}, start_index={start_index}, limit={args.limit}"
        )

    dashboard_url = args.dashboard_url
    if args.dashboard and dashboard_url is None:
        dashboard_url = f"http://{cfg['dashboard']['host']}:{int(cfg['dashboard']['port'])}/api/events"

    print(f"Run name: {args.run_name}")
    print(f"UART: {args.port} @ {args.baudrate}")
    print(f"Test rows available: {total_available}")
    print(f"Sending rows: {start_index}..{end_index - 1}")
    print(f"Feature count: {len(feature_order)}")
    print("Host -> STM32 line format: row_id,f1,f2,...,f15")
    print("STM32 -> Host expected format: pred_id,confidence OR row_id,pred_id,confidence")

    correct = 0
    sent = 0
    posted = 0

    with serial.Serial(args.port, args.baudrate, timeout=args.timeout) as ser:
        time.sleep(2.0)
        ser.reset_input_buffer()
        ser.reset_output_buffer()

        for row_id in range(start_index, end_index):
            scaled_row = scaled_test[row_id].tolist()
            truth_label = str(test_df.iloc[row_id]["Label_mapped"])

            ser.write(encode_row(row_id, scaled_row))
            raw_response = ser.readline().decode("utf-8", errors="replace").strip()
            if not raw_response:
                print(f"row={row_id} timeout waiting for STM32 response")
                continue

            try:
                echoed_row_id, pred_id, confidence = parse_response(raw_response)
            except Exception as exc:  # pragma: no cover - runtime device parsing
                print(f"row={row_id} invalid response={raw_response!r} error={exc}")
                continue

            if echoed_row_id is not None and int(echoed_row_id) != row_id:
                print(f"row={row_id} mismatched echoed row_id={echoed_row_id}, response={raw_response!r}")
                continue

            if pred_id < 0 or pred_id >= len(class_names):
                print(f"row={row_id} invalid pred_id={pred_id}, response={raw_response!r}")
                continue

            predicted_label = class_names[pred_id]
            is_correct = predicted_label == truth_label
            correct += int(is_correct)
            sent += 1

            print(
                f"row={row_id} truth={truth_label} pred={predicted_label} "
                f"pred_id={pred_id} conf={confidence:.4f} ok={is_correct}"
            )

            if args.dashboard and dashboard_url:
                event = build_event(
                    row_id=row_id,
                    truth_label=truth_label,
                    predicted_label=predicted_label,
                    confidence=confidence,
                    scaled_row=scaled_row,
                )
                try:
                    requests.post(dashboard_url, json=event, timeout=3)
                    posted += 1
                except requests.RequestException as exc:
                    print(f"row={row_id} failed to post dashboard event: {exc}")

            if args.delay_ms > 0:
                time.sleep(float(args.delay_ms) / 1000.0)

    accuracy = 0.0 if sent == 0 else correct / float(sent)
    print(f"Done. rows_scored={sent} correct={correct} accuracy={accuracy:.4f} dashboard_posted={posted}")


if __name__ == "__main__":
    main()
