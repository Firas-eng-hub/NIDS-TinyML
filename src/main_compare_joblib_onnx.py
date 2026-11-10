"""Compare joblib and ONNXRuntime predictions on the saved test split."""

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd

from .config import load_config, resolve_split_path
from .data.replay import rebuild_training_dataframe
from .features.preprocess import load_preprocessor, transform

try:
    import onnxruntime as ort
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "onnxruntime is required. Install it with: pip install onnxruntime"
    ) from exc


def load_test_split(df: pd.DataFrame, splits_path: Path) -> pd.DataFrame:
    with open(splits_path, "r", encoding="utf-8") as handle:
        split_indices = json.load(handle)
    return df.iloc[split_indices["test"]].reset_index(drop=True)


def resolve_run_paths(
    cfg: dict, root: Path, run_name: str
) -> Tuple[Path, Path, Path, Path]:
    model_path = root / cfg["artifacts"]["models_dir"] / run_name / "model.joblib"
    onnx_path = root / cfg["artifacts"]["models_dir"] / run_name / "model.onnx"
    preprocessor_path = (
        root / cfg["artifacts"]["scalers_dir"] / run_name / "preprocessor.json"
    )
    metadata_path = root / cfg["artifacts"]["metadata_dir"] / run_name / "metadata.json"
    return model_path, onnx_path, preprocessor_path, metadata_path


def extract_onnx_outputs(
    session: ort.InferenceSession, input_name: str, x: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    outputs = session.run(None, {input_name: x.astype(np.float32)})

    pred_ids = None
    probs = None
    for output in outputs:
        arr = np.asarray(output)
        if arr.dtype.kind in ("i", "u"):
            pred_ids = arr.reshape(-1)
        elif arr.dtype.kind == "f":
            probs = arr

    if pred_ids is None:
        raise ValueError("ONNX output did not contain integer class IDs")
    if probs is None:
        raise ValueError("ONNX output did not contain float probability scores")

    if probs.ndim == 3 and probs.shape[1] == 1:
        probs = probs[:, 0, :]
    elif probs.ndim == 2:
        pass
    else:
        probs = probs.reshape(probs.shape[0], -1)

    return pred_ids.astype(int), probs.astype(np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--run_name", required=True, help="Run name, e.g. rf_run_04")
    parser.add_argument(
        "--limit", type=int, default=1000, help="How many test rows to compare"
    )
    parser.add_argument(
        "--start_index", type=int, default=0, help="Offset inside the test split"
    )
    args = parser.parse_args()

    cfg, root = load_config(args.config)
    model_path, onnx_path, preprocessor_path, metadata_path = resolve_run_paths(
        cfg, root, args.run_name
    )
    splits_path = resolve_split_path(cfg, root, args.run_name)

    for path in (model_path, onnx_path, preprocessor_path, metadata_path, splits_path):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    print("Rebuilding cleaned dataset...")
    df = rebuild_training_dataframe(cfg, root, args.run_name)
    test_df = load_test_split(df, splits_path)

    preprocessor = load_preprocessor(str(preprocessor_path))
    x_all = transform(test_df, preprocessor)

    start_index = max(0, int(args.start_index))
    end_index = min(len(x_all), start_index + max(0, int(args.limit)))
    if start_index >= end_index:
        raise ValueError("Requested comparison range is empty")

    x = x_all[start_index:end_index]
    y_true = test_df.iloc[start_index:end_index]["Label_mapped"].tolist()

    model = joblib.load(model_path)
    class_names: List[str] = [str(name) for name in model.classes_]

    joblib_probs = model.predict_proba(x)
    joblib_class_order = [str(name) for name in model.classes_]
    joblib_pred_labels = [str(name) for name in model.predict(x)]
    joblib_label_to_id = {name: idx for idx, name in enumerate(joblib_class_order)}
    joblib_pred_ids = np.array(
        [joblib_label_to_id[label] for label in joblib_pred_labels], dtype=int
    )

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    onnx_pred_ids, onnx_probs = extract_onnx_outputs(session, input_name, x)
    onnx_pred_labels = [class_names[idx] for idx in onnx_pred_ids]

    same_labels = np.array(joblib_pred_labels) == np.array(onnx_pred_labels)
    same_top_idx = joblib_pred_ids == onnx_pred_ids
    prob_max_abs_diff = float(np.max(np.abs(joblib_probs - onnx_probs)))
    prob_mean_abs_diff = float(np.mean(np.abs(joblib_probs - onnx_probs)))
    agreement = float(np.mean(same_labels))
    accuracy_joblib = float(np.mean(np.array(joblib_pred_labels) == np.array(y_true)))
    accuracy_onnx = float(np.mean(np.array(onnx_pred_labels) == np.array(y_true)))

    print(f"Rows compared: {len(x)}")
    print(f"Joblib accuracy on slice: {accuracy_joblib:.6f}")
    print(f"ONNXRuntime accuracy on slice: {accuracy_onnx:.6f}")
    print(f"Prediction agreement: {agreement:.6f}")
    print(f"Top-class ID agreement: {float(np.mean(same_top_idx)):.6f}")
    print(f"Probability max abs diff: {prob_max_abs_diff:.10f}")
    print(f"Probability mean abs diff: {prob_mean_abs_diff:.10f}")

    mismatches = np.where(~same_labels)[0]
    if len(mismatches) == 0:
        print("Result: joblib and ONNXRuntime predictions match on the tested rows.")
        return

    print(f"Mismatches: {len(mismatches)}")
    preview_count = min(10, len(mismatches))
    print("First mismatches:")
    for idx in mismatches[:preview_count]:
        row_id = start_index + int(idx)
        print(
            f"row={row_id} true={y_true[idx]} "
            f"joblib={joblib_pred_labels[idx]} onnx={onnx_pred_labels[idx]} "
            f"joblib_probs={joblib_probs[idx].tolist()} onnx_probs={onnx_probs[idx].tolist()}"
        )


if __name__ == "__main__":
    main()
