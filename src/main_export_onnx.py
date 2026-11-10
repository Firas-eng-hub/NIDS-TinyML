"""Export a trained scikit-learn run to ONNX."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from .config import load_config


def resolve_run_paths(cfg: dict, root: Path, run_name: str):
    models_dir = root / cfg["artifacts"]["models_dir"] / run_name
    scalers_dir = root / cfg["artifacts"]["scalers_dir"] / run_name
    metadata_dir = root / cfg["artifacts"]["metadata_dir"] / run_name

    model_path = models_dir / "model.joblib"
    preprocessor_path = scalers_dir / "preprocessor.json"
    metadata_path = metadata_dir / "metadata.json"
    onnx_path = models_dir / "model.onnx"
    onnx_metadata_path = metadata_dir / "onnx_export.json"
    return model_path, preprocessor_path, metadata_path, onnx_path, onnx_metadata_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument(
        "--run_name", required=True, help="Trained run to export, e.g. rf_run_03"
    )
    parser.add_argument(
        "--input_name", default="float_input", help="ONNX input tensor name"
    )
    parser.add_argument(
        "--target_opset", type=int, default=17, help="ONNX opset version to export"
    )
    args = parser.parse_args()

    cfg, root = load_config(args.config)
    model_path, preprocessor_path, metadata_path, onnx_path, onnx_metadata_path = (
        resolve_run_paths(cfg, root, args.run_name)
    )

    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    if not preprocessor_path.exists():
        raise FileNotFoundError(f"Preprocessor artifact not found: {preprocessor_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata artifact not found: {metadata_path}")

    model = joblib.load(model_path)

    with open(preprocessor_path, "r", encoding="utf-8") as handle:
        preprocessor = json.load(handle)
    with open(metadata_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    feature_order = list(preprocessor["feature_order"])
    original_class_names = [str(name) for name in getattr(model, "classes_", [])]
    if not original_class_names:
        original_class_names = [str(name) for name in metadata.get("class_names", [])]
    if not hasattr(model, "classes_"):
        raise ValueError(
            "Loaded model does not expose classes_; cannot export classifier labels."
        )
    if not original_class_names:
        raise ValueError("No class names found in metadata or model.classes_.")

    # X-CUBE-AI code generation requires integer class labels in the ONNX classifier output.
    model.classes_ = np.arange(len(original_class_names), dtype=np.int64)

    initial_type = [(str(args.input_name), FloatTensorType([None, len(feature_order)]))]

    onnx_model = convert_sklearn(
        model,
        initial_types=initial_type,
        options={id(model): {"zipmap": False}},
        target_opset=int(args.target_opset),
    )
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    onnx_path.write_bytes(onnx_model.SerializeToString())

    export_info = {
        "run_name": args.run_name,
        "model_type": metadata.get("model_type"),
        "feature_count": len(feature_order),
        "feature_order": feature_order,
        "class_names": original_class_names,
        "class_ids": list(range(len(original_class_names))),
        "input_name": str(args.input_name),
        "target_opset": int(args.target_opset),
        "zipmap": False,
        "model_joblib_path": str(model_path),
        "preprocessor_path": str(preprocessor_path),
        "onnx_path": str(onnx_path),
        "note": (
            "This ONNX file expects the same preprocessed float input used during training. "
            "Apply exact feature order, log1p features, and MinMax scaling first."
        ),
    }

    onnx_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(onnx_metadata_path, "w", encoding="utf-8") as handle:
        json.dump(export_info, handle, indent=2)

    print(f"Exported ONNX model: {onnx_path}")
    print(f"Feature count: {len(feature_order)}")
    print(f"Target opset: {args.target_opset}")
    print("ZipMap disabled for ONNX probability output.")
    print("Class labels exported as int64 IDs for X-CUBE-AI compatibility.")
    print("Important: the ONNX input expects already-preprocessed features.")


if __name__ == "__main__":
    main()
