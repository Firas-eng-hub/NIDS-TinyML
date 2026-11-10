"""Export a trained RandomForest run to emlearn C code."""

import argparse
import json
import shutil
from pathlib import Path

import joblib

from .config import load_config

try:
    import emlearn
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "emlearn is required. Install it with: pip install emlearn"
    ) from exc


from .config import resolve_model_paths as resolve_run_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument(
        "--run_name", required=True, help="Trained RF run to export, e.g. rf_run_04"
    )
    parser.add_argument(
        "--model_name", default="nids_model", help="C identifier for the exported model"
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory. Defaults to artifacts/embedded/<run_name>",
    )
    parser.add_argument(
        "--method",
        default="loadable",
        choices=["loadable", "inline"],
        help="emlearn export method",
    )
    args = parser.parse_args()

    cfg, root = load_config(args.config)
    model_path, preprocessor_path, metadata_path = resolve_run_paths(
        cfg, root, args.run_name
    )

    for path in (model_path, preprocessor_path, metadata_path):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    model = joblib.load(model_path)
    if model.__class__.__name__ != "RandomForestClassifier":
        raise TypeError(
            f"Expected RandomForestClassifier, got {model.__class__.__name__}. "
            "This exporter is intended for RF deployment."
        )

    with open(preprocessor_path, "r", encoding="utf-8") as handle:
        preprocessor = json.load(handle)
    with open(metadata_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    feature_order = list(preprocessor["feature_order"])
    class_names = [
        str(name)
        for name in getattr(model, "classes_", metadata.get("class_names", []))
    ]
    if not class_names:
        raise ValueError("No class names available from model or metadata.")

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else (root / "artifacts" / "embedded" / args.run_name)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    header_path = output_dir / f"{args.model_name}.h"
    preprocessor_copy_path = output_dir / "preprocessor.json"
    class_map_path = output_dir / "class_map.json"
    export_info_path = output_dir / "emlearn_export.json"

    cmodel = emlearn.convert(model, method=args.method)
    try:
        cmodel.save(file=str(header_path), name=args.model_name, include_proba=True)
    except TypeError:
        cmodel.save(file=str(header_path), name=args.model_name)

    shutil.copy2(preprocessor_path, preprocessor_copy_path)

    class_map = {
        "run_name": args.run_name,
        "model_name": args.model_name,
        "feature_order": feature_order,
        "class_names": class_names,
        "class_ids": list(range(len(class_names))),
    }
    with open(class_map_path, "w", encoding="utf-8") as handle:
        json.dump(class_map, handle, indent=2)

    export_info = {
        "run_name": args.run_name,
        "model_type": metadata.get("model_type"),
        "export_backend": "emlearn",
        "export_method": args.method,
        "model_name": args.model_name,
        "feature_count": len(feature_order),
        "feature_order": feature_order,
        "class_names": class_names,
        "class_ids": list(range(len(class_names))),
        "model_joblib_path": str(model_path),
        "header_path": str(header_path),
        "preprocessor_path": str(preprocessor_copy_path),
        "class_map_path": str(class_map_path),
        "note": (
            "This export preserves the trained RandomForest class order. "
            "For first validation, feed already-preprocessed 15-feature float inputs in the exact feature_order."
        ),
    }
    with open(export_info_path, "w", encoding="utf-8") as handle:
        json.dump(export_info, handle, indent=2)

    print(f"Exported emlearn header: {header_path}")
    print(f"Copied preprocessor: {preprocessor_copy_path}")
    print(f"Exported class map: {class_map_path}")
    print(f"Export metadata: {export_info_path}")
    print(f"Feature count: {len(feature_order)}")
    print(f"Class order: {class_names}")


if __name__ == "__main__":
    main()
