"""Export a trained scikit-learn RandomForest to C using m2cgen."""

import argparse
import json
from pathlib import Path

import joblib

from .config import load_config

try:
    import m2cgen as m2c
except ImportError as exc:
    raise ImportError(
        "m2cgen is required. Install it with: pip install m2cgen"
    ) from exc


from .config import resolve_model_paths as resolve_run_paths


def generate_wrapper(model_name: str, class_names: list, feature_order: list) -> str:
    """Generate a thin wrapper header that adds predict() on top of m2cgen's score()."""
    n_classes = len(class_names)
    class_comment = ", ".join(f"{i}={c}" for i, c in enumerate(class_names))
    feature_comment = ", ".join(feature_order)

    lines = []
    lines.append(f"#ifndef {model_name.upper()}_WRAPPER_H")
    lines.append(f"#define {model_name.upper()}_WRAPPER_H")
    lines.append("")
    lines.append("#include <stdint.h>")
    lines.append("#include <float.h>")
    lines.append("")
    lines.append(f"#define {model_name.upper()}_FEATURE_COUNT {len(feature_order)}")
    lines.append(f"#define {model_name.upper()}_CLASS_COUNT   {n_classes}")
    lines.append("")
    lines.append(f"/* Class order : {class_comment} */")
    lines.append(f"/* Feature order: {feature_comment} */")
    lines.append("")
    lines.append("/* Forward declaration of m2cgen-generated scorer. */")
    lines.append("void score(double *input, double *output);")
    lines.append("")
    lines.append(f"static inline int32_t {model_name}_predict(const float *features)")
    lines.append("{")
    lines.append(f"    double input[{len(feature_order)}];")
    lines.append(f"    double output[{n_classes}];")
    lines.append("    int32_t best = 0;")
    lines.append("    double  best_score = -DBL_MAX;")
    lines.append("    int32_t i;")
    lines.append(
        f"    for (i = 0; i < {len(feature_order)}; i++) input[i] = (double)features[i];"
    )
    lines.append("    score(input, output);")
    lines.append(f"    for (i = 0; i < {n_classes}; i++) {{")
    lines.append(
        "        if (output[i] > best_score) { best_score = output[i]; best = i; }"
    )
    lines.append("    }")
    lines.append("    return best;")
    lines.append("}")
    lines.append("")
    lines.append(f"#endif /* {model_name.upper()}_WRAPPER_H */")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a trained RandomForest to C code using m2cgen."
    )
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument(
        "--run_name", required=True, help="Trained RF run, e.g. rf_run_04"
    )
    parser.add_argument(
        "--model_name", default="m2c_model", help="C identifier for the model"
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory. Defaults to artifacts/embedded_m2c/<run_name>",
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
            f"Expected RandomForestClassifier, got {model.__class__.__name__}"
        )

    with open(preprocessor_path, "r", encoding="utf-8") as handle:
        preprocessor = json.load(handle)
    with open(metadata_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    feature_order = list(preprocessor["feature_order"])
    class_names = [
        str(c) for c in getattr(model, "classes_", metadata.get("class_names", []))
    ]
    if not class_names:
        raise ValueError("No class names found in model or metadata.")

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else root / "artifacts" / "embedded_m2c" / args.run_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- m2cgen export ---
    c_code = m2c.export_to_c(model)

    source_path = output_dir / f"{args.model_name}.c"
    wrapper_path = output_dir / f"{args.model_name}_wrapper.h"
    class_map_path = output_dir / "class_map.json"
    export_info_path = output_dir / "m2cgen_export.json"

    source_path.write_text(c_code, encoding="utf-8")
    wrapper_path.write_text(
        generate_wrapper(args.model_name, class_names, feature_order),
        encoding="utf-8",
    )

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
        "export_backend": "m2cgen",
        "model_name": args.model_name,
        "model_type": metadata.get("model_type"),
        "feature_count": len(feature_order),
        "feature_order": feature_order,
        "class_names": class_names,
        "class_ids": list(range(len(class_names))),
        "tree_count": len(model.estimators_),
        "source_path": str(source_path),
        "wrapper_path": str(wrapper_path),
        "class_map_path": str(class_map_path),
        "note": (
            "m2cgen generates a score(double *input, double *output) function. "
            "The wrapper header exposes predict(float *features) -> int32_t. "
            "Feed already-preprocessed 15-feature float inputs in exact feature_order."
        ),
    }
    with open(export_info_path, "w", encoding="utf-8") as handle:
        json.dump(export_info, handle, indent=2)

    print(f"[m2cgen] C source   : {source_path}")
    print(f"[m2cgen] Wrapper    : {wrapper_path}")
    print(f"[m2cgen] Class map  : {class_map_path}")
    print(f"[m2cgen] Export info: {export_info_path}")
    print(f"[m2cgen] Trees      : {len(model.estimators_)}")
    print(f"[m2cgen] Classes    : {class_names}")


if __name__ == "__main__":
    main()
