"""Export a trained scikit-learn RandomForest to plain C."""

import argparse
import json
import re
from pathlib import Path
from typing import List

import joblib
import numpy as np

from .config import load_config
from .config import resolve_model_paths as resolve_run_paths


def sanitize_c_identifier(name: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")
    if not value:
        raise ValueError("Model name is empty after sanitization")
    if value[0].isdigit():
        value = f"rf_{value}"
    return value


def format_float(value: float) -> str:
    if np.isnan(value) or np.isinf(value):
        raise ValueError(f"Invalid float value for C export: {value}")
    text = f"{float(value):.9g}"
    if "e" not in text and "E" not in text and "." not in text:
        text += ".0"
    return f"{text}f"


def format_int_array(values: np.ndarray) -> str:
    return ", ".join(str(int(v)) for v in values.tolist())


def format_float_matrix(values: np.ndarray) -> str:
    rows = []
    for row in values.tolist():
        rows.append("    {" + ", ".join(format_float(v) for v in row) + "}")
    return ",\n".join(rows)


def generate_header(
    model_name: str, class_names: List[str], feature_order: List[str], tree_count: int
) -> str:
    guard = f"{model_name.upper()}_H"
    class_comment = ", ".join(f"{idx}={name}" for idx, name in enumerate(class_names))
    feature_comment = ", ".join(feature_order)
    return f"""#ifndef {guard}
#define {guard}

#include <stdint.h>

#define {model_name.upper()}_FEATURE_COUNT {len(feature_order)}
#define {model_name.upper()}_CLASS_COUNT {len(class_names)}
#define {model_name.upper()}_TREE_COUNT {tree_count}

/* Class order: {class_comment} */
/* Feature order: {feature_comment} */

int32_t {model_name}_predict(const float *features);
void {model_name}_predict_proba(const float *features, float *out_probs);

#endif
"""


def generate_source(model, model_name: str, class_names: List[str]) -> str:
    lines: List[str] = []
    n_classes = len(class_names)
    estimators = list(model.estimators_)

    lines.append(f'#include "{model_name}.h"')
    lines.append("")
    lines.append("#include <stddef.h>")
    lines.append("")

    for tree_idx, estimator in enumerate(estimators):
        tree = estimator.tree_
        node_count = int(tree.node_count)
        if node_count > np.iinfo(np.int16).max:
            raise ValueError(
                "Tree exceeds the int16 node-index limit; train a smaller forest"
            )
        children_left = tree.children_left.astype(np.int16)
        children_right = tree.children_right.astype(np.int16)
        feature = tree.feature.astype(np.int16)
        threshold = tree.threshold.astype(np.float64)

        leaf_probs = np.zeros((node_count, n_classes), dtype=np.float32)
        for node_idx in range(node_count):
            values = tree.value[node_idx][0].astype(np.float64)
            total = float(np.sum(values))
            if total > 0.0:
                leaf_probs[node_idx] = (values / total).astype(np.float32)

        lines.append(
            f"static const int16_t tree_{tree_idx}_left[{node_count}] = {{{format_int_array(children_left)}}};"
        )
        lines.append(
            f"static const int16_t tree_{tree_idx}_right[{node_count}] = {{{format_int_array(children_right)}}};"
        )
        lines.append(
            f"static const int16_t tree_{tree_idx}_feature[{node_count}] = {{{format_int_array(feature)}}};"
        )
        lines.append(
            f"static const double tree_{tree_idx}_threshold[{node_count}] = "
            f"{{{', '.join(format(float(v), '.17e') for v in threshold.tolist())}}};"
        )
        lines.append(
            f"static const float tree_{tree_idx}_leaf_probs[{node_count}][{n_classes}] = {{\n"
            f"{format_float_matrix(leaf_probs)}\n}};"
        )
        lines.append("")

    lines.append(
        f"void {model_name}_predict_proba(const float *features, float *out_probs)"
    )
    lines.append("{")
    lines.append("    int tree_idx = 0;")
    lines.append(
        f"    for (int class_idx = 0; class_idx < {n_classes}; class_idx++) {{"
    )
    lines.append("        out_probs[class_idx] = 0.0f;")
    lines.append("    }")
    lines.append("")
    lines.append(f"    for (tree_idx = 0; tree_idx < {len(estimators)}; tree_idx++) {{")
    lines.append("        int node = 0;")
    lines.append("        while (1) {")
    lines.append("            int16_t feature_idx = -1;")
    lines.append("            double threshold = 0.0;")
    lines.append("            int16_t left = -1;")
    lines.append("            int16_t right = -1;")
    lines.append("")
    lines.append("            switch (tree_idx) {")

    for tree_idx, estimator in enumerate(estimators):
        node_count = int(estimator.tree_.node_count)
        lines.append(f"            case {tree_idx}:")
        lines.append(f"                feature_idx = tree_{tree_idx}_feature[node];")
        lines.append(f"                threshold = tree_{tree_idx}_threshold[node];")
        lines.append(f"                left = tree_{tree_idx}_left[node];")
        lines.append(f"                right = tree_{tree_idx}_right[node];")
        lines.append("                break;")

    lines.append("            default:")
    lines.append("                return;")
    lines.append("            }")
    lines.append("")
    lines.append("            if (left == -1 && right == -1) {")
    lines.append(
        f"                for (int class_idx = 0; class_idx < {n_classes}; class_idx++) {{"
    )
    lines.append("                    switch (tree_idx) {")
    for tree_idx, estimator in enumerate(estimators):
        lines.append(f"                    case {tree_idx}:")
        lines.append(
            f"                        out_probs[class_idx] += tree_{tree_idx}_leaf_probs[node][class_idx];"
        )
        lines.append("                        break;")
    lines.append("                    default:")
    lines.append("                        break;")
    lines.append("                    }")
    lines.append("                }")
    lines.append("                break;")
    lines.append("            }")
    lines.append("")
    lines.append("            if (features[feature_idx] <= threshold) {")
    lines.append("                node = left;")
    lines.append("            } else {")
    lines.append("                node = right;")
    lines.append("            }")
    lines.append("        }")
    lines.append("    }")
    lines.append("")
    lines.append(
        f"    for (int class_idx = 0; class_idx < {n_classes}; class_idx++) {{"
    )
    lines.append(f"        out_probs[class_idx] /= (float){len(estimators)};")
    lines.append("    }")
    lines.append("}")
    lines.append("")
    lines.append(f"int32_t {model_name}_predict(const float *features)")
    lines.append("{")
    lines.append(f"    float probs[{n_classes}];")
    lines.append("    int32_t best_idx = 0;")
    lines.append("    float best_score = 0.0f;")
    lines.append("")
    lines.append(f"    {model_name}_predict_proba(features, probs);")
    lines.append("    best_idx = 0;")
    lines.append("    best_score = probs[0];")
    lines.append(
        f"    for (int32_t class_idx = 1; class_idx < {n_classes}; class_idx++) {{"
    )
    lines.append("        if (probs[class_idx] > best_score) {")
    lines.append("            best_score = probs[class_idx];")
    lines.append("            best_idx = class_idx;")
    lines.append("        }")
    lines.append("    }")
    lines.append("")
    lines.append("    return best_idx;")
    lines.append("}")
    lines.append("")

    return "\n".join(lines)


def generate_preprocessor(preprocessor: dict) -> str:
    """Generate the selected run's transform; never embed a particular run in firmware."""
    features = preprocessor["feature_order"]
    mask = [int(f in preprocessor["log1p_features"]) for f in features]
    mins = [preprocessor["min_vals"][f] for f in features]
    maxs = [preprocessor["max_vals"][f] for f in features]
    return f"""#ifndef RF_PREPROCESSOR_H
#define RF_PREPROCESSOR_H
#include <math.h>
#define RF_PREPROCESSOR_FEATURE_COUNT {len(features)}
static inline void rf_preprocess(const float *input, float *output)
{{
    static const unsigned char logs[{len(features)}] = {{{", ".join(map(str, mask))}}};
    static const double mins[{len(features)}] = {{{", ".join(format(float(v), ".17e") for v in mins)}}};
    static const double maxs[{len(features)}] = {{{", ".join(format(float(v), ".17e") for v in maxs)}}};
    for (int i = 0; i < {len(features)}; i++) {{
        float x = logs[i] ? log1pf(input[i]) : input[i];
        double denom = maxs[i] - mins[i];
        output[i] = (float)(((double)x - mins[i]) / (denom == 0.0 ? 1.0 : denom));
    }}
}}
#endif
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument(
        "--run_name", required=True, help="Trained RF run to export, e.g. rf_run_04"
    )
    parser.add_argument(
        "--model_name",
        default="rf_model",
        help="Base C identifier/file name for the export",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory. Defaults to artifacts/embedded_c/<run_name>",
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
        str(name)
        for name in getattr(model, "classes_", metadata.get("class_names", []))
    ]
    if not class_names:
        raise ValueError("No class names available from model or metadata.")

    model_name = sanitize_c_identifier(args.model_name)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else (root / "artifacts" / "embedded_c" / args.run_name)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    header_path = output_dir / f"{model_name}.h"
    source_path = output_dir / f"{model_name}.c"
    class_map_path = output_dir / "class_map.json"
    export_info_path = output_dir / "rf_c_export.json"

    header_path.write_text(
        generate_header(
            model_name=model_name,
            class_names=class_names,
            feature_order=feature_order,
            tree_count=len(model.estimators_),
        ),
        encoding="utf-8",
    )
    source_path.write_text(
        generate_source(model=model, model_name=model_name, class_names=class_names),
        encoding="utf-8",
    )

    (output_dir / "rf_preprocessor.h").write_text(
        generate_preprocessor(preprocessor), encoding="utf-8"
    )
    (output_dir / "preprocessor.json").write_text(
        json.dumps(preprocessor, indent=2), encoding="utf-8"
    )

    class_map = {
        "run_name": args.run_name,
        "model_name": model_name,
        "feature_order": feature_order,
        "class_names": class_names,
        "class_ids": list(range(len(class_names))),
    }
    with open(class_map_path, "w", encoding="utf-8") as handle:
        json.dump(class_map, handle, indent=2)

    export_info = {
        "run_name": args.run_name,
        "export_backend": "plain_c_rf",
        "model_name": model_name,
        "model_type": metadata.get("model_type"),
        "feature_count": len(feature_order),
        "feature_order": feature_order,
        "class_names": class_names,
        "class_ids": list(range(len(class_names))),
        "tree_count": len(model.estimators_),
        "header_path": str(header_path),
        "source_path": str(source_path),
        "class_map_path": str(class_map_path),
        "note": "Model expects preprocessed floats; rf_preprocessor.h implements this run's raw-input transform. Validate export agreement before deployment.",
    }
    with open(export_info_path, "w", encoding="utf-8") as handle:
        json.dump(export_info, handle, indent=2)

    print(f"Exported C header: {header_path}")
    print(f"Exported C source: {source_path}")
    print(f"Class map: {class_map_path}")
    print(f"Export metadata: {export_info_path}")
    print(f"Tree count: {len(model.estimators_)}")
    print(f"Class order: {class_names}")


if __name__ == "__main__":
    main()
