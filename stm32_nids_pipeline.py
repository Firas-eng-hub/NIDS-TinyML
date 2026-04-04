#!/usr/bin/env python3
"""End-to-end STM32-oriented NIDS MLOps pipeline."""

import argparse
import re
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

try:
    import emlearn
except ImportError as exc:
    raise ImportError(
        "The 'emlearn' package is required for Phase 5. Install it with: pip install emlearn"
    ) from exc

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split


FEATURE_COLUMNS = [
    "SYN Flag Cnt",
    "RST Flag Cnt",
    "ACK Flag Cnt",
    "PSH Flag Cnt",
    "Down/Up Ratio",
    "Pkt Len Min",
    "Pkt Len Max",
]
LABEL_NAMES = {0: "Normal", 1: "DoS", 2: "BruteForce"}
INT16_MIN = int(np.iinfo(np.int16).min)
INT16_MAX = int(np.iinfo(np.int16).max)
RATIO_SCALE = 100
QUALITY_GATE_MACRO_F1 = 0.85


def log_phase(phase_number: int, title: str) -> None:
    print(f"\n[Phase {phase_number}/5] {title}")


def validate_model_name(model_name: str) -> None:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", model_name):
        raise ValueError(
            f"Invalid C identifier '{model_name}'. Use letters, numbers, and underscores only."
        )


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized.columns = [str(col).strip() for col in normalized.columns]
    return normalized


def clip_to_int16(df: pd.DataFrame) -> pd.DataFrame:
    overflow_mask = (df < INT16_MIN) | (df > INT16_MAX)
    overflow_count = int(overflow_mask.sum().sum())
    if overflow_count:
        print(
            f"[WARN] Clipped {overflow_count} feature values into the int16 range "
            f"[{INT16_MIN}, {INT16_MAX}] for emlearn compatibility."
        )
    return df.clip(lower=INT16_MIN, upper=INT16_MAX).astype(np.int16)


def prepare_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df)
    missing = [column for column in FEATURE_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            "Baseline CSV is missing required STM32 feature columns: "
            + ", ".join(missing)
        )

    features = df[FEATURE_COLUMNS].copy()
    features = features.replace([np.inf, -np.inf], np.nan)

    for column in FEATURE_COLUMNS:
        features[column] = pd.to_numeric(features[column], errors="coerce")

    features = features.fillna(0)

    # Convert the ratio into a fixed-point integer so the exported model can
    # use int16_t inputs on-device instead of floating-point values.
    features["Down/Up Ratio"] = np.rint(features["Down/Up Ratio"] * RATIO_SCALE)

    for column in FEATURE_COLUMNS:
        features[column] = np.rint(features[column]).astype(np.int32)

    features["Pkt Len Min"] = features["Pkt Len Min"].clip(lower=0)
    features["Pkt Len Max"] = features[["Pkt Len Min", "Pkt Len Max"]].max(axis=1)

    return clip_to_int16(features)


def load_baseline_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Baseline CSV not found: {csv_path}. Pass --baseline_csv with a valid file."
        )

    print(f"[INFO] Loading baseline traffic from: {csv_path}")
    raw_df = pd.read_csv(csv_path)
    if raw_df.empty:
        raise ValueError(f"Baseline CSV is empty: {csv_path}")

    normal_df = prepare_feature_frame(raw_df)
    normal_df["Label"] = 0

    print(f"[INFO] Loaded {len(normal_df)} normal baseline rows.")
    print(f"[INFO] Selected STM32 feature order: {FEATURE_COLUMNS}")
    print(
        f"[INFO] Down/Up Ratio is quantized offline as an integer scaled by x{RATIO_SCALE}."
    )
    return normal_df


def generate_synthetic_attacks(
    normal_df: pd.DataFrame, random_state: int = 42
) -> pd.DataFrame:
    if normal_df.empty:
        raise ValueError("Normal baseline dataframe is empty; cannot generate attacks.")

    rng = np.random.default_rng(random_state)
    sample_count = len(normal_df)
    normal_features = normal_df[FEATURE_COLUMNS].copy()

    dos_df = (
        normal_features.sample(n=sample_count, replace=True, random_state=random_state)
        .reset_index(drop=True)
        .astype(np.int32)
    )
    dos_pkt_min = np.minimum(dos_df["Pkt Len Min"], rng.integers(0, 48, size=sample_count))
    dos_df["SYN Flag Cnt"] = np.maximum(
        dos_df["SYN Flag Cnt"], rng.integers(18, 64, size=sample_count)
    )
    dos_df["RST Flag Cnt"] = rng.integers(0, 3, size=sample_count)
    dos_df["ACK Flag Cnt"] = rng.integers(0, 3, size=sample_count)
    dos_df["PSH Flag Cnt"] = rng.integers(0, 2, size=sample_count)
    dos_df["Down/Up Ratio"] = 0
    dos_df["Pkt Len Min"] = dos_pkt_min
    dos_df["Pkt Len Max"] = np.maximum(
        dos_pkt_min + rng.integers(20, 220, size=sample_count),
        dos_df["Pkt Len Max"],
    )
    dos_df = clip_to_int16(dos_df)
    dos_df["Label"] = 1

    bruteforce_df = (
        normal_features.sample(
            n=sample_count, replace=True, random_state=random_state + 1
        )
        .reset_index(drop=True)
        .astype(np.int32)
    )
    brute_pkt_min = np.maximum(
        bruteforce_df["Pkt Len Min"], rng.integers(32, 96, size=sample_count)
    )
    bruteforce_df["SYN Flag Cnt"] = rng.integers(1, 5, size=sample_count)
    bruteforce_df["RST Flag Cnt"] = rng.integers(0, 4, size=sample_count)
    bruteforce_df["ACK Flag Cnt"] = np.maximum(
        bruteforce_df["ACK Flag Cnt"], rng.integers(14, 48, size=sample_count)
    )
    bruteforce_df["PSH Flag Cnt"] = np.maximum(
        bruteforce_df["PSH Flag Cnt"], rng.integers(10, 36, size=sample_count)
    )
    bruteforce_df["Down/Up Ratio"] = rng.integers(80, 220, size=sample_count)
    bruteforce_df["Pkt Len Min"] = brute_pkt_min
    bruteforce_df["Pkt Len Max"] = np.maximum(
        brute_pkt_min + rng.integers(64, 320, size=sample_count),
        bruteforce_df["Pkt Len Max"],
    )
    bruteforce_df = clip_to_int16(bruteforce_df)
    bruteforce_df["Label"] = 2

    merged_df = pd.concat([normal_df, dos_df, bruteforce_df], ignore_index=True)
    merged_df = merged_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    print(f"[INFO] Normal rows: {len(normal_df)}")
    print(f"[INFO] Synthetic DoS rows: {len(dos_df)}")
    print(f"[INFO] Synthetic BruteForce rows: {len(bruteforce_df)}")
    print("[INFO] Merged class distribution:")
    print(merged_df["Label"].value_counts().sort_index().rename(index=LABEL_NAMES))

    return merged_df


def split_dataset(
    dataset_df: pd.DataFrame, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    class_counts = dataset_df["Label"].value_counts()
    if class_counts.min() < 2:
        raise ValueError(
            "Each class must have at least 2 samples before the 80/20 train/test split."
        )

    X = dataset_df[FEATURE_COLUMNS].to_numpy(dtype=np.int16)
    y = dataset_df["Label"].to_numpy(dtype=np.int8)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )

    print(f"[INFO] Train samples: {len(X_train)}")
    print(f"[INFO] Test samples: {len(X_test)}")
    return X_train, X_test, y_train, y_test


def train_model(
    X_train: np.ndarray, y_train: np.ndarray, random_state: int = 42
) -> RandomForestClassifier:
    print("[INFO] Training RandomForestClassifier with STM32 size constraints...")
    model = RandomForestClassifier(
        n_estimators=20,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def validate_model(
    model: RandomForestClassifier, X_test: np.ndarray, y_test: np.ndarray
) -> float:
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print(f"[INFO] Accuracy: {accuracy:.4f}")
    print(f"[INFO] Macro F1-Score: {macro_f1:.4f}")
    print("[INFO] Classification report:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1, 2],
            target_names=[LABEL_NAMES[0], LABEL_NAMES[1], LABEL_NAMES[2]],
            zero_division=0,
        )
    )

    if macro_f1 < QUALITY_GATE_MACRO_F1:
        raise RuntimeError(
            f"Quality gate failed: Macro F1-Score {macro_f1:.4f} is below "
            f"{QUALITY_GATE_MACRO_F1:.2f}. Aborting C export."
        )

    print("[INFO] Quality gate passed.")
    return macro_f1


def export_emlearn_header(
    model: RandomForestClassifier, output_header: Path, model_name: str
) -> None:
    output_header.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"[INFO] Converting model to emlearn loadable C code and writing header: {output_header}"
    )
    cmodel = emlearn.convert(model, method="loadable")

    try:
        cmodel.save(file=str(output_header), name=model_name, include_proba=False)
    except TypeError:
        print(
            "[WARN] This emlearn version does not expose include_proba=False on save(); "
            "exporting with default probability support."
        )
        cmodel.save(file=str(output_header), name=model_name)

    print("[INFO] Export complete.")

    # Firmware integration note for STM32:
    #   1. Copy the generated nids_model.h together with the emlearn runtime headers
    #      (for loadable trees this includes eml_trees.h and eml_common.h) into
    #      your STM32 project include path.
    #   2. Build the input vector in the exact FEATURE_COLUMNS order below:
    #      [SYN Flag Cnt, RST Flag Cnt, ACK Flag Cnt, PSH Flag Cnt,
    #       Down/Up Ratio x100, Pkt Len Min, Pkt Len Max]
    #   3. Run inference from firmware with int16_t features:
    #        int16_t features[7] = {syn, rst, ack, psh, down_up_ratio_x100, pkt_min, pkt_max};
    #        int32_t cls = eml_trees_predict(&nids_model, features, 7);
    #      Returned labels are: 0=Normal, 1=DoS, 2=BruteForce.
    print(
        "[INFO] STM32 call pattern: int32_t cls = "
        f"eml_trees_predict(&{model_name}, features, {len(FEATURE_COLUMNS)});"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and export an STM32-friendly Random Forest NIDS model."
    )
    parser.add_argument(
        "--baseline_csv",
        default="local_baseline.csv",
        help="Path to the CSV file containing normal baseline traffic.",
    )
    parser.add_argument(
        "--output_header",
        default="nids_model.h",
        help="Path for the exported emlearn header file.",
    )
    parser.add_argument(
        "--model_name",
        default="nids_model",
        help="C identifier used as the exported model name inside the header.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed used for attack generation, train/test split, and model training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_model_name(args.model_name)

    baseline_csv = Path(args.baseline_csv)
    output_header = Path(args.output_header)

    log_phase(1, "Data Ingestion & Feature Selection")
    normal_df = load_baseline_csv(baseline_csv)

    log_phase(2, "Synthetic Attack Injection")
    dataset_df = generate_synthetic_attacks(normal_df, random_state=args.random_state)

    log_phase(3, "Hardware-Constrained Model Training")
    X_train, X_test, y_train, y_test = split_dataset(
        dataset_df, random_state=args.random_state
    )
    model = train_model(X_train, y_train, random_state=args.random_state)

    log_phase(4, "Validation & Quality Gate")
    validate_model(model, X_test, y_test)

    log_phase(5, "AOT Compilation to C Code (emlearn)")
    export_emlearn_header(model, output_header=output_header, model_name=args.model_name)

    print("\n[INFO] STM32 NIDS pipeline completed successfully.")


if __name__ == "__main__":
    main()
