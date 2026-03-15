"""Evaluation utilities."""

from typing import Dict, List

import json
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_curve,
)
from sklearn.preprocessing import label_binarize


def _align_proba_columns(y_proba: np.ndarray, model, class_names: List[str]) -> np.ndarray:
    """Align probability columns to the requested class_names order."""
    if not hasattr(model, "classes_"):
        return y_proba

    model_classes = [str(c) for c in model.classes_]
    class_names = [str(c) for c in class_names]
    cls_to_idx = {name: idx for idx, name in enumerate(model_classes)}
    missing = [name for name in class_names if name not in cls_to_idx]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"Model probabilities do not contain expected classes: {missing_text}")

    indices = [cls_to_idx[name] for name in class_names]
    return y_proba[:, indices]


def _plot_multiclass_roc(
    y_true: np.ndarray, y_proba: np.ndarray, class_names: List[str], out_path: str
) -> Dict[str, float]:
    y_bin = label_binarize(y_true, classes=class_names)
    fig, ax = plt.subplots(figsize=(9, 7))
    auc_by_class: Dict[str, float] = {}

    for idx, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_bin[:, idx], y_proba[:, idx])
        roc_auc = float(auc(fpr, tpr))
        auc_by_class[class_name] = roc_auc
        ax.plot(fpr, tpr, label=f"{class_name} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_title("Multiclass ROC Curves")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)
    return auc_by_class


def _plot_multiclass_pr(
    y_true: np.ndarray, y_proba: np.ndarray, class_names: List[str], out_path: str
) -> Dict[str, float]:
    y_bin = label_binarize(y_true, classes=class_names)
    auc_by_class: Dict[str, float] = {}
    fig, ax = plt.subplots(figsize=(9, 7))

    for idx, class_name in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(y_bin[:, idx], y_proba[:, idx])
        pr_auc = float(auc(recall, precision))
        auc_by_class[class_name] = pr_auc
        ax.plot(recall, precision, label=f"{class_name} (AUC={pr_auc:.3f})")

    ax.set_title("Multiclass Precision-Recall Curves")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)
    return auc_by_class


def evaluate_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    class_names: List[str],
    reports_dir: str,
    prefix: str = "test",
) -> Dict[str, object]:
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X) if hasattr(model, "predict_proba") else None
    if y_proba is not None:
        y_proba = _align_proba_columns(y_proba, model, class_names)

    acc = float(accuracy_score(y, y_pred))
    macro_f1 = float(f1_score(y, y_pred, average="macro"))

    report = classification_report(y, y_pred, labels=class_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(y, y_pred, labels=class_names)

    os.makedirs(reports_dir, exist_ok=True)

    report_path = os.path.join(reports_dir, f"{prefix}_report.json")
    payload = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class": report,
        "confusion_matrix": cm.tolist(),
    }

    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", colorbar=True, xticks_rotation=45)
    plt.tight_layout()
    cm_path = os.path.join(reports_dir, f"{prefix}_confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close(fig)

    roc_path = None
    pr_path = None
    roc_auc_by_class = None
    pr_auc_by_class = None

    if y_proba is not None and y_proba.shape[1] == len(class_names):
        roc_path = os.path.join(reports_dir, f"{prefix}_roc_curve.png")
        pr_path = os.path.join(reports_dir, f"{prefix}_pr_curve.png")
        roc_auc_by_class = _plot_multiclass_roc(y, y_proba, class_names, roc_path)
        pr_auc_by_class = _plot_multiclass_pr(y, y_proba, class_names, pr_path)
        payload["roc_auc_by_class"] = roc_auc_by_class
        payload["pr_auc_by_class"] = pr_auc_by_class

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "report_path": report_path,
        "confusion_matrix_path": cm_path,
        "roc_curve_path": roc_path,
        "pr_curve_path": pr_path,
        "roc_auc_by_class": roc_auc_by_class,
        "pr_auc_by_class": pr_auc_by_class,
    }
