"""Evaluation metrics: QWK, classification reports, adjacent-error analysis."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from dr_detect import CLASS_NAMES


def quadratic_weighted_kappa(y_true, y_pred) -> float:
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))


def full_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "qwk": quadratic_weighted_kappa(y_true, y_pred),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=[CLASS_NAMES[i] for i in sorted(CLASS_NAMES)],
            zero_division=0,
        ),
    }


def adjacent_error_stats(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    diff = np.abs(y_true - y_pred)
    n = len(diff)
    return {
        "n": int(n),
        "exact": float((diff == 0).mean()),
        "adjacent_error": float(((diff == 1).sum()) / n),
        "far_error": float(((diff >= 2).sum()) / n),
        "mean_abs_error": float(diff.mean()),
    }


def plot_confusion_matrix(y_true, y_pred, path: Path, title: str = "Confusion Matrix") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [CLASS_NAMES[i] for i in range(5)]
    cm = confusion_matrix(y_true, y_pred, labels=list(range(5)))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_history(
    history: dict,
    path: Path,
    title: str = "Training Curves",
    phase_boundary: int | None = None,
) -> Path:
    """Plot merged two-phase curves with an optional vertical marker at fine-tune start."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if phase_boundary is None:
        phase_boundary = history.get("phase_boundary") or history.get("phase1_epochs")

    metric_keys = [
        k
        for k in ("accuracy", "val_accuracy", "qwk", "val_qwk")
        if k in history and isinstance(history[k], list) and len(history[k]) > 0
    ]

    n_cols = 2 if metric_keys else 1
    plt.figure(figsize=(10 if n_cols == 2 else 6, 4))

    ax1 = plt.subplot(1, n_cols, 1)
    for k in ("loss", "val_loss"):
        if k in history and isinstance(history[k], list):
            ax1.plot(history[k], label=k)
    if phase_boundary is not None and int(phase_boundary) > 0:
        ax1.axvline(int(phase_boundary) - 0.5, color="gray", linestyle="--", label="fine-tune start")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch (merged phases)")
    ax1.legend()

    if metric_keys:
        ax2 = plt.subplot(1, n_cols, 2)
        for k in metric_keys:
            ax2.plot(history[k], label=k)
        if phase_boundary is not None and int(phase_boundary) > 0:
            ax2.axvline(int(phase_boundary) - 0.5, color="gray", linestyle="--", label="fine-tune start")
        ax2.set_title("Accuracy / QWK")
        ax2.set_xlabel("Epoch (merged phases)")
        ax2.legend()

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def save_metrics_json(metrics: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {k: v for k, v in metrics.items() if k != "classification_report"}
    serializable["classification_report"] = metrics.get("classification_report", "")
    path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
