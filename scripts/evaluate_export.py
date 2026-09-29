#!/usr/bin/env python3
"""Evaluate Softmax vs CORAL on held-out test; export best model + meta.json."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from dr_detect import CLASS_NAMES
from dr_detect.config import CFG, ROOT
from dr_detect.data import FundusSequence
from dr_detect.explain import make_gradcam_heatmap, overlay_gradcam
from dr_detect.metrics import (
    adjacent_error_stats,
    full_metrics,
    plot_confusion_matrix,
    save_metrics_json,
)
from dr_detect.models import coral_logits_to_label
from dr_detect.preprocess import preprocess_fundus, preprocess_steps_gallery
import cv2


def _load(path: Path):
    return tf.keras.models.load_model(
        path,
        custom_objects={"coral_loss": __import__("dr_detect.losses", fromlist=["coral_loss"]).coral_loss},
        compile=False,
    )


def _predict_tta(model, x: np.ndarray) -> np.ndarray:
    """Average the forward pass with a horizontal flip. Scoring only — not training labels."""
    p = model.predict(x, verbose=0)
    p_flip = model.predict(np.flip(x, axis=2), verbose=0)
    return 0.5 * (p + p_flip)


def eval_model(model, seq, coral: bool, tta: bool = True):
    ys, preds = [], []
    for i in range(len(seq)):
        x, y = seq[i]
        p = _predict_tta(model, x) if tta else model.predict(x, verbose=0)
        if coral:
            pred = coral_logits_to_label(p)
            true = y.sum(axis=-1).astype(int)
        else:
            pred = np.argmax(p, axis=-1)
            true = y
        ys.append(true)
        preds.append(pred)
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(preds)
    return y_true, y_pred, full_metrics(y_true, y_pred), adjacent_error_stats(y_true, y_pred)


def main():
    CFG.ensure_dirs()
    test_df = pd.read_csv(CFG.processed_dir / "test.csv")
    soft_path = CFG.models_dir / "softmax_effb0.keras"
    coral_path = CFG.models_dir / "coral_effb0.keras"

    results = {}

    if soft_path.exists():
        soft = _load(soft_path)
        seq = FundusSequence(test_df, batch_size=CFG.batch_size, shuffle=False, augment=False, coral=False)
        yt, yp, metrics, adj = eval_model(soft, seq, coral=False)
        plot_confusion_matrix(yt, yp, CFG.figures_dir / "cm_softmax.png", "Softmax Test CM")
        save_metrics_json({**metrics, "adjacent": adj}, CFG.artifacts_dir / "softmax_test_metrics.json")
        results["softmax"] = {**{k: v for k, v in metrics.items() if k != "classification_report"}, "adjacent": adj}
        print("Softmax\n", metrics["classification_report"], "QWK", metrics["qwk"])

    if coral_path.exists():
        coral = _load(coral_path)
        seq_c = FundusSequence(test_df, batch_size=CFG.batch_size, shuffle=False, augment=False, coral=True)
        yt, yp, metrics, adj = eval_model(coral, seq_c, coral=True)
        plot_confusion_matrix(yt, yp, CFG.figures_dir / "cm_coral.png", "CORAL Test CM")
        save_metrics_json({**metrics, "adjacent": adj}, CFG.artifacts_dir / "coral_test_metrics.json")
        results["coral"] = {**{k: v for k, v in metrics.items() if k != "classification_report"}, "adjacent": adj}
        print("CORAL\n", metrics["classification_report"], "QWK", metrics["qwk"])

    # Pick winner by QWK
    winner = None
    best_qwk = -1.0
    for name, m in results.items():
        if m["qwk"] > best_qwk:
            best_qwk = m["qwk"]
            winner = name

    if winner is None:
        raise SystemExit("No models found to evaluate. Train first.")

    src = soft_path if winner == "softmax" else coral_path
    dst = CFG.models_dir / "best_model.keras"
    dst.write_bytes(src.read_bytes())

    meta = {
        "winner": winner,
        "backbone": CFG.backbone,
        "img_size": CFG.img_size,
        "class_names": CLASS_NAMES,
        "coral": winner == "coral",
        "metrics": results[winner],
        "all_results": results,
        "preprocess": {
            "clahe_clip": CFG.clahe_clip,
            "graham_sigma": CFG.graham_sigma,
            "unsharp_amount": CFG.unsharp_amount,
        },
    }
    (CFG.models_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Comparison chart
    if len(results) >= 1:
        labels = list(results.keys())
        qwks = [results[k]["qwk"] for k in labels]
        accs = [results[k]["accuracy"] for k in labels]
        f1s = [results[k]["f1_macro"] for k in labels]
        x = np.arange(len(labels))
        w = 0.25
        plt.figure(figsize=(8, 4))
        plt.bar(x - w, qwks, w, label="QWK")
        plt.bar(x, accs, w, label="Accuracy")
        plt.bar(x + w, f1s, w, label="F1 macro")
        plt.xticks(x, labels)
        plt.ylim(0, 1)
        plt.title("Softmax vs CORAL (test)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(CFG.figures_dir / "softmax_vs_coral.png", dpi=150)
        plt.close()

    # Grad-CAM gallery on a few test images
    model = _load(dst)
    coral_flag = winner == "coral"
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    sample = test_df.sample(n=min(4, len(test_df)), random_state=CFG.seed)
    for i, (_, row) in enumerate(sample.iterrows()):
        bgr = cv2.imread(row["image_path"])
        steps = preprocess_steps_gallery(bgr)
        x = preprocess_fundus(bgr, is_bgr=True, for_model=True)[None, ...]
        heat = make_gradcam_heatmap(x, model, coral=coral_flag)
        overlay = overlay_gradcam(steps["unsharp"], heat)
        axes[0, i].imshow(steps["raw"])
        axes[0, i].set_title(f"True {CLASS_NAMES[int(row['label'])]}")
        axes[0, i].axis("off")
        axes[1, i].imshow(overlay)
        axes[1, i].set_title("Grad-CAM")
        axes[1, i].axis("off")
    plt.tight_layout()
    plt.savefig(CFG.figures_dir / "gradcam_gallery.png", dpi=150)
    plt.close()

    print(f"Winner={winner} QWK={best_qwk:.4f} -> {dst}")

    with (ROOT / "docs" / "EXPERIMENT_LOG.md").open("a", encoding="utf-8") as f:
        f.write("\n## EXP-003 — Test evaluation & export\n\n")
        f.write(f"- Winner: **{winner}** (QWK={best_qwk:.4f})\n")
        f.write(f"- meta: `models/meta.json`\n")
        f.write(f"- Results: `{json.dumps(results)}`\n")


if __name__ == "__main__":
    main()
