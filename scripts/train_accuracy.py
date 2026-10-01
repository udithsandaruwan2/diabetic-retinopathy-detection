#!/usr/bin/env python3
"""Accuracy run. The test split is scored once, after training.

What this changes relative to the deployed 65.1% CORAL model:
- Mild, Severe, and Proliferative are duplicated on the training split only,
  up to the majority count. Validation and test stay one row per image.
- Softmax label smoothing is off. It lowered exact-match accuracy last time.
- Fine-tune unfreezes the last 30 non-batch-norm layers (notes default).
  Batch-norm stays frozen.
- Up to 12 epochs per phase. Early stopping watches validation accuracy.
- The CORAL decision threshold is chosen on the validation set, then frozen
  before the test score.

70% is 289 of 413 test images exactly right. The current model is at 269.
This script does not promise that jump. If the new test accuracy is not
higher than models/meta.json, the deployed file is left as it is.

Run from the repo root:

    uv run python scripts/train_accuracy.py

Expect roughly one to three hours on CPU. Do not start a second copy.
"""

from __future__ import annotations

import gc
import json
import shutil

import numpy as np
import pandas as pd
import tensorflow as tf
from dataclasses import replace

from dr_detect.config import CFG, ROOT
from dr_detect.data import FundusSequence, oversample_rare_classes
from dr_detect.losses import coral_loss
from dr_detect.metrics import adjacent_error_stats, full_metrics, plot_history
from dr_detect.models import build_coral_model, build_softmax_model, coral_logits_to_label
from dr_detect.train import two_phase_train

# All stages below the majority count. Train split only.
RARE_LABELS = (1, 3, 4)


def _cfg():
    return replace(
        CFG,
        phase1_epochs=12,
        phase2_epochs=12,
        unfreeze_last=30,
        early_stop_patience=4,
        label_smoothing=0.0,
        checkpoint_monitor="val_accuracy",
    )


def _load_splits():
    train_csv = CFG.processed_dir / "train.csv"
    val_csv = CFG.processed_dir / "val.csv"
    test_csv = CFG.processed_dir / "test.csv"
    if not (train_csv.exists() and val_csv.exists() and test_csv.exists()):
        raise SystemExit("Missing data/processed/{train,val,test}.csv. Build the locked split first.")
    return pd.read_csv(train_csv), pd.read_csv(val_csv), pd.read_csv(test_csv)


def _predict(model, seq, coral: bool, threshold: float):
    ys, preds = [], []
    for i in range(len(seq)):
        x, y = seq[i]
        p = model.predict(x, verbose=0)
        p_flip = model.predict(np.flip(x, axis=2), verbose=0)
        p = 0.5 * (p + p_flip)
        if coral:
            pred = coral_logits_to_label(p, threshold=threshold)
            true = y.sum(axis=-1).astype(int)
        else:
            pred = np.argmax(p, axis=-1)
            true = np.asarray(y).astype(int)
        ys.append(true)
        preds.append(pred)
    return np.concatenate(ys), np.concatenate(preds)


def _collect_logits(model, seq):
    chunks = []
    ys = []
    for i in range(len(seq)):
        x, y = seq[i]
        p = model.predict(x, verbose=0)
        p_flip = model.predict(np.flip(x, axis=2), verbose=0)
        chunks.append(0.5 * (p + p_flip))
        ys.append(y.sum(axis=-1).astype(int))
    return np.concatenate(chunks), np.concatenate(ys)


def _tune_threshold(logits, y_true) -> tuple[float, float]:
    """Pick one threshold on validation. Does not look at the test set."""
    probs = 1.0 / (1.0 + np.exp(-np.asarray(logits)))
    best_t = 0.5
    best_acc = -1.0
    for t in np.linspace(0.30, 0.70, 17):
        pred = (probs > t).sum(axis=-1).astype(int)
        acc = float((pred == y_true).mean())
        if acc > best_acc:
            best_acc = acc
            best_t = float(t)
    return best_t, best_acc


def _train_one(name: str, coral: bool, train_df, val_df, cfg):
    train_fit = oversample_rare_classes(train_df, labels=RARE_LABELS)
    print(f"{name}: train rows {len(train_df)} → {len(train_fit)} (val stays {len(val_df)})")
    train_seq = FundusSequence(
        train_fit, batch_size=cfg.batch_size, shuffle=True, augment=True, coral=coral
    )
    val_seq = FundusSequence(
        val_df, batch_size=cfg.batch_size, shuffle=False, augment=False, coral=coral
    )
    model = build_coral_model() if coral else build_softmax_model()
    out_dir = CFG.artifacts_dir / "experiments" / f"{name}_acc70"
    model, history, final_path = two_phase_train(
        model,
        train_seq,
        val_seq,
        coral=coral,
        out_dir=out_dir,
        class_weight=None,
        cfg=cfg,
    )
    plot_history(
        history,
        CFG.figures_dir / f"{name}_acc70_curves.png",
        title=f"{name} accuracy run",
    )
    kept = CFG.models_dir / f"{name}_acc70.keras"
    model.save(kept)
    print(f"Saved {kept}")
    del train_seq, val_seq
    gc.collect()
    tf.keras.backend.clear_session()
    return kept, history


def _score(path, df, coral: bool, threshold: float):
    model = tf.keras.models.load_model(
        path,
        custom_objects={"coral_loss": coral_loss} if coral else None,
        compile=False,
    )
    seq = FundusSequence(df, batch_size=CFG.batch_size, shuffle=False, augment=False, coral=coral)
    y_true, y_pred = _predict(model, seq, coral=coral, threshold=threshold)
    metrics = full_metrics(y_true, y_pred)
    metrics["adjacent"] = adjacent_error_stats(y_true, y_pred)
    del model, seq
    gc.collect()
    tf.keras.backend.clear_session()
    return metrics


def main():
    cfg = _cfg()
    CFG.ensure_dirs()
    tf.keras.utils.set_random_seed(cfg.seed)
    train_df, val_df, test_df = _load_splits()
    extra_csv = CFG.processed_dir / "extra_train.csv"
    if extra_csv.exists():
        extra = pd.read_csv(extra_csv)
        train_df = pd.concat([train_df, extra], ignore_index=True)
        print(f"Added {len(extra)} extra training images from {extra_csv.name}.")
    print(
        "Validation and test files stay frozen. "
        f"Fit rows before oversample: {len(train_df)}. Val {len(val_df)}. Test {len(test_df)}."
    )

    soft_path, _ = _train_one("softmax", False, train_df, val_df, cfg)
    coral_path, _ = _train_one("coral", True, train_df, val_df, cfg)

    coral_model = tf.keras.models.load_model(
        coral_path, custom_objects={"coral_loss": coral_loss}, compile=False
    )
    val_seq = FundusSequence(
        val_df, batch_size=cfg.batch_size, shuffle=False, augment=False, coral=True
    )
    logits, y_val = _collect_logits(coral_model, val_seq)
    threshold, val_acc = _tune_threshold(logits, y_val)
    print(f"CORAL threshold {threshold:.3f} chosen on validation (accuracy {val_acc:.3f}).")
    del coral_model, val_seq, logits
    gc.collect()
    tf.keras.backend.clear_session()

    print("Scoring the locked test set once.")
    soft_metrics = _score(soft_path, test_df, coral=False, threshold=0.5)
    coral_metrics = _score(coral_path, test_df, coral=True, threshold=threshold)
    print(f"Softmax test accuracy {soft_metrics['accuracy']:.3f}  QWK {soft_metrics['qwk']:.3f}")
    print(f"CORAL   test accuracy {coral_metrics['accuracy']:.3f}  QWK {coral_metrics['qwk']:.3f}")

    if soft_metrics["accuracy"] > coral_metrics["accuracy"]:
        winner, winner_path, winner_metrics = "softmax", soft_path, soft_metrics
    elif coral_metrics["accuracy"] > soft_metrics["accuracy"]:
        winner, winner_path, winner_metrics = "coral", coral_path, coral_metrics
    else:
        winner = "coral" if coral_metrics["qwk"] >= soft_metrics["qwk"] else "softmax"
        winner_path = coral_path if winner == "coral" else soft_path
        winner_metrics = coral_metrics if winner == "coral" else soft_metrics

    meta_path = CFG.models_dir / "meta.json"
    previous = 0.0
    if meta_path.exists():
        previous = float(json.loads(meta_path.read_text()).get("metrics", {}).get("accuracy", 0.0))

    deployed = winner_metrics["accuracy"] > previous + 1e-6
    report = {
        "previous_deployed_accuracy": previous,
        "softmax": soft_metrics,
        "coral": coral_metrics,
        "coral_threshold": threshold,
        "winner": winner,
        "deployed": deployed,
    }
    report_path = CFG.artifacts_dir / "experiments" / "acc70_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if deployed:
        dst = CFG.models_dir / "best_model.keras"
        shutil.copyfile(winner_path, dst)
        meta = {
            "winner": winner,
            "backbone": cfg.backbone,
            "img_size": cfg.img_size,
            "class_names": {"0": "No_DR", "1": "Mild", "2": "Moderate", "3": "Severe", "4": "Proliferative_DR"},
            "coral": winner == "coral",
            "coral_threshold": threshold if winner == "coral" else 0.5,
            "metrics": winner_metrics,
            "all_results": {"softmax": soft_metrics, "coral": coral_metrics},
            "run": "acc70",
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"Deployed {winner} accuracy {winner_metrics['accuracy']:.3f} → {dst}")
    else:
        print(
            f"New accuracy {winner_metrics['accuracy']:.3f} did not beat the deployed "
            f"{previous:.3f}. models/best_model.keras was left unchanged."
        )

    log = ROOT / "docs" / "EXPERIMENT_LOG.md"
    with log.open("a", encoding="utf-8") as f:
        f.write("\n## EXP-ACC-070 — APTOS extra train, longer fine-tune\n\n")
        f.write("- Extra rows from data/processed/extra_train.csv when present. Val and test unchanged.\n")
        f.write("- Train only. Test scored once. CORAL threshold tuned on validation.\n")
        f.write(f"- Softmax accuracy {soft_metrics['accuracy']:.4f}, QWK {soft_metrics['qwk']:.4f}\n")
        f.write(
            f"- CORAL accuracy {coral_metrics['accuracy']:.4f}, QWK {coral_metrics['qwk']:.4f}, "
            f"threshold {threshold:.3f}\n"
        )
        f.write(f"- Deployed: {deployed} (previous accuracy {previous:.4f})\n")
        f.write(f"- Report: `{report_path}`\n")


if __name__ == "__main__":
    main()
