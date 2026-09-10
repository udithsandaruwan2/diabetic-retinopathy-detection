#!/usr/bin/env python3
"""Train CORAL ordinal EfficientNetB0 (two-phase TL). Reuses existing splits."""

from __future__ import annotations

import json

import pandas as pd
import tensorflow as tf

from dr_detect.config import CFG, ROOT
from dr_detect.data import FundusSequence, download_kaggle_dataset
from dr_detect.metrics import plot_history
from dr_detect.models import build_coral_model
from dr_detect.train import two_phase_train


def main():
    CFG.ensure_dirs()
    tf.keras.utils.set_random_seed(CFG.seed)
    download_kaggle_dataset(CFG)

    train_df = pd.read_csv(CFG.processed_dir / "train.csv")
    val_df = pd.read_csv(CFG.processed_dir / "val.csv")

    train_seq = FundusSequence(
        train_df, batch_size=CFG.batch_size, shuffle=True, augment=True, coral=True
    )
    val_seq = FundusSequence(
        val_df, batch_size=CFG.batch_size, shuffle=False, augment=False, coral=True
    )

    model = build_coral_model()
    out_dir = CFG.artifacts_dir / "experiments" / "coral_stable"
    model, history, final_path = two_phase_train(
        model,
        train_seq,
        val_seq,
        coral=True,
        out_dir=out_dir,
        class_weight=None,
        cfg=CFG,
    )

    plot_history(history, CFG.figures_dir / "coral_curves.png", title="CORAL Training (stable)")
    model.save(CFG.models_dir / "coral_effb0.keras")
    print(f"Saved {final_path}")

    log_path = ROOT / "docs" / "EXPERIMENT_LOG.md"
    with log_path.open("a", encoding="utf-8") as f:
        f.write("\n## EXP-STABLE-001b — CORAL EfficientNetB0 (BN-safe)\n\n")
        f.write(f"- Config: `{json.dumps(CFG.to_dict())}`\n")
        f.write(f"- Artifact: `{final_path}`\n")
        f.write(f"- Curves: `docs/figures/coral_curves.png`\n")
        f.write(f"- Selection: `{out_dir / 'best_selection.json'}`\n")


if __name__ == "__main__":
    main()
