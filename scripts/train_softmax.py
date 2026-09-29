#!/usr/bin/env python3
"""Train Softmax EfficientNetB0 baseline (two-phase TL). Reuses existing splits when present."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import tensorflow as tf

from dr_detect.config import CFG, ROOT
from dr_detect.data import (
    FundusSequence,
    build_dataframe,
    class_weights_dict,
    download_kaggle_dataset,
    oversample_rare_classes,
    stratified_splits,
)
from dr_detect.metrics import plot_history
from dr_detect.models import build_softmax_model
from dr_detect.train import two_phase_train


def main():
    CFG.ensure_dirs()
    tf.keras.utils.set_random_seed(CFG.seed)

    download_kaggle_dataset(CFG)
    train_csv = CFG.processed_dir / "train.csv"
    val_csv = CFG.processed_dir / "val.csv"
    test_csv = CFG.processed_dir / "test.csv"

    if train_csv.exists() and val_csv.exists() and test_csv.exists():
        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)
        print(f"Reusing existing splits: train={len(train_df)} val={len(val_df)}")
    else:
        df = build_dataframe(CFG.raw_dir)
        train_df, val_df, test_df = stratified_splits(df, CFG)
        CFG.processed_dir.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(train_csv, index=False)
        val_df.to_csv(val_csv, index=False)
        test_df.to_csv(test_csv, index=False)

    cw_path = CFG.processed_dir / "class_weights.json"
    if cw_path.exists():
        cw = {int(k): float(v) for k, v in json.loads(cw_path.read_text()).items()}
    else:
        cw = class_weights_dict(train_df)
        cw_path.write_text(json.dumps(cw, indent=2))

    # Class weights stay on the original train counts. Oversample Mild/Severe for fitting only.
    train_fit = oversample_rare_classes(train_df)
    print(f"Train oversample (Mild+Severe → majority): {len(train_df)} → {len(train_fit)}")
    train_seq = FundusSequence(train_fit, batch_size=CFG.batch_size, shuffle=True, augment=True)
    val_seq = FundusSequence(val_df, batch_size=CFG.batch_size, shuffle=False, augment=False)

    model = build_softmax_model()
    out_dir = CFG.artifacts_dir / "experiments" / "softmax_acc"
    model, history, final_path = two_phase_train(
        model,
        train_seq,
        val_seq,
        coral=False,
        out_dir=out_dir,
        class_weight=cw,
        cfg=CFG,
    )

    plot_history(history, CFG.figures_dir / "softmax_curves.png", title="Softmax Training (acc)")
    model.save(CFG.models_dir / "softmax_effb0.keras")
    print(f"Saved {final_path} and models/softmax_effb0.keras")

    log_path = ROOT / "docs" / "EXPERIMENT_LOG.md"
    with log_path.open("a", encoding="utf-8") as f:
        f.write("\n## EXP-ACC-001a — Softmax richer head + oversample\n\n")
        f.write(f"- Config: `{json.dumps(CFG.to_dict())}`\n")
        f.write(f"- Artifact: `{final_path}`\n")
        f.write(f"- Curves: `docs/figures/softmax_curves.png`\n")
        f.write(f"- Selection: `{out_dir / 'best_selection.json'}`\n")


if __name__ == "__main__":
    main()
