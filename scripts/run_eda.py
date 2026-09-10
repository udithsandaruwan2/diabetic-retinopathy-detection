#!/usr/bin/env python3
"""EDA + preprocess/aug figures + stratified splits (notebook 01 equivalent)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from dr_detect import CLASS_NAMES
from dr_detect.augment import augment_image
from dr_detect.config import CFG, ROOT
from dr_detect.data import (
    build_dataframe,
    class_weights_dict,
    download_kaggle_dataset,
    stratified_splits,
)
from dr_detect.preprocess import preprocess_steps_gallery


def main():
    CFG.ensure_dirs()
    download_kaggle_dataset(CFG)
    df = build_dataframe(CFG.raw_dir)
    print(df["label_name"].value_counts())

    # Class histogram
    order = [CLASS_NAMES[i] for i in range(5)]
    counts = df["label_name"].value_counts().reindex(order).fillna(0)
    plt.figure(figsize=(8, 4))
    sns.barplot(x=list(counts.index), y=list(counts.values), color="#2c7fb8")
    plt.title("Class distribution (full dataset)")
    plt.ylabel("Count")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(CFG.figures_dir / "class_distribution.png", dpi=150)
    plt.close()

    # Sample gallery
    fig, axes = plt.subplots(1, 5, figsize=(14, 3))
    for i, name in CLASS_NAMES.items():
        subset = df[df["label"] == i]
        if len(subset) == 0:
            axes[i].axis("off")
            continue
        path = subset.sample(1, random_state=CFG.seed).iloc[0]["image_path"]
        bgr = cv2.imread(path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        axes[i].imshow(rgb)
        axes[i].set_title(name)
        axes[i].axis("off")
    plt.suptitle("Sample fundus image per DR stage")
    plt.tight_layout()
    plt.savefig(CFG.figures_dir / "sample_gallery.png", dpi=150)
    plt.close()

    # Preprocess before/after
    sample_path = df.sample(1, random_state=CFG.seed).iloc[0]["image_path"]
    bgr = cv2.imread(sample_path)
    steps = preprocess_steps_gallery(bgr)
    fig, axes = plt.subplots(1, 5, figsize=(14, 3))
    for ax, (k, v) in zip(axes, steps.items()):
        ax.imshow(v)
        ax.set_title(k)
        ax.axis("off")
    plt.suptitle("Preprocess pipeline stages")
    plt.tight_layout()
    plt.savefig(CFG.figures_dir / "preprocess_stages.png", dpi=150)
    plt.close()

    # Augmentation mosaic
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.ravel()
    axes[0].imshow(cv2.resize(rgb, (224, 224)))
    axes[0].set_title("original")
    axes[0].axis("off")
    rng = np.random.default_rng(CFG.seed)
    for i in range(1, 8):
        aug = augment_image(rgb, rng)
        axes[i].imshow(cv2.resize(aug, (224, 224)))
        axes[i].set_title(f"aug {i}")
        axes[i].axis("off")
    plt.suptitle("Train-time augmentation examples")
    plt.tight_layout()
    plt.savefig(CFG.figures_dir / "augmentation_mosaic.png", dpi=150)
    plt.close()

    train_df, val_df, test_df = stratified_splits(df, CFG)
    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        part.to_csv(CFG.processed_dir / f"{name}.csv", index=False)
    df.to_csv(CFG.processed_dir / "all_images.csv", index=False)

    cw = class_weights_dict(train_df)
    (CFG.processed_dir / "class_weights.json").write_text(json.dumps(cw, indent=2))

    split_stats = {
        "n_total": len(df),
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "train_counts": train_df["label_name"].value_counts().to_dict(),
        "val_counts": val_df["label_name"].value_counts().to_dict(),
        "test_counts": test_df["label_name"].value_counts().to_dict(),
        "class_weights": cw,
    }
    (CFG.processed_dir / "split_stats.json").write_text(json.dumps(split_stats, indent=2))

    # Split stacked bar
    split_df = pd.DataFrame(
        {
            "train": train_df["label"].value_counts(),
            "val": val_df["label"].value_counts(),
            "test": test_df["label"].value_counts(),
        }
    ).fillna(0).sort_index()
    split_df.index = [CLASS_NAMES[i] for i in split_df.index]
    split_df.plot(kind="bar", figsize=(8, 4), title="Stratified split counts")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(CFG.figures_dir / "split_counts.png", dpi=150)
    plt.close()

    # Decision log append
    with (ROOT / "docs" / "DECISION_LOG.md").open("a", encoding="utf-8") as f:
        f.write(
            f"\n| 2026-09-09 | EDA complete: n={len(df)}; splits written; figures saved | — | See `data/processed/split_stats.json` |\n"
        )

    print(json.dumps(split_stats, indent=2))
    print("Figures written to", CFG.figures_dir)


if __name__ == "__main__":
    main()
