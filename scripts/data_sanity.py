#!/usr/bin/env python3
"""Quick data sanity check before training (unreadable / near-black / label hist)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from dr_detect.config import CFG, ROOT


def main(max_scan: int = 500):
    train = pd.read_csv(CFG.processed_dir / "train.csv")
    sample = train.sample(n=min(max_scan, len(train)), random_state=CFG.seed)
    unreadable = 0
    near_black = 0
    for path in sample["image_path"]:
        img = cv2.imread(str(path))
        if img is None:
            unreadable += 1
            continue
        if float(img.mean()) < 5.0:
            near_black += 1

    hist = train["label"].value_counts().sort_index().to_dict()
    report = {
        "train_n": int(len(train)),
        "scanned": int(len(sample)),
        "unreadable": unreadable,
        "near_black": near_black,
        "label_histogram": {str(k): int(v) for k, v in hist.items()},
    }
    out = CFG.processed_dir / "data_sanity.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {out}")
    return report


if __name__ == "__main__":
    main()
