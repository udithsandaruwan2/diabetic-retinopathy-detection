#!/usr/bin/env python3
"""Download the primary Kaggle DR dataset."""

from dr_detect.config import CFG
from dr_detect.data import download_kaggle_dataset, build_dataframe

if __name__ == "__main__":
    CFG.ensure_dirs()
    path = download_kaggle_dataset(CFG)
    df = build_dataframe(path)
    print(f"Downloaded under: {path}")
    print(df["label_name"].value_counts())
    out = CFG.processed_dir / "all_images.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {out} ({len(df)} rows)")
