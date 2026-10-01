#!/usr/bin/env python3
"""Add APTOS 2019 (224px) as extra training rows.

Does not modify data/processed/train.csv, val.csv, or test.csv.
Drops any new image whose file hash or 16x16 average hash matches a
picture already in those three splits.

    uv run python scripts/ingest_aptos.py
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from dr_detect import CLASS_NAMES, LABEL_ALIASES
from dr_detect.config import CFG, ROOT

APTOS_SLUG = "sovitrath/diabetic-retinopathy-224x224-2019-data"
DEST = ROOT / "data" / "external" / "aptos2019"


def _ahash(path: Path) -> str | None:
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None
    small = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA)
    bits = (small > float(small.mean())).astype(np.uint8).ravel()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:064x}"


def _file_hash(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    images = list(DEST.rglob("*.png")) + list(DEST.rglob("*.jpg")) + list(DEST.rglob("*.jpeg"))
    if images:
        print(f"APTOS images already present: {len(images)}")
        return
    env = os.environ.copy()
    cred = ROOT / ".kaggle" / "kaggle.json"
    if cred.exists():
        env["KAGGLE_CONFIG_DIR"] = str(ROOT / ".kaggle")
    subprocess.run(
        [
            "uv",
            "run",
            "kaggle",
            "datasets",
            "download",
            "-d",
            APTOS_SLUG,
            "-p",
            str(DEST),
            "--unzip",
        ],
        check=True,
        env=env,
        cwd=str(ROOT),
    )


def _label_from_name(name: str) -> int | None:
    key = name.strip().lower().replace(" ", "_").replace("-", "_")
    if key in LABEL_ALIASES:
        return LABEL_ALIASES[key]
    for alias, idx in LABEL_ALIASES.items():
        if alias in key or key in alias:
            return idx
    return None


def _rows_from_folders() -> list[dict]:
    rows = []
    for path in DEST.rglob("*"):
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        label = _label_from_name(path.parent.name)
        if label is None:
            label = _label_from_name(path.parent.parent.name)
        if label is None:
            continue
        rows.append({"image_path": str(path.resolve()), "label": int(label), "label_name": CLASS_NAMES[int(label)]})
    return rows


def _rows_from_csv() -> list[dict]:
    rows = []
    for csv_path in DEST.rglob("*.csv"):
        frame = pd.read_csv(csv_path)
        cols = {c.lower(): c for c in frame.columns}
        if "diagnosis" not in cols:
            continue
        id_col = cols.get("id_code") or cols.get("image") or cols.get("id")
        if id_col is None:
            continue
        images = {
            p.stem: p
            for p in DEST.rglob("*")
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        }
        for _, rec in frame.iterrows():
            stem = str(rec[id_col])
            path = images.get(stem) or images.get(Path(stem).stem)
            if path is None:
                continue
            label = int(rec[cols["diagnosis"]])
            if label not in CLASS_NAMES:
                continue
            rows.append(
                {
                    "image_path": str(path.resolve()),
                    "label": label,
                    "label_name": CLASS_NAMES[label],
                }
            )
        if rows:
            return rows
    return rows


def main() -> None:
    _download()
    fresh = _rows_from_csv() or _rows_from_folders()
    if not fresh:
        raise SystemExit(f"No labeled APTOS images found under {DEST}")

    locked = pd.concat(
        [
            pd.read_csv(CFG.processed_dir / "train.csv"),
            pd.read_csv(CFG.processed_dir / "val.csv"),
            pd.read_csv(CFG.processed_dir / "test.csv"),
        ],
        ignore_index=True,
    )
    file_hashes: set[str] = set()
    visual_hashes: set[str] = set()
    for path in locked["image_path"]:
        file = Path(path)
        if not file.exists():
            continue
        file_hashes.add(_file_hash(file))
        visual = _ahash(file)
        if visual:
            visual_hashes.add(visual)

    kept = []
    dropped = 0
    unreadable = 0
    for row in fresh:
        file = Path(row["image_path"])
        digest = _file_hash(file)
        visual = _ahash(file)
        if visual is None:
            unreadable += 1
            continue
        if digest in file_hashes or visual in visual_hashes:
            dropped += 1
            continue
        file_hashes.add(digest)
        visual_hashes.add(visual)
        kept.append(row)

    out = pd.DataFrame(kept)
    dest = CFG.processed_dir / "extra_train.csv"
    out.to_csv(dest, index=False)
    report = {
        "source": APTOS_SLUG,
        "found": len(fresh),
        "kept": len(kept),
        "dropped_duplicates": dropped,
        "unreadable": unreadable,
        "by_label": out["label_name"].value_counts().to_dict() if len(out) else {},
        "csv": str(dest),
    }
    report_path = CFG.processed_dir / "aptos_ingest.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
