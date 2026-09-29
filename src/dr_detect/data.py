"""Dataset download, dataframe construction, stratified splits, Keras Sequence."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import Sequence

from dr_detect import CLASS_NAMES, LABEL_ALIASES
from dr_detect.config import CFG, Config
from dr_detect.preprocess import load_and_preprocess, preprocess_fundus


def download_kaggle_dataset(cfg: Config = CFG) -> Path:
    """Download and unzip the primary Kaggle dataset into data/raw/."""
    from dr_detect.config import ROOT

    cfg.ensure_dirs()
    # Already extracted?
    if any(cfg.raw_dir.rglob("*.png")) or any(cfg.raw_dir.rglob("*.jpeg")) or any(cfg.raw_dir.rglob("*.jpg")):
        return cfg.raw_dir

    env = os.environ.copy()
    cred = ROOT / ".kaggle" / "kaggle.json"
    if cred.exists():
        env["KAGGLE_CONFIG_DIR"] = str(ROOT / ".kaggle")

    cmd = [
        "uv",
        "run",
        "kaggle",
        "datasets",
        "download",
        "-d",
        cfg.kaggle_dataset,
        "-p",
        str(cfg.raw_dir),
        "--unzip",
    ]
    subprocess.run(cmd, check=True, env=env, cwd=str(ROOT))
    return cfg.raw_dir


def _normalize_label(name: str) -> int | None:
    key = name.strip().lower().replace(" ", "_").replace("-", "_")
    if key in LABEL_ALIASES:
        return LABEL_ALIASES[key]
    # fuzzy
    for alias, idx in LABEL_ALIASES.items():
        if alias in key or key in alias:
            return idx
    return None


def build_dataframe(raw_dir: Path | None = None) -> pd.DataFrame:
    """Scan image folders / csv and return path, label, label_name."""
    raw_dir = Path(raw_dir or CFG.raw_dir)
    rows: list[dict] = []

    # Prefer train.csv if present (APTOS style)
    csv_candidates = list(raw_dir.rglob("*.csv"))
    image_files = (
        list(raw_dir.rglob("*.png"))
        + list(raw_dir.rglob("*.jpg"))
        + list(raw_dir.rglob("*.jpeg"))
    )

    # Folder-structured: .../No_DR/img.png
    for img in image_files:
        parent = img.parent.name
        label = _normalize_label(parent)
        if label is None:
            # try grandparent
            label = _normalize_label(img.parent.parent.name)
        if label is None:
            continue
        rows.append(
            {
                "image_path": str(img.resolve()),
                "label": int(label),
                "label_name": CLASS_NAMES[int(label)],
            }
        )

    if not rows and csv_candidates:
        # Map id_code + diagnosis
        for csv_path in csv_candidates:
            df = pd.read_csv(csv_path)
            cols = {c.lower(): c for c in df.columns}
            id_col = cols.get("id_code") or cols.get("image") or cols.get("id")
            y_col = cols.get("diagnosis") or cols.get("level") or cols.get("label")
            if not id_col or not y_col:
                continue
            by_stem = {p.stem: p for p in image_files}
            for _, r in df.iterrows():
                stem = str(r[id_col])
                path = by_stem.get(stem) or by_stem.get(stem.replace(".png", ""))
                if path is None:
                    # search
                    matches = [p for p in image_files if stem in p.stem]
                    if not matches:
                        continue
                    path = matches[0]
                label = int(r[y_col])
                rows.append(
                    {
                        "image_path": str(path.resolve()),
                        "label": label,
                        "label_name": CLASS_NAMES.get(label, str(label)),
                    }
                )

    if not rows:
        raise RuntimeError(
            f"No labeled images found under {raw_dir}. "
            "Check dataset extraction layout."
        )

    out = pd.DataFrame(rows).drop_duplicates(subset=["image_path"]).reset_index(drop=True)
    return out


def stratified_splits(
    df: pd.DataFrame,
    cfg: Config = CFG,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """70/15/15 stratified by label."""
    assert abs(cfg.train_frac + cfg.val_frac + cfg.test_frac - 1.0) < 1e-6
    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - cfg.train_frac),
        stratify=df["label"],
        random_state=cfg.seed,
    )
    relative_test = cfg.test_frac / (cfg.val_frac + cfg.test_frac)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test,
        stratify=temp_df["label"],
        random_state=cfg.seed,
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def oversample_rare_classes(
    df: pd.DataFrame,
    labels: tuple[int, ...] = (1, 3),
    seed: int = CFG.seed,
) -> pd.DataFrame:
    """Duplicate Mild (1) and Severe (3) until each matches the majority class count.

    Use on the training split only. Validation and test stay one row per image.
    """
    counts = df["label"].value_counts()
    target = int(counts.max())
    rng = np.random.default_rng(seed)
    parts = [df]
    for lab in labels:
        subset = df[df["label"] == int(lab)]
        n = len(subset)
        if n == 0 or n >= target:
            continue
        extra_idx = rng.choice(subset.index.to_numpy(), size=target - n, replace=True)
        parts.append(df.loc[extra_idx])
    out = pd.concat(parts, ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def class_weights_dict(train_df: pd.DataFrame) -> dict[int, float]:
    classes = np.unique(train_df["label"].values)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=train_df["label"].values,
    )
    return {int(c): float(w) for c, w in zip(classes, weights)}


class FundusSequence(Sequence):
    """Keras Sequence that loads + preprocesses (+ optional augment)."""

    def __init__(
        self,
        df: pd.DataFrame,
        batch_size: int = CFG.batch_size,
        shuffle: bool = True,
        augment: bool = False,
        coral: bool = False,
        seed: int = CFG.seed,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.df = df.reset_index(drop=True)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.coral = coral
        self.seed = seed
        self.indices = np.arange(len(self.df))
        self.on_epoch_end()

    def __len__(self) -> int:
        return int(np.ceil(len(self.df) / self.batch_size))

    def on_epoch_end(self) -> None:
        if self.shuffle:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(self.indices)
            self.seed += 1

    def __getitem__(self, idx: int):
        from dr_detect.augment import augment_image

        batch_idx = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch = self.df.iloc[batch_idx]
        xs, ys = [], []
        for _, row in batch.iterrows():
            bgr = cv2.imread(row["image_path"], cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            # augment in uint8 RGB space before final model preprocess
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if self.augment:
                rgb = augment_image(rgb)
            x = preprocess_fundus(rgb, is_bgr=False, for_model=True)
            xs.append(x)
            ys.append(int(row["label"]))
        x_arr = np.stack(xs).astype(np.float32)
        y_arr = np.asarray(ys, dtype=np.int32)
        if self.coral:
            # levels: (batch, K-1) where levels[k] = 1 if y > k
            k = CFG.num_classes - 1
            levels = np.zeros((len(y_arr), k), dtype=np.float32)
            for i, y in enumerate(y_arr):
                levels[i, :y] = 1.0
            return x_arr, levels
        return x_arr, y_arr
