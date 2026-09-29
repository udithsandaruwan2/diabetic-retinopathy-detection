"""Central configuration for training and inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Config:
    """Hyperparameters and paths. Document every change in docs/EXPERIMENT_LOG.md."""

    seed: int = 42
    img_size: int = 224
    batch_size: int = 8
    num_classes: int = 5
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15

    # Preprocess
    crop_tol: int = 7
    clahe_clip: float = 2.0
    clahe_tile: int = 8
    graham_sigma: float = 10.0
    unsharp_sigma: float = 1.0
    unsharp_amount: float = 1.5

    # Training (stable TL defaults; raise epochs/batch on Kaggle GPU)
    backbone: str = "EfficientNetB0"
    dropout: float = 0.3
    phase1_lr: float = 3e-4
    phase2_lr: float = 1e-5
    phase1_epochs: int = 8
    phase2_epochs: int = 8
    unfreeze_last: int = 20  # non-BN layers only (see set_fine_tune)
    early_stop_patience: int = 3
    reduce_lr_patience: int = 2
    reduce_lr_factor: float = 0.5
    clipnorm: float = 1.0
    freeze_bn_on_finetune: bool = True
    label_smoothing: float = 0.1  # Softmax only
    qwk_each_epoch: bool = False  # True on GPU; full-val predict is slow on CPU

    # Kaggle dataset
    kaggle_dataset: str = "sachinkumar413/diabetic-retinopathy-dataset"

    # Paths
    data_dir: Path = field(default_factory=lambda: ROOT / "data")
    raw_dir: Path = field(default_factory=lambda: ROOT / "data" / "raw")
    processed_dir: Path = field(default_factory=lambda: ROOT / "data" / "processed")
    models_dir: Path = field(default_factory=lambda: ROOT / "models")
    figures_dir: Path = field(default_factory=lambda: ROOT / "docs" / "figures")
    artifacts_dir: Path = field(default_factory=lambda: ROOT / "artifacts")

    def ensure_dirs(self) -> None:
        for p in (
            self.raw_dir,
            self.processed_dir,
            self.models_dir,
            self.figures_dir,
            self.artifacts_dir / "experiments",
        ):
            p.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = str(v)
        return d


CFG = Config()
