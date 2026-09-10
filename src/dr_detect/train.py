"""Training helpers: compile, callbacks, two-phase fit, prediction."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import cohen_kappa_score
from tensorflow.keras import callbacks

from dr_detect.config import CFG, Config
from dr_detect.losses import coral_loss
from dr_detect.models import coral_logits_to_label, freeze_batch_norm, set_fine_tune


class QWKCallback(callbacks.Callback):
    """Compute quadratic weighted kappa on a validation Sequence each epoch."""

    def __init__(self, val_seq, coral: bool = False):
        super().__init__()
        self.val_seq = val_seq
        self.coral = coral
        self.val_qwk: list[float] = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        ys, preds = [], []
        for i in range(len(self.val_seq)):
            x, y = self.val_seq[i]
            p = self.model.predict(x, verbose=0)
            if self.coral:
                pred = coral_logits_to_label(p)
                # y are levels → reconstruct labels
                true = y.sum(axis=-1).astype(int)
            else:
                pred = np.argmax(p, axis=-1)
                true = y
            ys.append(true)
            preds.append(pred)
        y_true = np.concatenate(ys)
        y_pred = np.concatenate(preds)
        qwk = float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
        logs["val_qwk"] = qwk
        self.val_qwk.append(qwk)
        print(f" — val_qwk: {qwk:.4f}")


def make_callbacks(
    out_dir: Path,
    val_seq,
    coral: bool = False,
    cfg: Config = CFG,
) -> list:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / ("best_coral.keras" if coral else "best_softmax.keras")
    cbs = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=cfg.early_stop_patience,
            restore_best_weights=True,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=cfg.reduce_lr_factor,
            patience=cfg.reduce_lr_patience,
            min_lr=1e-7,
        ),
        callbacks.ModelCheckpoint(
            filepath=str(ckpt),
            monitor="val_loss",
            save_best_only=True,
        ),
        callbacks.CSVLogger(str(out_dir / "history.csv")),
    ]
    if cfg.qwk_each_epoch:
        cbs.insert(0, QWKCallback(val_seq, coral=coral))
    return cbs


def _adam(lr: float, cfg: Config = CFG):
    return tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=cfg.clipnorm)


def compile_softmax(model, lr: float, cfg: Config = CFG):
    model.compile(
        optimizer=_adam(lr, cfg),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def compile_coral(model, lr: float, cfg: Config = CFG):
    # No Keras accuracy on raw CORAL logits — it is misleading vs multi-label levels.
    # Prefer val_loss for scheduling; optional QWKCallback for ordinal quality.
    model.compile(
        optimizer=_adam(lr, cfg),
        loss=coral_loss,
        metrics=[],
    )
    return model


def _best_val_loss_from_csv(history_csv: Path) -> float | None:
    if not history_csv.exists():
        return None
    try:
        import pandas as pd

        df = pd.read_csv(history_csv)
        if "val_loss" not in df.columns or df.empty:
            return None
        return float(df["val_loss"].min())
    except Exception:
        return None


def _pick_best_checkpoint(out_dir: Path, coral: bool) -> tuple[Path | None, float | None, str]:
    """Return (checkpoint_path, best_val_loss, phase_name) across phase1/phase2."""
    name = "best_coral.keras" if coral else "best_softmax.keras"
    best_path: Path | None = None
    best_loss: float | None = None
    best_phase = ""
    for phase in ("phase1", "phase2"):
        ckpt = out_dir / phase / name
        vl = _best_val_loss_from_csv(out_dir / phase / "history.csv")
        if not ckpt.exists() or vl is None:
            continue
        if best_loss is None or vl < best_loss:
            best_loss = vl
            best_path = ckpt
            best_phase = phase
    return best_path, best_loss, best_phase


def two_phase_train(
    model,
    train_seq,
    val_seq,
    *,
    coral: bool,
    out_dir: Path,
    class_weight: dict | None = None,
    cfg: Config = CFG,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")

    # Phase 1 — frozen backbone (BN already frozen with base.trainable=False)
    if coral:
        compile_coral(model, cfg.phase1_lr, cfg)
    else:
        compile_softmax(model, cfg.phase1_lr, cfg)
    if cfg.freeze_bn_on_finetune:
        freeze_batch_norm(model)
    cbs = make_callbacks(out_dir / "phase1", val_seq, coral=coral, cfg=cfg)
    hist1 = model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=cfg.phase1_epochs,
        callbacks=cbs,
        class_weight=None if coral else class_weight,
        verbose=1,
    )
    p1_epochs = len(hist1.history.get("loss", []))

    # Phase 2 — gentle fine-tune (non-BN top layers only)
    set_fine_tune(model, cfg.unfreeze_last, freeze_bn=cfg.freeze_bn_on_finetune)
    if coral:
        compile_coral(model, cfg.phase2_lr, cfg)
    else:
        compile_softmax(model, cfg.phase2_lr, cfg)
    cbs2 = make_callbacks(out_dir / "phase2", val_seq, coral=coral, cfg=cfg)
    hist2 = model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=cfg.phase2_epochs,
        callbacks=cbs2,
        class_weight=None if coral else class_weight,
        verbose=1,
    )

    # Merge histories with phase boundary metadata
    history = {k: list(v) for k, v in hist1.history.items()}
    for k, v in hist2.history.items():
        history.setdefault(k, [])
        history[k].extend(list(v))
    history["phase1_epochs"] = p1_epochs
    history["phase_boundary"] = p1_epochs

    # Deploy the better of phase1 vs phase2 checkpoints (never last crashed epoch)
    best_ckpt, best_vl, best_phase = _pick_best_checkpoint(out_dir, coral=coral)
    final_path = out_dir / ("coral_final.keras" if coral else "softmax_final.keras")
    custom = {"coral_loss": coral_loss} if coral else None
    if best_ckpt is not None:
        best_model = tf.keras.models.load_model(best_ckpt, custom_objects=custom, compile=False)
        best_model.save(final_path)
        model = best_model
        meta = {
            "selected_phase": best_phase,
            "best_val_loss": best_vl,
            "checkpoint": str(best_ckpt),
        }
        (out_dir / "best_selection.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"Selected {best_phase} checkpoint (val_loss={best_vl:.4f}) → {final_path}")
    else:
        model.save(final_path)
        print(f"No phase checkpoints found; saved current weights → {final_path}")

    def _jsonify(v):
        if isinstance(v, list):
            return [float(x) for x in v]
        if isinstance(v, (np.integer, int)):
            return int(v)
        if isinstance(v, (np.floating, float)):
            return float(v)
        return v

    (out_dir / "history.json").write_text(
        json.dumps({k: _jsonify(v) for k, v in history.items()}, indent=2),
        encoding="utf-8",
    )
    return model, history, final_path


def predict_labels(model, seq, coral: bool = False) -> tuple[np.ndarray, np.ndarray]:
    ys, preds = [], []
    for i in range(len(seq)):
        x, y = seq[i]
        p = model.predict(x, verbose=0)
        if coral:
            pred = coral_logits_to_label(p)
            true = y.sum(axis=-1).astype(int)
        else:
            pred = np.argmax(p, axis=-1)
            true = y
        ys.append(true)
        preds.append(pred)
    return np.concatenate(ys), np.concatenate(preds)
