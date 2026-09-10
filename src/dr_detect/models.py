"""EfficientNetB0 backbones with Softmax or CORAL heads."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0

from dr_detect.config import CFG


def _backbone(img_size: int = CFG.img_size, trainable: bool = False):
    base = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(img_size, img_size, 3),
    )
    base.trainable = trainable
    return base


def build_softmax_model(
    img_size: int = CFG.img_size,
    dropout: float = CFG.dropout,
    num_classes: int = CFG.num_classes,
) -> tf.keras.Model:
    base = _backbone(img_size, trainable=False)
    x = layers.GlobalAveragePooling2D(name="gap")(base.output)
    x = layers.Dropout(dropout, name="dropout")(x)
    out = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    model = models.Model(inputs=base.input, outputs=out, name="dr_softmax_effb0")
    model._backbone = base  # type: ignore[attr-defined]
    return model


def build_coral_model(
    img_size: int = CFG.img_size,
    dropout: float = CFG.dropout,
    num_classes: int = CFG.num_classes,
) -> tf.keras.Model:
    """CORAL head: num_classes-1 logits (no activation); sigmoid applied in loss/metrics."""
    base = _backbone(img_size, trainable=False)
    x = layers.GlobalAveragePooling2D(name="gap")(base.output)
    x = layers.Dropout(dropout, name="dropout")(x)
    # Bias-only rank thresholds via Dense without activation
    out = layers.Dense(num_classes - 1, activation=None, name="coral_logits")(x)
    model = models.Model(inputs=base.input, outputs=out, name="dr_coral_effb0")
    model._backbone = base  # type: ignore[attr-defined]
    return model


def _find_backbone(model: tf.keras.Model) -> tf.keras.Model | None:
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and layer.name.startswith("efficientnet"):
            return layer
    if hasattr(model, "_backbone"):
        return model._backbone  # type: ignore[attr-defined]
    return None


def freeze_batch_norm(model: tf.keras.Model) -> int:
    """Freeze all BatchNormalization layers (Keras TL recipe: keep BN in inference mode)."""
    n = 0
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            n += freeze_batch_norm(layer)
        elif isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
            n += 1
    return n


def set_fine_tune(
    model: tf.keras.Model,
    unfreeze_last: int = CFG.unfreeze_last,
    freeze_bn: bool = True,
) -> None:
    """Unfreeze the last N *non-BN* backbone layers; keep BatchNorm frozen if freeze_bn."""
    base = _find_backbone(model)
    if base is None:
        # fallback: unfreeze last N non-BN layers of whole model except head
        candidates = [
            layer
            for layer in model.layers[:-1]
            if not isinstance(layer, layers.BatchNormalization)
        ]
        for layer in model.layers:
            layer.trainable = False
        for layer in candidates[-unfreeze_last:]:
            layer.trainable = True
        if freeze_bn:
            freeze_batch_norm(model)
        return

    base.trainable = True
    for layer in base.layers:
        layer.trainable = False

    # Walk from the top of the backbone; unfreeze only non-BN layers.
    thawed = 0
    for layer in reversed(base.layers):
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
            continue
        if thawed >= unfreeze_last:
            continue
        layer.trainable = True
        thawed += 1

    if freeze_bn:
        freeze_batch_norm(model)


def coral_logits_to_label(logits, threshold: float = 0.5):
    import numpy as np

    if hasattr(logits, "numpy"):
        logits = logits.numpy()
    probs = 1.0 / (1.0 + np.exp(-np.asarray(logits)))
    return (probs > threshold).sum(axis=-1).astype(np.int32)
