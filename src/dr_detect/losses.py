"""Loss functions including CORAL ordinal loss."""

from __future__ import annotations

import tensorflow as tf


def coral_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Binary cross-entropy on cumulative levels.
    y_true: (batch, K-1) float levels; y_pred: (batch, K-1) logits.
    """
    y_true = tf.cast(y_true, tf.float32)
    return tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    )
