"""Grad-CAM explainability for EfficientNet-based models."""

from __future__ import annotations

import cv2
import numpy as np
import tensorflow as tf


def find_last_conv_layer(model: tf.keras.Model) -> str:
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model):
            for sub in reversed(layer.layers):
                if isinstance(sub, tf.keras.layers.Conv2D):
                    return sub.name
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    # EfficientNet nested
    for layer in model.layers:
        if "efficientnet" in layer.name:
            for sub in reversed(layer.layers):
                if isinstance(sub, tf.keras.layers.Conv2D):
                    return sub.name
    raise ValueError("No Conv2D layer found for Grad-CAM")


def make_gradcam_heatmap(
    img_array: np.ndarray,
    model: tf.keras.Model,
    last_conv_layer_name: str | None = None,
    pred_index: int | None = None,
    coral: bool = False,
) -> np.ndarray:
    """
    img_array: (1,H,W,3) preprocessed.
    For Softmax: uses class score. For CORAL: uses sum of logits as severity proxy.
    """
    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer(model)

    # Resolve nested layer path
    conv_layer = None
    for layer in model.layers:
        if layer.name == last_conv_layer_name:
            conv_layer = layer
            break
        if hasattr(layer, "layers"):
            for sub in layer.layers:
                if sub.name == last_conv_layer_name:
                    conv_layer = sub
                    # build model that maps input -> nested conv
                    break

    # Prefer getting activations via a submodel on the backbone
    backbone = None
    for layer in model.layers:
        if "efficientnet" in layer.name:
            backbone = layer
            break

    if backbone is not None:
        last_conv = None
        for sub in reversed(backbone.layers):
            if isinstance(sub, tf.keras.layers.Conv2D):
                last_conv = sub
                break
        grad_model = tf.keras.Model(
            [model.inputs],
            [backbone.get_layer(last_conv.name).output, model.output],
        )
    else:
        grad_model = tf.keras.Model(
            [model.inputs],
            [model.get_layer(last_conv_layer_name).output, model.output],
        )

    img = tf.convert_to_tensor(img_array)
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img)
        if coral:
            loss = tf.reduce_sum(predictions, axis=-1)
        else:
            if pred_index is None:
                pred_index = int(tf.argmax(predictions[0]))
            loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled, conv_outputs), axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_gradcam(
    rgb_uint8: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    heat = cv2.resize(heatmap, (rgb_uint8.shape[1], rgb_uint8.shape[0]))
    heat = np.uint8(255 * heat)
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
    overlay = np.clip((1 - alpha) * rgb_uint8 + alpha * heat_color, 0, 255).astype(np.uint8)
    return overlay
