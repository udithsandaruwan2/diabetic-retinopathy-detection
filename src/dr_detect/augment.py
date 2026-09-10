"""Train-time data augmentation (never apply to val/test)."""

from __future__ import annotations

import cv2
import numpy as np


def augment_image(rgb: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """Lightweight geometric + photometric aug on uint8 RGB."""
    rng = rng or np.random.default_rng()
    img = rgb.copy()

    if rng.random() < 0.5:
        img = cv2.flip(img, 1)

    angle = float(rng.uniform(-15, 15))
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img = cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    # zoom via crop/resize
    zoom = float(rng.uniform(0.9, 1.1))
    nh, nw = int(h / zoom), int(w / zoom)
    y0 = max((h - nh) // 2, 0)
    x0 = max((w - nw) // 2, 0)
    crop = img[y0 : y0 + nh, x0 : x0 + nw]
    img = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)

    # brightness / contrast
    alpha = float(rng.uniform(0.85, 1.15))  # contrast
    beta = float(rng.uniform(-20, 20))  # brightness
    img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    # small translation
    tx = int(rng.uniform(-0.05, 0.05) * w)
    ty = int(rng.uniform(-0.05, 0.05) * h)
    m2 = np.float32([[1, 0, tx], [0, 1, ty]])
    img = cv2.warpAffine(img, m2, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    return img
