"""Fundus preprocessing: crop, CLAHE, Ben Graham, unsharp, EfficientNet normalize."""

from __future__ import annotations

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input

from dr_detect.config import CFG


def crop_dark_borders(img: np.ndarray, tol: int = CFG.crop_tol) -> np.ndarray:
    """Remove near-black borders around the retinal disk."""
    if img.ndim == 2:
        mask = img > tol
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray > tol
    if not mask.any():
        return img
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return img[rmin : rmax + 1, cmin : cmax + 1]


def apply_clahe(img: np.ndarray, clip: float = CFG.clahe_clip, tile: int = CFG.clahe_tile) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalization on LAB L-channel."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    l2 = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2RGB)


def ben_graham(img: np.ndarray, sigma: float = CFG.graham_sigma) -> np.ndarray:
    """Illumination normalization used by Ben Graham (Kaggle DR)."""
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    out = cv2.addWeighted(img, 4, blur, -4, 128)
    return np.clip(out, 0, 255).astype(np.uint8)


def unsharp_mask(
    img: np.ndarray,
    sigma: float = CFG.unsharp_sigma,
    amount: float = CFG.unsharp_amount,
) -> np.ndarray:
    """Edge enhancement via unsharp masking."""
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    sharp = cv2.addWeighted(img, 1.0 + amount, blur, -amount, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def preprocess_fundus(
    img_bgr_or_rgb: np.ndarray,
    *,
    is_bgr: bool = True,
    img_size: int = CFG.img_size,
    for_model: bool = True,
) -> np.ndarray:
    """
    Full pipeline: RGB → crop → CLAHE → Graham → unsharp → resize → (EfficientNet preprocess).

    Returns float32 tensor (H,W,3) ready for the model when for_model=True,
    else uint8 RGB for visualization.
    """
    img = img_bgr_or_rgb
    if is_bgr:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = crop_dark_borders(img)
    if img.size == 0:
        img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    img = apply_clahe(img)
    img = ben_graham(img)
    img = unsharp_mask(img)
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    if for_model:
        return preprocess_input(img.astype(np.float32))
    return img


def load_and_preprocess(path: str | bytes, **kwargs) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8) if isinstance(path, str) else np.frombuffer(path, dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return preprocess_fundus(bgr, is_bgr=True, **kwargs)


def preprocess_steps_gallery(img_bgr: np.ndarray, img_size: int = CFG.img_size) -> dict[str, np.ndarray]:
    """Return intermediate RGB uint8 stages for documentation figures."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    cropped = crop_dark_borders(rgb)
    clahe = apply_clahe(cropped)
    graham = ben_graham(clahe)
    sharp = unsharp_mask(graham)
    final = cv2.resize(sharp, (img_size, img_size))
    return {
        "raw": cv2.resize(rgb, (img_size, img_size)),
        "cropped": cv2.resize(cropped, (img_size, img_size)),
        "clahe": cv2.resize(clahe, (img_size, img_size)),
        "graham": cv2.resize(graham, (img_size, img_size)),
        "unsharp": final,
    }
