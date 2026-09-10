#!/usr/bin/env python3
"""Render architecture diagrams from Mermaid (via matplotlib fallback schematics)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from dr_detect.config import CFG


def _box(ax, xy, text, w=1.8, h=0.55, color="#dceaf7"):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.05",
        facecolor=color, edgecolor="#1f4e79", linewidth=1.2,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8, wrap=True)
    return (x + w / 2, y + h / 2)


def _arrow(ax, p, q):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="->", color="#333", lw=1.2))


def save_flow(name: str, nodes: list[str], title: str):
    fig, ax = plt.subplots(figsize=(12, 2.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 2)
    ax.axis("off")
    ax.set_title(title)
    centers = []
    gap = 12 / max(len(nodes), 1)
    for i, label in enumerate(nodes):
        c = _box(ax, (0.3 + i * gap, 0.7), label, w=min(gap - 0.2, 2.0))
        centers.append(c)
    for a, b in zip(centers, centers[1:]):
        _arrow(ax, (a[0] + 0.7, a[1]), (b[0] - 0.7, b[1]))
    path = CFG.figures_dir / name
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close()
    print("wrote", path)


def main():
    CFG.ensure_dirs()
    save_flow(
        "diagram_system.png",
        ["Kaggle APTOS", "Split 70/15/15", "Preprocess", "Train Softmax/CORAL", "Pick by QWK", "Streamlit"],
        "End-to-end system architecture",
    )
    save_flow(
        "diagram_preprocess.png",
        ["Raw", "Crop", "CLAHE", "Graham", "Unsharp", "Resize 224", "EffNet prep"],
        "Preprocessing pipeline",
    )
    save_flow(
        "diagram_softmax.png",
        ["Input", "EfficientNetB0", "GAP", "Dropout", "Dense 5 Softmax"],
        "Softmax model architecture",
    )
    save_flow(
        "diagram_coral.png",
        ["Features", "GAP", "Dropout", "4 logits", "Sigmoid P(y>k)", "Decode grade"],
        "CORAL ordinal head",
    )
    save_flow(
        "diagram_tl_phases.png",
        ["Freeze base", "Train head lr=1e-3", "Unfreeze top 40", "Fine-tune lr=1e-5", "Checkpoint"],
        "Transfer-learning phases",
    )
    save_flow(
        "diagram_gradcam.png",
        ["Image", "CNN", "Last conv", "Gradients", "Heatmap", "Overlay"],
        "Grad-CAM explainability flow",
    )
    save_flow(
        "diagram_web_ux.png",
        ["Upload", "Validate", "Preprocess", "Predict", "Grad-CAM", "Risk banner"],
        "Web demo UX flow",
    )
    save_flow(
        "diagram_hardware.png",
        ["Local uv/Jupyter", "Kaggle GPU train", "Export .keras", "Local Streamlit CPU"],
        "Train vs serve hardware",
    )


if __name__ == "__main__":
    main()
