# Decision Log

Chronological record of design choices. Update whenever a non-trivial decision is made.

| Date | Decision | Alternatives considered | Rationale |
|------|----------|-------------------------|-----------|
| 2026-09-09 | Primary dataset = `sachinkumar413/diabetic-retinopathy-dataset` (5 folders: Healthy/Mild/Moderate/Severe/Proliferate) | sovitrath APTOS-224 (slow download), pkdarabi (binary only) | Multi-class, complete download on this network; maps Healthy→No_DR |
| 2026-09-09 | Backbone = EfficientNetB0 | B3, MobileNetV2, ResNet50 | No local GPU / 7 GB RAM; B0 still ImageNet TL; Kaggle GPU friendly |
| 2026-09-09 | Dual heads: Softmax baseline + CORAL ordinal | Softmax only; focal loss only; ensemble | Softmax satisfies CW; CORAL is ordinal novelty aligned with QWK |
| 2026-09-09 | Model selection metric = val QWK | val accuracy / val loss | Clinical ordinal severity; CW still reports Acc/P/R/F1 |
| 2026-09-09 | Preprocess = crop → CLAHE → Graham → unsharp → 224 → EfficientNet preprocess | Raw /255 only; gaussian-filtered dataset | Covers contrast + illumination + edge enhancement rubric |
| 2026-09-09 | Split 70/15/15 stratified, SEED=42 | 70/10/20 from notes | Larger val for QWK model selection |
| 2026-09-09 | Train on Kaggle GPU; demo on local Streamlit | Local CPU full train | Hardware constraint |
| 2026-09-09 | Package manager = `uv` | pip/venv, conda | User requirement; fast lockfile reproducibility |
| 2026-09-09 | Web demo = Streamlit | FastAPI+React, Gradio | Simple real-time demo for CW video |
| 2026-09-09 | Class imbalance = balanced class weights (+ optional rare-class oversample) | SMOTE on pixels; ignore imbalance | Standard, reproducible, works with Keras |
| 2026-09-09 | Augment train only | Augment all splits | Prevent leakage / optimistic metrics |

| 2026-09-09 | EDA complete: n=5676; splits written; figures saved | — | See `data/processed/split_stats.json` |

| 2026-09-09 | Replaced pkdarabi dataset after EDA found single-class only (`No_DR`) | continue with single-class | Multi-stage CW requires 5-class; selected `sachinkumar413/diabetic-retinopathy-dataset` |

| 2026-09-09 | EDA complete: n=2750; splits written; figures saved | — | See `data/processed/split_stats.json` |

| 2026-09-10 | Mid-run collapse root cause = Phase-2 BN fine-tune shock (not epoch-3 data bug) | data shuffle bug; LR only | Phase-1 histories improve; train_loss spikes at Phase-2 start when last 40 layers incl. BatchNorm become trainable (Keras TL warning) |
| 2026-09-10 | Fine-tune: freeze all BN; unfreeze last 20 **non-BN** layers | unfreeze 40 incl. BN; `training=False` backbone only | Matches Keras EfficientNet recipe; prevents destroying ImageNet BN stats |
| 2026-09-10 | Adam `clipnorm=1.0`; phase1_lr=`3e-4`; epochs 8+8 | clipnorm unset; lr=`1e-3`; 3+2 | Softens Phase-2 shock; gives ReduceLR/EarlyStopping room to work |
| 2026-09-10 | Export best checkpoint across phases by `val_loss` | always save last Phase-2 weights | Prior runs shipped broken final weights even when Phase-1 was better |
| 2026-09-10 | Remove Keras `accuracy` metric from CORAL compile | keep accuracy on logits | Raw CORAL logits vs multi-hot levels make Keras accuracy misleading (~0.9→0.5 looks like collapse) |
