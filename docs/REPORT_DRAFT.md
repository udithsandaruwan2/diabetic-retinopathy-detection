# Diabetic Retinopathy Stage Detection using Transfer Learning and Ordinal CORAL Regression

**Course:** Computer Vision — BSCCOMP24.2P  
**Author:** Coursework submission pack  
**Prototype video URL:** `VIDEO_URL=TBD` (see `docs/VIDEO_PLAN.md`)

## Abstract
This project implements 5-stage diabetic retinopathy (DR) grading from fundus photographs. A hybrid medical preprocessing pipeline (crop → CLAHE → Ben Graham illumination normalization → unsharp edge enhancement) feeds an EfficientNetB0 transfer-learning backbone. We compare a Softmax classifier with a novel **CORAL ordinal regression** head. On the held-out test set, Softmax reached QWK **0.662** and CORAL reached QWK **0.718** (winner). A Streamlit app provides real-time inference with Grad-CAM explainability.

## 1. Introduction
Diabetic retinopathy damages retinal microvasculature and is a leading cause of preventable blindness. Specialist grading capacity is limited; automated screening assistants can prioritize high-risk cases. This system classifies ICDR stages 0–4 and is explicitly framed as decision support, not autonomous diagnosis.

## 2. Dataset justification
Primary dataset: `sachinkumar413/diabetic-retinopathy-dataset` (Kaggle).
Folders map to stages: Healthy→No_DR, Mild DR, Moderate DR, Severe DR, Proliferate DR.
Total images: 2750. Stratified split 70/15/15 (SEED=42): train 1924 / val 413 / test 413.
Class imbalance is severe (Severe only 190 images) → balanced class weights.
Ethics: public secondary data; no PHI; screening-assistant disclaimer; label noise and domain shift remain limitations.

## 3. Preprocessing
1. Dark-border crop (`tol=7`)
2. CLAHE on LAB L-channel (`clipLimit=2.0`, tiles 8×8) — contrast
3. Ben Graham illumination (`addWeighted` 4/−4 + Gaussian σ=10)
4. Unsharp mask (σ=1.0, amount=1.5) — edge enhancement
5. Resize 224×224 + EfficientNet `preprocess_input`

Evidence: `docs/figures/preprocess_stages.png`

## 4. Augmentation & balancing
Train-only: flip, ±15° rotation, zoom, brightness/contrast, small translation.
Class weights (train): see `data/processed/class_weights.json`.
Evidence: `docs/figures/augmentation_mosaic.png`, `split_counts.png`.

## 5. Softmax architecture & transfer learning
EfficientNetB0 (ImageNet, include_top=False) → GAP → Dropout(0.3) → Dense(5, softmax).
Phase 1: freeze base, Adam 1e-3, 3 epochs. Phase 2: unfreeze last 40 layers, Adam 1e-5, 2 epochs.
Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger.
Test: Acc=0.533, macro-F1=0.439, QWK=0.662.

## 6. Novel CORAL ordinal head
DR grades are ordered. CORAL predicts P(y>k) for k=0..3 with rank-consistent logits.
Same backbone/schedule; CORAL loss = BCE on cumulative levels.
Test: Acc=0.533, macro-F1=0.422, QWK=0.718 (**selected**).
Adjacent-error rate=0.312; far-error=0.155.

## 7. Results
| Model | Acc | Macro-F1 | Weighted-F1 | QWK |
|-------|-----|----------|-------------|-----|
| Softmax | 0.533 | 0.439 | 0.523 | 0.662 |
| CORAL | 0.533 | 0.422 | 0.537 | 0.718 |

Figures: training curves, confusion matrices, Softmax vs CORAL bar chart, Grad-CAM gallery under `docs/figures/`.

Interpretation: accuracy alone understates clinical usefulness; CORAL’s higher QWK means fewer severe ordinal mistakes.

## 8. Prototype web demo
`uv run streamlit run app/streamlit_app.py`
Features: upload/sample, preprocess preview, stage + risk banner, Grad-CAM, clinical disclaimer.
Record and host video per `docs/VIDEO_PLAN.md`, then replace `VIDEO_URL=TBD`.

## 9. Impact, limitations, future work
Impact: triage support for scarce specialist capacity; explainable heatmaps aid trust.
Limits: short local CPU training schedule; class imbalance; no patient-level IDs; camera domain shift.
Future: longer Kaggle-GPU training, EfficientNetB3, external Messidor-2 test, clinician pilot.

## 10. Conclusion
We delivered a reproducible DR staging pipeline with documented preprocessing, Softmax baseline, ordinal CORAL innovation, full metric suite, Grad-CAM, and a Streamlit prototype — aligned to all CW rubric criteria.

## References
- APTOS / ICDR clinical grading scale
- Tan & Le, EfficientNet (2019)
- Cao et al., CORAL ordinal regression
- Ben Graham Kaggle DR preprocessing
- Selvaraju et al., Grad-CAM
- Project docs: RESEARCH.md, ARCHITECTURE.md, DECISION_LOG.md, EXPERIMENT_LOG.md
