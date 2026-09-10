# Research Log — Diabetic Retinopathy Stage Detection

**Module:** NIBM BSc (Hons) Computer Science — Computer Vision (BSCCOMP24.2P)  
**Assessment:** CW1 Individual Project (100 marks)  
**Last updated:** 2026-09-09

---

## 1. Problem statement

Diabetic Retinopathy (DR) is a microvascular complication of diabetes that damages the retina. Untreated progression can cause irreversible vision loss. Screening requires specialists to grade fundus photographs on the International Clinical Diabetic Retinopathy (ICDR) scale:

| Grade | Name | Clinical meaning |
|------:|------|------------------|
| 0 | No DR | No diabetic retinal lesions |
| 1 | Mild | Microaneurysms only |
| 2 | Moderate | More than mild; less than severe |
| 3 | Severe | Extensive hemorrhages / venous beading / IRMA |
| 4 | Proliferative DR | Neovascularization or vitreous/preretinal hemorrhage |

**System role:** automated **screening assistant** — prioritizes patients for clinical review. Not a replacement for ophthalmologists.

Clinical signs models must learn: microaneurysms, hemorrhages, cotton-wool spots, exudates, abnormal new vessels.

---

## 2. Learning outcomes mapped

| LO | How this project addresses it |
|----|-------------------------------|
| Image models & representations | RGB tensors, EfficientNet feature hierarchy, Grad-CAM saliency |
| Acquisition / display / transmission | Kaggle fundus archive, PNG/JPEG, web demo display of overlays |
| Filtering, segmentation-ish prep, labelling, features | CLAHE, Graham, unsharp, class labels, CNN features |
| Object / pattern recognition | 5-stage pattern classification + ordinal CORAL |

---

## 3. Dataset research

### Candidates compared (Kaggle API, Sep 2026)

| Dataset | Size | Notes | Decision |
|---------|------|-------|----------|
| `sachinkumar413/diabetic-retinopathy-dataset` | ~350 MB | 5 folders Healthy→Proliferate | **PRIMARY (executed)** |
| `sovitrath/diabetic-retinopathy-224x224-gaussian-filtered` | ~427 MB | Pre-Graham | Ablation only |
| `mariaherrerot/aptos2019` | ~8 GB | Full-res APTOS | Too heavy locally |
| `tanlikesmath/diabetic-retinopathy-resized` | ~7.3 GB | EyePACS resized | Overkill for CW |
| Combined EyePACS+APTOS packs | 18–35 GB | Research scale | Out of scope |

### Why primary dataset

1. Coursework-sized (~3.6k labeled images, APTOS distribution).  
2. Already 224×224 — fits EfficientNet transfer learning.  
3. **Not** heavily pre-filtered — we own CLAHE / Graham / edge enhancement for rubric marks.  
4. High usability rating; widely cited for DR grading benchmarks.

### Expected class imbalance (APTOS)

~49% No DR, ~10% Mild, ~27% Moderate, ~5% Severe, ~8% Proliferative → accuracy alone is unsafe; use class weights + **Quadratic Weighted Kappa (QWK)**.

### Ethics & limitations

- Public secondary data; no PHI stored in repo.  
- Labels can be noisy; single-grader bias possible.  
- Domain shift across cameras/clinics.  
- No guaranteed patient-level split IDs in this pack → document as limitation.  
- Outputs are decision support only.

---

## 4. Method research

### Baseline (meets CW)

EfficientNet + ImageNet transfer learning, two-phase freeze/fine-tune, augmentation, class weights, precision/recall/F1/CM/curves.

### Transfer-learning stability (2026-09-10)

Observed Softmax/CORAL “epoch-3” collapse was a **merged-curve artifact**: Phase-1 was stable; Phase-2 start spiked train loss when the last 40 backbone layers (including BatchNormalization) became trainable. Keras docs warn that fine-tuning BN destroys pretrained running statistics.

**Stabilized protocol:**

- Freeze all `BatchNormalization` during fine-tune; unfreeze last **20 non-BN** layers only.
- Adam with `clipnorm=1.0`; Phase-1 LR `3e-4`, Phase-2 LR `1e-5`; up to 8+8 epochs with EarlyStopping / ReduceLROnPlateau.
- After both phases, deploy the **better** of `phase1/best_*.keras` vs `phase2/best_*.keras` by `val_loss` (never last-epoch weights).
- Do **not** trust Keras `accuracy` on raw CORAL logits; schedule on `val_loss` and report QWK at eval.

### Novel upgrades (exceed CW)

| Method | Rationale |
|--------|-----------|
| **CORAL ordinal regression** | DR grades are ordered; adjacent mistakes less severe than far mistakes (aligns with QWK) |
| **Hybrid fundus preprocess** | CLAHE (contrast) + Ben Graham illumination + unsharp (edge enhancement) |
| **QWK-first checkpointing** | Select model by clinical ordinal metric, not only accuracy |
| **Grad-CAM in live web UI** | Explainability for trust |

### Backbone choice

**EfficientNetB0** primary (local RAM ~7 GB, no NVIDIA GPU; Kaggle T4 for full training). EfficientNetB3 optional stretch if GPU time allows.

### Related literature / sources

- APTOS 2019 Blindness Detection (Kaggle competition).  
- Tan & Le, EfficientNet (ICML 2019).  
- Cao, Mirjalili, Raschka — CORAL ordinal regression (Pattern Recognition Letters).  
- Ben Graham — Kaggle DR winner preprocessing (illumination normalization).  
- Selvaraju et al. — Grad-CAM.  
- Course notes cache: `Notes/KNOWLEDGE_CACHE.md`, CNN Blueprint OCR.

---

## 5. Hardware & training venue

| Resource | Observation |
|----------|-------------|
| Local GPU | None |
| Local RAM | ~7 GB |
| Disk free | ~132 GB |
| Decision | Train on **Kaggle Notebooks GPU**; serve Streamlit demo on local CPU |

---

## 6. Evaluation protocol

- Stratified split 70% / 15% / 15% (train/val/test), `SEED=42`.  
- Test used **once** for final reporting.  
- Metrics: accuracy, precision, recall, F1 (macro & weighted), confusion matrix, loss/acc curves, **QWK**.  
- Error analysis: adjacent (±1 grade) vs far (≥2) errors.  
- Softmax vs CORAL comparison on identical split.

---

## 7. Deliverables checklist

- [x] Research documentation (this file)  
- [x] Modular codebase + notebooks  
- [x] Trained model artifacts (CORAL winner)  
- [x] Streamlit real-time demo  
- [x] Architecture diagrams & figures  
- [x] Report draft/HTML (`docs/REPORT_DRAFT.md`, `docs/REPORT.html`)
- [ ] Hosted prototype video URL (record via `docs/VIDEO_PLAN.md`)  


## 8. Executed results snapshot

- Dataset: sachinkumar413/diabetic-retinopathy-dataset (2750 images, 5 classes).
- Winner: CORAL EfficientNetB0, test QWK ≈ 0.718 (Softmax QWK ≈ 0.662).
- Artifacts: `models/best_model.keras`, `models/meta.json`.
- Demo: `uv run streamlit run app/streamlit_app.py`.
