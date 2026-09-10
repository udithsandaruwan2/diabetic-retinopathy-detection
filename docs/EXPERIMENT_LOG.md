# Experiment Log

Record every training / evaluation run. Copy a blank row template for each experiment.

## Template

```text
ID:
Date:
Notebook/script:
Hardware:
Dataset + split commit/hash:
Model: Softmax | CORAL
Config: (paste config block)
Epochs completed:
Best val metric(s):
Test metrics (if final):
Artifact paths:
Notes / failures:
```

---

## EXP-000 — Project bootstrap

- **Date:** 2026-09-09  
- **Notes:** Repository scaffolding, research docs, no training yet.

---

## Runs

_(Training runs appended below as they complete.)_

## EXP-004 — EDA and preprocessing evidence

- Dataset downloaded and indexed (`data/processed/all_images.csv`)
- Split stats: `data/processed/split_stats.json`
- Figures: class_distribution, sample_gallery, preprocess_stages, augmentation_mosaic, split_counts

## EXP-001 — Softmax EfficientNetB0

- Config: `{"seed": 42, "img_size": 224, "batch_size": 8, "num_classes": 5, "train_frac": 0.7, "val_frac": 0.15, "test_frac": 0.15, "crop_tol": 7, "clahe_clip": 2.0, "clahe_tile": 8, "graham_sigma": 10.0, "unsharp_sigma": 1.0, "unsharp_amount": 1.5, "backbone": "EfficientNetB0", "dropout": 0.3, "phase1_lr": 0.001, "phase2_lr": 1e-05, "phase1_epochs": 3, "phase2_epochs": 2, "unfreeze_last": 40, "early_stop_patience": 3, "reduce_lr_patience": 2, "reduce_lr_factor": 0.5, "qwk_each_epoch": false, "kaggle_dataset": "sachinkumar413/diabetic-retinopathy-dataset", "data_dir": "/home/neon-cultivator/us/diabetic-retinopathy-detection/data", "raw_dir": "/home/neon-cultivator/us/diabetic-retinopathy-detection/data/raw", "processed_dir": "/home/neon-cultivator/us/diabetic-retinopathy-detection/data/processed", "models_dir": "/home/neon-cultivator/us/diabetic-retinopathy-detection/models", "figures_dir": "/home/neon-cultivator/us/diabetic-retinopathy-detection/docs/figures", "artifacts_dir": "/home/neon-cultivator/us/diabetic-retinopathy-detection/artifacts"}`
- Artifact: `/home/neon-cultivator/us/diabetic-retinopathy-detection/artifacts/experiments/softmax/softmax_final.keras`
- Curves: `docs/figures/softmax_curves.png`

## EXP-002 — CORAL EfficientNetB0

- Config: `{"seed": 42, "img_size": 224, "batch_size": 8, "num_classes": 5, "train_frac": 0.7, "val_frac": 0.15, "test_frac": 0.15, "crop_tol": 7, "clahe_clip": 2.0, "clahe_tile": 8, "graham_sigma": 10.0, "unsharp_sigma": 1.0, "unsharp_amount": 1.5, "backbone": "EfficientNetB0", "dropout": 0.3, "phase1_lr": 0.001, "phase2_lr": 1e-05, "phase1_epochs": 3, "phase2_epochs": 2, "unfreeze_last": 40, "early_stop_patience": 3, "reduce_lr_patience": 2, "reduce_lr_factor": 0.5, "qwk_each_epoch": false, "kaggle_dataset": "sachinkumar413/diabetic-retinopathy-dataset", "data_dir": "/home/neon-cultivator/us/diabetic-retinopathy-detection/data", "raw_dir": "/home/neon-cultivator/us/diabetic-retinopathy-detection/data/raw", "processed_dir": "/home/neon-cultivator/us/diabetic-retinopathy-detection/data/processed", "models_dir": "/home/neon-cultivator/us/diabetic-retinopathy-detection/models", "figures_dir": "/home/neon-cultivator/us/diabetic-retinopathy-detection/docs/figures", "artifacts_dir": "/home/neon-cultivator/us/diabetic-retinopathy-detection/artifacts"}`
- Artifact: `/home/neon-cultivator/us/diabetic-retinopathy-detection/artifacts/experiments/coral/coral_final.keras`
- Curves: `docs/figures/coral_curves.png`

## EXP-003 — Test evaluation & export

- Winner: **coral** (QWK=0.7178)
- meta: `models/meta.json`
- Results: `{"softmax": {"accuracy": 0.5326876513317191, "precision_macro": 0.4951857789978763, "recall_macro": 0.5033535353535353, "f1_macro": 0.4387531433334397, "precision_weighted": 0.6580256173744412, "recall_weighted": 0.5326876513317191, "f1_weighted": 0.5229679131968157, "qwk": 0.6620645541807368, "adjacent": {"n": 413, "exact": 0.5326876513317191, "adjacent_error": 0.29055690072639223, "far_error": 0.17675544794188863, "mean_abs_error": 0.7239709443099274}}, "coral": {"accuracy": 0.5326876513317191, "precision_macro": 0.45128926890800136, "recall_macro": 0.47474651274651264, "f1_macro": 0.4218043973741981, "precision_weighted": 0.5904790766664105, "recall_weighted": 0.5326876513317191, "f1_weighted": 0.536666324034593, "qwk": 0.7177696900732531, "adjacent": {"n": 413, "exact": 0.5326876513317191, "adjacent_error": 0.31234866828087166, "far_error": 0.1549636803874092, "mean_abs_error": 0.648910411622276}}}`


## EXP-001 Softmax — COMPLETE
- Test Acc=0.533, QWK=0.662
- Curves: docs/figures/softmax_curves.png
- CM: docs/figures/cm_softmax.png

## EXP-002 CORAL — COMPLETE (WINNER)
- Test Acc=0.533, QWK=0.718
- Curves: docs/figures/coral_curves.png
- CM: docs/figures/cm_coral.png
- Exported: models/best_model.keras + models/meta.json

## EXP-003 Evaluation notes
- Softmax vs CORAL chart: docs/figures/softmax_vs_coral.png
- Grad-CAM gallery: docs/figures/gradcam_gallery.png
- CORAL improves QWK despite similar accuracy (ordinal clinical alignment)

---

## EXP-STABLE-000 — Root-cause diagnosis (no retrain)

- **Date:** 2026-09-10
- **Evidence:** `artifacts/experiments/{softmax,coral}/phase*/history.csv`
- **Finding:** Phase-1 Softmax/CORAL improve; collapse aligns with Phase-2 start (merged plot looks like “epoch 3”).
  - Softmax P1→P2 train_loss ≈ 1.12 → 1.26
  - CORAL P1→P2 train_loss ≈ 0.30 → 0.51; Keras “acc” on logits ~0.84 → 0.58 (misleading metric)
- **Primary cause:** `set_fine_tune(unfreeze_last=40)` made BatchNorm trainable → BN fine-tune shock.
- **Secondary:** `two_phase_train` saved last Phase-2 weights, not best across phases.
- **Tertiary:** 3+2 epochs too short for ReduceLR/ES to recover.
- **Not primary:** per-epoch shuffle already in `FundusSequence.on_epoch_end`.

## EXP-STABLE-001 — Stabilized retraining (COMPLETE)

- **Date:** 2026-09-10
- **Changes:** BN freeze on fine-tune; unfreeze_last=20 non-BN; Adam clipnorm=1.0; phase1_lr=3e-4; epochs 8+8; best-across-phases export; CORAL without Keras accuracy; phase-boundary curve plots.
- **Data sanity:** `data/processed/data_sanity.json` — 0 unreadable / 0 near-black in 500-scan; same splits reused.
- **Phase-2 train-loss spike vs end-P1:** Softmax **−1.2%**, CORAL **−2.1%** (was +12% / +70% before).
- **Softmax test:** Acc=**0.625**, QWK=**0.713** (prior unstable Acc=0.533, QWK=0.662)
- **CORAL test:** Acc=**0.627**, QWK=**0.719** (prior Acc=0.533, QWK=0.718) — QWK ≥ 0.718 met
- **Winner:** CORAL → `models/best_model.keras` + `models/meta.json`
- **Artifacts:** `artifacts/experiments/softmax_stable/`, `artifacts/experiments/coral_stable/`, `docs/figures/softmax_curves.png`, `docs/figures/coral_curves.png`
- **Selection:** Softmax phase2 val_loss=0.906; CORAL phase2 val_loss=0.271

## EXP-003 Evaluation notes (superseded by EXP-STABLE-001 export)
- Softmax vs CORAL chart: docs/figures/softmax_vs_coral.png
- Grad-CAM gallery: docs/figures/gradcam_gallery.png
- CORAL improves QWK despite similar accuracy (ordinal clinical alignment)


## EXP-STABLE-001a — Softmax EfficientNetB0 (BN-safe)

- Config: `{"seed": 42, "img_size": 224, "batch_size": 8, "num_classes": 5, "train_frac": 0.7, "val_frac": 0.15, "test_frac": 0.15, "crop_tol": 7, "clahe_clip": 2.0, "clahe_tile": 8, "graham_sigma": 10.0, "unsharp_sigma": 1.0, "unsharp_amount": 1.5, "backbone": "EfficientNetB0", "dropout": 0.3, "phase1_lr": 0.0003, "phase2_lr": 1e-05, "phase1_epochs": 8, "phase2_epochs": 8, "unfreeze_last": 20, "early_stop_patience": 3, "reduce_lr_patience": 2, "reduce_lr_factor": 0.5, "clipnorm": 1.0, "freeze_bn_on_finetune": true, "qwk_each_epoch": false, "kaggle_dataset": "sachinkumar413/diabetic-retinopathy-dataset", "data_dir": "/home/neon-cultivator/us/diabetic-retinopathy-detection/data", "raw_dir": "/home/neon-cultivator/us/diabetic-retinopathy-detection/data/raw", "processed_dir": "/home/neon-cultivator/us/diabetic-retinopathy-detection/data/processed", "models_dir": "/home/neon-cultivator/us/diabetic-retinopathy-detection/models", "figures_dir": "/home/neon-cultivator/us/diabetic-retinopathy-detection/docs/figures", "artifacts_dir": "/home/neon-cultivator/us/diabetic-retinopathy-detection/artifacts"}`
- Artifact: `/home/neon-cultivator/us/diabetic-retinopathy-detection/artifacts/experiments/softmax_stable/softmax_final.keras`
- Curves: `docs/figures/softmax_curves.png`
- Selection: `/home/neon-cultivator/us/diabetic-retinopathy-detection/artifacts/experiments/softmax_stable/best_selection.json`

## EXP-STABLE-001b — CORAL EfficientNetB0 (BN-safe)

- Config: `{"seed": 42, "img_size": 224, "batch_size": 8, "num_classes": 5, "train_frac": 0.7, "val_frac": 0.15, "test_frac": 0.15, "crop_tol": 7, "clahe_clip": 2.0, "clahe_tile": 8, "graham_sigma": 10.0, "unsharp_sigma": 1.0, "unsharp_amount": 1.5, "backbone": "EfficientNetB0", "dropout": 0.3, "phase1_lr": 0.0003, "phase2_lr": 1e-05, "phase1_epochs": 8, "phase2_epochs": 8, "unfreeze_last": 20, "early_stop_patience": 3, "reduce_lr_patience": 2, "reduce_lr_factor": 0.5, "clipnorm": 1.0, "freeze_bn_on_finetune": true, "qwk_each_epoch": false, "kaggle_dataset": "sachinkumar413/diabetic-retinopathy-dataset", "data_dir": "/home/neon-cultivator/us/diabetic-retinopathy-detection/data", "raw_dir": "/home/neon-cultivator/us/diabetic-retinopathy-detection/data/raw", "processed_dir": "/home/neon-cultivator/us/diabetic-retinopathy-detection/data/processed", "models_dir": "/home/neon-cultivator/us/diabetic-retinopathy-detection/models", "figures_dir": "/home/neon-cultivator/us/diabetic-retinopathy-detection/docs/figures", "artifacts_dir": "/home/neon-cultivator/us/diabetic-retinopathy-detection/artifacts"}`
- Artifact: `/home/neon-cultivator/us/diabetic-retinopathy-detection/artifacts/experiments/coral_stable/coral_final.keras`
- Curves: `docs/figures/coral_curves.png`
- Selection: `/home/neon-cultivator/us/diabetic-retinopathy-detection/artifacts/experiments/coral_stable/best_selection.json`

## EXP-003 — Test evaluation & export

- Winner: **coral** (QWK=0.7190)
- meta: `models/meta.json`
- Results: `{"softmax": {"accuracy": 0.6246973365617433, "precision_macro": 0.5110558416915153, "recall_macro": 0.5163131313131313, "f1_macro": 0.5083679028672438, "precision_weighted": 0.6281587265852213, "recall_weighted": 0.6246973365617433, "f1_weighted": 0.6228369577488038, "qwk": 0.7129535144885886, "adjacent": {"n": 413, "exact": 0.6246973365617433, "adjacent_error": 0.24213075060532688, "far_error": 0.13317191283292978, "mean_abs_error": 0.5399515738498789}}, "coral": {"accuracy": 0.6271186440677966, "precision_macro": 0.41805066072573016, "recall_macro": 0.45734391534391533, "f1_macro": 0.4355096098479826, "precision_weighted": 0.571629919696417, "recall_weighted": 0.6271186440677966, "f1_weighted": 0.5960260621352537, "qwk": 0.7190319332179493, "adjacent": {"n": 413, "exact": 0.6271186440677966, "adjacent_error": 0.26150121065375304, "far_error": 0.11138014527845036, "mean_abs_error": 0.5012106537530266}}}`
