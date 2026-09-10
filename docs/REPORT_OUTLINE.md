# Report Outline (≤20 pages) — Mapped to Rubric

**Title:** Diabetic Retinopathy Stage Detection using Transfer Learning and Ordinal CORAL Regression  
**Course:** Computer Vision — BSCCOMP24.2P  
**Constraint:** PDF ≤ 20 pages + hosted prototype video URL

| Section | Suggested pages | Rubric |
|---------|----------------:|--------|
| 1. Introduction & clinical motivation | 1.5 | 1, 9 |
| 2. Dataset justification (Kaggle APTOS-224) | 1.5 | 1 |
| 3. Preprocessing pipeline (+ figures) | 2 | 2 |
| 4. Augmentation & class balancing | 1.5 | 3 |
| 5. CNN architecture & transfer learning | 2.5 | 4 |
| 6. Novel CORAL ordinal head | 1.5 | 4, 9 |
| 7. Training strategy & experiments | 2 | 5 |
| 8. Results & error analysis | 3 | 6 |
| 9. Prototype web demo & video | 1 | 8 |
| 10. Ethics, limitations, future work | 1.5 | 9 |
| 11. Conclusion | 0.5 | 8 |
| References | 0.5 | 8 |
| **Total** | **~19** | |

## Required figures (from `docs/figures/`)

1. System architecture  
2. Class distribution histogram  
3. Sample gallery per stage  
4. Preprocess before/after strip  
5. Augmentation mosaic  
6. Softmax architecture  
7. TL freeze/fine-tune timeline  
8. CORAL ordinal diagram  
9. Accuracy/loss curves  
10. Confusion matrices (Softmax & CORAL)  
11. Softmax vs CORAL metric bar chart  
12. Grad-CAM examples  
13. Streamlit screenshots  
14. Deployment hardware diagram  

## Prototype video

- Record Streamlit demo (upload → preprocess → prediction → Grad-CAM).  
- Host unlisted (YouTube/Drive).  
- Paste URL here when available: `VIDEO_URL=TBD`
