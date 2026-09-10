# Notes Knowledge Cache

Machine-readable memory distilled from `Notes/` for this diabetic-retinopathy / deep-learning project.
**Source of truth for agents:** prefer this file over re-reading large notebooks/PDFs unless detail is missing.
Last built from notebooks, the DR PDF guide, Colab eye HTML, transfer-learning labs, related tutorials, and OCR of `The_CNN_Blueprint.pdf` (see `Notes/CNN_BLUEPRINT_CACHE.md`). PPTX is title-only.

---

## Index of sources

| Source | Role |
|--------|------|
| `Diabetic Retinopathy Detection Using Deep Learning.pdf` | Primary DR product/guide: problem, pipeline, EfficientNetB3, kappa, Grad-CAM |
| `Colab_Eye_Disease_Prediction.ipynb.html` | Colab DR pipeline (setup → viz; Ben Graham optional) |
| `transfer_learning_tf*.ipynb` | Flowers + MobileNetV2 two-phase TL pattern |
| `data_augmentation_comparison.ipynb` | CIFAR aug vs baseline (PyTorch) |
| `Deep_Learning_Image_Classification*.ipynb` | Fashion-MNIST / MNIST CNN labs |
| `Deep_Neural_Network*.ipynb` | Tabular diabetes MLP |
| `Tumor_prediction_with_image_Masks.ipynb` | Mini U-Net + Dice |
| `siamese_network_tutorial.ipynb` | Contrastive Siamese (AT&T faces) |
| `Face_Recognition_Based_Attendance.ipynb` | ArcFace + OpenCV SSD attendance |
| `01_Hand_written_Character_Recognition*.ipynb` | MNIST CNN |
| `02_imdb_moview_review_classification_tf_hub.ipynb` | TF Hub text classifier |
| `Defense_Multi_Agent_LLM_fast.ipynb` | LangGraph multi-agent defense LLM |
| `The_CNN_Blueprint*.pdf` | Visual CNN primer → distilled in `CNN_BLUEPRINT_CACHE.md`; full OCR in `The_CNN_Blueprint_OCR.md` |
| `Convolutional Neural Networks- CNN*.pptx` | Lecture deck (minimal extractable text) |

---

## Domain: Diabetic Retinopathy

- **Role of system:** screening assistant for doctors — **not** a replacement clinician; trusted assistant, not opaque black box.
- **Signs:** microaneurysms, hemorrhages, cotton-wool spots, abnormal new vessels.
- **5 grades:** `0 No_DR`, `1 Mild`, `2 Moderate`, `3 Severe`, `4 Proliferative_DR`.
- **Datasets:** preferred pre-resized `sovitrath/diabetic-retinopathy-224x224-2019-data`; also APTOS / resized variants; full competition ~80GB.
- **Stack:** NumPy, Pandas, TF/Keras, OpenCV, scikit-learn; Kaggle via `kaggle.json` + `chmod 600`.

### Config (canonical from notes)

```text
IMG_SIZE=224  BATCH_SIZE=32  EPOCHS=30  NUM_CLASSES=5
LEARNING_RATE=1e-4 (phase1) → 1e-5 (phase2)
DROPOUT_RATE=0.3
split: train 70% / val 10% / test 20%  (test never in training)
```

### Preprocessing

- Default: BGR→RGB, resize 224, `/255.0`.
- Optional **Ben Graham**: crop black borders (`tol=7`) → resize → `cv2.addWeighted(img, 4, GaussianBlur(σ=10), -4, 128)`.
- Labels: folder/DataFrame with map No_DR…Proliferative and `"0"`–`"4"`.

### Class imbalance

- Healthy ≫ severe → naive model predicts No_DR.
- Fix: `sklearn.utils.class_weight.compute_class_weight('balanced', ...)`.

### Model: EfficientNetB3 + head

- Base: `EfficientNetB3(weights='imagenet', include_top=False, input_shape=(224,224,3))`.
- Head: `GlobalAveragePooling2D` → `Dense(512, relu)` → `Dense(256, relu)` → `Dense(5, softmax)`.
- **Phase 1:** freeze base; Adam `1e-4`; `sparse_categorical_crossentropy`.
- **Phase 2:** unfreeze last **30** layers; Adam `1e-5` (avoid catastrophic forgetting).

### Metrics & explainability

- Prefer **quadratic weighted kappa** + confusion matrix + precision/recall over accuracy alone.
- Quote from notes: *“Kappa rewards close mistakes and punishes dangerous ones.”*
- **Grad-CAM** on last conv layer; heatmaps should hit vessels/lesions, not corners/artifacts.
- Inference: acquire → standardize → analyze → risk-based output; persist model for screening.

---

## CNN fundamentals (from labs + blueprint)

Full visual primer: `Notes/CNN_BLUEPRINT_CACHE.md`.

- Image = H×W×C numeric tensor (RGB channels); Conv2D needs channel dim — MNIST `(N,28,28)` → `(N,28,28,1)`.
- Why CNNs: dense nets explode parameters and discard spatial layout; local filters preserve it.
- Four pillars: **Conv** (sliding match → feature heatmaps) → **ReLU** (neg→0) → **MaxPool** (2×2/stride2) → **FC** (flatten → class scores).
- Hierarchy: edges → corners/shapes → objects via stacked Conv→ReLU→Pool.
- **MNIST CNN pattern:** Conv32→Pool → Conv64→Pool → Flatten → Dense128 → Dropout(0.5) → Dense10; Adam 0.001; batch 128.
- Fashion-MNIST MLP and tabular diabetes DNN also in notes (see hyperparams table).

---

## Transfer learning pattern (flowers / MobileNetV2 — same recipe as DR)

1. ImageNet base `include_top=False`.
2. Train head with base frozen (`training=False` for BN); higher LR.
3. Unfreeze top N layers; lower LR.
4. Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard, CSVLogger.
5. Normalize with backbone’s `preprocess_input` when required (MobileNetV2 → **[-1, 1]**).

Flowers: 5 classes; split 0.70/0.15/0.15; `SEED=42`; Phase1 LR `1e-3` ≤30 ep; Phase2 unfreeze 30 @ `1e-5` ≤20 ep.

---

## Data augmentation

- **Train only** — never augment val/test.
- DR conceptual: rotate, shift, hflip, slight zoom.
- ImageDataGenerator (flowers): hflip, rotation 20, shift 0.10, zoom 0.20, shear 0.15, brightness [0.70,1.30], `fill_mode=nearest`.
- CIFAR study: aug improves test acc and shrinks train–test gap; models train slower on harder examples.

---

## Other lab projects (retained memory)

### Tumor MRI masks (U-Net)
- `mateuszbuda/lgg-mri-segmentation`; 128² grayscale; Dice loss; mini U-Net 16→32→64 with skips; batch 8, 20 ep; threshold 0.5.

### Siamese (AT&T)
- Shared CNN → 128-D L2 embedding; Euclidean distance; labels **0=same, 1=different**; contrastive loss margin 1.0; 112×92×1; threshold search for one-shot.

### Face attendance
- OpenCV Caffe SSD + ArcFace ONNX 512-D; cosine thr **0.40** (lower=stricter); det conf 0.80; face ≥80px; preprocess `(x-127.5)/128.0` @ 112².

### IMDB TF Hub
- `gnews-swivel-20dim` → Dense16→Dense1 sigmoid; batch 512; 20 ep.

### Multi-agent defense LLM
- LangGraph + Ollama; parallel ISR/Cyber/Logistics → Threat → Planner → Debate → Governance → Commander; decision support only, no autonomous actions.

---

## Preferred code patterns

- Central Config cell; `SEED=42`; Kaggle download+unzip.
- TF/Keras for most vision; PyTorch for CIFAR aug comparison.
- Medical: class weights + quadratic kappa + Grad-CAM + optional Ben Graham.
- Generators: `ImageDataGenerator` / path DataFrames + OpenCV.
- Power-of-two batch sizes that fit GPU/RAM.

---

## Hyperparameter cheat sheet

| Context | Key values |
|---------|------------|
| DR | 224², BS32, 30ep, LR 1e-4→1e-5, EfficientNetB3, unfreeze last 30 |
| Flowers TL | 224², BS32, P1 30ep/1e-3, P2 20ep/1e-5, dropout 0.30, ES patience 7 |
| CIFAR aug | BS128, 30ep, AdamW 1e-3 wd 1e-4, CosineAnnealing |
| Siamese | 112×92, emb128, margin1.0, BS32, 50ep, LR1e-4 |
| Tumor U-Net | 128², BS8, 20ep, Dice |
| Face ArcFace | 112², conf0.80, cos0.40 |
| MNIST CNN | Adam0.001, BS128, Dropout0.5, 10ep |

---

## Non-negotiables when building on these notes

1. Do not treat accuracy as the main DR metric — use quadratic kappa + CM.
2. Always handle class imbalance (class weights).
3. Two-phase freeze → fine-tune at lower LR.
4. Never leak test data; never augment val/test.
5. Match preprocessing to backbone.
6. Prefer explainability (Grad-CAM) for medical outputs.
7. Frame outputs as clinical decision support, not autonomous diagnosis.
