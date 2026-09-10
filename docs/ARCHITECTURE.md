# Architecture

Living architecture document. Mermaid sources below; rendered PNGs live in [`figures/`](figures/).

---

## 1. End-to-end system

```mermaid
flowchart LR
  subgraph data [Data]
    Kaggle[Kaggle_APTOS_224]
    Split[Stratified_70_15_15]
  end
  subgraph train [Train_Kaggle_GPU]
    Prep[CLAHE_Graham_Unsharp]
    Aug[Train_Aug_Only]
    B0[EfficientNetB0_TL]
    Soft[Softmax_Head]
    Coral[CORAL_Head]
    Pick[Select_by_Val_QWK]
  end
  subgraph demo [Local_Demo]
    Web[Streamlit_App]
    Artifacts[Saved_Model_plus_Meta]
    Cam[Grad_CAM]
  end
  Kaggle --> Split --> Prep --> Aug --> B0
  B0 --> Soft
  B0 --> Coral
  Soft --> Pick
  Coral --> Pick
  Pick --> Artifacts --> Web --> Cam
```

---

## 2. Dataset and split flow

```mermaid
flowchart TD
  DL[Download_Kaggle_Dataset] --> DF[Build_DataFrame_path_label]
  DF --> Map[Map_folder_names_to_0_4]
  Map --> Strat[StratifiedShuffleSplit_SEED_42]
  Strat --> Train[Train_70pct]
  Strat --> Val[Val_15pct]
  Strat --> Test[Test_15pct]
  Train --> AugOnly[Augmentation_ON]
  Val --> NoAug[Augmentation_OFF]
  Test --> NoAug
```

---

## 3. Preprocessing pipeline

```mermaid
flowchart LR
  Raw[Raw_RGB] --> Crop[Border_Crop]
  Crop --> CLAHE[CLAHE_Contrast]
  CLAHE --> Graham[Ben_Graham_Illum]
  Graham --> Unsharp[Unsharp_Edge_Enhance]
  Unsharp --> Resize[Resize_224]
  Resize --> EffPrep[EfficientNet_preprocess_input]
  EffPrep --> Tensor[Model_Tensor]
```

**Parameters (defaults):**

| Step | Key params |
|------|------------|
| Border crop | `tol=7` on dark pixels |
| CLAHE | `clipLimit=2.0`, `tileGridSize=(8,8)` on L channel (LAB) |
| Ben Graham | `addWeighted(img, 4, GaussianBlur(σ=10), -4, 128)` |
| Unsharp | Gaussian blur σ=1.0, amount=1.5 |
| Resize | 224×224 bilinear |

---

## 4. Augmentation (train only)

```mermaid
flowchart TD
  Batch[Train_Batch] --> Flip[Random_HorizontalFlip]
  Flip --> Rot[Random_Rotation_pm15]
  Rot --> Zoom[Random_Zoom_pm10pct]
  Zoom --> Bright[Random_Brightness_Contrast]
  Bright --> Shift[Random_Translation_small]
  Shift --> Out[Augmented_Batch]
```

---

## 5. Softmax model architecture

```mermaid
flowchart TD
  In[Input_224x224x3] --> Eff[EfficientNetB0_ImageNet_include_top_False]
  Eff --> GAP[GlobalAveragePooling2D]
  GAP --> Drop[Dropout_0.3]
  Drop --> Dense5[Dense_5_softmax]
  Dense5 --> Pred[Stage_0_to_4]
```

---

## 6. Transfer-learning phases

```mermaid
flowchart LR
  P1[Phase1_Freeze_Base] -->|Adam_1e-3| Head[Train_Head]
  Head --> P2[Phase2_Unfreeze_Top_40]
  P2 -->|Adam_1e-5| Fine[Fine_Tune]
  Fine --> Best[Checkpoint_Best_Val_QWK]
```

---

## 7. CORAL ordinal architecture

```mermaid
flowchart TD
  Feat[EfficientNetB0_Features] --> GAP2[GAP]
  GAP2 --> Drop2[Dropout]
  Drop2 --> CoralL[Dense_4_logits_no_activation]
  CoralL --> Sig[Sigmoid_P_severity_gt_k]
  Sig --> Decode[Count_thresholds_crossed]
  Decode --> Grade[Predicted_Grade_0_to_4]
```

CORAL models \(P(y > k)\) for \(k \in \{0,1,2,3\}\). Loss encourages rank-consistent cumulative probabilities.

---

## 8. Training loop and callbacks

```mermaid
flowchart TD
  Fit[model.fit] --> ES[EarlyStopping]
  Fit --> RLR[ReduceLROnPlateau]
  Fit --> CKPT[ModelCheckpoint_val_qwk]
  Fit --> CSV[CSVLogger]
  CKPT --> Art[artifacts_and_models]
```

---

## 9. Evaluation dashboard

```mermaid
flowchart LR
  Test[HeldOut_Test] --> Preds[Predictions]
  Preds --> Metrics[Acc_P_R_F1_QWK]
  Preds --> CM[Confusion_Matrix]
  Preds --> Err[Adjacent_vs_Far_Errors]
  Preds --> Curves[Loss_Acc_Curves]
```

---

## 10. Grad-CAM explainability

```mermaid
flowchart TD
  Img[Preprocessed_Image] --> Model[CNN]
  Model --> LastConv[Last_Conv_Activations]
  Model --> Score[Class_Score]
  Score --> Grads[Gradients_w_r_t_Conv]
  Grads --> Pool[Global_Average_Pool_Grads]
  Pool --> Heat[Weighted_Sum_ReLU_Normalize]
  Heat --> Overlay[Heatmap_on_Fundus]
```

---

## 11. Web demo UX

```mermaid
flowchart TD
  Upload[User_Upload] --> Valid[Validate_Image]
  Valid --> PrepPrev[Optional_Preprocess_Preview]
  PrepPrev --> Infer[Model_Inference]
  Infer --> Stage[Show_Stage_and_Confidence]
  Infer --> CAM[Grad_CAM_Overlay]
  Stage --> Risk[Risk_Banner_if_Severe]
  Risk --> Disc[Clinical_Disclaimer]
```

---

## 12. Hardware train / serve

```mermaid
flowchart LR
  LocalDev[Local_uv_Jupyter] --> Push[Push_Notebook_or_Script]
  Push --> KaggleGPU[Kaggle_Notebook_GPU]
  KaggleGPU --> Weights[best_model.keras]
  Weights --> LocalCPU[Local_Streamlit_CPU]
  LocalCPU --> User[Demo_User]
```
