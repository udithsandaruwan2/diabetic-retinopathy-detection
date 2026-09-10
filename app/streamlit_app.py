"""Streamlit real-time DR stage screening demo with Grad-CAM."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf

from dr_detect import CLASS_NAMES
from dr_detect.config import CFG, ROOT
from dr_detect.explain import make_gradcam_heatmap, overlay_gradcam
from dr_detect.models import coral_logits_to_label
from dr_detect.preprocess import preprocess_fundus, preprocess_steps_gallery

st.set_page_config(page_title="DR Stage Screening Demo", layout="wide")

DISCLAIMER = (
    "**Disclaimer:** This prototype is a coursework screening assistant only. "
    "It is **not** a medical device and must not be used for diagnosis or treatment decisions."
)


@st.cache_resource
def load_model_and_meta():
    meta_path = CFG.models_dir / "meta.json"
    model_path = CFG.models_dir / "best_model.keras"
    if not model_path.exists():
        # fallbacks
        for cand in ("softmax_effb0.keras", "coral_effb0.keras"):
            p = CFG.models_dir / cand
            if p.exists():
                model_path = p
                break
    if not model_path.exists():
        return None, None

    custom = {}
    try:
        from dr_detect.losses import coral_loss

        custom["coral_loss"] = coral_loss
    except Exception:
        pass
    model = tf.keras.models.load_model(model_path, custom_objects=custom, compile=False)
    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        meta = {"coral": "coral" in model_path.name, "class_names": CLASS_NAMES}
    return model, meta


def predict(model, meta, bgr: np.ndarray):
    coral = bool(meta.get("coral", False))
    x = preprocess_fundus(bgr, is_bgr=True, for_model=True)[None, ...].astype(np.float32)
    raw = model.predict(x, verbose=0)
    if coral:
        grade = int(coral_logits_to_label(raw)[0])
        # severity score = expected grade from sigmoid probs
        probs_levels = 1 / (1 + np.exp(-raw[0]))
        conf = float(np.mean(np.abs(probs_levels - 0.5) * 2))  # crude certainty
        class_probs = None
    else:
        class_probs = raw[0]
        grade = int(np.argmax(class_probs))
        conf = float(class_probs[grade])
    steps = preprocess_steps_gallery(bgr)
    heat = make_gradcam_heatmap(x, model, pred_index=grade if not coral else None, coral=coral)
    overlay = overlay_gradcam(steps["unsharp"], heat)
    return grade, conf, class_probs, steps, overlay


def main():
    st.title("Diabetic Retinopathy Stage Detection")
    st.caption("EfficientNetB0 + hybrid fundus preprocess · Softmax / CORAL · Grad-CAM")
    st.info(DISCLAIMER)

    model, meta = load_model_and_meta()
    if model is None:
        st.error("No trained model found in `models/`. Run training + evaluate_export first.")
        st.stop()

    st.sidebar.header("How it works")
    st.sidebar.markdown(
        """
1. Upload a retinal fundus photo  
2. Pipeline: crop → CLAHE → Ben Graham → unsharp → 224px  
3. EfficientNetB0 predicts ICDR stage 0–4  
4. Grad-CAM highlights influential regions  
        """
    )
    show_prep = st.sidebar.checkbox("Show preprocess stages", value=True)
    show_cam = st.sidebar.checkbox("Show Grad-CAM", value=True)

    col_u, col_r = st.columns(2)
    with col_u:
        uploaded = st.file_uploader("Upload fundus image", type=["png", "jpg", "jpeg"])
        sample_dir = CFG.processed_dir
        # optional sample from test csv
        test_csv = CFG.processed_dir / "test.csv"
        if test_csv.exists() and st.button("Use random test sample"):
            import pandas as pd

            row = pd.read_csv(test_csv).sample(1, random_state=None).iloc[0]
            st.session_state["sample_path"] = row["image_path"]
            st.session_state["sample_true"] = int(row["label"])

    bgr = None
    true_label = None
    if uploaded is not None:
        data = np.frombuffer(uploaded.read(), dtype=np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    elif st.session_state.get("sample_path"):
        bgr = cv2.imread(st.session_state["sample_path"])
        true_label = st.session_state.get("sample_true")

    if bgr is None:
        st.write("Upload an image or click **Use random test sample** to begin.")
        st.stop()

    grade, conf, class_probs, steps, overlay = predict(model, meta, bgr)
    name = CLASS_NAMES.get(grade, str(grade))

    with col_r:
        st.subheader(f"Predicted stage: **{grade} — {name}**")
        st.metric("Confidence proxy", f"{conf:.2%}")
        if true_label is not None:
            st.write(f"Ground truth (sample): {true_label} — {CLASS_NAMES[true_label]}")
        if grade >= 3:
            st.error("High-risk grade (≥ Severe). Prioritize urgent clinical review.")
        elif grade >= 2:
            st.warning("Moderate DR suggested. Schedule clinical follow-up.")
        else:
            st.success("Lower grade suggested. Continue routine screening protocols.")

        if class_probs is not None:
            st.bar_chart({CLASS_NAMES[i]: float(class_probs[i]) for i in range(5)})

    c1, c2 = st.columns(2)
    with c1:
        st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), caption="Input", use_container_width=True)
    with c2:
        if show_cam:
            st.image(overlay, caption="Grad-CAM overlay", use_container_width=True)

    if show_prep:
        st.subheader("Preprocess preview")
        pcs = st.columns(5)
        for ax, key in zip(pcs, ["raw", "cropped", "clahe", "graham", "unsharp"]):
            ax.image(steps[key], caption=key, use_container_width=True)

    st.markdown("---")
    st.markdown(DISCLAIMER)


if __name__ == "__main__":
    main()
