"""Load the tuned model and summarize real layer activity for the showcase."""

from __future__ import annotations

import base64
import json
import threading

import cv2
import numpy as np
import tensorflow as tf

from dr_detect import CLASS_NAMES
from dr_detect.config import CFG
from dr_detect.explain import make_gradcam_heatmap, overlay_gradcam
from dr_detect.models import coral_logits_to_label
from dr_detect.preprocess import preprocess_fundus, preprocess_steps_gallery

DISCLAIMER = (
    "Screening assistant only. Not a medical device. Do not use this for diagnosis or treatment."
)

# Visual columns. Each tuple is (id, label, name predicate).
_STAGE_RULES = [
    ("stem", "Stem", lambda n: n.startswith("stem")),
    ("early", "Blocks 1–2", lambda n: n.startswith(("block1", "block2"))),
    ("mid", "Blocks 3–4", lambda n: n.startswith(("block3", "block4"))),
    ("deep", "Blocks 5–6", lambda n: n.startswith(("block5", "block6"))),
    ("top", "Block 7 + top", lambda n: n.startswith(("block7", "top_"))),
    ("dense512", "Dense 512", lambda n: n in {"dense_512", "dropout_512"}),
    ("dense256", "Dense 256", lambda n: n in {"dense_256", "dropout_256"}),
    ("logits", "Stage logits", lambda n: n in {"coral_logits", "predictions"}),
]

_lock = threading.Lock()
_cache: dict = {}


def _load():
    if _cache:
        return _cache
    with _lock:
        if _cache:
            return _cache
        meta_path = CFG.models_dir / "meta.json"
        model_path = CFG.models_dir / "best_model.keras"
        custom = {}
        try:
            from dr_detect.losses import coral_loss

            custom["coral_loss"] = coral_loss
        except Exception:
            pass
        model = tf.keras.models.load_model(model_path, custom_objects=custom, compile=False)
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        stages = _describe_stages(model)
        probe_names = [s["probe"] for s in stages if s["probe"]]
        probe = tf.keras.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(name).output for name in probe_names],
            name="activity_probe",
        )
        _cache.update(model=model, meta=meta, stages=stages, probe=probe, probe_names=probe_names)
        return _cache


def _describe_stages(model) -> list[dict]:
    by_name = {layer.name: layer for layer in model.layers}
    stages = []
    for sid, label, pred in _STAGE_RULES:
        members = [layer for name, layer in by_name.items() if pred(name)]
        if not members:
            continue
        params = 0
        for layer in members:
            try:
                params += int(layer.count_params())
            except Exception:
                pass
        probe = members[-1].name
        for layer in reversed(members):
            if layer.__class__.__name__ in {"Conv2D", "Dense", "Activation"} or layer.name in {
                "coral_logits",
                "predictions",
                "top_activation",
                "dense_512",
                "dense_256",
            }:
                probe = layer.name
                break
        units = _unit_count(by_name[probe])
        stages.append(
            {
                "id": sid,
                "label": label,
                "layers": len(members),
                "params": params,
                "units": units,
                "probe": probe,
            }
        )
    return stages


def _unit_count(layer) -> int:
    if hasattr(layer, "filters") and layer.filters:
        return int(layer.filters)
    if hasattr(layer, "units") and layer.units:
        return int(layer.units)
    try:
        shape = layer.output.shape
        if shape[-1] is not None:
            return int(shape[-1])
    except Exception:
        pass
    return 0


def _jpeg_b64(rgb: np.ndarray) -> str:
    bgr = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def screen_image(bgr: np.ndarray) -> dict:
    bundle = _load()
    model = bundle["model"]
    meta = bundle["meta"]
    coral = bool(meta.get("coral", True))
    x = preprocess_fundus(bgr, is_bgr=True, for_model=True)[None, ...].astype(np.float32)
    raw = model.predict(x, verbose=0)
    if coral:
        grade = int(coral_logits_to_label(raw, threshold=float(meta.get("coral_threshold", 0.5)))[0])
        level_probs = (1.0 / (1.0 + np.exp(-raw[0]))).astype(float).tolist()
        confidence = float(np.mean(np.abs(np.array(level_probs) - 0.5) * 2))
    else:
        probs = raw[0]
        grade = int(np.argmax(probs))
        level_probs = [float(v) for v in probs]
        confidence = float(probs[grade])

    acts = bundle["probe"].predict(x, verbose=0)
    if len(bundle["probe_names"]) == 1:
        acts = [acts]
    energies = [float(np.mean(np.abs(a))) for a in acts]
    peak = max(energies) or 1.0
    stages = []
    for stage, energy in zip(bundle["stages"], energies):
        stages.append(
            {
                **{k: stage[k] for k in ("id", "label", "layers", "params", "units", "probe")},
                "energy": energy,
                "intensity": energy / peak,
            }
        )

    steps = preprocess_steps_gallery(bgr)
    heat = make_gradcam_heatmap(x, model, pred_index=None if coral else grade, coral=coral)
    overlay = overlay_gradcam(steps["unsharp"], heat)
    total_params = int(sum(layer.count_params() for layer in model.layers if hasattr(layer, "count_params")))
    try:
        total_params = int(model.count_params())
    except Exception:
        pass

    return {
        "grade": grade,
        "grade_name": CLASS_NAMES.get(grade, str(grade)),
        "confidence": confidence,
        "coral": coral,
        "level_probs": level_probs,
        "disclaimer": DISCLAIMER,
        "backbone": meta.get("backbone", "EfficientNetB0"),
        "layer_count": len(model.layers),
        "param_count": total_params,
        "stages": stages,
        "images": {
            "original": _jpeg_b64(steps["raw"]),
            "preprocessed": _jpeg_b64(steps["unsharp"]),
            "gradcam": _jpeg_b64(overlay),
        },
    }
