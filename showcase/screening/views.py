from __future__ import annotations

import threading
from pathlib import Path

import cv2
import numpy as np
from django.http import HttpRequest, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.decorators.http import require_GET, require_POST

from dr_detect.config import CFG
from screening.assistant import remember, reply, write_note
from screening.inference import DISCLAIMER, screen_image

DEVELOPER_URL = "https://udithsandaruwan.com"
SAMPLES = Path(__file__).resolve().parent / "samples"
_sample_lock = threading.Lock()
_sample_i = 0


def _load_sample():
    """Next bundled test photograph. These files ship with the app."""
    global _sample_i
    if not SAMPLES.is_dir():
        return None
    files = sorted(
        path
        for path in SAMPLES.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not files:
        return None
    with _sample_lock:
        path = files[_sample_i % len(files)]
        _sample_i += 1
    return cv2.imread(str(path))


def _ctx(page: str) -> dict:
    return {"disclaimer": DISCLAIMER, "page": page, "developer_url": DEVELOPER_URL}


def home(request: HttpRequest):
    return render(request, "screening/home.html", _ctx("home"))


@ensure_csrf_cookie
def product(request: HttpRequest):
    return render(request, "screening/product.html", _ctx("product"))


def about(request: HttpRequest):
    return render(request, "screening/about.html", _ctx("about"))


@require_GET
def health(request: HttpRequest):
    ready = (CFG.models_dir / "best_model.keras").exists()
    return JsonResponse({"ok": True, "model": ready})


@require_POST
def screen(request: HttpRequest):
    upload = request.FILES.get("image")
    bgr = None
    if upload is not None:
        data = np.frombuffer(upload.read(), dtype=np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    elif request.POST.get("sample") == "1":
        bgr = _load_sample()
    if bgr is None:
        return JsonResponse({"error": "Upload a fundus image or use a sample."}, status=400)
    try:
        payload = screen_image(bgr)
    except Exception as exc:
        return JsonResponse({"error": str(exc)}, status=500)
    payload["reading_id"] = remember(payload)
    return JsonResponse(payload)


@require_POST
def note(request: HttpRequest):
    reading_id = request.POST.get("reading_id", "")
    body = write_note(reading_id)
    if body is None:
        return JsonResponse({"error": "That reading has expired. Analyse the photograph again."}, status=404)
    return JsonResponse(body)


@require_POST
def chat(request: HttpRequest):
    reading_id = request.POST.get("reading_id", "")
    message = request.POST.get("message", "")
    body = reply(reading_id, message)
    if body is None:
        return JsonResponse({"error": "That reading has expired. Analyse the photograph again."}, status=404)
    return JsonResponse(body)
