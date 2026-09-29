from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
from django.http import HttpRequest, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.decorators.http import require_GET, require_POST

from dr_detect.config import CFG
from screening.inference import DISCLAIMER, screen_image

DEVELOPER_URL = "https://udithsandaruwan.com"


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
        test_csv = CFG.processed_dir / "test.csv"
        if test_csv.exists():
            row = pd.read_csv(test_csv).sample(1).iloc[0]
            bgr = cv2.imread(str(row["image_path"]))
    if bgr is None:
        return JsonResponse({"error": "Upload a fundus image or use a sample."}, status=400)
    try:
        payload = screen_image(bgr)
    except Exception as exc:
        return JsonResponse({"error": str(exc)}, status=500)
    return JsonResponse(payload)
