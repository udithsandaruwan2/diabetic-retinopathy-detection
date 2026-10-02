"""Written note and chat for one screening. The stage is already decided."""

from __future__ import annotations

import base64
import json
import os
import threading
import time
import uuid

from screening.inference import DISCLAIMER

MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")
FALLBACK_MODEL = "gemini-3.1-flash-lite"
TTL_SECONDS = 30 * 60

SYSTEM = (
    "You are the written assistant on a diabetic retinopathy screening desk. "
    "You are not a doctor. You do not diagnose, and you do not name drugs, doses, or treatment plans. "
    "The stage number and name for the current photograph are fixed by a separate model. Never change them. "
    "Answer questions about this photograph, the five stages, what diabetic retinopathy is, "
    "what the model score means, how the network and CORAL head work, the project metrics, "
    "and what a person should do next. "
    "You may also answer other questions. Give a useful plain-language answer, "
    "and if the question is about health, keep it as screening guidance rather than a personal diagnosis. "
    "Practical advice is: keep routine eye checks, or ask an eye clinician to review the photograph soon. "
    "Do not invent lesions, measurements, or findings that are not in the supplied facts when you talk about this photograph. "
    "The score is the model's own score, not a calibrated probability. "
    "Proliferative recall on the locked test set is about 0.02, so a No DR or Mild result is not clearance. "
    "If the stage is Severe or Proliferative, say an eye clinician should review the photograph soon. "
    "For other stages, say to keep routine eye checks and to ask a clinician if the person is unsure. "
    "Never present your reply as a diagnosis."
)

NOTE_SCHEMA = {
    "type": "object",
    "properties": {
        "stage_line": {"type": "string"},
        "score_line": {"type": "string"},
        "next_step": {"type": "string"},
        "limit": {"type": "string"},
    },
    "required": ["stage_line", "score_line", "next_step", "limit"],
}

_lock = threading.Lock()
_readings: dict[str, dict] = {}


def _unavailable(stage_name: str) -> dict:
    return {
        "available": False,
        "stage_line": f"The model called this photograph {stage_name.replace('_', ' ')}.",
        "score_line": "The written assistant is unavailable. The stage above is unchanged.",
        "next_step": "Treat this as a screening hint. Ask an eye clinician if you are unsure.",
        "limit": DISCLAIMER,
    }


def _jpeg(data_url: str) -> bytes:
    raw = data_url.split(",", 1)[-1]
    return base64.b64decode(raw)


def _prune(now: float) -> None:
    stale = [key for key, row in _readings.items() if now - row["created"] > TTL_SECONDS]
    for key in stale:
        _readings.pop(key, None)


def remember(payload: dict) -> str:
    now = time.time()
    images = payload.get("images") or {}
    reading_id = uuid.uuid4().hex
    with _lock:
        _prune(now)
        _readings[reading_id] = {
            "created": now,
            "grade": int(payload.get("grade", 0)),
            "grade_name": str(payload.get("grade_name", "")),
            "confidence": float(payload.get("confidence", 0.0)),
            "level_probs": list(payload.get("level_probs") or []),
            "prepared": _jpeg(images["preprocessed"]) if images.get("preprocessed") else b"",
            "attention": _jpeg(images["gradcam"]) if images.get("gradcam") else b"",
            "note": None,
            "turns": [],
        }
    return reading_id


def _row(reading_id: str) -> dict | None:
    now = time.time()
    with _lock:
        _prune(now)
        row = _readings.get(reading_id)
        if row is None:
            return None
        return row


def _facts(row: dict) -> str:
    name = row["grade_name"].replace("_", " ")
    probs = ", ".join(f"{value:.2f}" for value in row["level_probs"])
    return (
        f"Fixed stage: {row['grade']} ({name}). "
        f"Model score: {row['confidence']:.2f}. This is not a calibrated probability. "
        f"Four cumulative threshold scores: {probs or 'not available'}. "
        "Proliferative recall on the locked test set is 0.07."
    )


def _client():
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        return None
    from google import genai

    return genai.Client(api_key=key)


def _generate(client, **kwargs):
    models = [MODEL]
    if FALLBACK_MODEL not in models:
        models.append(FALLBACK_MODEL)
    last = None
    for model in models:
        for attempt in range(3):
            try:
                return client.models.generate_content(model=model, **kwargs)
            except Exception as exc:
                last = exc
                text = str(exc)
                busy = any(token in text for token in ("503", "UNAVAILABLE"))
                missing = any(token in text for token in ("404", "NOT_FOUND"))
                if busy and attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                if busy or missing:
                    break
                raise
    raise last


def write_note(reading_id: str) -> dict | None:
    row = _row(reading_id)
    if row is None:
        return None
    client = _client()
    if client is None:
        note = _unavailable(row["grade_name"])
        row["note"] = note
        return note
    try:
        from google.genai import types

        parts = [types.Part.from_text(text=_facts(row))]
        if row["prepared"]:
            parts.append(types.Part.from_bytes(data=row["prepared"], mime_type="image/jpeg"))
        if row["attention"]:
            parts.append(types.Part.from_bytes(data=row["attention"], mime_type="image/jpeg"))
        parts.append(
            types.Part.from_text(
                text=(
                    "Write the four note fields. stage_line must repeat the fixed stage name. "
                    "score_line explains the model score. next_step is the plain next action. "
                    "limit states this is not a diagnosis and that proliferative disease is often missed."
                )
            )
        )
        response = _generate(
            client,
            contents=parts,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM,
                response_mime_type="application/json",
                response_json_schema=NOTE_SCHEMA,
                temperature=0.2,
            ),
        )
        data = json.loads(response.text or "{}")
        note = {
            "available": True,
            "stage_line": str(data["stage_line"]),
            "score_line": str(data["score_line"]),
            "next_step": str(data["next_step"]),
            "limit": str(data["limit"]),
        }
    except Exception:
        note = _unavailable(row["grade_name"])
    row["note"] = note
    return note


def reply(reading_id: str, message: str) -> dict | None:
    row = _row(reading_id)
    if row is None:
        return None
    text = " ".join(message.split())
    if not text:
        return {"reply": "Ask about this photograph."}
    if len(text) > 500:
        text = text[:500]
    client = _client()
    if client is None:
        return {
            "reply": "The written assistant is unavailable. The stage on screen is unchanged. " + DISCLAIMER
        }
    note = row.get("note") or {}
    prior = list(row["turns"])
    try:
        from google.genai import types

        contents = []
        for turn in prior[-8:]:
            contents.append(
                types.Content(role=turn["role"], parts=[types.Part.from_text(text=turn["text"])])
            )
        contents.append(types.Content(role="user", parts=[types.Part.from_text(text=text)]))
        response = _generate(
            client,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM + "\n" + _facts(row) + "\nNote: " + json.dumps(note),
                temperature=0.3,
            ),
        )
        answer = (response.text or "").strip() or (
            "I can only talk about this screening result. " + DISCLAIMER
        )
    except Exception:
        answer = "The written assistant is unavailable. The stage on screen is unchanged. " + DISCLAIMER
    row["turns"].append({"role": "user", "text": text})
    row["turns"].append({"role": "model", "text": answer})
    row["turns"] = row["turns"][-8:]
    return {"reply": answer}
