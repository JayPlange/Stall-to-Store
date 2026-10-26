"""Stage 2 adapter: Muse Voice Transcribe.

Documented (dev.meta.ai/docs/speech-to-text and /api-reference/voice/transcribe, read 2026-10-26):
POST /v1/asr/transcribe, multipart parts `request` (JSON) and `audio` (WAV: mono 16-bit PCM, 16 or 24 kHz),
max 10 minutes and 32 MB. Response JSON: sessionId, transcript, audioDurationMs, turns[] with
turnId, startMs, endMs, transcript, speaker. NO word-level timestamps exist. `turns` is EMPTY in the
default PUSH_TO_TALK mode, so we request ENDPOINTING to get turn timings (segment granularity).
Documented but not yet observed live: normalise() validates strictly and refuses anything else.
"""
import json
from pathlib import Path

from app.models.base import AdapterNotVerified, CallContext, MetaClient
from app.models.pricing import transcribe_cost
from app.models.types import Transcript, Unit

MAX_AUDIO_BYTES = 32 * 1024 * 1024


def build_request(model: str, mode: str = "ENDPOINTING", keywords: list[str] | None = None) -> dict:
    req = {"model": model, "audioEncoding": "WAV", "mode": mode, "languageBias": ["English"]}
    if keywords:
        req["keywords"] = keywords
    return req


def normalise(body: object) -> Transcript:
    shape_ok = isinstance(body, dict) and isinstance(body.get("transcript"), str) \
        and isinstance(body.get("turns", []), list)
    if not shape_ok:
        keys = sorted(body.keys()) if isinstance(body, dict) else type(body).__name__
        raise AdapterNotVerified(f"TranscribeResponse differs from docs; observed top-level keys: {keys}")
    units = []
    for turn in body.get("turns") or []:
        start, end, text = turn.get("startMs"), turn.get("endMs"), turn.get("transcript")
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)) or not isinstance(text, str):
            raise AdapterNotVerified(f"turn differs from docs; keys: {sorted(turn)}")
        if text.strip():
            units.append(Unit(text.strip(), round(start / 1000, 3), round(end / 1000, 3)))
    if units:
        return Transcript(body["transcript"], "segment", units)
    text = body["transcript"].strip()
    return Transcript(body["transcript"], "none", [Unit(text)] if text else [])


def transcribe(client: MetaClient, ctx: CallContext, audio_path: Path, model: str) -> tuple[Transcript, str]:
    if audio_path.suffix.lower() != ".wav":
        raise ValueError("Muse Voice Transcribe accepts WAV only")
    size = audio_path.stat().st_size
    if size > MAX_AUDIO_BYTES:
        raise ValueError("audio exceeds the documented 32 MB limit")
    req = build_request(model)
    with audio_path.open("rb") as fh:
        result = client.request(
            ctx, model, "POST", "/asr/transcribe",
            files={"request": (None, json.dumps(req), "application/json"),
                   "audio": ("audio.wav", fh, "audio/wav")},
            request_summary={"request": req, "audio_bytes": size},
            estimate_cost=lambda b: transcribe_cost(b.get("audioDurationMs") if isinstance(b, dict) else None))
    return normalise(result.body), result.raw_path
