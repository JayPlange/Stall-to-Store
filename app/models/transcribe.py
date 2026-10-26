"""Stage 2 adapter: Muse Voice Transcribe.

Documented (dev.meta.ai/docs/api-reference/voice/transcribe, read 2026-10-26):
POST /v1/asr/transcribe, multipart parts `request` (JSON settings) and `audio` (audio/wav);
WAV only, max 10 minutes, max 32 MB body. The example request JSON uses model, audioEncoding, mode.
NOT documented in what we could read: the JSON TranscribeResponse fields (text, words, timings).
So normalisation refuses to guess until a real response has been inspected.
"""
import json
from pathlib import Path

from app.models.base import AdapterNotVerified, CallContext, MetaClient
from app.models.types import Transcript

MAX_AUDIO_BYTES = 32 * 1024 * 1024


def build_request(model: str, mode: str = "PUSH_TO_TALK") -> dict:
    return {"model": model, "audioEncoding": "WAV", "mode": mode}


def normalise(body: object) -> Transcript:
    """Map a TranscribeResponse to Transcript. Update this ONLY from an observed raw artefact."""
    keys = sorted(body.keys()) if isinstance(body, dict) else type(body).__name__
    raise AdapterNotVerified(
        f"TranscribeResponse shape not yet verified; observed top-level keys: {keys}. "
        "Inspect the raw artefact and implement this mapping.")


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
            request_summary={"request": req, "audio_bytes": size})
    return normalise(result.body), result.raw_path
