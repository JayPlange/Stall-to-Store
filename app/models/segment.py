"""Stage 3 adapter: SAM 3.1 tracking.

SPEC: "called as sam-3.1 through the Responses API ... still to confirm". The pages we could read
(2026-10-26) list sam-3.1 and say SAM accepts video, with files uploaded via POST /v1/files, but do not
document the request or the box/mask/track response. Do not assume a video call behaves like a
sequence of frame calls. This adapter makes no network call until the operation is verified.

Internal contract once verified: return list[Track]; each FrameHit.mask_path is a PNG (white = object)
written under runs/{run_id}/artefacts/masks/ by this adapter.
"""
from pathlib import Path

from app.models.base import AdapterNotVerified, CallContext, MetaClient
from app.models.types import Track


def track(client: MetaClient, ctx: CallContext, video_path: Path, prompts: list[str], model: str) -> list[Track]:
    raise AdapterNotVerified(
        "SAM 3.1 video request/response not verified. Run scripts/smoke_meta.py against the official "
        "SAM docs first, then implement this adapter from the saved raw response.")
