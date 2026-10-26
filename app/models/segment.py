"""Stage 3 adapter: SAM 3.1 video tracking via the Responses API.

Documented (dev.meta.ai/docs/sam/segmenting and /sam/reading-segmentation, read 2026-10-26):
- POST /v1/responses, model "sam-3.1", one user message with an input_text noun phrase (ONE concept per
  request; lists return zero matches) and an input_video part (video_url as https or base64 data URL).
- MP4/MOV with H.264/H.265/MPEG-4; up to 15,000 frames; up to 16 tracked objects per frame.
- Output is text in the output_text lane: one line per frame, "<Nf>" then records of
  object id + box token (source-pixel xyxy; the parser returns half-open bounds, x2+1) + mask token (box-local base85 raster). Object ids are stable
  while an object stays in view. Zero matches is an empty output, not an error.
- Billed per frame sent, minimum 50 frames per request.
Parsing uses Meta's official parser package (meta-sam-parser 0.1.0), not a hand-written grammar.

NOT yet observed live: the non-streaming Responses JSON envelope. extract_output_text() follows the
documented Responses shape (output[].content[] of type output_text) and refuses anything else.
"""
import base64
from collections import defaultdict
from pathlib import Path

from meta_sam_parser import CompletedOutcome, decode_mask_to_raster, video_segmentation_format
from PIL import Image

from app.models.base import AdapterNotVerified, CallContext, MetaClient
from app.models.pricing import sam_video_cost
from app.models.types import FrameHit, Track

MAX_INLINE_VIDEO_BYTES = 20 * 1024 * 1024


def build_request(model: str, prompt: str, video_data_url: str) -> dict:
    if not prompt.strip() or "," in prompt:
        raise ValueError("SAM takes one short noun phrase per request")
    return {
        "model": model,
        "input": [{"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": prompt.strip()},
            {"type": "input_video", "video_url": video_data_url}]}],
        "metadata": {"include_confidence": "true", "mask_encoding": "one_bit"},
        "store": False,
    }


def extract_output_text(body: object) -> str:
    if isinstance(body, dict) and isinstance(body.get("output_text"), str):
        return body["output_text"]
    if isinstance(body, dict) and isinstance(body.get("output"), list):
        parts = [c.get("text", "") for item in body["output"] if isinstance(item, dict)
                 for c in (item.get("content") or []) if isinstance(c, dict) and c.get("type") == "output_text"]
        if parts or body.get("status") == "completed":
            return "".join(parts)
    keys = sorted(body.keys()) if isinstance(body, dict) else type(body).__name__
    raise AdapterNotVerified(f"Responses envelope differs from docs; top-level keys: {keys}")


def parse_tracks(output_text: str, fps: float, prompt: str, prompt_index: int,
                 mask_dir: Path | None) -> tuple[list[Track], list[str]]:
    """Official parser -> our Tracks. Masks are written as box-local 1-bit PNGs when mask_dir is given."""
    parser = video_segmentation_format().create_parser()
    parser.push(output_text, emit=False)
    result = parser.finish(CompletedOutcome()).result
    boxes, masks = {}, {}
    for rec in result.records:
        if rec.kind == "text" or rec.frame is None:
            continue
        key = (rec.object_id, rec.frame.frame_index)
        (boxes if rec.kind == "box" else masks)[key] = rec
    hits: dict[str, list[FrameHit]] = defaultdict(list)
    for (obj, frame_idx), box in sorted(boxes.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        track_id = f"p{prompt_index}o{obj}"
        mask_path, mask_box = None, None
        m = masks.get((obj, frame_idx))
        if m is not None and mask_dir is not None:
            raster = decode_mask_to_raster(m.mask)
            img = Image.frombytes("L", (m.mask.width, m.mask.height), bytes(v * 255 for v in raster))
            mask_dir.mkdir(parents=True, exist_ok=True)
            out = mask_dir / f"{track_id}_{frame_idx:05d}.png"
            img.convert("1").save(out)
            mask_path = out.as_posix()
            mask_box = (m.bounds.left, m.bounds.top, m.bounds.right, m.bounds.bottom)
        hits[track_id].append(FrameHit(frame_idx, round(frame_idx / fps, 3),
                                       (box.left, box.top, box.right, box.bottom), mask_path,
                                       box.confidence, mask_box))
    diagnostics = [f"{d.severity}:{d.code}:{d.message}" for d in result.diagnostics]
    return [Track(tid, prompt, h) for tid, h in hits.items()], diagnostics


def track(client: MetaClient, ctx: CallContext, video_path: Path, prompts: list[str], model: str,
          fps: float, frames_sent: int, mask_dir: Path | None = None) -> tuple[list[Track], list[str]]:
    """One request per prompt (each billed separately). video_path must be the normalised video."""
    data = video_path.read_bytes()
    if len(data) > MAX_INLINE_VIDEO_BYTES:
        raise ValueError("normalised video too large to send inline; use the Files API path")
    data_url = f"data:video/mp4;base64,{base64.b64encode(data).decode()}"
    tracks, diagnostics = [], []
    for i, prompt in enumerate(prompts):
        body = build_request(model, prompt, data_url)
        result = client.request(CallContext(ctx.run_id, f"{ctx.stage}-p{i}"), model, "POST", "/responses",
                                json_body=body, estimate_cost=lambda _b: sam_video_cost(frames_sent))
        found, diags = parse_tracks(extract_output_text(result.body), fps, prompt, i, mask_dir)
        tracks += found
        diagnostics += [f"p{i}:{d}" for d in diags]
    return tracks, diagnostics
