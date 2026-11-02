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
import re
from collections import defaultdict
from pathlib import Path

from meta_sam_parser import CompletedOutcome, SegmentationMask, decode_mask_to_raster, video_segmentation_format
from PIL import Image

from app.models.base import AdapterNotVerified, CallContext, MetaClient
from app.models.pricing import sam_video_cost
from app.models.types import FrameHit, Track

MAX_INLINE_VIDEO_BYTES = 20 * 1024 * 1024
# A valid one_bit 1x1 mask, used to skip the parser's per-mask decode (meta-sam-parser is pinned).
MASK_PLACEHOLDER = '!!!!!&!!!!!'
# Grammar per meta-sam-parser: mask payloads never contain "|" or ";".
_FRAME_RE = re.compile(r"<([0-9]+)f(?:;[^>]*)?>")
_RECORD_RE = re.compile(r"([0-9]+)((?:<\|[^|]*\|>)+)")
_MASK_DATA_RE = re.compile(r"data=([0-9]+),([0-9]+),([!~][^|;]+)")


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


def index_masks(output_text: str) -> dict[tuple[str, int], SegmentationMask]:
    """Map (object id, frame) to its undecoded mask. Decoding is slow, so it happens only on request."""
    masks = {}
    for line in output_text.splitlines():
        frame = _FRAME_RE.match(line)
        if not frame:
            continue
        for record in _RECORD_RE.finditer(line, frame.end()):
            data = _MASK_DATA_RE.search(record.group(2))
            if data:
                height, width, payload = int(data.group(1)), int(data.group(2)), data.group(3)
                encoding = "one_bit" if payload[0] == "!" else "lossless"
                masks[record.group(1), int(frame.group(1))] = SegmentationMask(encoding, payload, width, height)
    return masks


def save_mask(mask: SegmentationMask, out: Path) -> Path:
    """Decode one mask with Meta's strict decoder and write it as a box-local 1-bit PNG."""
    raster = decode_mask_to_raster(mask)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.frombytes("L", (mask.width, mask.height), bytes(v * 255 for v in raster)).convert("1").save(out)
    return out


def parse_tracks(output_text: str, fps: float, prompt: str, prompt_index: int) -> tuple[list[Track], list[str]]:
    """Official parser -> our Tracks, with boxes only.

    The parser decodes and re-encodes every mask in pure Python (about 0.7 s each; 18 minutes for one 67 s
    clip), so masks are swapped for a 1x1 placeholder first. Frames, ids, boxes, confidence and malformed-line
    diagnostics still come from the official parser. Use index_masks() and save_mask() for the frames we need.
    """
    stripped = _MASK_DATA_RE.sub(f"data=1,1,{MASK_PLACEHOLDER}", output_text)
    parser = video_segmentation_format().create_parser()
    parser.push(stripped, emit=False)
    result = parser.finish(CompletedOutcome()).result
    boxes = [rec for rec in result.records if rec.kind == "box" and rec.frame is not None]
    hits: dict[str, list[FrameHit]] = defaultdict(list)
    for box in sorted(boxes, key=lambda b: (b.frame.frame_index, b.object_id)):
        frame_idx = box.frame.frame_index
        xyxy = (box.left, box.top, box.right, box.bottom)
        hits[f"p{prompt_index}o{box.object_id}"].append(
            FrameHit(frame_idx, round(frame_idx / fps, 3), xyxy, None, box.confidence, xyxy))
    diagnostics = [f"{d.severity}:{d.code}:{d.message}" for d in result.diagnostics]
    return [Track(tid, prompt, h) for tid, h in hits.items()], diagnostics


def track(client: MetaClient, ctx: CallContext, video_path: Path, prompts: list[str], model: str,
          fps: float, frames_sent: int, output_dir: Path) -> tuple[list[Track], list[str]]:
    """One request per prompt (each billed separately). video_path must be the normalised video.

    Each prompt's output text is saved to output_dir/sam_output_p{i}.txt; masks are decoded from it later.
    """
    data = video_path.read_bytes()
    if len(data) > MAX_INLINE_VIDEO_BYTES:
        raise ValueError("normalised video too large to send inline; use the Files API path")
    data_url = f"data:video/mp4;base64,{base64.b64encode(data).decode()}"
    tracks, diagnostics = [], []
    for i, prompt in enumerate(prompts):
        body = build_request(model, prompt, data_url)
        result = client.request(CallContext(ctx.run_id, f"{ctx.stage}-p{i}"), model, "POST", "/responses",
                                json_body=body, estimate_cost=lambda _b: sam_video_cost(frames_sent))
        text = extract_output_text(result.body)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"sam_output_p{i}.txt").write_text(text, encoding="utf-8")
        found, diags = parse_tracks(text, fps, prompt, i)
        tracks += found
        diagnostics += [f"p{i}:{d}" for d in diags]
    return tracks, diagnostics
