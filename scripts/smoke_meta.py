"""Day 1 access gate (SPEC G1a): ONE deliberate call per invocation, never retried.

    python scripts/smoke_meta.py check                 # env var names only, no network
    python scripts/smoke_meta.py models   --yes        # GET /v1/models (documented list-models)
    python scripts/smoke_meta.py chat     --yes        # tiny text chat to STS_VISION_MODEL
    python scripts/smoke_meta.py transcribe --yes [--audio speech.wav]   # default: 2 s tone (proves access only)
    python scripts/smoke_meta.py sam --yes --video clip.mp4 --prompt "bottle"  # ONE SAM 3.1 video request

Raw sanitised responses go to runs/_smoke/raw/ (gitignored); a summary line is appended to
runs/_smoke/smoke_log.jsonl and the call is logged in model_calls. Each response's top-level keys are
printed so adapters can be written from what the API actually returns.
A successful models/chat call proves account access only. It does NOT prove SAM video segmentation or
transcription work: run those checks separately and read their outputs.
The sam check normalises the clip like ingest does (2 fps, <=1280 px) and is billed per frame (min 50).
"""
import argparse
import base64
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.artefacts import safe_join  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.db import connect, init_db, iso, utcnow  # noqa: E402
from app.models.base import AdapterNotVerified, CallContext, LiveCallsDisabled, MetaClient, ProviderError  # noqa: E402
from app.models.pricing import sam_video_cost, spark_cost, transcribe_cost  # noqa: E402
from app.models.segment import build_request as sam_request, extract_output_text, parse_tracks  # noqa: E402
from app.models.transcribe import build_request, normalise as transcribe_normalise  # noqa: E402
from app.pipeline.ingest import extract_frames, normalise_video  # noqa: E402


def _shape(body) -> object:
    if isinstance(body, dict):
        return {k: _shape(v) if k in ("usage", "error") else type(v).__name__ for k, v in body.items()}
    return type(body).__name__


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["check", "models", "chat", "transcribe", "sam"])
    ap.add_argument("--yes", action="store_true", help="confirm this single live call")
    ap.add_argument("--audio", help="transcribe: a WAV with real speech (mono 16-bit, 16/24 kHz)")
    ap.add_argument("--video", help="sam: a short MP4/MOV clip of products (no faces)")
    ap.add_argument("--prompt", default="product package", help="sam: ONE short noun phrase")
    args = ap.parse_args()
    s = get_settings()
    print(f"MODEL_API_KEY present: {s.api_key_present}; STS_ALLOW_LIVE_MODEL_CALLS={s.allow_live_model_calls}; "
          f"base={s.meta_api_base}")
    if args.what == "check":
        return 0
    if not args.yes:
        print("refusing: pass --yes to make exactly one live call")
        return 1
    conn = connect(s.db_path)
    init_db(conn)
    client = MetaClient(s, conn)
    smoke_dir = safe_join(s.runs_dir, "_smoke")
    smoke_dir.mkdir(parents=True, exist_ok=True)
    ctx = CallContext(None, f"smoke-{args.what}")
    model = {"models": "-", "chat": s.vision_model, "transcribe": s.transcribe_model, "sam": s.segment_model}[args.what]
    if args.what == "sam" and not args.video:
        print("refusing: sam needs --video")
        return 1
    entry = {"at": iso(utcnow()), "what": args.what, "model": model, "base": s.meta_api_base}
    try:
        if args.what == "models":
            r = client.request(ctx, model, "GET", "/models")
        elif args.what == "chat":
            r = client.request(ctx, model, "POST", "/chat/completions", json_body={
                "model": model, "messages": [{"role": "user", "content": "Reply with the single word: ok"}]},
                estimate_cost=lambda b: spark_cost(model, b.get("usage")))
        elif args.what == "transcribe":
            if args.audio:
                wav, label = Path(args.audio), "user-supplied speech sample"
            else:
                wav, label = smoke_dir / "tone.wav", "2 s generated 440 Hz tone"
                subprocess.run([s.ffmpeg_bin, "-nostdin", "-y", "-v", "error", "-f", "lavfi", "-i",
                                "sine=frequency=440:sample_rate=24000", "-t", "2", "-ac", "1", "-c:a", "pcm_s16le",
                                str(wav)], check=True)
            req = build_request(model)
            with wav.open("rb") as fh:
                r = client.request(ctx, model, "POST", "/asr/transcribe", files={
                    "request": (None, json.dumps(req), "application/json"), "audio": ("audio.wav", fh, "audio/wav")},
                    request_summary={"request": req, "audio": label},
                    estimate_cost=lambda b: transcribe_cost(b.get("audioDurationMs")))
            entry["normalised"] = asdict(transcribe_normalise(r.body))
        else:
            clip = normalise_video(Path(args.video), smoke_dir / "sam_clip.mp4", s.frame_fps, s.ffmpeg_bin)
            frames = len(extract_frames(clip, smoke_dir / "sam_frames", s.frame_fps, s.ffmpeg_bin))
            body = sam_request(model, args.prompt, "data:video/mp4;base64," + base64.b64encode(clip.read_bytes()).decode())
            r = client.request(ctx, model, "POST", "/responses", json_body=body,
                               request_summary={**body, "video": f"{frames} frames at {s.frame_fps} fps"},
                               estimate_cost=lambda _b: sam_video_cost(frames))
            tracks, diags = parse_tracks(extract_output_text(r.body), s.frame_fps, args.prompt, 0, None)
            entry["sam"] = {"frames_sent": frames, "tracks": len(tracks), "diagnostics": diags[:10],
                            "hits_per_track": {t.track_id: len(t.frames) for t in tracks}}
        cost = conn.execute("SELECT cost_usd FROM model_calls ORDER BY id DESC LIMIT 1").fetchone()[0]
        entry.update(ok=True, status=r.status, latency_ms=r.latency_ms, units=r.units, raw=r.raw_path,
                     estimated_cost_usd=cost,
                     shape=_shape(r.body))
    except AdapterNotVerified as exc:
        entry.update(ok=False, error_type="schema_mismatch", message=str(exc))
    except LiveCallsDisabled as exc:
        print(f"not called: {exc}")
        return 1
    except ProviderError as exc:
        entry.update(ok=False, status=exc.status, error_type=exc.error_type, message=exc.safe_message, raw=exc.raw_path)
    with (smoke_dir / "smoke_log.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    print(json.dumps(entry, indent=2))
    return 0 if entry.get("ok") else 3


if __name__ == "__main__":
    raise SystemExit(main())
