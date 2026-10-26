"""Day 1 access gate (SPEC G1a): ONE deliberate call per invocation, never retried.

    python scripts/smoke_meta.py check                 # env var names only, no network
    python scripts/smoke_meta.py models   --yes        # GET /v1/models (documented list-models)
    python scripts/smoke_meta.py chat     --yes        # tiny text chat to STS_VISION_MODEL
    python scripts/smoke_meta.py transcribe --yes      # 2 s generated tone WAV to Muse Voice Transcribe

Raw sanitised responses go to runs/_smoke/raw/ (gitignored); a summary line is appended to
runs/_smoke/smoke_log.jsonl and the call is logged in model_calls. Each response's top-level keys are
printed so adapters can be written from what the API actually returns.
SAM 3.1 has no smoke call here: its request format is not documented in pages we could read.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.artefacts import safe_join  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.db import connect, init_db, iso, utcnow  # noqa: E402
from app.models.base import CallContext, LiveCallsDisabled, MetaClient, ProviderError  # noqa: E402
from app.models.transcribe import build_request  # noqa: E402


def _shape(body) -> object:
    if isinstance(body, dict):
        return {k: _shape(v) if k in ("usage", "error") else type(v).__name__ for k, v in body.items()}
    return type(body).__name__


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["check", "models", "chat", "transcribe"])
    ap.add_argument("--yes", action="store_true", help="confirm this single live call")
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
    model = {"models": "-", "chat": s.vision_model, "transcribe": s.transcribe_model}[args.what]
    entry = {"at": iso(utcnow()), "what": args.what, "model": model, "base": s.meta_api_base}
    try:
        if args.what == "models":
            r = client.request(ctx, model, "GET", "/models")
        elif args.what == "chat":
            r = client.request(ctx, model, "POST", "/chat/completions", json_body={
                "model": model, "messages": [{"role": "user", "content": "Reply with the single word: ok"}]})
        else:
            wav = smoke_dir / "tone.wav"
            subprocess.run([s.ffmpeg_bin, "-nostdin", "-y", "-v", "error", "-f", "lavfi", "-i",
                            "sine=frequency=440:sample_rate=16000", "-t", "2", "-ac", "1", "-c:a", "pcm_s16le",
                            str(wav)], check=True)
            req = build_request(model)
            with wav.open("rb") as fh:
                r = client.request(ctx, model, "POST", "/asr/transcribe", files={
                    "request": (None, json.dumps(req), "application/json"), "audio": ("tone.wav", fh, "audio/wav")},
                    request_summary={"request": req, "audio": "2 s generated 440 Hz tone"})
        entry.update(ok=True, status=r.status, latency_ms=r.latency_ms, units=r.units, raw=r.raw_path,
                     shape=_shape(r.body))
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
