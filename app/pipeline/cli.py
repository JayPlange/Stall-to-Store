"""Run one pipeline stage at a time. Model stages need --live AND STS_ALLOW_LIVE_MODEL_CALLS=true.

    python -m app.pipeline.cli ingest     <run_id>
    python -m app.pipeline.cli transcribe <run_id> --live
    python -m app.pipeline.cli prices     <run_id> --live | --regex
    python -m app.pipeline.cli track      <run_id> --live
    python -m app.pipeline.cli keyframes  <run_id>
    python -m app.pipeline.cli identify   <run_id> --live
    python -m app.pipeline.cli associate  <run_id>

Artefacts in runs/{run_id}/artefacts/: transcript.json, price_mentions.json, tracks.json,
keyframes.json, guesses.json, association.json. Each stage can be re-run alone.
"""
import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from app import catalogue
from app.artefacts import relative_to_runs, run_dir
from app.config import get_settings
from app.db import connect, init_db
from app.logging_setup import configure_logging
from app.models import segment, transcribe, vision
from app.models.base import AdapterNotVerified, CallContext, LiveCallsDisabled, MetaClient, ProviderError
from app.models.types import Crop, PriceMention, Track, Transcript
from app.pipeline.associate import associate
from app.pipeline.ingest import ingest
from app.pipeline.keyframes import save_keyframe
from app.pipeline.prices_regex import regex_mentions

LIVE_STAGES = {"transcribe", "track", "identify"}


def _read(path: Path):
    if not path.is_file():
        raise SystemExit(f"missing artefact {path.name}; run the earlier stage first")
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {path.name}")


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["ingest", "transcribe", "prices", "track", "keyframes", "identify", "associate"])
    ap.add_argument("run_id")
    ap.add_argument("--live", action="store_true", help="allow this stage to make a paid model call")
    ap.add_argument("--regex", action="store_true", help="prices: use the regex fallback, no model call")
    ap.add_argument("--prompts", default="product,packet,bottle,tin,box", help="track: SAM text prompts")
    args = ap.parse_args(argv)

    s = get_settings()
    conn = connect(s.db_path)
    init_db(conn)
    run = catalogue.get_run(conn, args.run_id)
    if not run:
        raise SystemExit("unknown run")
    if run["source"] != "real":
        raise SystemExit("refusing to run the pipeline on a fixture run")
    rdir = run_dir(s.runs_dir, args.run_id, create=True)
    art = rdir / "artefacts"
    if args.stage in ("ingest", "track") and not run["video_path"]:
        raise SystemExit("run has no uploaded video")
    needs_live = args.stage in LIVE_STAGES or (args.stage == "prices" and not args.regex)
    if needs_live and not args.live:
        raise SystemExit(f"{args.stage} makes a paid model call: pass --live to confirm")
    client = MetaClient(s, conn)
    ctx = CallContext(args.run_id, args.stage)
    try:
        if args.stage == "ingest":
            ingest(s.runs_dir / run["video_path"], rdir, s.frame_fps, s.ffmpeg_bin, s.ffprobe_bin)
        elif args.stage == "transcribe":
            t, raw = transcribe.transcribe(client, ctx, rdir / "audio.wav", s.transcribe_model)
            _write(art / "transcript.json", {**t.to_dict(), "raw": raw})
        elif args.stage == "prices":
            t = Transcript.from_dict(_read(art / "transcript.json"))
            ms = regex_mentions(t) if args.regex else vision.parse_prices(client, ctx, t, s.vision_model)
            _write(art / "price_mentions.json", {"granularity": t.granularity, "mentions": [asdict(m) for m in ms]})
        elif args.stage == "track":
            tracks = segment.track(client, ctx, s.runs_dir / run["video_path"], args.prompts.split(","), s.segment_model)
            _write(art / "tracks.json", {"tracks": [asdict(t) for t in tracks]})
        elif args.stage == "keyframes":
            tracks = [Track.from_dict(t) for t in _read(art / "tracks.json")["tracks"]]
            frames = {f["frame_idx"]: rdir / "frames" / f["file"] for f in _read(rdir / "frames.json")["frames"]}
            out = {}
            for t in tracks:
                kf = save_keyframe(t, lambda i: frames.get(i, Path("missing")), rdir / "keyframes")
                out[t.track_id] = relative_to_runs(s.runs_dir, kf) if kf else None
            _write(art / "keyframes.json", out)
        elif args.stage == "identify":
            kfs = _read(art / "keyframes.json")
            crops = [Crop(tid, str(s.runs_dir / p)) for tid, p in kfs.items() if p]
            guesses = vision.identify(client, ctx, crops, s.vision_model)
            _write(art / "guesses.json", [asdict(g) for g in guesses])
        elif args.stage == "associate":
            meta = _read(rdir / "metadata.json")
            tracks = [Track.from_dict(t) for t in _read(art / "tracks.json")["tracks"]]
            pm = _read(art / "price_mentions.json")
            mentions = [PriceMention(**m) for m in pm["mentions"]]
            guesses = {g["track_id"]: g for g in _read(art / "guesses.json")}
            kfs = _read(art / "keyframes.json")
            names = {tid: (g.get("name") or None) for tid, g in guesses.items()}
            decisions, per_track = associate(mentions, tracks, pm["granularity"], s.frame_fps,
                                             meta["width"], meta["height"], names)
            proposals = [catalogue.ProductProposal(t.track_id, guesses.get(t.track_id, {}).get("brand"),
                                                   guesses.get(t.track_id, {}).get("name"),
                                                   guesses.get(t.track_id, {}).get("size"), kfs.get(t.track_id),
                                                   guesses.get(t.track_id, {}).get("confidence"))
                         for t in tracks]
            _write(art / "association.json", {"mentions": [asdict(d) for d in decisions],
                                              "tracks": {k: asdict(v) for k, v in per_track.items()}})
            catalogue.store_draft(conn, args.run_id, proposals, mentions, decisions, per_track)
        catalogue.set_run_status(conn, args.run_id, catalogue.get_run(conn, args.run_id)["status"], args.stage)
    except (LiveCallsDisabled, AdapterNotVerified) as exc:
        print(f"{args.stage}: {exc}", file=sys.stderr)
        return 2
    except ProviderError as exc:
        catalogue.set_run_status(conn, args.run_id, "failed", args.stage, exc.safe_message)
        print(f"{args.stage}: provider error {exc.status} {exc.error_type}: {exc.safe_message} "
              f"(raw: {exc.raw_path}). Not retried.", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
