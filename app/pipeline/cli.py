"""Run one pipeline stage at a time. Model stages need --live AND STS_ALLOW_LIVE_MODEL_CALLS=true.

    python -m app.pipeline.cli ingest     <run_id>
    python -m app.pipeline.cli transcribe <run_id> --live
    python -m app.pipeline.cli prices     <run_id> --live | --regex
    python -m app.pipeline.cli track      <run_id> --live
    python -m app.pipeline.cli keyframes  <run_id>
    python -m app.pipeline.cli identify   <run_id> --live
    python -m app.pipeline.cli link       <run_id> --live   # optional Muse Spark price matching
    python -m app.pipeline.cli associate  <run_id>
    python -m app.pipeline.cli cutouts    <run_id>
    python -m app.pipeline.cli overlay    <run_id>

Artefacts in runs/{run_id}/artefacts/: transcript.json, price_mentions.json, tracks.json,
keyframes.json, guesses.json, association.json, cutouts.json. Each stage can be re-run alone.
`track` sends ONE request per --prompts entry (each billed per frame): start with a single prompt.
"""
import argparse
import json
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from app import catalogue
from app.artefacts import relative_to_runs, run_dir
from app.config import get_settings
from app.db import connect, init_db
from app.logging_setup import configure_logging
from app.models import segment, transcribe, vision
from app.models.base import AdapterNotVerified, CallContext, LiveCallsDisabled, MetaClient, ProviderError
from app.models.types import Crop, PriceLink, PriceMention, Track, Transcript
from app.pipeline.associate import apply_price_links, associate, is_identifiable, merge_tracks
from app.pipeline import overlay as overlay_stage
from app.pipeline.ingest import ingest
from app.pipeline.keyframes import (PHOTO_CANDIDATES, make_contact_sheet, make_cutout, mask_fill, ranked_frames,
                                    save_keyframe)
from app.pipeline.prices_regex import regex_mentions

LIVE_STAGES = {"transcribe", "track", "identify", "link"}


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
    ap.add_argument("stage", choices=["ingest", "transcribe", "prices", "track", "keyframes", "identify", "associate",
                                      "cutouts", "overlay", "link"])
    ap.add_argument("run_id")
    ap.add_argument("--live", action="store_true", help="allow this stage to make a paid model call")
    ap.add_argument("--regex", action="store_true", help="prices: use the regex fallback, no model call")
    ap.add_argument("--prompts", default="product package", help="track: ';'-separated SAM noun phrases")
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
            meta = _read(rdir / "metadata.json")
            prompts = [p.strip() for p in args.prompts.split(";") if p.strip()]
            tracks, diags = segment.track(client, ctx, rdir / meta["normalised"], prompts, s.segment_model,
                                          meta["frame_fps"], meta["frame_count"], art)
            tracks, dropped = merge_tracks(tracks)
            _write(art / "tracks.json", {"prompts": prompts, "diagnostics": diags, "dropped_tracks": dropped,
                                         "tracks": [asdict(t) for t in tracks]})
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
            filtered = [t.track_id for t in tracks if t.track_id in guesses and not is_identifiable(
                guesses[t.track_id].get("brand"), guesses[t.track_id].get("name"),
                guesses[t.track_id].get("confidence"))]
            tracks = [t for t in tracks if t.track_id not in filtered]
            names = {tid: " ".join(x for x in (g.get("brand"), g.get("name")) if x) or None
                     for tid, g in guesses.items()}
            decisions, per_track = associate(mentions, tracks, pm["granularity"], meta["frame_fps"],
                                             meta["frame_width"], meta["frame_height"], names)
            proposals = [catalogue.ProductProposal(t.track_id, guesses.get(t.track_id, {}).get("brand"),
                                                   guesses.get(t.track_id, {}).get("name"),
                                                   guesses.get(t.track_id, {}).get("size"), kfs.get(t.track_id),
                                                   guesses.get(t.track_id, {}).get("confidence"))
                         for t in tracks]
            if (art / "price_links.json").is_file():  # optional: publishing never depends on this step
                links = [PriceLink(**link) for link in _read(art / "price_links.json")["links"]]
                apply_price_links(decisions, per_track, mentions, links)
            _write(art / "association.json", {"mentions": [asdict(d) for d in decisions],
                                              "tracks": {k: asdict(v) for k, v in per_track.items()},
                                              "filtered_tracks": filtered})
            catalogue.store_draft(conn, args.run_id, proposals, mentions, decisions, per_track)
        elif args.stage == "cutouts":
            tracks = {t["track_id"]: Track.from_dict(t) for t in _read(art / "tracks.json")["tracks"]}
            frames = {f["frame_idx"]: rdir / "frames" / f["file"] for f in _read(rdir / "frames.json")["frames"]}
            made, outputs = {}, {}
            filtered = set(_read(art / "association.json").get("filtered_tracks", []))
            for p in catalogue.list_products(conn, args.run_id):
                t = tracks.get(p["track_id"])
                if not t:
                    continue
                others_by_frame = defaultdict(list)
                for other in tracks.values():
                    if other.track_id != t.track_id and other.track_id not in filtered:
                        for h in other.frames:
                            others_by_frame[h.frame_idx].append(h.box_xyxy)
                ranked = ranked_frames(t, lambda i: frames.get(i, Path("missing")), others_by_frame)
                prompt_index, _, obj = t.track_id[1:].partition("o")
                if prompt_index not in outputs:
                    outputs[prompt_index] = segment.index_masks(
                        (art / f"sam_output_p{prompt_index}.txt").read_text(encoding="utf-8"))
                options = []
                for hit, score in ranked[:PHOTO_CANDIDATES]:
                    mask = outputs[prompt_index].get((obj, hit.frame_idx))
                    if mask is not None and hit.mask_box:
                        png = segment.save_mask(mask, art / "masks" / f"{t.track_id}_{hit.frame_idx:05d}.png")
                        options.append((score * mask_fill(png), hit, png))
                if not options:
                    continue
                _, hit, mask_png = max(options, key=lambda x: x[0])
                out = make_cutout(frames[hit.frame_idx], mask_png, hit.mask_box, rdir / "cutouts" / f"{t.track_id}.jpg")
                made[t.track_id] = relative_to_runs(s.runs_dir, out)
                conn.execute("UPDATE products SET cutout_path = ? WHERE id = ?", (made[t.track_id], p["id"]))
            conn.commit()
            _write(art / "cutouts.json", made)
        elif args.stage == "link":
            kfs = _read(art / "keyframes.json")
            products, tiles = [], []
            for g in _read(art / "guesses.json"):
                tid, path = g["track_id"], kfs.get(g["track_id"])
                if path and is_identifiable(g.get("brand"), g.get("name"), g.get("confidence")):
                    number = len(products) + 1
                    cutout = rdir / "cutouts" / f"{tid}.jpg"
                    label = " ".join(x for x in (g.get("brand"), g.get("name"), g.get("size")) if x)
                    products.append({"number": number, "track_id": tid, "label": label})
                    tiles.append((number, cutout if cutout.is_file() else s.runs_dir / path))
            sheet = make_contact_sheet(tiles, art / "contact_sheet.jpg")
            pm = _read(art / "price_mentions.json")
            links = vision.match_prices(client, ctx, sheet, Transcript.from_dict(_read(art / "transcript.json")).text,
                                        [PriceMention(**m) for m in pm["mentions"]], products, s.vision_model)
            _write(art / "price_links.json", {"products": products, "links": [asdict(x) for x in links]})
        elif args.stage == "overlay":
            meta = _read(rdir / "metadata.json")
            filtered = set(_read(art / "association.json").get("filtered_tracks", []))
            tracks = [t for t in map(Track.from_dict, _read(art / "tracks.json")["tracks"])
                      if t.track_id not in filtered]
            labels = {g["track_id"]: g.get("brand") or g.get("name") or "product" for g in _read(art / "guesses.json")}
            outputs = {}

            def mask_png(track_id: str, frame_idx: int) -> Path | None:
                png = art / "masks" / f"{track_id}_{frame_idx:05d}.png"
                if png.is_file():
                    return png
                prompt_index, _, obj = track_id[1:].partition("o")
                if prompt_index not in outputs:
                    outputs[prompt_index] = segment.index_masks(
                        (art / f"sam_output_p{prompt_index}.txt").read_text(encoding="utf-8"))
                mask = outputs[prompt_index].get((obj, frame_idx))
                return segment.save_mask(mask, png) if mask is not None else None

            overlay_stage.render(s.runs_dir / run["video_path"], tracks, labels, mask_png, art / "tracking.mp4",
                                 (meta["frame_width"], meta["frame_height"]), s.ffmpeg_bin)
            print("wrote tracking.mp4")
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
