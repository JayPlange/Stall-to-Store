"""Every pipeline stage, shared by the command line and the app. Reads and writes runs/{run_id}/artefacts/."""
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from app import catalogue
from app.artefacts import relative_to_runs, run_dir
from app.config import Settings
from app.models import segment, transcribe, vision
from app.models.base import CallContext, MetaClient
from app.models.types import Crop, PriceLink, PriceMention, Track, Transcript
from app.pipeline import overlay as overlay_stage
from app.pipeline.associate import apply_price_links, associate, merge_tracks, unusable_tracks
from app.pipeline.ingest import ingest
from app.pipeline.keyframes import (PHOTO_CANDIDATES, make_contact_sheet, make_cutout, mask_fill, ranked_frames,
                                    save_keyframe)
from app.pipeline.prices_regex import regex_mentions

LIVE_STAGES = {"transcribe", "track", "identify", "link"}
OVERLAY_MASK_EVERY = 2  # decode one mask per second of video (SAM tracks at 2 fps)


class StageError(RuntimeError):
    """A stage cannot run, e.g. an earlier stage's artefact is missing."""


def _read(path: Path):
    if not path.is_file():
        raise StageError(f"missing artefact {path.name}; run the earlier stage first")
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {path.name}")


def run_stage(stage: str, run: dict, s: Settings, conn, client: MetaClient, prompts: str | None = None,
              regex: bool = False) -> None:
    run_id = run["id"]
    rdir = run_dir(s.runs_dir, run_id, create=True)
    art = rdir / "artefacts"
    art.mkdir(exist_ok=True)
    prompts = prompts or s.sam_prompts
    ctx = CallContext(run_id, stage)
    if stage == "ingest":
        ingest(s.runs_dir / run["video_path"], rdir, s.frame_fps, s.ffmpeg_bin, s.ffprobe_bin)
    elif stage == "transcribe":
        t, raw = transcribe.transcribe(client, ctx, rdir / "audio.wav", s.transcribe_model)
        _write(art / "transcript.json", {**t.to_dict(), "raw": raw})
    elif stage == "prices":
        t = Transcript.from_dict(_read(art / "transcript.json"))
        ms = regex_mentions(t) if regex else vision.parse_prices(client, ctx, t, s.vision_model)
        _write(art / "price_mentions.json", {"granularity": t.granularity, "mentions": [asdict(m) for m in ms]})
    elif stage == "track":
        meta = _read(rdir / "metadata.json")
        prompts = [p.strip() for p in prompts.split(";") if p.strip()]
        tracks, diags = segment.track(client, ctx, rdir / meta["normalised"], prompts, s.segment_model,
                                      meta["frame_fps"], meta["frame_count"], art)
        tracks, dropped = merge_tracks(tracks)
        _write(art / "tracks.json", {"prompts": prompts, "diagnostics": diags, "dropped_tracks": dropped,
                                     "tracks": [asdict(t) for t in tracks]})
    elif stage == "keyframes":
        tracks = [Track.from_dict(t) for t in _read(art / "tracks.json")["tracks"]]
        frames = {f["frame_idx"]: rdir / "frames" / f["file"] for f in _read(rdir / "frames.json")["frames"]}
        out = {}
        for t in tracks:
            kf = save_keyframe(t, lambda i: frames.get(i, Path("missing")), rdir / "keyframes")
            out[t.track_id] = relative_to_runs(s.runs_dir, kf) if kf else None
        _write(art / "keyframes.json", out)
    elif stage == "identify":
        kfs = _read(art / "keyframes.json")
        crops = [Crop(tid, str(s.runs_dir / p)) for tid, p in kfs.items() if p]
        guesses = vision.identify(client, ctx, crops, s.vision_model)
        _write(art / "guesses.json", [asdict(g) for g in guesses])
    elif stage == "associate":
        meta = _read(rdir / "metadata.json")
        tracks = [Track.from_dict(t) for t in _read(art / "tracks.json")["tracks"]]
        pm = _read(art / "price_mentions.json")
        mentions = [PriceMention(**m) for m in pm["mentions"]]
        guesses = {g["track_id"]: g for g in _read(art / "guesses.json")}
        kfs = _read(art / "keyframes.json")
        unusable = unusable_tracks(tracks, guesses)
        filtered = sorted(unusable)
        tracks = [t for t in tracks if t.track_id not in unusable]
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
                                          "filtered_tracks": filtered, "filter_reasons": unusable})
        catalogue.store_draft(conn, run_id, proposals, mentions, decisions, per_track)
    elif stage == "cutouts":
        tracks = {t["track_id"]: Track.from_dict(t) for t in _read(art / "tracks.json")["tracks"]}
        frames = {f["frame_idx"]: rdir / "frames" / f["file"] for f in _read(rdir / "frames.json")["frames"]}
        made, outputs = {}, {}
        filtered = set(_read(art / "association.json").get("filtered_tracks", []))
        for p in catalogue.list_products(conn, run_id):
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
                    png = art / "masks" / f"{t.track_id}_{hit.frame_idx:05d}.png"  # often made by the overlay
                    png = png if png.is_file() else segment.save_mask(mask, png)
                    options.append((score * mask_fill(png), hit, png))
            if not options:
                continue
            _, hit, mask_png = max(options, key=lambda x: x[0])
            out = make_cutout(frames[hit.frame_idx], mask_png, hit.mask_box, rdir / "cutouts" / f"{t.track_id}.jpg")
            made[t.track_id] = relative_to_runs(s.runs_dir, out)
            conn.execute("UPDATE products SET cutout_path = ? WHERE id = ?", (made[t.track_id], p["id"]))
        conn.commit()
        _write(art / "cutouts.json", made)
    elif stage == "link":
        kfs = _read(art / "keyframes.json")
        products, tiles = [], []
        guesses = _read(art / "guesses.json")
        tracks = [Track.from_dict(t) for t in _read(art / "tracks.json")["tracks"]]
        unusable = unusable_tracks(tracks, {g["track_id"]: g for g in guesses})
        for g in guesses:
            tid, path = g["track_id"], kfs.get(g["track_id"])
            if path and tid not in unusable:
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
    elif stage == "overlay":
        meta = _read(rdir / "metadata.json")
        filtered = set(_read(art / "association.json").get("filtered_tracks", []))
        tracks = [t for t in map(Track.from_dict, _read(art / "tracks.json")["tracks"])
                  if t.track_id not in filtered]
        labels = {g["track_id"]: g.get("brand") or g.get("name") or "product" for g in _read(art / "guesses.json")}
        # Boxes follow every SAM frame. Masks decode slowly, so every other frame's mask is decoded (in
        # parallel) and each frame borrows the nearest one: the video is ready about a minute after the draft.
        outputs, decoded, jobs = {}, defaultdict(list), []
        for t in tracks:
            prompt_index, _, obj = t.track_id[1:].partition("o")
            if prompt_index not in outputs:
                outputs[prompt_index] = segment.index_masks(
                    (art / f"sam_output_p{prompt_index}.txt").read_text(encoding="utf-8"))
            for hit in t.frames[::OVERLAY_MASK_EVERY]:
                mask = outputs[prompt_index].get((obj, hit.frame_idx))
                if mask is None:
                    continue
                png = art / "masks" / f"{t.track_id}_{hit.frame_idx:05d}.png"
                decoded[t.track_id].append(hit.frame_idx)
                if not png.is_file():
                    jobs.append((mask, png))
        segment.save_masks(jobs)

        def mask_png(track_id: str, frame_idx: int) -> Path | None:
            if not decoded[track_id]:
                return None
            nearest = min(decoded[track_id], key=lambda f: abs(f - frame_idx))
            return art / "masks" / f"{track_id}_{nearest:05d}.png"

        overlay_stage.render(s.runs_dir / run["video_path"], tracks, labels, mask_png, art / "tracking.mp4",
                             (meta["frame_width"], meta["frame_height"]), s.ffmpeg_bin)
        print("wrote tracking.mp4")
