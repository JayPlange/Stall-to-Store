"""Tracking video: SAM's real boxes and masks drawn over the seller's footage. Local only, no model calls.

SAM ran at 2 fps, so boxes glide between tracked frames for playback. They only bridge short gaps; when an
object has left the frame nothing is drawn, so the video never shows a detection SAM did not make.
"""
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from app.models.types import FrameHit, Track

VIDEO_FPS = 12
MAX_GAP_S = 1.0     # bridge SAM frames at most this far apart
HOLD_S = 0.25       # show an isolated detection for this long either side
MASK_ALPHA = 95
PALETTE = [(234, 168, 27), (30, 169, 82), (193, 68, 30), (47, 128, 237), (155, 81, 224), (0, 168, 168), (232, 62, 140)]


def box_at(track: Track, t: float) -> tuple[tuple[float, float, float, float], FrameHit] | None:
    """Box at time t, and the nearest real hit whose mask to show. None when SAM had nothing there."""
    hits = track.frames
    before = next((h for h in reversed(hits) if h.t <= t), None)
    after = next((h for h in hits if h.t >= t), None)
    if before and after and before is not after and after.t - before.t <= MAX_GAP_S:
        k = (t - before.t) / (after.t - before.t)
        box = tuple(a + (b - a) * k for a, b in zip(before.box_xyxy, after.box_xyxy))
        return box, before if k < 0.5 else after
    for hit in (before, after):
        if hit and abs(hit.t - t) <= HOLD_S:
            return hit.box_xyxy, hit
    return None


def _draw_track(frame: Image.Image, box, label: str, colour, mask_path: Path | None, font) -> None:
    x0, y0, x1, y1 = (int(v) for v in box)
    if x1 <= x0 or y1 <= y0:
        return
    if mask_path and mask_path.is_file():
        with Image.open(mask_path) as m:
            alpha = m.convert("L").resize((x1 - x0, y1 - y0), Image.NEAREST).point(lambda v: MASK_ALPHA if v else 0)
        frame.paste(Image.new("RGB", alpha.size, colour), (x0, y0), alpha)
    draw = ImageDraw.Draw(frame)
    draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=colour, width=3)
    left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
    w, h = right - left + 12, bottom - top + 8
    ty = y0 - h if y0 - h >= 0 else y0
    draw.rectangle((x0, ty, x0 + w, ty + h), fill=colour)
    draw.text((x0 + 6, ty + 3 - top), label, font=font, fill=(255, 255, 255))


def render(video_path: Path, tracks: list[Track], labels: dict[str, str], mask_path_for, out_path: Path,
           size: tuple[int, int], ffmpeg_bin: str = "ffmpeg") -> Path:
    """Write an H.264 video with the seller's audio. mask_path_for(track_id, frame_idx) may return None."""
    work = out_path.parent / "overlay_frames"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    w, h = size
    try:
        subprocess.run([ffmpeg_bin, "-nostdin", "-y", "-v", "error", "-i", str(video_path),
                        "-vf", f"fps={VIDEO_FPS},scale={w}:{h}", "-q:v", "3", str(work / "%05d.jpg")], check=True)
        colours = {t.track_id: PALETTE[i % len(PALETTE)] for i, t in enumerate(tracks)}
        font = ImageFont.load_default(size=16)
        for i, path in enumerate(sorted(work.glob("*.jpg"))):
            t = i / VIDEO_FPS
            with Image.open(path) as img:
                frame = img.convert("RGB")
            for track in tracks:
                found = box_at(track, t)
                if found:
                    box, hit = found
                    _draw_track(frame, box, labels.get(track.track_id, "product"), colours[track.track_id],
                                mask_path_for(track.track_id, hit.frame_idx), font)
            frame.save(path, quality=90)
        subprocess.run([ffmpeg_bin, "-nostdin", "-y", "-v", "error", "-framerate", str(VIDEO_FPS),
                        "-i", str(work / "%05d.jpg"), "-i", str(video_path), "-map", "0:v", "-map", "1:a?",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-c:a", "aac", "-shortest",
                        "-movflags", "+faststart", str(out_path)], check=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return out_path
