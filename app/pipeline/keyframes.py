"""Stage 4 keyframes and stage 7 cutouts. Real pixels only; no generative edits here."""
from pathlib import Path

from PIL import Image, ImageFilter, ImageStat

from app.models.types import FrameHit, Track

PAD = 0.08


def sharpness(img: Image.Image) -> float:
    """Variance of an edge-filtered greyscale image: higher is sharper."""
    edges = img.convert("L").filter(ImageFilter.FIND_EDGES)
    return ImageStat.Stat(edges).var[0]


def _crop_box(hit: FrameHit, w: int, h: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = hit.box_xyxy
    px, py = (x1 - x0) * PAD, (y1 - y0) * PAD
    return (max(0, int(x0 - px)), max(0, int(y0 - py)), min(w, int(x1 + px)), min(h, int(y1 + py)))


def best_frame(track: Track, frame_path_for) -> tuple[FrameHit, float] | None:
    """Pick the hit maximising normalised area x normalised sharpness."""
    scored = []
    for hit in track.frames:
        path = frame_path_for(hit.frame_idx)
        if not Path(path).is_file():
            continue
        with Image.open(path) as img:
            box = _crop_box(hit, *img.size)
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            area = (box[2] - box[0]) * (box[3] - box[1])
            scored.append((hit, area, sharpness(img.crop(box))))
    if not scored:
        return None
    max_a = max(a for _, a, _ in scored) or 1
    max_s = max(s for _, _, s in scored) or 1
    hit, a, s = max(scored, key=lambda x: (x[1] / max_a) * (x[2] / max_s))
    return hit, round((a / max_a) * (s / max_s), 4)


def save_keyframe(track: Track, frame_path_for, out_dir: Path) -> Path | None:
    chosen = best_frame(track, frame_path_for)
    if chosen is None:
        return None
    hit, _ = chosen
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{track.track_id}.jpg"
    with Image.open(frame_path_for(hit.frame_idx)) as img:
        img.crop(_crop_box(hit, *img.size)).convert("RGB").save(out, quality=90)
    return out


def make_cutout(frame_path: Path, mask_path: Path, mask_box, out_path: Path) -> Path:
    """Real pixels inside SAM's mask on white. The mask raster covers mask_box (half-open, frame pixels)."""
    left, top, right, bottom = (int(round(v)) for v in mask_box)
    with Image.open(frame_path) as frame, Image.open(mask_path) as mask:
        region = frame.convert("RGB").crop((left, top, right, bottom))
        alpha = mask.convert("L").resize(region.size, Image.NEAREST)
        out = Image.new("RGB", region.size, "white")
        out.paste(region, (0, 0), alpha)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.save(out_path, quality=92)
    return out_path
