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


def make_cutout(frame_path: Path, mask_path: Path, box_xyxy, out_path: Path) -> Path:
    """Composite the masked object on white and crop to its box. Mask: white = object."""
    with Image.open(frame_path) as frame, Image.open(mask_path) as mask:
        frame = frame.convert("RGB")
        mask = mask.convert("L").resize(frame.size)
        white = Image.new("RGB", frame.size, "white")
        composed = Image.composite(frame, white, mask)
        hit = FrameHit(0, 0.0, tuple(box_xyxy))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        composed.crop(_crop_box(hit, *frame.size)).save(out_path, quality=92)
    return out_path
