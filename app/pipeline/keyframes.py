"""Stage 4 keyframes and stage 7 cutouts. Real pixels only; no generative edits here."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageStat

from app.models.types import FrameHit, Track

PAD = 0.08
EDGE_PX = 2
EDGE_PENALTY = 0.5
PHOTO_CANDIDATES = 5  # masks decode slowly, so only the best few frames are checked


def sharpness(img: Image.Image) -> float:
    """Variance of an edge-filtered greyscale image: higher is sharper."""
    edges = img.convert("L").filter(ImageFilter.FIND_EDGES)
    return ImageStat.Stat(edges).var[0]


def _crop_box(hit: FrameHit, w: int, h: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = hit.box_xyxy
    px, py = (x1 - x0) * PAD, (y1 - y0) * PAD
    return (max(0, int(x0 - px)), max(0, int(y0 - py)), min(w, int(x1 + px)), min(h, int(y1 + py)))


def _overlap(a, b) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(0, w) * max(0, h)


def _clear_view(hit: FrameHit, other_boxes: list, w: int, h: int) -> float:
    """1.0 for an unobstructed product; lower if other products cover it or it runs off the frame."""
    x0, y0, x1, y1 = hit.box_xyxy
    area = max(1, (x1 - x0) * (y1 - y0))
    covered = min(1.0, sum(_overlap(hit.box_xyxy, b) for b in other_boxes) / area)
    at_edge = x0 <= EDGE_PX or y0 <= EDGE_PX or x1 >= w - EDGE_PX or y1 >= h - EDGE_PX
    return (1 - covered) * (EDGE_PENALTY if at_edge else 1.0)


def ranked_frames(track: Track, frame_path_for, others_by_frame: dict | None = None) -> list[tuple[FrameHit, float]]:
    """Hits best first: normalised area x normalised sharpness x how clear the view is."""
    others_by_frame = others_by_frame or {}
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
            clear = _clear_view(hit, others_by_frame.get(hit.frame_idx, []), *img.size)
            scored.append((hit, area, sharpness(img.crop(box)), clear))
    if not scored:
        return []
    max_a = max(a for _, a, _, _ in scored) or 1
    max_s = max(s for _, _, s, _ in scored) or 1
    ranked = [(hit, round((a / max_a) * (s / max_s) * c, 4)) for hit, a, s, c in scored]
    return sorted(ranked, key=lambda x: -x[1])


def best_frame(track: Track, frame_path_for) -> tuple[FrameHit, float] | None:
    ranked = ranked_frames(track, frame_path_for)
    return ranked[0] if ranked else None


def mask_fill(mask_path: Path) -> float:
    """Share of the box the mask covers: low when a hand or another item hides part of the product."""
    with Image.open(mask_path) as mask:
        return ImageStat.Stat(mask.convert("L")).mean[0] / 255


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


def make_contact_sheet(items: list[tuple[int, Path]], out_path: Path, columns: int = 4, tile: int = 300) -> Path:
    """Numbered product tiles for Muse Spark. The numbers are how it refers to products."""
    rows = (len(items) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile, rows * tile), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=30)
    for i, (number, path) in enumerate(items):
        x, y = (i % columns) * tile, (i // columns) * tile
        with Image.open(path) as img:
            photo = img.convert("RGB")
        photo.thumbnail((tile - 20, tile - 20))
        sheet.paste(photo, (x + (tile - photo.width) // 2, y + (tile - photo.height) // 2))
        draw.rectangle((x, y, x + 46, y + 42), fill=(10, 63, 44))
        draw.text((x + 10, y + 4), str(number), font=font, fill="white")
        draw.rectangle((x, y, x + tile - 1, y + tile - 1), outline=(200, 200, 200))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=90)
    return out_path


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
