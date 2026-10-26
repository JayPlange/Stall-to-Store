"""DEVELOPMENT FIXTURE. Hand-written tracks and price mentions, NOT model output.

Used only to build and test the review screen, shop page and ordering before real inference works.
Fixture runs have runs.source = 'fixture', every page shows a fixture banner, and eval/run_eval.py
excludes them. The association step below is the real deterministic code run on fake inputs.
"""
import sqlite3
from pathlib import Path

from PIL import Image, ImageDraw

from app.artefacts import relative_to_runs, run_dir
from app.catalogue import ProductProposal, create_run, store_draft
from app.models.types import FrameHit, PriceMention, Track
from app.pipeline.associate import associate

FRAME_W, FRAME_H, FPS = 1280, 720, 2.0

# (track_id, brand, name, size, visible_from_s, visible_to_s, box)
FIXTURE_TRACKS = [
    ("t1", "FIXTURE", "Evaporated Milk", "160 g", 1.0, 5.0, (440, 160, 840, 600)),
    ("t2", "FIXTURE", "Tomato Paste", "70 g", 5.5, 9.0, (480, 200, 800, 560)),
    ("t3", "FIXTURE", "Bar Soap", "200 g", 9.5, 13.0, (300, 250, 600, 500)),
    ("t4", "FIXTURE", "Bar Soap", "200 g", 9.5, 13.0, (680, 250, 980, 500)),
    ("t5", "FIXTURE", "Body Lotion", "400 ml", 13.5, 17.5, (250, 150, 600, 620)),
    ("t6", "FIXTURE", "Toothpaste", "120 g", 13.5, 17.5, (680, 150, 1030, 620)),
    ("t7", "FIXTURE", "Sardines", "125 g", 18.0, 20.0, (440, 200, 840, 560)),
]

FIXTURE_MENTIONS = [
    PriceMention("the milk is 12 cedis", 1200, 3.0, 3.8, "milk", "fixture"),
    PriceMention("tomato paste, 5 cedis", 500, 7.0, 7.6, "tomato paste", "fixture"),
    PriceMention("these ones, 8 cedis", 800, 11.0, 11.6, "these ones", "fixture"),
    PriceMention("this one 35 cedis", 3500, 15.5, 16.0, "this one", "fixture"),
]


def fixture_tracks() -> list[Track]:
    tracks = []
    for tid, _b, name, _s, t0, t1, box in FIXTURE_TRACKS:
        hits, i = [], 0
        while (t := round(i / FPS, 3)) <= 20.0:
            if t0 <= t <= t1:
                hits.append(FrameHit(i, t, box, None, 0.9))
            i += 1
        tracks.append(Track(tid, name, hits))
    return tracks


def _placeholder(path: Path, title: str, colour: tuple[int, int, int]) -> None:
    img = Image.new("RGB", (360, 360), colour)
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, 359, 44), fill=(180, 20, 20))
    draw.text((12, 14), "DEV FIXTURE - NOT A REAL PHOTO", fill="white")
    draw.text((20, 170), title, fill="black")
    img.save(path, quality=85)


def load_fixture(conn: sqlite3.Connection, runs_root: Path) -> dict:
    run = create_run(conn, source="fixture", meta={"note": "development fixture, not model output"})
    rdir = run_dir(runs_root, run["id"], create=True)
    tracks = fixture_tracks()
    names = {tid: name for tid, _b, name, *_ in FIXTURE_TRACKS}
    decisions, per_track = associate(FIXTURE_MENTIONS, tracks, "word", FPS, FRAME_W, FRAME_H, names)
    proposals = []
    for i, (tid, brand, name, size, *_rest) in enumerate(FIXTURE_TRACKS):
        img = rdir / "keyframes" / f"{tid}.jpg"
        _placeholder(img, f"{name} {size}", (230 - i * 15, 220, 190 + i * 8))
        proposals.append(ProductProposal(tid, brand, name, size, relative_to_runs(runs_root, img), None))
    store_draft(conn, run["id"], proposals, FIXTURE_MENTIONS, decisions, per_track)
    return run
