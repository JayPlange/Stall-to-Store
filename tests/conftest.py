import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.catalogue import ProductProposal, create_run, publish, set_status, store_draft
from app.config import Settings
from app.db import connect, init_db
from app.main import create_app

HAS_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
needs_ffmpeg = pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    # _env_file=None: tests never read a developer's .env (and never its API key).
    return Settings(_env_file=None, runs_dir=tmp_path / "runs", db_path=tmp_path / "test.db",
                    model_api_key=None, allow_live_model_calls=False, max_upload_mb=5)


@pytest.fixture
def conn(settings):
    c = connect(settings.db_path)
    init_db(c)
    yield c
    c.close()


@pytest.fixture
def client(settings):
    return TestClient(create_app(settings))


def make_draft(conn, products: list[dict], source: str = "real") -> dict:
    """Insert a run with draft products directly (no pipeline). Each dict: name, price (pesewas or None)."""
    run = create_run(conn, source=source)
    proposals = [ProductProposal(f"t{i}", p.get("brand"), p["name"], p.get("size"), None) for i, p in enumerate(products)]
    store_draft(conn, run["id"], proposals, [], [], {})
    rows = conn.execute("SELECT id FROM products WHERE run_id = ? ORDER BY id", (run["id"],)).fetchall()
    ids = [r["id"] for r in rows]
    for pid, p in zip(ids, products):
        if p.get("price"):
            conn.execute("UPDATE products SET price_pesewas = ?, price_source = 'seller' WHERE id = ?", (p["price"], pid))
    conn.commit()
    return {"run": run, "ids": ids}


@pytest.fixture
def published(conn):
    """Shop with: [0] approved 12.00, [1] approved 5.50, [2] hidden 3.00, [3] draft 9.00."""
    d = make_draft(conn, [{"name": "Milk", "brand": "Acme", "price": 1200}, {"name": "Soap", "price": 550},
                          {"name": "Hidden thing", "price": 300}, {"name": "Draft thing", "price": 900}])
    rid = d["run"]["id"]
    set_status(conn, rid, d["ids"][0], "approved")
    set_status(conn, rid, d["ids"][1], "approved")
    set_status(conn, rid, d["ids"][2], "hidden")
    shop = publish(conn, rid, "Auntie Ama Provisions", "+233 24 000 0000", "auntie-ama")
    return {**d, "shop": shop}


@pytest.fixture(scope="session")
def sample_video(tmp_path_factory) -> Path:
    if not HAS_FFMPEG:
        pytest.skip("ffmpeg not installed")
    out = tmp_path_factory.mktemp("media") / "stall.mp4"
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=10",
                    "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000", "-t", "6",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(out)], check=True)
    return out


@pytest.fixture(scope="session")
def silent_video(tmp_path_factory) -> Path:
    if not HAS_FFMPEG:
        pytest.skip("ffmpeg not installed")
    out = tmp_path_factory.mktemp("media") / "noaudio.mp4"
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=10",
                    "-t", "6", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)], check=True)
    return out
