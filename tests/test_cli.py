import json
from dataclasses import asdict

import pytest

from app.artefacts import run_dir
from app.catalogue import create_run, list_products
from app.fixtures import FIXTURE_MENTIONS, load_fixture
from app.pipeline import cli


@pytest.fixture
def cli_env(settings, conn, monkeypatch):
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    return settings


def test_paid_stage_needs_live_flag(cli_env, conn):
    run = create_run(conn)
    with pytest.raises(SystemExit, match="--live"):
        cli.main(["transcribe", run["id"]])
    with pytest.raises(SystemExit, match="--live"):
        cli.main(["prices", run["id"]])


def test_live_flag_still_blocked_when_calls_disabled(cli_env, conn):
    run = create_run(conn, video_path="placeholder/input/source.mp4")
    rd = run_dir(cli_env.runs_dir, run["id"], create=True)
    (rd / "audio.wav").write_bytes(b"RIFF")
    assert cli.main(["transcribe", run["id"], "--live"]) == 2
    assert conn.execute("SELECT COUNT(*) FROM model_calls").fetchone()[0] == 0
    assert cli.main(["track", run["id"], "--live"]) == 2


def test_refuses_fixture_runs(cli_env, conn):
    run = load_fixture(conn, cli_env.runs_dir)
    with pytest.raises(SystemExit, match="fixture"):
        cli.main(["associate", run["id"]])


def test_offline_stages_from_artefacts(cli_env, conn):
    """Regex prices + associate on hand-written test artefacts (test data, not model output)."""
    run = create_run(conn)
    rd = run_dir(cli_env.runs_dir, run["id"], create=True)
    art = rd / "artefacts"
    (rd / "metadata.json").write_text(json.dumps({"width": 1280, "height": 720}))
    (art / "transcript.json").write_text(json.dumps({"text": "milk 12 cedis", "granularity": "word", "units": [
        {"text": "the milk is 12 cedis", "t_start": 2.0, "t_end": 3.0, "confidence": None}]}))
    (art / "tracks.json").write_text(json.dumps({"tracks": [{"track_id": "t1", "label": "product", "frames": [
        {"frame_idx": i, "t": i / 2, "box_xyxy": [400, 150, 880, 600]} for i in range(0, 10)]}]}))
    (art / "guesses.json").write_text(json.dumps([{"track_id": "t1", "brand": "Acme", "name": "Milk", "size": None,
                                                   "confidence": 0.7}]))
    (art / "keyframes.json").write_text(json.dumps({"t1": None}))
    assert cli.main(["prices", run["id"], "--regex"]) == 0
    assert json.loads((art / "price_mentions.json").read_text())["mentions"][0]["amount_pesewas"] == 1200
    assert cli.main(["associate", run["id"]]) == 0
    p = list_products(conn, run["id"])[0]
    assert p["match_status"] == "auto" and p["price_pesewas"] == 1200 and p["status"] == "draft"
    assert asdict(FIXTURE_MENTIONS[0])  # fixtures stay importable, untouched
