"""Whole-pipeline flow with MOCKED provider responses. Proves wiring and safety, not model accuracy."""
import json
import re

import httpx
import pytest
from fastapi.testclient import TestClient
from meta_sam_parser._mask_codec import _encode_raster
from pydantic import SecretStr

from app.catalogue import create_run, get_run, list_products, set_run_status
from app.artefacts import new_run_id, run_dir
from app.db import connect
from app.main import create_app
from app.models.base import MetaClient
from app.pipeline.ingest import ingest
from app.pipeline.orchestrate import run_pipeline, step_states
from tests.conftest import needs_ffmpeg

W, H = 4, 3
PAYLOAD = _encode_raster(bytes([1] * 12), W, H)
SAM = "".join(f"<{i}f>1<|box;x1=20;y1=20;x2=120;y2=120;w=320;h=240;c=0.9|><|mask;x=0;y=0;data={H},{W},{PAYLOAD}|>\n"
              for i in range(10))


def respond(calls, fail=None):
    def handler(request):
        path = request.url.path
        calls.append(path)
        if fail and fail in path:
            return httpx.Response(402, json={"error": {"message": "Billing account issue.", "type": "billing_error"}})
        if path.endswith("/asr/transcribe"):
            return httpx.Response(200, json={"sessionId": "s", "transcript": "the milk is 12 cedis", "audioDurationMs": 6000,
                                             "turns": [{"turnId": 0, "startMs": 0, "endMs": 5000, "transcript": "the milk is 12 cedis"}]})
        if path.endswith("/responses"):
            first = sum(c.endswith("/responses") for c in calls) == 1
            return httpx.Response(200, json={"status": "completed", "output": [
                {"type": "message", "content": [{"type": "output_text", "text": SAM if first else ""}]}]})
        body = json.loads(request.content)
        kind = body["response_format"]["json_schema"]["name"]
        content = {"price_mentions": {"mentions": [{"unit_start": 0, "unit_end": 0, "quote": "12 cedis", "amount_cedis": "12", "referent": "milk"}]},
                   "products": {"products": [{"index": 0, "brand": "Acme", "name": "Milk", "size": None, "confidence": 0.9}]},
                   "price_links": {"links": [{"price_index": 0, "product_numbers": [1], "confidence": 0.9, "reason": "named"}]}}[kind]
        return httpx.Response(200, json={"choices": [{"index": 0, "message": {"role": "assistant", "content": json.dumps(content)}}],
                                         "usage": {"prompt_tokens": 10, "completion_tokens": 5}})
    return handler


@pytest.fixture
def live_settings(settings):
    return settings.model_copy(update={"allow_live_model_calls": True, "model_api_key": SecretStr("k-test")})


@pytest.fixture
def ingested(conn, live_settings, sample_video):
    run_id = new_run_id()
    rdir = run_dir(live_settings.runs_dir, run_id, create=True)
    src = rdir / "input" / "source.mp4"
    src.write_bytes(sample_video.read_bytes())
    create_run(conn, "real", run_id, f"{run_id}/input/source.mp4")
    ingest(src, rdir, live_settings.frame_fps)
    set_run_status(conn, run_id, "ingested", "ingest")
    return run_id


def factory(settings, calls, fail=None):
    return lambda conn: MetaClient(settings, conn, httpx.MockTransport(respond(calls, fail)))


@needs_ffmpeg
def test_whole_pipeline_makes_a_draft_and_never_approves(live_settings, conn, ingested):
    calls = []
    assert run_pipeline(ingested, live_settings, factory(live_settings, calls)) is True
    assert sum(c.endswith("/responses") for c in calls) == 2 and len(calls) == 6  # one request per stage or prompt
    run = get_run(conn, ingested)
    assert (run["status"], run["stage"]) == ("draft_ready", "done")
    products = list_products(conn, ingested)
    assert len(products) == 1 and products[0]["status"] == "draft"
    assert products[0]["price_pesewas"] == 1200 and products[0]["cutout_path"]
    assert (live_settings.runs_dir / ingested / "artefacts" / "tracking.mp4").is_file()
    assert conn.execute("SELECT COUNT(*) FROM model_calls WHERE run_id = ?", (ingested,)).fetchone()[0] == 6


@needs_ffmpeg
def test_required_failure_stops_once_with_a_safe_message(live_settings, conn, ingested):
    calls = []
    assert run_pipeline(ingested, live_settings, factory(live_settings, calls, fail="/asr/transcribe")) is False
    assert len(calls) == 1  # no retry and no later paid stage
    run = get_run(conn, ingested)
    assert run["status"] == "failed" and "Billing account issue" in run["error"] and "k-test" not in run["error"]
    assert list_products(conn, ingested) == []


def test_step_states_follow_the_current_stage():
    states = [s["state"] for s in step_states({"stage": "prices"})]
    assert states[:3] == ["done", "active", "waiting"]


def _app(settings, started):
    app = create_app(settings)
    app.state.start_pipeline = started.append
    return TestClient(app)


@needs_ffmpeg
def test_upload_starts_processing_only_when_live_calls_are_on(settings, live_settings, sample_video):
    started = []
    c = _app(settings, started)
    c.post("/runs", files={"video": ("a.mp4", sample_video.read_bytes(), "video/mp4")})
    assert started == []
    c = _app(live_settings, started)
    r = c.post("/runs", files={"video": ("a.mp4", sample_video.read_bytes(), "video/mp4")})
    assert len(started) == 1 and "Working on your video" in r.text and 'http-equiv="refresh"' in r.text


def test_double_click_cannot_start_twice(live_settings, conn):
    run = create_run(conn, "real")
    set_run_status(conn, run["id"], "ingested", "ingest")
    started = []
    c = _app(live_settings, started)
    for _ in range(2):
        c.post(f"/r/{run['review_token']}/process")
    assert started == [run["id"]]


def test_process_refused_when_live_calls_are_off(settings, conn):
    run = create_run(conn, "real")
    set_run_status(conn, run["id"], "ingested", "ingest")
    started = []
    r = _app(settings, started).post(f"/r/{run['review_token']}/process")
    assert r.status_code == 400 and started == []


@needs_ffmpeg
def test_finishing_late_steps_never_undoes_a_publish(live_settings, conn, ingested):
    from app.pipeline import orchestrate
    real_run_stage = orchestrate.run_stage

    def publish_during_photos(stage, run, s, c, client, *a, **k):
        real_run_stage(stage, run, s, c, client, *a, **k)
        if stage == "cutouts":
            set_run_status(c, run["id"], "published", "publish")

    orchestrate.run_stage = publish_during_photos
    try:
        assert run_pipeline(ingested, live_settings, factory(live_settings, [])) is True
    finally:
        orchestrate.run_stage = real_run_stage
    assert get_run(conn, ingested)["status"] == "published"


def test_page_stops_refreshing_once_a_draft_exists(live_settings, conn):
    run = create_run(conn, "real")
    set_run_status(conn, run["id"], "draft_ready", "overlay")
    r = _app(live_settings, []).get(f"/r/{run['review_token']}")
    assert 'http-equiv="refresh"' not in r.text
