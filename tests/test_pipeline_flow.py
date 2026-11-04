"""Whole-pipeline flow with MOCKED provider responses. Proves wiring and safety, not model accuracy."""
import json
import re
import time

import httpx
import pytest
from fastapi.testclient import TestClient
from meta_sam_parser._mask_codec import _encode_raster
from pydantic import SecretStr

from app.catalogue import create_run, get_run, list_products, set_run_status
from app.artefacts import new_run_id, run_dir
from app.db import connect, iso, utcnow
from app.main import create_app
from app.models.base import MetaClient
from app.pipeline.ingest import ingest
from app.pipeline.orchestrate import MAX_ATTEMPTS, can_process, make_starter, run_pipeline, step_states
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
    return settings.model_copy(update={"allow_live_model_calls": True, "model_api_key": SecretStr("k-test"),
                                       "upload_code": SecretStr("market-code")})


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


@pytest.fixture
def queued(conn, ingested):
    """An ingested run claimed for processing, as the upload and process routes leave it."""
    set_run_status(conn, ingested, "processing", None)
    return ingested


def factory(settings, calls, fail=None):
    return lambda conn: MetaClient(settings, conn, httpx.MockTransport(respond(calls, fail)))


@needs_ffmpeg
def test_whole_pipeline_makes_a_draft_and_never_approves(live_settings, conn, queued):
    calls = []
    assert run_pipeline(queued, live_settings, factory(live_settings, calls)) is True
    assert sum(c.endswith("/responses") for c in calls) == 2 and len(calls) == 6  # one request per stage or prompt
    run = get_run(conn, queued)
    assert (run["status"], run["stage"]) == ("draft_ready", "done")
    products = list_products(conn, queued)
    assert len(products) == 1 and products[0]["status"] == "draft"
    assert products[0]["price_pesewas"] == 1200 and products[0]["cutout_path"]
    assert (live_settings.runs_dir / queued / "artefacts" / "tracking.mp4").is_file()
    assert conn.execute("SELECT COUNT(*) FROM model_calls WHERE run_id = ?", (queued,)).fetchone()[0] == 6


@needs_ffmpeg
def test_required_failure_stops_once_with_a_safe_message(live_settings, conn, queued):
    calls = []
    assert run_pipeline(queued, live_settings, factory(live_settings, calls, fail="/asr/transcribe")) is False
    assert len(calls) == 1  # no retry and no later paid stage
    run = get_run(conn, queued)
    assert run["status"] == "failed" and "Billing account issue" in run["error"] and "k-test" not in run["error"]
    assert list_products(conn, queued) == []


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
    r = c.post("/runs", files={"video": ("a.mp4", sample_video.read_bytes(), "video/mp4")},
               data={"access_code": "market-code"})
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
def test_finishing_late_steps_never_undoes_a_publish(live_settings, conn, queued):
    from app.pipeline import orchestrate
    real_run_stage = orchestrate.run_stage

    def publish_during_photos(stage, run, s, c, client, *a, **k):
        real_run_stage(stage, run, s, c, client, *a, **k)
        if stage == "cutouts":
            set_run_status(c, run["id"], "published", "publish")

    orchestrate.run_stage = publish_during_photos
    try:
        assert run_pipeline(queued, live_settings, factory(live_settings, [])) is True
    finally:
        orchestrate.run_stage = real_run_stage
    assert get_run(conn, queued)["status"] == "published"


def test_page_stops_refreshing_once_a_draft_exists(live_settings, conn):
    run = create_run(conn, "real")
    set_run_status(conn, run["id"], "draft_ready", "overlay")
    r = _app(live_settings, []).get(f"/r/{run['review_token']}")
    assert 'http-equiv="refresh"' not in r.text


def test_paid_runs_need_an_access_code(live_settings):
    open_server = live_settings.model_copy(update={"upload_code": None})
    assert can_process(live_settings) and not can_process(open_server)


def _log_call(conn, run_id, stage, cost):
    conn.execute("INSERT INTO model_calls(run_id, stage, model, provider, endpoint, ok, cost_usd, created_at)"
                 " VALUES (?,?,?,?,?,?,?,?)", (run_id, stage, "m", "meta", "/x", 1, cost, iso(utcnow())))
    conn.commit()


def test_daily_budget_stops_new_runs(live_settings, conn):
    run = create_run(conn, "real")
    set_run_status(conn, run["id"], "ingested", "ingest")
    _log_call(conn, None, "other", live_settings.max_daily_usd)
    started = []
    r = _app(live_settings, started).post(f"/r/{run['review_token']}/process")
    assert r.status_code == 429 and "budget" in r.text and started == []


def test_a_failed_video_cannot_be_retried_forever(live_settings, conn):
    run = create_run(conn, "real")
    set_run_status(conn, run["id"], "failed", "identify", "x")
    for _ in range(MAX_ATTEMPTS):
        _log_call(conn, run["id"], "transcribe", 0.01)
    started = []
    r = _app(live_settings, started).post(f"/r/{run['review_token']}/process")
    assert r.status_code == 429 and started == []


def test_restart_releases_a_stuck_run(settings, conn):
    run = create_run(conn, "real")
    set_run_status(conn, run["id"], "processing", "track")
    create_app(settings)
    after = get_run(conn, run["id"])
    assert after["status"] == "failed" and "restarted" in after["error"]


def test_real_background_process_runs_and_reports(live_settings, conn):
    """A real worker process, with paid calls off, so it stops at the first paid stage without any spending."""
    run = create_run(conn, "real")
    set_run_status(conn, run["id"], "processing", None)
    off = live_settings.model_copy(update={"allow_live_model_calls": False})
    make_starter(off)(run["id"])
    deadline = time.monotonic() + 60
    while get_run(conn, run["id"])["status"] == "processing" and time.monotonic() < deadline:
        time.sleep(0.2)
    final = get_run(conn, run["id"])
    assert final["status"] == "failed" and final["stage"] == "transcribe"
    assert conn.execute("SELECT COUNT(*) FROM model_calls").fetchone()[0] == 0


def test_a_run_stopped_while_queued_never_calls_meta(live_settings, conn):
    run = create_run(conn, "real")
    set_run_status(conn, run["id"], "failed", None, "Stopped before it started")
    calls = []
    assert run_pipeline(run["id"], live_settings, factory(live_settings, calls)) is False
    assert calls == [] and get_run(conn, run["id"])["status"] == "failed"


def test_tracking_video_is_promised_while_it_is_made_then_shown(live_settings, conn):
    run = create_run(conn, "real")
    art = run_dir(live_settings.runs_dir, run["id"], create=True) / "artefacts"
    art.mkdir(parents=True, exist_ok=True)
    (art / "tracks.json").write_text(json.dumps({"tracks": []}), encoding="utf-8")
    set_run_status(conn, run["id"], "draft_ready", "overlay")
    c = _app(live_settings, [])
    page = c.get(f"/r/{run['review_token']}").text
    assert 'id="tracking-wait"' in page and "<video" not in page
    (art / "tracking.mp4").write_bytes(b"video")
    set_run_status(conn, run["id"], "draft_ready", "done")
    page = c.get(f"/r/{run['review_token']}").text
    assert "<video" in page and 'id="tracking-wait"' not in page
    assert c.get(f"/r/{run['review_token']}/tracking.mp4").status_code == 200
