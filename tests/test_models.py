"""Adapter plumbing with a mocked transport. No network; these are NOT evidence of real API behaviour."""
import json

import httpx
import pytest
from pydantic import SecretStr

from app.artefacts import new_run_id
from app.models import transcribe, vision
from app.models.pricing import spark_cost
from app.models.base import AdapterNotVerified, CallContext, LiveCallsDisabled, MetaClient, ProviderError, sanitise
from app.models.types import Crop, Transcript, Unit

FAKE_KEY = "test-key-not-real-0123456789"


class Recorder:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.responses.pop(0)


def live(settings):
    return settings.model_copy(update={"allow_live_model_calls": True, "model_api_key": SecretStr(FAKE_KEY)})


def test_calls_disabled_by_default(settings, conn):
    rec = Recorder([])
    client = MetaClient(settings, conn, httpx.MockTransport(rec))
    with pytest.raises(LiveCallsDisabled):
        client.request(CallContext(None, "smoke"), "m", "GET", "/models")
    assert rec.requests == []


def test_calls_disabled_without_key(settings, conn):
    s = settings.model_copy(update={"allow_live_model_calls": True})
    with pytest.raises(LiveCallsDisabled, match="MODEL_API_KEY"):
        MetaClient(s, conn).request(CallContext(None, "smoke"), "m", "GET", "/models")


def _chat_body(content: dict) -> dict:
    return {"id": "x", "object": "chat.completion", "created": 0, "model": "muse-spark-1.3",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": json.dumps(content)}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}


def test_success_saves_sanitised_raw_and_logs_call(settings, conn, tmp_path):
    s = live(settings)
    rec = Recorder([httpx.Response(200, json=_chat_body({"products": [
        {"index": 0, "brand": "Acme", "name": "Milk", "size": "160 g", "confidence": 0.8}]}))])
    img = tmp_path / "crop.jpg"
    img.write_bytes(b"\xff\xd8fakejpeg")
    run_id = new_run_id()
    guesses = vision.identify(MetaClient(s, conn, httpx.MockTransport(rec)), CallContext(run_id, "identify"),
                              [Crop("t1", str(img))], "muse-spark-1.3")
    assert guesses[0].name == "Milk" and guesses[0].track_id == "t1"
    assert rec.requests[0].headers["Authorization"] == f"Bearer {FAKE_KEY}"
    raw_file = s.runs_dir / run_id / "raw" / "identify.json"
    raw = raw_file.read_text(encoding="utf-8")
    assert FAKE_KEY not in raw and "base64," not in raw and "data-url redacted" in raw
    row = conn.execute("SELECT * FROM model_calls").fetchone()
    assert row["ok"] == 1 and row["stage"] == "identify" and json.loads(row["units_json"])["total_tokens"] == 15
    # Estimate from Meta's published price (10 prompt + 5 completion tokens), never a model's claim.
    assert row["cost_usd"] == spark_cost("muse-spark-1.3", {"prompt_tokens": 10, "completion_tokens": 5})


def test_billing_error_is_recorded_and_not_retried(settings, conn):
    s = live(settings)
    rec = Recorder([httpx.Response(402, json={"error": {"message": "Billing account issue.", "type": "billing_error",
                                                        "param": None, "code": None}})])
    with pytest.raises(ProviderError) as e:
        MetaClient(s, conn, httpx.MockTransport(rec)).request(CallContext(None, "smoke"), "m", "GET", "/models")
    assert e.value.status == 402 and e.value.error_type == "billing_error"
    assert len(rec.requests) == 1
    row = conn.execute("SELECT ok, http_status, error_type FROM model_calls").fetchone()
    assert tuple(row) == (0, 402, "billing_error")


def test_raw_files_do_not_overwrite(settings, conn):
    s = live(settings)
    rec = Recorder([httpx.Response(200, json={"data": []}), httpx.Response(200, json={"data": []})])
    c = MetaClient(s, conn, httpx.MockTransport(rec))
    a = c.request(CallContext(None, "models"), "-", "GET", "/models")
    b = c.request(CallContext(None, "models"), "-", "GET", "/models")
    assert a.raw_path != b.raw_path


def test_unexpected_chat_shape_is_not_guessed(settings, conn):
    s = live(settings)
    rec = Recorder([httpx.Response(200, json={"output": "something else"})])
    t = Transcript("milk 12 cedis", "word", [Unit("milk", 0.0, 0.5), Unit("12 cedis", 0.5, 1.0)])
    with pytest.raises(AdapterNotVerified):
        vision.parse_prices(MetaClient(s, conn, httpx.MockTransport(rec)), CallContext(None, "prices"), t, "m")


def test_parse_prices_takes_timings_from_transcript(settings, conn):
    s = live(settings)
    body = _chat_body({"mentions": [
        {"unit_start": 1, "unit_end": 2, "quote": "milk 12 cedis", "amount_cedis": "12", "referent": "milk"},
        {"unit_start": 5, "unit_end": 9, "quote": "out of range", "amount_cedis": "3", "referent": None},
        {"unit_start": 0, "unit_end": 0, "quote": "bad", "amount_cedis": "twelve", "referent": None}]})
    rec = Recorder([httpx.Response(200, json=body)])
    t = Transcript("x", "word", [Unit("now", 0.0, 0.4), Unit("milk", 1.0, 1.4), Unit("12 cedis", 1.5, 2.0)])
    ms = vision.parse_prices(MetaClient(s, conn, httpx.MockTransport(rec)), CallContext(None, "prices"), t, "m")
    assert len(ms) == 1 and ms[0].amount_pesewas == 1200 and (ms[0].t_start, ms[0].t_end) == (1.0, 2.0)


def test_transcribe_normalisation_refuses_until_verified(settings, conn, tmp_path):
    s = live(settings)
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF0000WAVE")
    rec = Recorder([httpx.Response(200, json={"some": "shape"})])
    with pytest.raises(AdapterNotVerified, match="some"):
        transcribe.transcribe(MetaClient(s, conn, httpx.MockTransport(rec)), CallContext(None, "transcribe"),
                              wav, "muse-voice-transcribe-1.0")
    assert b"multipart/form-data" in rec.requests[0].headers["content-type"].encode()
    with pytest.raises(ValueError):
        transcribe.transcribe(MetaClient(s, conn), CallContext(None, "t"), tmp_path / "a.mp3", "m")


def test_sanitise():
    out = sanitise({"Authorization": "Bearer x", "nested": [{"url": "data:image/png;base64,AAAA"}],
                    "msg": f"key {FAKE_KEY}"}, FAKE_KEY)
    assert out["Authorization"] == "<redacted>" and "redacted" in out["nested"][0]["url"]
    assert FAKE_KEY not in out["msg"]


def test_sanitise_keeps_usage_but_redacts_credentials():
    usage = {"prompt_tokens": 15, "completion_tokens": 154, "total_tokens": 169,
             "completion_tokens_details": {"reasoning_tokens": 144}, "prompt_tokens_details": {"cached_tokens": 0}}
    assert sanitise({"usage": usage})["usage"] == usage
    creds = {"Authorization": "x", "api_key": "x", "x-api-key": "x", "access_token": "x", "refresh_token": "x",
             "client_secret": "x", "password": "x", "token": "x", "session_token": "x"}
    assert all(v == "<redacted>" for v in sanitise(creds).values())
