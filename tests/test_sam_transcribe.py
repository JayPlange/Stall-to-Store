"""SAM and transcription adapters against DOCUMENTED shapes, using synthetic payloads.

SAM lines are built with meta-sam-parser's own encoder so masks are valid. None of this is evidence
that the live API behaves this way; the smoke script is what confirms that.
"""
import json

import httpx
import pytest
from meta_sam_parser._mask_codec import _encode_raster
from PIL import Image
from pydantic import SecretStr

from app.artefacts import new_run_id
from app.models import segment, transcribe
from app.models.base import AdapterNotVerified, CallContext, MetaClient
from app.models.pricing import sam_video_cost, spark_cost, transcribe_cost
from app.pipeline.keyframes import make_cutout

W, H = 4, 3
RASTER = bytes([0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0])
PAYLOAD = _encode_raster(RASTER, W, H)


def rec(obj, x1, y1, x2, y2, c=None):
    conf = f";c={c}" if c is not None else ""
    return (f"{obj}<|box;x1={x1};y1={y1};x2={x2};y2={y2};w=320;h=240{conf}|>"
            f"<|mask;x=0;y=0;data={H},{W},{PAYLOAD}|>")


SAM_TEXT = "\n".join([
    f"<0f>{rec(3, 10, 20, 50, 60, 0.9)},{rec(7, 100, 20, 150, 60)}",
    f"<1f>{rec(3, 12, 22, 52, 62)}",
    f"<4f>{rec(3, 14, 24, 54, 64)}",  # frames may skip
    "<5f>garbage that is not a record",
]) + "\n"


def live(settings):
    return settings.model_copy(update={"allow_live_model_calls": True, "model_api_key": SecretStr("k-test-123")})


def test_parse_tracks_groups_by_stable_object_id(tmp_path):
    tracks, diags = segment.parse_tracks(SAM_TEXT, 2.0, "bottle", 0, tmp_path / "masks")
    by_id = {t.track_id: t for t in tracks}
    assert set(by_id) == {"p0o3", "p0o7"}
    hits = by_id["p0o3"].frames
    assert [h.frame_idx for h in hits] == [0, 1, 4] and [h.t for h in hits] == [0.0, 0.5, 2.0]
    # meta-sam-parser reports half-open pixel bounds: wire x2=50 -> right=51.
    assert hits[0].box_xyxy == (10, 20, 51, 61) and hits[0].score == 0.9 and hits[1].score is None
    with Image.open(hits[0].mask_path) as m:
        assert m.size == (W, H) and m.convert("L").getpixel((1, 0)) == 255
    assert diags  # the garbage line is reported, not silently dropped


def test_empty_output_is_zero_tracks():
    assert segment.parse_tracks("", 2.0, "x", 0, None) == ([], [])


def test_extract_output_text_shapes():
    assert segment.extract_output_text({"output_text": "<0f>"}) == "<0f>"
    body = {"status": "completed", "output": [{"type": "message", "content": [{"type": "output_text", "text": "a"},
                                                                            {"type": "output_text", "text": "b"}]}]}
    assert segment.extract_output_text(body) == "ab"
    assert segment.extract_output_text({"status": "completed", "output": []}) == ""
    with pytest.raises(AdapterNotVerified):
        segment.extract_output_text({"result": "?"})


def test_build_request_one_concept_only():
    body = segment.build_request("sam-3.1", "bottle", "data:video/mp4;base64,AA")
    content = body["input"][0]["content"]
    assert content[0] == {"type": "input_text", "text": "bottle"} and content[1]["type"] == "input_video"
    with pytest.raises(ValueError):
        segment.build_request("sam-3.1", "bottle, tin", "data:")


def test_track_one_request_per_prompt_with_cost(settings, conn, tmp_path):
    video = tmp_path / "n.mp4"
    video.write_bytes(b"\x00\x00\x00\x18ftypmp42")
    sent = []

    def handler(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"status": "completed", "output": [
            {"type": "message", "content": [{"type": "output_text", "text": SAM_TEXT}]}]})

    run_id = new_run_id()
    client = MetaClient(live(settings), conn, httpx.MockTransport(handler))
    tracks, _ = segment.track(client, CallContext(run_id, "track"), video, ["bottle", "tin"], "sam-3.1",
                              2.0, 30, tmp_path / "masks")
    assert len(sent) == 2 and [b["input"][0]["content"][0]["text"] for b in sent] == ["bottle", "tin"]
    assert {t.track_id for t in tracks} == {"p0o3", "p0o7", "p1o3", "p1o7"}
    rows = conn.execute("SELECT stage, cost_usd FROM model_calls ORDER BY id").fetchall()
    assert [r["stage"] for r in rows] == ["track-p0", "track-p1"]
    assert all(r["cost_usd"] == sam_video_cost(50) for r in rows)  # 30 frames billed at the 50 minimum
    raw = (settings.runs_dir / run_id / "raw" / "track-p0.json").read_text()
    assert "base64," not in raw and "k-test-123" not in raw


def test_cutout_uses_box_local_mask(tmp_path):
    frame = tmp_path / "f.jpg"
    Image.new("RGB", (40, 30), (200, 0, 0)).save(frame)
    mask = tmp_path / "m.png"
    m = Image.new("L", (4, 3), 0)
    m.putpixel((1, 1), 255)
    m.save(mask)
    out = make_cutout(frame, mask, (10, 10, 30, 25), tmp_path / "c.jpg")
    with Image.open(out) as img:
        assert img.size == (20, 15)
        assert img.getpixel((0, 0))[1] > 200          # outside mask: white
        r, g, b = img.getpixel((7, 7))                # inside mask: real (red) pixels
        assert r > 150 and g < 80


def test_transcribe_normalise_documented_shape():
    body = {"sessionId": "s", "transcript": "milk twelve cedis. soap eight", "audioDurationMs": 6400,
            "turns": [{"turnId": 0, "startMs": 300, "endMs": 2100, "transcript": "milk twelve cedis.", "speaker": None},
                      {"turnId": 1, "startMs": 2500, "endMs": 4000, "transcript": "soap eight", "speaker": None}]}
    t = transcribe.normalise(body)
    assert t.granularity == "segment" and [(u.t_start, u.t_end) for u in t.units] == [(0.3, 2.1), (2.5, 4.0)]
    t = transcribe.normalise({"transcript": "milk twelve", "turns": []})
    assert t.granularity == "none" and t.units[0].text == "milk twelve"
    assert transcribe.normalise({"transcript": "", "turns": []}).units == []
    for bad in ({"text": "x"}, {"transcript": "x", "turns": [{"start": 1}]}, "plain text"):
        with pytest.raises(AdapterNotVerified):
            transcribe.normalise(bad)


def test_transcribe_requests_turn_timings():
    req = transcribe.build_request("muse-voice-transcribe-1.0")
    assert req["mode"] == "ENDPOINTING" and req["audioEncoding"] == "WAV"


def test_pricing_estimates():
    assert sam_video_cost(120) == 0.024 and sam_video_cost(10) == 0.01
    assert transcribe_cost(60_999) == round(60 * 0.18 / 3600, 6)
    assert spark_cost("muse-spark-1.3", {"prompt_tokens": 1_000_000, "completion_tokens": 0}) == 1.25
    assert spark_cost("unknown", {"prompt_tokens": 1}) is None and transcribe_cost(None) is None
