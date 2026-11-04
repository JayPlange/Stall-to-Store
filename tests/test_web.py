import io
import json
import re
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi.testclient import TestClient

from app.catalogue import list_products, run_by_token
from app.db import connect
from app.main import create_app
from tests.conftest import needs_ffmpeg


def _token(resp) -> str:
    return re.search(r"/r/([A-Za-z0-9_\-]+)", resp.text).group(1)


def test_health_reports_names_not_values(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["model_api_key_present"] is False and body["live_model_calls_enabled"] is False
    assert r.headers["Referrer-Policy"] == "no-referrer"


def test_upload_rejects_bad_extension(client, settings):
    r = client.post("/runs", files={"video": ("../../evil.exe", b"MZ...", "application/octet-stream")})
    assert r.status_code == 400 and "unsupported file type" in r.text
    assert not any(settings.runs_dir.iterdir())


def test_upload_rejects_non_video(client, settings):
    r = client.post("/runs", files={"video": ("clip.mp4", b"not a video at all", "video/mp4")})
    assert r.status_code == 400 and "not a readable video" in r.text
    assert not any(settings.runs_dir.iterdir())


def test_upload_rejects_empty_and_oversize(client):
    assert client.post("/runs", files={"video": ("a.mp4", b"", "video/mp4")}).status_code == 400
    big = io.BytesIO(b"\0" * (5 * 1024 * 1024 + 1))
    r = client.post("/runs", files={"video": ("a.mp4", big, "video/mp4")})
    assert r.status_code == 400 and "larger than" in r.text


@needs_ffmpeg
def test_upload_rejects_too_short_and_silent(settings, sample_video, silent_video):
    c = TestClient(create_app(settings.model_copy(update={"min_duration_s": 10})))
    r = c.post("/runs", files={"video": ("a.mp4", sample_video.read_bytes(), "video/mp4")})
    assert r.status_code == 400 and "shorter" in r.text
    c = TestClient(create_app(settings))
    r = c.post("/runs", files={"video": ("a.mp4", silent_video.read_bytes(), "video/mp4")})
    assert r.status_code == 400 and "no audio" in r.text


@needs_ffmpeg
def test_valid_upload_ingests(client, settings, sample_video):
    r = client.post("/runs", files={"video": ("My Stall (1).MP4", sample_video.read_bytes(), "video/mp4")})
    assert r.status_code == 200 and "not run for this video" in r.text
    conn = connect(settings.db_path)
    run = run_by_token(conn, _token(r)) if "/r/" in r.text else None
    run = run or dict(conn.execute("SELECT * FROM runs").fetchone())
    assert run["status"] == "ingested" and run["source"] == "real"
    rdir = settings.runs_dir / run["id"]
    assert (rdir / "input" / "source.mp4").is_file() and (rdir / "audio.wav").is_file()
    frames = json.loads((rdir / "frames.json").read_text())
    assert frames["fps"] == 2.0 and len(frames["frames"]) in (12, 13)
    meta = json.loads((rdir / "metadata.json").read_text())
    assert meta["has_audio"] and 5.5 < meta["duration_s"] < 6.5
    # SAM receives normalised.mp4; its frame count must equal ours so frame N means the same image.
    from app.pipeline.ingest import probe
    norm = probe(rdir / "normalised.mp4")
    assert not norm.has_audio and (meta["frame_width"], meta["frame_height"]) == (320, 240)
    import subprocess
    count = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                            "stream=nb_read_frames", "-of", "csv=p=0", str(rdir / "normalised.mp4")],
                           capture_output=True, text=True).stdout.strip()
    assert int(count) == len(frames["frames"])
    import wave
    with wave.open(str(rdir / "audio.wav")) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (24000, 1, 2)


def test_fixture_route_disabled_by_default(client):
    assert client.post("/dev/fixture").status_code == 404


@pytest.fixture
def fixture_client(settings):
    return TestClient(create_app(settings.model_copy(update={"dev_fixtures": True})))


def test_fixture_review_publish_order_flow(fixture_client, settings):
    c = fixture_client
    r = c.post("/dev/fixture")
    assert r.status_code == 200 and "DEVELOPMENT FIXTURE" in r.text
    token = _token(r)
    conn = connect(settings.db_path)
    run = run_by_token(conn, token)
    products = list_products(conn, run["id"])
    by_track = {p["track_id"]: p for p in products}
    assert by_track["t1"]["match_status"] == "auto" and by_track["t5"]["match_status"] == "needs_seller"

    # Publishing with nothing approved fails.
    r = c.post(f"/r/{token}/publish", data={"seller_name": "Ama", "seller_whatsapp": "+233240000000", "slug": "ama"})
    assert r.status_code == 400 and "approve at least one" in r.text

    # HTMX approve returns a partial card.
    milk = by_track["t1"]["id"]
    r = c.post(f"/r/{token}/products/{milk}/status", data={"status": "approved"}, headers={"HX-Request": "true"})
    assert r.status_code == 200 and f'id="product-{milk}"' in r.text and "<html" not in r.text

    # Approving an unpriced product fails with a message, not a crash.
    lotion = by_track["t5"]["id"]
    r = c.post(f"/r/{token}/products/{lotion}/status", data={"status": "approved"}, headers={"HX-Request": "true"})
    assert "needs a price" in r.text
    # Seller picks a suggested price, then approves.
    mention = conn.execute("SELECT mention_id FROM candidates WHERE product_id = ? ORDER BY rank LIMIT 1",
                           (lotion,)).fetchone()[0]
    c.post(f"/r/{token}/products/{lotion}/suggestion", data={"mention_id": mention})
    c.post(f"/r/{token}/products/{lotion}/status", data={"status": "approved"})

    r = c.post(f"/r/{token}/publish", data={"seller_name": "Ama", "seller_whatsapp": "0240000000", "slug": "ama-shop"})
    assert r.status_code == 200 and "/s/ama-shop" in r.text

    shop = c.get("/s/ama-shop")
    assert shop.status_code == 200 and "DEVELOPMENT FIXTURE" in shop.text
    assert "Evaporated Milk" in shop.text and "Body Lotion" in shop.text and "Sardines" not in shop.text
    assert c.get(f"/s/ama-shop/img/{milk}").status_code == 200
    assert c.get(f"/s/ama-shop/img/{by_track['t7']['id']}").status_code == 404  # unapproved image not public

    r = c.post("/s/ama-shop/order", data={f"qty_{milk}": "2", f"qty_{lotion}": "1"})
    assert r.status_code == 200 and "has not been sent yet" in r.text
    link = re.search(r'href="(https://wa\.me/[^"]+)"', r.text).group(1).replace("&amp;", "&")
    text = parse_qs(urlparse(link).query)["text"][0]
    assert "Total: GH₵59.00" in text  # 2 x 12.00 + 1 x 35.00
    ref = re.search(r"S2S-[A-Z0-9]{6}", text).group(0)

    # Unapproved product and forged price fields are rejected / ignored.
    r = c.post("/s/ama-shop/order", data={f"qty_{by_track['t7']['id']}": "1"})
    assert r.status_code == 400
    r = c.post("/s/ama-shop/order", data={f"qty_{milk}": "1", "price": "1", "total": "1"})
    assert "GH₵12.00" in r.text

    # Seller confirms by reference, twice (idempotent).
    r = c.post(f"/r/{token}/orders/{ref}/confirm")
    assert r.status_code == 200 and f"{ref} confirmed" in r.text
    r = c.post(f"/r/{token}/orders/{ref}/confirm")
    assert "already confirmed" in r.text
    assert c.post(f"/r/{token}/orders/S2S-ZZZZZZ/confirm").status_code == 400


def test_save_and_approve_in_one_step(fixture_client, settings):
    c = fixture_client
    token = _token(c.post("/dev/fixture"))
    conn = connect(settings.db_path)
    p = next(p for p in list_products(conn, run_by_token(conn, token)["id"]) if p["track_id"] == "t6")
    form = {"brand": "FIXTURE", "name": "Toothpaste", "size": "120 g", "price": "14.5", "approve": "1"}
    r = c.post(f"/r/{token}/products/{p['id']}", data=form, headers={"HX-Request": "true"})
    assert "approved" in r.text and "price set by you" in r.text
    row = conn.execute("SELECT status, price_pesewas, price_source FROM products WHERE id = ?", (p["id"],)).fetchone()
    assert tuple(row) == ("approved", 1450, "seller")
    # Bad price: nothing approved, error shown in the card.
    q = next(x for x in list_products(conn, run_by_token(conn, token)["id"]) if x["track_id"] == "t7")
    r = c.post(f"/r/{token}/products/{q['id']}", data={**form, "price": "abc"}, headers={"HX-Request": "true"})
    assert "at most 2 decimals" in r.text
    assert conn.execute("SELECT status FROM products WHERE id = ?", (q["id"],)).fetchone()[0] == "draft"


def test_wrong_token_and_unknown_shop_404(client):
    assert client.get("/r/not-a-real-token").status_code == 404
    assert client.get("/s/nope").status_code == 404
    assert client.post("/r/x/products/1/status", data={"status": "approved"}).status_code == 404


def test_cannot_touch_other_runs_product(fixture_client, settings):
    c = fixture_client
    t1 = _token(c.post("/dev/fixture"))
    t2 = _token(c.post("/dev/fixture"))
    conn = connect(settings.db_path)
    other = list_products(conn, run_by_token(conn, t2)["id"])[0]["id"]
    assert c.post(f"/r/{t1}/products/{other}/status", data={"status": "approved"}).status_code == 404
    assert c.get(f"/r/{t1}/img/{other}").status_code == 404


def test_upload_code_blocks_strangers(settings, tmp_path):
    from pydantic import SecretStr
    c = TestClient(create_app(settings.model_copy(update={"upload_code": SecretStr("market-2026")})))
    assert "Access code" in c.get("/").text
    r = c.post("/runs", files={"video": ("a.mp4", b"x", "video/mp4")}, data={"access_code": "wrong"})
    assert r.status_code == 403 and not list(settings.runs_dir.iterdir())
    r = c.post("/runs", files={"video": ("a.mp4", b"x", "video/mp4")}, data={"access_code": "market-2026"})
    assert r.status_code == 400  # passes the code, then fails normal validation


def test_oversized_upload_is_refused_before_the_body_is_read(client):
    r = client.post("/runs", content=b"x" * 10, headers={"Content-Length": str(50 * 1024 * 1024),
                                                         "Content-Type": "multipart/form-data; boundary=X"})
    assert r.status_code == 413


def test_upload_works_from_a_phone_gallery(client):
    page = client.get("/").text
    assert "capture=" not in page and 'id="upload-form"' in page


def test_review_page_offers_a_way_to_save_the_link(fixture_client):
    r = fixture_client.post("/dev/fixture")
    assert "Send it to my WhatsApp" in r.text and "Copy link" in r.text


def test_home_is_the_seller_area_until_a_shop_is_published(client):
    assert 'id="upload-form"' in client.get("/").text
    assert 'id="upload-form"' in client.get("/seller").text


def test_home_shows_the_published_real_shop_with_a_seller_button(settings, conn, published):
    c = TestClient(create_app(settings))
    page = c.get("/").text
    assert "Auntie Ama Provisions" in page and 'href="/seller"' in page and 'id="upload-form"' not in page
    assert "Save &amp; approve" not in page  # no seller controls on the customer front


def test_fixture_shops_never_become_the_home_page(fixture_client):
    r = fixture_client.post("/dev/fixture")
    token = _token(r)
    for pid in re.findall(r'id="product-(\d+)"', r.text)[:2]:
        fixture_client.post(f"/r/{token}/products/{pid}/status", data={"status": "approved"})
    fixture_client.post(f"/r/{token}/publish", data={"seller_name": "Fixture", "seller_whatsapp": "+233200000001", "slug": "fixture-shop"})
    assert 'id="upload-form"' in fixture_client.get("/").text


def test_no_third_party_scripts(client):
    page = client.get("/seller").text
    assert "unpkg.com" not in page and "/static/vendor/htmx.min.js" in page
    assert client.get("/static/vendor/htmx.min.js").status_code == 200


def test_review_page_always_shares_its_own_address(settings, conn, published):
    c = TestClient(create_app(settings))
    token = published["run"]["review_token"]
    r = c.post(f"/r/{token}/publish", data={"seller_name": "Auntie Ama Provisions",
                                             "seller_whatsapp": "+233 24 000 0000", "slug": "auntie-ama"})
    assert r.status_code == 200 and "/publish" in str(r.url)
    shared = re.search(r'href="https://wa\.me/[^"]*text=([^"]+)"', r.text).group(1)
    assert shared.endswith(f"/r/{token}") and "publish" not in shared
    assert f'var home = "/r/{token}"' in r.text and "history.replaceState" in r.text


def test_live_shop_review_page_says_live_and_links_to_the_shop(settings, conn, published):
    page = TestClient(create_app(settings)).get(f"/r/{published['run']['review_token']}").text
    assert "Your shop is open." in page and "Check it. Then go live." not in page
    assert page.count('href="/s/auntie-ama"') >= 2


def test_seller_area_links_back_to_the_shop_once_one_is_live(settings, conn, published):
    assert "Back to the shop" in TestClient(create_app(settings)).get("/seller").text


def test_seller_area_has_no_shop_link_before_any_shop(client):
    assert "Back to the shop" not in client.get("/seller").text


def test_wrong_links_get_a_page_with_a_way_back(client):
    for r in (client.get("/s/nope"), client.get("/r/not-a-real-token/publish")):
        assert r.status_code in (404, 405) and "text/html" in r.headers["content-type"]
        assert 'href="/"' in r.text and "detail" not in r.text
    assert client.get("/health").json()  # JSON endpoints are unchanged


def test_publish_button_count_follows_approvals_without_a_reload(settings, conn, published):
    page = TestClient(create_app(settings)).get(f"/r/{published['run']['review_token']}").text
    assert 'id="publish-button">Publish 2 approved products' in page
    assert 'getElementById("publish-button")' in page  # recounted after each htmx save
