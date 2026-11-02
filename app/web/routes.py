"""HTTP routes: upload, seller review (/r/{token}), shop page (/s/{slug}), health.

The review token in the URL is the seller's only credential for their run; it is never logged.
HTMX requests get partials; plain form posts get a full page, so everything works without JS.
"""
import json
import logging
import sqlite3
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from app import catalogue
from app.artefacts import new_run_id, relative_to_runs, resolve_artefact, run_dir
from app.catalogue import CatalogueError
from app.config import Settings
from app.fixtures import load_fixture
from app.money import cedis_input_value, format_cedis
from app.orders import service as orders
from app.orders.links import order_message, wa_link
from app.pipeline.ingest import IngestError, ingest, tools_available
from app.pipeline.orchestrate import ESTIMATED_COST_USD, can_process, step_states
from app.pipeline.uploads import UploadError, accept_upload

log = logging.getLogger("sts.web")
WEB_DIR = Path(__file__).parent
STATIC_DIR = WEB_DIR / "static"
templates = Jinja2Templates(directory=WEB_DIR / "templates")
templates.env.filters["cedis"] = format_cedis
templates.env.filters["cedis_input"] = cedis_input_value

router = APIRouter()


def conn_dep() -> sqlite3.Connection:  # replaced in create_app
    raise RuntimeError("database dependency not configured")


def settings_of(request: Request) -> Settings:
    return request.app.state.settings


def is_htmx(request: Request) -> bool:
    return request.headers.get("HX-Request") == "true"


def _run_or_404(conn, token: str) -> dict:
    run = catalogue.run_by_token(conn, token)
    if not run:
        raise HTTPException(404, "Not found")
    return run


# -- health ---------------------------------------------------------------
@router.get("/health")
def health(request: Request, conn: sqlite3.Connection = Depends(conn_dep)):
    s = settings_of(request)
    conn.execute("SELECT 1").fetchone()
    return {"status": "ok", "db": "ok", "ffmpeg": tools_available(s.ffmpeg_bin, s.ffprobe_bin),
            "live_model_calls_enabled": s.allow_live_model_calls, "model_api_key_present": s.api_key_present,
            "dev_fixtures": s.dev_fixtures}


# -- upload ---------------------------------------------------------------
@router.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {"settings": settings_of(request), "error": None})


@router.post("/runs")
def upload(request: Request, video: UploadFile = File(...), conn: sqlite3.Connection = Depends(conn_dep)):
    s = settings_of(request)
    run_id = new_run_id()
    rdir = run_dir(s.runs_dir, run_id, create=True)
    try:
        src, meta = accept_upload(video.file, video.filename, rdir, s)
    except UploadError as exc:
        return templates.TemplateResponse(request, "index.html", {"settings": s, "error": str(exc)},
                                          status_code=400)
    run = catalogue.create_run(conn, "real", run_id, relative_to_runs(s.runs_dir, src))
    try:
        ingest(src, rdir, s.frame_fps, s.ffmpeg_bin, s.ffprobe_bin)
        catalogue.set_run_status(conn, run_id, "ingested", "ingest")
    except IngestError as exc:
        catalogue.set_run_status(conn, run_id, "failed", "ingest", str(exc))
    else:
        if can_process(s):
            _begin_processing(request, conn, run_id)
    return RedirectResponse(f"/r/{run['review_token']}", status_code=303)


def _begin_processing(request: Request, conn, run_id: str) -> bool:
    """Claim the run atomically, so a double click can never start (and bill) it twice."""
    claimed = conn.execute("UPDATE runs SET status = 'processing', stage = NULL, error = NULL"
                           " WHERE id = ? AND status IN ('ingested', 'failed') AND source = 'real'"
                           " AND NOT EXISTS (SELECT 1 FROM products WHERE run_id = ?)", (run_id, run_id)).rowcount
    conn.commit()
    if claimed:
        request.app.state.start_pipeline(run_id)
    return bool(claimed)


@router.post("/dev/fixture")
def dev_fixture(request: Request, conn: sqlite3.Connection = Depends(conn_dep)):
    s = settings_of(request)
    if not s.dev_fixtures:
        raise HTTPException(404, "Not found")
    run = load_fixture(conn, s.runs_dir)
    return RedirectResponse(f"/r/{run['review_token']}", status_code=303)


# -- review ---------------------------------------------------------------
def _card_ctx(conn, run: dict, product: dict, error: str | None = None) -> dict:
    return {"run": run, "p": product, "suggestions": catalogue.suggestions(conn, product["id"]), "error": error}


def _run_story(s: Settings, run: dict, metadata: dict | None) -> list[dict]:
    """What each Meta model really produced for this run, read from its saved artefacts. Never fixture data."""
    if run["source"] != "real":
        return []
    art = run_dir(s.runs_dir, run["id"]) / "artefacts"
    def read(name):
        path = art / name
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    story = []
    if (tracks := read("tracks.json")) is not None:
        frames = metadata["frame_count"] if metadata else 0
        story.append({"model": "SAM 3.1", "job": "Found and tracked products in the video",
                      "stat": f"{len(tracks['tracks'])} objects across {frames} frames"})
    if (transcript := read("transcript.json")) is not None:
        words = len(transcript["text"].split())
        story.append({"model": "Muse Voice Transcribe", "job": "Listened to the seller's narration",
                      "stat": f"{words} words transcribed", "quote": transcript["text"]})
    prices, guesses = read("price_mentions.json"), read("guesses.json")
    if prices is not None or guesses is not None:
        named = sum(1 for g in guesses or [] if g.get("brand") or g.get("name"))
        story.append({"model": "Muse Spark", "job": "Read prices from speech and names from product photos",
                      "stat": f"{len((prices or {}).get('mentions', []))} prices, {named} products named"})
    return story


def _review_ctx(conn, request: Request, run: dict, error: str | None = None, notice: str | None = None) -> dict:
    s = settings_of(request)
    products = catalogue.list_products(conn, run["id"])
    shop = catalogue.get_shop_for_run(conn, run["id"])
    metadata = None
    meta_path = run_dir(s.runs_dir, run["id"]) / "metadata.json"
    if meta_path.is_file():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    counts = {k: sum(1 for p in products if p["status"] == k) for k in ("draft", "approved", "hidden")}
    return {"run": run, "products": products, "shop": shop, "metadata": metadata, "counts": counts,
            "story": _run_story(s, run, metadata), "steps": step_states(run), "can_process": can_process(s),
            "estimated_cost": ESTIMATED_COST_USD,
            "has_tracking_video": (run_dir(s.runs_dir, run["id"]) / "artefacts" / "tracking.mp4").is_file(),
            "cards": [_card_ctx(conn, run, p) for p in products], "error": error, "notice": notice,
            "orders": orders.list_orders(conn, shop["id"]) if shop else []}


def _review_response(request, conn, run, error=None, notice=None, status_code=200):
    return templates.TemplateResponse(request, "review.html", _review_ctx(conn, request, run, error, notice),
                                      status_code=status_code)


@router.get("/r/{token}", response_class=HTMLResponse)
def review(request: Request, token: str, conn: sqlite3.Connection = Depends(conn_dep)):
    return _review_response(request, conn, _run_or_404(conn, token))


def _product_action(request, conn, token, product_id, action):
    run = _run_or_404(conn, token)
    error = None
    try:
        product = action(run)
    except CatalogueError as exc:
        error = str(exc)
        try:
            product = catalogue.get_product(conn, run["id"], product_id)
        except CatalogueError:
            raise HTTPException(404, "Not found")
    if is_htmx(request):
        return templates.TemplateResponse(request, "_product_card.html", _card_ctx(conn, run, product, error))
    return _review_response(request, conn, run, error=error, status_code=400 if error else 200)


@router.post("/r/{token}/products/{product_id}")
def edit_product(request: Request, token: str, product_id: int, brand: str = Form(""), name: str = Form(""),
                 size: str = Form(""), price: str = Form(""), approve: str = Form(""),
                 conn: sqlite3.Connection = Depends(conn_dep)):
    def action(run):
        product = catalogue.update_product(conn, run["id"], product_id, brand=brand, name=name, size=size, price=price)
        return catalogue.set_status(conn, run["id"], product_id, "approved") if approve else product
    return _product_action(request, conn, token, product_id, action)


@router.post("/r/{token}/products/{product_id}/status")
def product_status(request: Request, token: str, product_id: int, status: str = Form(...),
                   conn: sqlite3.Connection = Depends(conn_dep)):
    return _product_action(request, conn, token, product_id,
                           lambda run: catalogue.set_status(conn, run["id"], product_id, status))


@router.post("/r/{token}/products/{product_id}/suggestion")
def product_suggestion(request: Request, token: str, product_id: int, mention_id: int = Form(...),
                       conn: sqlite3.Connection = Depends(conn_dep)):
    return _product_action(request, conn, token, product_id,
                           lambda run: catalogue.use_suggestion(conn, run["id"], product_id, mention_id))


@router.post("/r/{token}/process")
def process_video(request: Request, token: str, conn: sqlite3.Connection = Depends(conn_dep)):
    run = _run_or_404(conn, token)
    if not can_process(settings_of(request)):
        return _review_response(request, conn, run, error="AI processing is switched off on this server.", status_code=400)
    _begin_processing(request, conn, run["id"])
    return RedirectResponse(f"/r/{token}", status_code=303)


@router.post("/r/{token}/publish")
def publish(request: Request, token: str, seller_name: str = Form(""), seller_whatsapp: str = Form(""),
            slug: str = Form(""), conn: sqlite3.Connection = Depends(conn_dep)):
    run = _run_or_404(conn, token)
    try:
        shop = catalogue.publish(conn, run["id"], seller_name, seller_whatsapp, slug)
    except CatalogueError as exc:
        return _review_response(request, conn, run, error=str(exc), status_code=400)
    run = catalogue.get_run(conn, run["id"])
    return _review_response(request, conn, run, notice=f"Published at /s/{shop['slug']}")


@router.post("/r/{token}/orders/{reference}/{action}")
def order_action(request: Request, token: str, reference: str, action: str,
                 conn: sqlite3.Connection = Depends(conn_dep)):
    run = _run_or_404(conn, token)
    shop = catalogue.get_shop_for_run(conn, run["id"])
    if not shop or action not in ("confirm", "cancel"):
        raise HTTPException(404, "Not found")
    try:
        if action == "confirm":
            order = orders.seller_confirm(conn, shop["id"], reference)
            notice = f"{order['reference']} confirmed" + (" (already confirmed)" if order["already_confirmed"] else "")
        else:
            order = orders.seller_cancel(conn, shop["id"], reference)
            notice = f"{order['reference']} cancelled"
    except orders.OrderError as exc:
        return _review_response(request, conn, run, error=str(exc), status_code=400)
    return _review_response(request, conn, run, notice=notice)


@router.get("/r/{token}/tracking.mp4")
def tracking_video(request: Request, token: str, conn: sqlite3.Connection = Depends(conn_dep)):
    run = _run_or_404(conn, token)
    path = run_dir(settings_of(request).runs_dir, run["id"]) / "artefacts" / "tracking.mp4"
    if not path.is_file():
        raise HTTPException(404, "Not found")
    return FileResponse(path, media_type="video/mp4")


@router.get("/r/{token}/img/{product_id}")
def review_image(request: Request, token: str, product_id: int, conn: sqlite3.Connection = Depends(conn_dep)):
    run = _run_or_404(conn, token)
    try:
        product = catalogue.get_product(conn, run["id"], product_id)
    except CatalogueError:
        raise HTTPException(404, "Not found")
    return _image(settings_of(request), product)


def _image(s: Settings, product: dict):
    for stored in (product["image_path"], product["cutout_path"], product["keyframe_path"]):
        path = resolve_artefact(s.runs_dir, stored)
        if path:
            return FileResponse(path)
    raise HTTPException(404, "Not found")


# -- shop -----------------------------------------------------------------
def _shop_or_404(conn, slug: str) -> dict:
    shop = catalogue.get_shop_by_slug(conn, slug)
    if not shop:
        raise HTTPException(404, "Not found")
    return shop


@router.get("/s/{slug}", response_class=HTMLResponse)
def shop_page(request: Request, slug: str, conn: sqlite3.Connection = Depends(conn_dep)):
    shop = _shop_or_404(conn, slug)
    return templates.TemplateResponse(request, "shop.html", {"shop": shop, "products": catalogue.shop_products(conn, shop),
                                                             "error": None})


@router.post("/s/{slug}/order", response_class=HTMLResponse)
async def shop_order(request: Request, slug: str, conn: sqlite3.Connection = Depends(conn_dep)):
    s = settings_of(request)
    shop = _shop_or_404(conn, slug)
    form = await request.form()
    lines = []
    for key, value in form.multi_items():
        if key.startswith("qty_") and str(value).strip():
            lines.append((key[4:], str(value).strip()))
    try:
        order = orders.propose_order(conn, shop, lines, ttl_minutes=s.click_to_chat_ttl_minutes,
                                     max_lines=s.max_order_lines, max_qty=s.max_line_quantity)
    except orders.OrderError as exc:
        return templates.TemplateResponse(request, "shop.html", {
            "shop": shop, "products": catalogue.shop_products(conn, shop), "error": str(exc)}, status_code=400)
    link = wa_link(shop["seller_whatsapp"], order_message(shop, order))
    return templates.TemplateResponse(request, "order.html", {"shop": shop, "order": order, "wa_link": link})


@router.get("/s/{slug}/img/{product_id}")
def shop_image(request: Request, slug: str, product_id: int, conn: sqlite3.Connection = Depends(conn_dep)):
    shop = _shop_or_404(conn, slug)
    product = next((p for p in catalogue.shop_products(conn, shop) if p["id"] == product_id), None)
    if not product:
        raise HTTPException(404, "Not found")
    return _image(settings_of(request), product)
