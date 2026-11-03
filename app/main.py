"""FastAPI app factory. Run: uvicorn app.main:create_app --factory"""
import logging
import secrets
import time
from collections.abc import Iterator

from fastapi import FastAPI, Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import Settings, get_settings
from app.db import connect, init_db
from app.logging_setup import configure_logging
from app.pipeline.orchestrate import make_starter, recover_interrupted
from app.web import routes

log = logging.getLogger("sts.http")
FORM_OVERHEAD = 1_000_000


def create_app(settings: Settings | None = None) -> FastAPI:
    configure_logging()
    settings = settings or get_settings()
    settings.runs_dir.mkdir(parents=True, exist_ok=True)
    conn = connect(settings.db_path)
    init_db(conn)
    recover_interrupted(conn)
    conn.close()

    app = FastAPI(title="Stall to Store", docs_url=None, redoc_url=None)
    app.state.settings = settings
    app.state.start_pipeline = make_starter(settings)

    def get_conn() -> Iterator:
        c = connect(settings.db_path)
        try:
            yield c
        finally:
            c.close()

    app.dependency_overrides[routes.conn_dep] = get_conn

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        if request.method == "POST":
            # Reject before the form is parsed: the parser would otherwise store the whole body first.
            size = request.headers.get("content-length")
            if size is None and request.url.path == "/runs":
                return PlainTextResponse("Please upload a file from the form.", status_code=411)
            if size is not None and (not size.isdigit() or int(size) > settings.max_upload_bytes + FORM_OVERHEAD):
                return PlainTextResponse(f"Videos can be up to {settings.max_upload_mb} MB.", status_code=413)
        request_id = secrets.token_hex(6)
        started = time.monotonic()
        response = await call_next(request)
        # Review URLs carry a secret token: never leak it through Referer.
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-Request-Id"] = request_id
        parts = request.url.path.split("/")
        if len(parts) > 2 and parts[1] == "r":
            parts[2] = "<token>"
        path = "/".join(parts)
        log.info("request", extra={"ctx": {"request_id": request_id, "method": request.method, "path": path,
                                           "status": response.status_code,
                                           "ms": int((time.monotonic() - started) * 1000)}})
        return response

    @app.exception_handler(StarletteHTTPException)
    async def friendly_errors(request: Request, exc: StarletteHTTPException):
        # People reach these by typing or bookmarking a link, so give them a page with a way back.
        pages = {404: ("Page not found", "This link does not lead to a shop or page. Check the address and try again."),
                 405: ("Nothing to show here", "This address only works when you press a button on the page."
                                               " Sellers: open your private link again.")}
        if exc.status_code not in pages:
            return await http_exception_handler(request, exc)
        heading, message = pages[exc.status_code]
        return routes.templates.TemplateResponse(request, "error.html", {"heading": heading, "message": message},
                                                 status_code=exc.status_code)

    app.mount("/static", StaticFiles(directory=routes.STATIC_DIR), name="static")
    app.include_router(routes.router)
    return app
