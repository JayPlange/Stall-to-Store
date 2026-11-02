"""FastAPI app factory. Run: uvicorn app.main:create_app --factory"""
import logging
import secrets
import threading
import time
from collections.abc import Iterator

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles

from app.config import Settings, get_settings
from app.db import connect, init_db
from app.logging_setup import configure_logging
from app.pipeline.orchestrate import run_pipeline
from app.web import routes

log = logging.getLogger("sts.http")


def create_app(settings: Settings | None = None) -> FastAPI:
    configure_logging()
    settings = settings or get_settings()
    settings.runs_dir.mkdir(parents=True, exist_ok=True)
    conn = connect(settings.db_path)
    init_db(conn)
    conn.close()

    app = FastAPI(title="Stall to Store", docs_url=None, redoc_url=None)
    app.state.settings = settings
    app.state.start_pipeline = lambda run_id: threading.Thread(
        target=run_pipeline, args=(run_id, settings), daemon=True).start()

    def get_conn() -> Iterator:
        c = connect(settings.db_path)
        try:
            yield c
        finally:
            c.close()

    app.dependency_overrides[routes.conn_dep] = get_conn

    @app.middleware("http")
    async def request_context(request: Request, call_next):
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

    app.mount("/static", StaticFiles(directory=routes.STATIC_DIR), name="static")
    app.include_router(routes.router)
    return app
