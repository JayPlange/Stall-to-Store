"""Meta Model API transport: explicit, logged, never retried.

Every call, success or failure, writes a sanitised raw artefact to runs/{run_id}/raw/{stage}.json
(then {stage}.2.json, ...) and one row in model_calls. Adapters normalise only after that.

Documented by Meta (dev.meta.ai/docs, read 2026-10-26): base https://api.meta.ai/v1, bearer auth,
error envelope {"error": {"message", "type", "param", "code"}}. 400/401/402/403/404/413 must not be
retried unchanged. We do not retry anything automatically.
"""
import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from app.artefacts import run_dir, safe_join, validate_run_id
from app.config import Settings
from app.db import iso, utcnow

log = logging.getLogger("sts.models")

PROVIDER = "meta"
KEEP_HEADERS = ("content-type", "retry-after", "x-request-id", "request-id")
SENSITIVE_KEYS = ("authorization", "api_key", "apikey", "token", "secret", "password")


class LiveCallsDisabled(RuntimeError):
    """Raised before any network I/O when live calls are not explicitly enabled."""


class AdapterNotVerified(RuntimeError):
    """The provider operation or response shape has not been verified against a real response."""


class ProviderError(RuntimeError):
    def __init__(self, status: int | None, error_type: str | None, message: str, raw_path: str | None):
        super().__init__(f"provider error {status} {error_type}: {message}")
        self.status = status
        self.error_type = error_type
        self.safe_message = message
        self.raw_path = raw_path


@dataclass
class CallContext:
    run_id: str | None
    stage: str


@dataclass
class CallResult:
    status: int
    body: Any
    raw_path: str
    latency_ms: int
    units: dict


def sanitise(value: Any, secret: str | None = None) -> Any:
    """Redact secrets and inline media (data URLs) from anything we persist."""
    if isinstance(value, dict):
        return {k: ("<redacted>" if any(s in k.lower() for s in SENSITIVE_KEYS) else sanitise(v, secret))
                for k, v in value.items()}
    if isinstance(value, list):
        return [sanitise(v, secret) for v in value]
    if isinstance(value, str):
        if value.startswith("data:"):
            return f"<data-url redacted, {len(value)} chars>"
        if secret and secret in value:
            return value.replace(secret, "<redacted>")
    return value


class MetaClient:
    def __init__(self, settings: Settings, conn: sqlite3.Connection, transport: httpx.BaseTransport | None = None):
        self.settings = settings
        self.conn = conn
        self.transport = transport

    # -- gating -----------------------------------------------------------
    def _key(self) -> str:
        if not self.settings.allow_live_model_calls:
            raise LiveCallsDisabled("live model calls are disabled (STS_ALLOW_LIVE_MODEL_CALLS=false)")
        if not self.settings.api_key_present:
            raise LiveCallsDisabled("MODEL_API_KEY is not set")
        return self.settings.model_api_key.get_secret_value().strip()

    # -- artefacts --------------------------------------------------------
    def _raw_target(self, ctx: CallContext) -> Path:
        root = self.settings.runs_dir
        base = run_dir(root, ctx.run_id, create=True) / "raw" if ctx.run_id else safe_join(root, "_smoke", "raw")
        base.mkdir(parents=True, exist_ok=True)
        stage = "".join(c for c in ctx.stage if c.isalnum() or c in "-_") or "call"
        path, n = base / f"{stage}.json", 1
        while path.exists():
            n += 1
            path = base / f"{stage}.{n}.json"
        return path

    def _record(self, ctx: CallContext, model: str, endpoint: str, request_summary: dict,
                status: int | None, headers: dict, body: Any, latency_ms: int, error_type: str | None,
                secret: str) -> tuple[str, dict]:
        units = body.get("usage") if isinstance(body, dict) and isinstance(body.get("usage"), dict) else {}
        path = self._raw_target(ctx)
        artefact = {
            "provider": PROVIDER, "endpoint": endpoint, "model": model, "stage": ctx.stage,
            "run_id": ctx.run_id, "created_at": iso(utcnow()), "http_status": status,
            "latency_ms": latency_ms, "request_summary": sanitise(request_summary, secret),
            "response_headers": {k: v for k, v in headers.items() if k.lower() in KEEP_HEADERS},
            "body": sanitise(body, secret),
        }
        path.write_text(json.dumps(artefact, indent=2, ensure_ascii=False), encoding="utf-8")
        raw_rel = path.resolve().relative_to(self.settings.runs_dir.resolve()).as_posix()
        self.conn.execute(
            "INSERT INTO model_calls(run_id, stage, model, provider, endpoint, ok, http_status, error_type,"
            " latency_ms, units_json, cost_usd, raw_path, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (ctx.run_id, ctx.stage, model, PROVIDER, endpoint, int(error_type is None), status, error_type,
             latency_ms, json.dumps(units), None, raw_rel, iso(utcnow())),
        )
        self.conn.commit()
        return raw_rel, units

    # -- transport --------------------------------------------------------
    def request(self, ctx: CallContext, model: str, method: str, endpoint: str, *,
                json_body: dict | None = None, files: dict | None = None, data: dict | None = None,
                accept: str = "application/json", request_summary: dict | None = None) -> CallResult:
        if ctx.run_id:
            validate_run_id(ctx.run_id)
        key = self._key()
        url = self.settings.meta_api_base.rstrip("/") + endpoint
        summary = request_summary if request_summary is not None else (json_body or {})
        started = time.monotonic()
        status, headers, body, error_type = None, {}, None, None
        try:
            with httpx.Client(timeout=self.settings.model_timeout_s, transport=self.transport) as client:
                resp = client.request(method, url, json=json_body, files=files, data=data,
                                      headers={"Authorization": f"Bearer {key}", "Accept": accept})
            status, headers = resp.status_code, dict(resp.headers)
            try:
                body = resp.json()
            except ValueError:
                body = resp.text
            if status >= 400:
                err = body.get("error") if isinstance(body, dict) else None
                error_type = (err or {}).get("type") or f"http_{status}"
        except httpx.HTTPError as exc:
            error_type, body = "transport_error", {"error": {"message": type(exc).__name__}}
        latency_ms = int((time.monotonic() - started) * 1000)
        raw_rel, units = self._record(ctx, model, endpoint, summary, status, headers, body,
                                      latency_ms, error_type, key)
        log.info("model_call", extra={"ctx": {"run_id": ctx.run_id, "stage": ctx.stage, "model": model,
                                              "endpoint": endpoint, "status": status, "latency_ms": latency_ms,
                                              "error_type": error_type}})
        if error_type:
            err = body.get("error") if isinstance(body, dict) else None
            message = (err or {}).get("message") if isinstance(err, dict) else None
            raise ProviderError(status, error_type, sanitise(message or "request failed", key), raw_rel)
        return CallResult(status, body, raw_rel, latency_ms, units)
