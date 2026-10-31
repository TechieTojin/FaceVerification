"""Structured logging with per-request correlation IDs."""
from __future__ import annotations

import contextvars
import json
import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

_req_log = logging.getLogger("faceverification.requests")


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
        }
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out)


def setup_logging(level: str = "INFO", fmt: str = "text") -> None:
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    handler = logging.StreamHandler()
    handler.addFilter(_RequestIdFilter())
    if fmt == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s [%(request_id)s]: %(message)s"))
    root.handlers = [handler]


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assigns a request ID, logs method/path/status/duration (never bodies)."""

    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:16]
        if len(rid) > 64 or not rid.replace("-", "").isalnum():
            rid = uuid.uuid4().hex[:16]
        token = request_id_var.set(rid)
        start = time.perf_counter()
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = rid
            if request.url.path != "/api/v1/stream":
                _req_log.info("%s %s -> %d (%.1fms)", request.method, request.url.path,
                              response.status_code, (time.perf_counter() - start) * 1000)
            return response
        finally:
            request_id_var.reset(token)
