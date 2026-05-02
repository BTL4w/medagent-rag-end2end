"""Access logging (no body / no secrets) and simple per-IP rate limit for /api only."""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Callable, MutableMapping
from uuid import uuid4

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp

logger = logging.getLogger("medagent.api.access")

_RATE_WINDOW_SEC = 60.0


class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable):
        rid = request.headers.get("X-Request-Id") or f"req_{uuid4().hex}"
        request.state.request_id = rid
        response = await call_next(request)
        response.headers["X-Request-Id"] = rid
        return response


class AccessLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable):
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = int((time.perf_counter() - start) * 1000)
        rid = getattr(request.state, "request_id", "-")
        client = request.client.host if request.client else "-"
        logger.info(
            "request_done request_id=%s method=%s path=%s status=%s duration_ms=%s client_ip=%s",
            rid,
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            client,
        )
        return response


class SimpleRateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, limit_per_minute: int) -> None:
        super().__init__(app)
        self._limit = max(1, limit_per_minute)
        self._hits: MutableMapping[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next: Callable):
        path = request.url.path
        if not path.startswith("/api/"):
            return await call_next(request)

        now = time.time()
        window_start = now - _RATE_WINDOW_SEC
        client_ip = request.client.host if request.client else "unknown"
        window = self._hits[client_ip]
        window[:] = [t for t in window if t >= window_start]
        if len(window) >= self._limit:
            rid = getattr(request.state, "request_id", "-")
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "code": "RATE_LIMIT_EXCEEDED",
                        "message": "Too many requests. Try again shortly.",
                        "details": {"limit_per_minute": self._limit},
                    },
                    "request_id": rid,
                },
            )
        window.append(now)
        return await call_next(request)
