from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from src.api.env_loader import load_app_env

load_app_env()

from src.api.middleware.http_extra import AccessLogMiddleware, RequestIdMiddleware, SimpleRateLimitMiddleware
from src.api.routes.appointment import router as appointment_router
from src.api.routes.chat import router as chat_router
from src.api.routes.system import router as system_router
from src.api.schemas import ErrorResponse

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

_cors_raw = os.getenv("CORS_ORIGINS", "").strip()
_cors_origins = (
    ["*"]
    if not _cors_raw
    else [o.strip() for o in _cors_raw.split(",") if o.strip()]
)

_rate_limit = int(os.getenv("RATE_LIMIT_API_PER_MINUTE", "120"))

app = FastAPI(title="MedAgent RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(AccessLogMiddleware)
app.add_middleware(SimpleRateLimitMiddleware, limit_per_minute=_rate_limit)
app.add_middleware(RequestIdMiddleware)

app.include_router(system_router, prefix="/api/v1")
app.include_router(chat_router, prefix="/api/v1")
app.include_router(appointment_router, prefix="/api/v1")

_STATIC_CHAT_DIR = Path(__file__).resolve().parents[2] / "frontend" / "static-chat"
if _STATIC_CHAT_DIR.is_dir():
    app.mount("/chat-ui", StaticFiles(directory=str(_STATIC_CHAT_DIR), html=True), name="chat_ui")


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    payload = ErrorResponse(
        error={
            "code": "HTTP_ERROR",
            "message": str(exc.detail),
            "details": {"status_code": exc.status_code},
        },
        request_id=getattr(request.state, "request_id", "unknown"),
    )
    return JSONResponse(status_code=exc.status_code, content=payload.model_dump())


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    payload = ErrorResponse(
        error={
            "code": "VALIDATION_ERROR",
            "message": "Request validation failed.",
            "details": {"errors": exc.errors()},
        },
        request_id=getattr(request.state, "request_id", "unknown"),
    )
    return JSONResponse(status_code=422, content=payload.model_dump())


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logging.getLogger("medagent.api").exception(
        "unhandled_exception request_id=%s path=%s",
        getattr(request.state, "request_id", "unknown"),
        request.url.path,
    )
    payload = ErrorResponse(
        error={
            "code": "INTERNAL_ERROR",
            "message": "Internal server error.",
            "details": {"type": type(exc).__name__},
        },
        request_id=getattr(request.state, "request_id", "unknown"),
    )
    return JSONResponse(status_code=500, content=payload.model_dump())
