from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from src.api.routes.appointment import router as appointment_router
from src.api.routes.chat import router as chat_router
from src.api.routes.system import router as system_router
from src.api.schemas import ErrorResponse

app = FastAPI(title="MedAgent RAG API")

_cors_origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(system_router, prefix="/api/v1")
app.include_router(chat_router, prefix="/api/v1")
app.include_router(appointment_router, prefix="/api/v1")

_STATIC_CHAT_DIR = Path(__file__).resolve().parents[2] / "frontend" / "static-chat"
if _STATIC_CHAT_DIR.is_dir():
    app.mount("/chat-ui", StaticFiles(directory=str(_STATIC_CHAT_DIR), html=True), name="chat_ui")


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-Id", f"req_{uuid4().hex}")
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-Id"] = request_id
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    payload = ErrorResponse(
        error={
            "code": "HTTP_ERROR",
            "message": str(exc.detail),
            "details": {"status_code": exc.status_code},
        },
        request_id=getattr(request.state, "request_id", f"req_{uuid4().hex}"),
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
        request_id=getattr(request.state, "request_id", f"req_{uuid4().hex}"),
    )
    return JSONResponse(status_code=422, content=payload.model_dump())


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    payload = ErrorResponse(
        error={
            "code": "INTERNAL_ERROR",
            "message": "Internal server error.",
            "details": {"type": type(exc).__name__},
        },
        request_id=getattr(request.state, "request_id", f"req_{uuid4().hex}"),
    )
    return JSONResponse(status_code=500, content=payload.model_dump())

