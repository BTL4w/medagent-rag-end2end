from __future__ import annotations

import os
from datetime import datetime

from fastapi import APIRouter

from src.agents.llm import OptionalLLM
from src.api.schemas import HealthResponse, PublicConfigResponse, ReadinessStatus

router = APIRouter(tags=["system"])


def _calendar_ready() -> ReadinessStatus:
    calendar_id = (os.getenv("GOOGLE_CALENDAR_ID", "") or os.getenv("CALENDAR_ID", "")).strip()
    cred_path = (os.getenv("GOOGLE_CREDENTIALS_PATH", "credentials.json") or "").strip().strip("\"'")
    has_calendar_id = bool(calendar_id)
    has_cred_file = os.path.exists(cred_path)
    ok = has_calendar_id and has_cred_file
    detail = "ok" if ok else "Missing GOOGLE_CALENDAR_ID/CALENDAR_ID or credentials.json"
    return ReadinessStatus(ok=ok, detail=detail)


def _vector_ready() -> ReadinessStatus:
    api_key = os.getenv("PINECONE_API_KEY", "").strip()
    index_name = os.getenv("PINECONE_INDEX_NAME", "").strip()
    ok = bool(api_key and index_name)
    detail = "ok" if ok else "Missing PINECONE_API_KEY or PINECONE_INDEX_NAME"
    return ReadinessStatus(ok=ok, detail=detail)


def _llm_ready() -> ReadinessStatus:
    llm = OptionalLLM()
    ok = llm.enabled
    detail = "ok" if ok else "Missing OPENAI_API_KEY"
    return ReadinessStatus(ok=ok, detail=detail)


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    readiness = {
        "vector_db": _vector_ready(),
        "llm": _llm_ready(),
        "calendar_adapter": _calendar_ready(),
    }
    status = "ok" if all(item.ok for item in readiness.values()) else "degraded"
    return HealthResponse(
        status=status,
        service="medagent-rag-api",
        version=os.getenv("APP_VERSION", "1.0.0"),
        time=datetime.utcnow(),
        readiness=readiness,
    )


@router.get("/config/public", response_model=PublicConfigResponse)
def public_config() -> PublicConfigResponse:
    return PublicConfigResponse(
        locale=os.getenv("APP_LOCALE", "vi-VN"),
        max_message_length=int(os.getenv("MAX_MESSAGE_LENGTH", "4000")),
        feature_flags={
            "chat": True,
            "appointments": True,
            "feedback": True,
            "citations_toggle": True,
        },
    )
