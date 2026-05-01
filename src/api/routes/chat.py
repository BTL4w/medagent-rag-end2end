from __future__ import annotations

from typing import Any, Dict, Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query

from src.agents.orchestrator import run_agent
from src.api.schemas import (
    AppointmentView,
    ChatMessageCreateRequest,
    ChatMessageCreateResponse,
    MessageItem,
    MessageListResponse,
    SessionCreateRequest,
    SessionCreateResponse,
    SessionMetadataResponse,
)
from src.api.store import store, utcnow

router = APIRouter(prefix="/chat", tags=["chat"])
ALLOWED_ROUTES = {"simple_qa", "complex_qa", "appointment", "chitchat", "unsupported", "clarify"}


@router.post("/sessions", response_model=SessionCreateResponse)
def create_session(payload: SessionCreateRequest) -> SessionCreateResponse:
    session = store.create_session(locale=payload.locale)
    return SessionCreateResponse(
        session_id=session.session_id,
        guest_id=session.guest_id,
        created_at=session.created_at,
        expires_at=session.expires_at,
    )


@router.get("/sessions/{session_id}", response_model=SessionMetadataResponse)
def get_session(session_id: str) -> SessionMetadataResponse:
    session = store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    status = "expired" if session.expires_at <= utcnow() else "active"
    return SessionMetadataResponse(
        session_id=session.session_id,
        status=status,
        created_at=session.created_at,
        last_active_at=session.last_active_at,
        message_count=len(session.messages),
        locale=session.locale,
    )


def _build_debug_payload(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "query": result.get("query"),
        "sub_queries": result.get("sub_queries", []),
        "appointment_result": result.get("appointment_result"),
    }


@router.post("/sessions/{session_id}/messages", response_model=ChatMessageCreateResponse)
def create_message(session_id: str, payload: ChatMessageCreateRequest) -> ChatMessageCreateResponse:
    session = store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.expires_at <= utcnow():
        raise HTTPException(status_code=410, detail="Session expired")

    store.append_message(
        session_id,
        role="user",
        content=payload.message,
        route=None,
        message_id=payload.client_message_id or f"msg_cli_{uuid4().hex}",
    )
    result = run_agent(
        payload.message,
        top_k=payload.top_k,
        retrieval_filter=payload.retrieval_filter,
        thread_id=session_id,
    )
    answer = str(result.get("answer") or "")
    route = str(result.get("route") or "clarify")
    if route not in ALLOWED_ROUTES:
        route = "clarify"
    route_reason = str(result.get("route_reason") or "")
    assistant_msg = store.append_message(
        session_id,
        role="assistant",
        content=answer,
        route=route,
    )

    citations = result.get("citations") if payload.include_citations else []
    debug_payload: Optional[Dict[str, Any]] = _build_debug_payload(result) if payload.include_debug else None
    return ChatMessageCreateResponse(
        message_id=assistant_msg["message_id"],
        session_id=session_id,
        answer=answer,
        route=route,
        route_reason=route_reason,
        appointment=AppointmentView(
            status=(result.get("appointment_result") or {}).get("status"),
            draft=result.get("appointment_draft"),
        ),
        citations=citations or [],
        created_at=assistant_msg["created_at"],
        debug=debug_payload,
    )


@router.get("/sessions/{session_id}/messages", response_model=MessageListResponse)
def list_messages(
    session_id: str,
    limit: int = Query(default=50, ge=1, le=200),
    cursor: Optional[str] = None,
) -> MessageListResponse:
    session = store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    cursor_idx = int(cursor) if cursor else 0
    result = store.list_messages(session_id, limit=limit, cursor=cursor_idx)
    return MessageListResponse(
        items=[MessageItem(**item) for item in result["items"]],
        next_cursor=result["next_cursor"],
    )
