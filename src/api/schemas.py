from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ReadinessStatus(BaseModel):
    ok: bool
    detail: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    service: str
    version: str
    time: datetime
    readiness: Dict[str, ReadinessStatus]


class SessionCreateRequest(BaseModel):
    device_id: Optional[str] = None
    locale: str = "vi-VN"


class SessionCreateResponse(BaseModel):
    session_id: str
    guest_id: str
    created_at: datetime
    expires_at: datetime


class SessionMetadataResponse(BaseModel):
    session_id: str
    status: Literal["active", "expired"]
    created_at: datetime
    last_active_at: datetime
    message_count: int
    locale: str = "vi-VN"


class ChatMessageCreateRequest(BaseModel):
    message: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    retrieval_filter: Optional[Dict[str, Any]] = None
    client_message_id: Optional[str] = None
    include_citations: bool = False
    include_debug: bool = False


class AppointmentView(BaseModel):
    status: Optional[str] = None
    draft: Optional[Dict[str, Any]] = None


class ChatMessageCreateResponse(BaseModel):
    message_id: str
    session_id: str
    answer: str
    route: str
    route_reason: str = ""
    appointment: AppointmentView
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    created_at: datetime
    debug: Optional[Dict[str, Any]] = None


class MessageItem(BaseModel):
    message_id: str
    role: Literal["user", "assistant"]
    content: str
    route: Optional[str] = None
    created_at: datetime


class MessageListResponse(BaseModel):
    items: List[MessageItem]
    next_cursor: Optional[str] = None


class AppointmentSubmitRequest(BaseModel):
    session_id: str
    message: str = Field(min_length=1)


class AppointmentConfirmRequest(BaseModel):
    confirm: bool


class AppointmentResponse(BaseModel):
    request_id: str
    session_id: str
    status: str
    message: str
    draft: Optional[Dict[str, Any]] = None
    data: Optional[Dict[str, Any]] = None
    updated_at: datetime


class FeedbackRequest(BaseModel):
    session_id: str
    message_id: str
    rating: Literal["up", "down"]
    comment: Optional[str] = None


class FeedbackResponse(BaseModel):
    feedback_id: str
    accepted: bool
    created_at: datetime


class PublicConfigResponse(BaseModel):
    locale: str = "vi-VN"
    feature_flags: Dict[str, bool]
    max_message_length: int = 4000
