from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import Lock
from typing import Any, Dict, List, Optional
from uuid import uuid4


def utcnow() -> datetime:
    return datetime.utcnow()


@dataclass
class SessionRecord:
    session_id: str
    guest_id: str
    created_at: datetime
    expires_at: datetime
    last_active_at: datetime
    locale: str = "vi-VN"
    messages: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class AppointmentRecord:
    request_id: str
    session_id: str
    status: str
    message: str
    draft: Dict[str, Any]
    data: Optional[Dict[str, Any]]
    updated_at: datetime


class InMemoryStore:
    def __init__(self) -> None:
        self._lock = Lock()
        self.sessions: Dict[str, SessionRecord] = {}
        self.appointments: Dict[str, AppointmentRecord] = {}
        self.feedbacks: List[Dict[str, Any]] = []

    def create_session(self, locale: str = "vi-VN") -> SessionRecord:
        now = utcnow()
        session = SessionRecord(
            session_id=f"sess_{uuid4().hex}",
            guest_id=f"guest_{uuid4().hex}",
            created_at=now,
            expires_at=now + timedelta(days=7),
            last_active_at=now,
            locale=locale,
        )
        with self._lock:
            self.sessions[session.session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[SessionRecord]:
        return self.sessions.get(session_id)

    def touch_session(self, session_id: str) -> None:
        with self._lock:
            session = self.sessions.get(session_id)
            if session:
                session.last_active_at = utcnow()

    def append_message(
        self,
        session_id: str,
        *,
        role: str,
        content: str,
        route: Optional[str] = None,
        message_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        item = {
            "message_id": message_id or f"msg_{uuid4().hex}",
            "role": role,
            "content": content,
            "route": route,
            "created_at": utcnow(),
        }
        with self._lock:
            session = self.sessions[session_id]
            session.messages.append(item)
            session.last_active_at = item["created_at"]
        return item

    def list_messages(self, session_id: str, limit: int = 50, cursor: Optional[int] = None) -> Dict[str, Any]:
        session = self.sessions[session_id]
        start = cursor or 0
        end = start + limit
        items = session.messages[start:end]
        next_cursor = str(end) if end < len(session.messages) else None
        return {"items": items, "next_cursor": next_cursor}

    def upsert_appointment(
        self,
        *,
        request_id: Optional[str],
        session_id: str,
        status: str,
        message: str,
        draft: Optional[Dict[str, Any]],
        data: Optional[Dict[str, Any]],
    ) -> AppointmentRecord:
        now = utcnow()
        rid = request_id or f"appt_req_{uuid4().hex}"
        record = AppointmentRecord(
            request_id=rid,
            session_id=session_id,
            status=status,
            message=message,
            draft=draft or {},
            data=data,
            updated_at=now,
        )
        with self._lock:
            self.appointments[rid] = record
        return record

    def get_appointment(self, request_id: str) -> Optional[AppointmentRecord]:
        return self.appointments.get(request_id)

    def get_latest_appointment_for_session(self, session_id: str) -> Optional[AppointmentRecord]:
        candidates = [appt for appt in self.appointments.values() if appt.session_id == session_id]
        if not candidates:
            return None
        return max(candidates, key=lambda item: item.updated_at)

    def add_feedback(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        item = {
            "feedback_id": f"fb_{uuid4().hex}",
            "created_at": utcnow(),
            **payload,
        }
        with self._lock:
            self.feedbacks.append(item)
        return item


store = InMemoryStore()
