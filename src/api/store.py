from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import Any, Dict, List, Optional, Protocol
from uuid import uuid4

from dotenv import load_dotenv
from psycopg import Connection
from psycopg.rows import dict_row


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


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


class StoreProtocol(Protocol):
    def create_session(self, locale: str = "vi-VN") -> SessionRecord: ...
    def get_session(self, session_id: str) -> Optional[SessionRecord]: ...
    def touch_session(self, session_id: str) -> None: ...
    def append_message(
        self,
        session_id: str,
        *,
        role: str,
        content: str,
        route: Optional[str] = None,
        message_id: Optional[str] = None,
    ) -> Dict[str, Any]: ...
    def list_messages(self, session_id: str, limit: int = 50, cursor: Optional[int] = None) -> Dict[str, Any]: ...
    def upsert_appointment(
        self,
        *,
        request_id: Optional[str],
        session_id: str,
        status: str,
        message: str,
        draft: Optional[Dict[str, Any]],
        data: Optional[Dict[str, Any]],
    ) -> AppointmentRecord: ...
    def get_appointment(self, request_id: str) -> Optional[AppointmentRecord]: ...
    def get_latest_appointment_for_session(self, session_id: str) -> Optional[AppointmentRecord]: ...
    def add_feedback(self, payload: Dict[str, Any]) -> Dict[str, Any]: ...


class InMemoryStore(StoreProtocol):
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


class PostgresStore(StoreProtocol):
    def __init__(self, database_url: str) -> None:
        self.database_url = database_url
        self._setup()

    def _connect(self) -> Connection:
        return Connection.connect(self.database_url, autocommit=True, row_factory=dict_row)

    def _setup(self) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS api_sessions (
                    session_id TEXT PRIMARY KEY,
                    guest_id TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL,
                    expires_at TIMESTAMPTZ NOT NULL,
                    last_active_at TIMESTAMPTZ NOT NULL,
                    locale TEXT NOT NULL
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS api_messages (
                    id BIGSERIAL PRIMARY KEY,
                    message_id TEXT UNIQUE NOT NULL,
                    session_id TEXT NOT NULL REFERENCES api_sessions(session_id) ON DELETE CASCADE,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    route TEXT NULL,
                    created_at TIMESTAMPTZ NOT NULL
                );
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_api_messages_session_id_id ON api_messages(session_id, id);")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS api_appointments (
                    request_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES api_sessions(session_id) ON DELETE CASCADE,
                    status TEXT NOT NULL,
                    message TEXT NOT NULL,
                    draft JSONB NOT NULL DEFAULT '{}'::jsonb,
                    data JSONB NULL,
                    updated_at TIMESTAMPTZ NOT NULL
                );
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_api_appointments_session_updated ON api_appointments(session_id, updated_at DESC);"
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS api_feedback (
                    feedback_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES api_sessions(session_id) ON DELETE CASCADE,
                    message_id TEXT NOT NULL,
                    rating TEXT NOT NULL,
                    comment TEXT NULL,
                    created_at TIMESTAMPTZ NOT NULL
                );
                """
            )

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
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO api_sessions(session_id, guest_id, created_at, expires_at, last_active_at, locale)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    session.session_id,
                    session.guest_id,
                    session.created_at,
                    session.expires_at,
                    session.last_active_at,
                    session.locale,
                ),
            )
        return session

    def get_session(self, session_id: str) -> Optional[SessionRecord]:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT session_id, guest_id, created_at, expires_at, last_active_at, locale
                FROM api_sessions
                WHERE session_id = %s
                """,
                (session_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            cur.execute(
                """
                SELECT message_id, role, content, route, created_at
                FROM api_messages
                WHERE session_id = %s
                ORDER BY id ASC
                """,
                (session_id,),
            )
            messages = [dict(item) for item in (cur.fetchall() or [])]
        return SessionRecord(
            session_id=row["session_id"],
            guest_id=row["guest_id"],
            created_at=row["created_at"],
            expires_at=row["expires_at"],
            last_active_at=row["last_active_at"],
            locale=row["locale"],
            messages=messages,
        )

    def touch_session(self, session_id: str) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("UPDATE api_sessions SET last_active_at = %s WHERE session_id = %s", (utcnow(), session_id))

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
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO api_messages(message_id, session_id, role, content, route, created_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    item["message_id"],
                    session_id,
                    item["role"],
                    item["content"],
                    item["route"],
                    item["created_at"],
                ),
            )
            cur.execute("UPDATE api_sessions SET last_active_at = %s WHERE session_id = %s", (item["created_at"], session_id))
        return item

    def list_messages(self, session_id: str, limit: int = 50, cursor: Optional[int] = None) -> Dict[str, Any]:
        start = cursor or 0
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, message_id, role, content, route, created_at
                FROM api_messages
                WHERE session_id = %s
                ORDER BY id ASC
                OFFSET %s LIMIT %s
                """,
                (session_id, start, limit),
            )
            items = cur.fetchall() or []
            cur.execute("SELECT COUNT(*) AS total FROM api_messages WHERE session_id = %s", (session_id,))
            total = int((cur.fetchone() or {}).get("total", 0))
        normalized_items = [
            {
                "message_id": row["message_id"],
                "role": row["role"],
                "content": row["content"],
                "route": row.get("route"),
                "created_at": row["created_at"],
            }
            for row in items
        ]
        next_cursor = str(start + limit) if (start + limit) < total else None
        return {"items": normalized_items, "next_cursor": next_cursor}

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
        payload_draft = draft or {}
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO api_appointments(request_id, session_id, status, message, draft, data, updated_at)
                VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb, %s)
                ON CONFLICT (request_id) DO UPDATE
                SET status = EXCLUDED.status,
                    message = EXCLUDED.message,
                    draft = EXCLUDED.draft,
                    data = EXCLUDED.data,
                    updated_at = EXCLUDED.updated_at
                """,
                (
                    rid,
                    session_id,
                    status,
                    message,
                    json.dumps(payload_draft, ensure_ascii=False),
                    json.dumps(data, ensure_ascii=False) if data is not None else None,
                    now,
                ),
            )
        return AppointmentRecord(
            request_id=rid,
            session_id=session_id,
            status=status,
            message=message,
            draft=payload_draft,
            data=data,
            updated_at=now,
        )

    def get_appointment(self, request_id: str) -> Optional[AppointmentRecord]:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT request_id, session_id, status, message, draft, data, updated_at
                FROM api_appointments
                WHERE request_id = %s
                """,
                (request_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None
        return AppointmentRecord(
            request_id=row["request_id"],
            session_id=row["session_id"],
            status=row["status"],
            message=row["message"],
            draft=row["draft"] or {},
            data=row["data"],
            updated_at=row["updated_at"],
        )

    def get_latest_appointment_for_session(self, session_id: str) -> Optional[AppointmentRecord]:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT request_id, session_id, status, message, draft, data, updated_at
                FROM api_appointments
                WHERE session_id = %s
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (session_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None
        return AppointmentRecord(
            request_id=row["request_id"],
            session_id=row["session_id"],
            status=row["status"],
            message=row["message"],
            draft=row["draft"] or {},
            data=row["data"],
            updated_at=row["updated_at"],
        )

    def add_feedback(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        item = {
            "feedback_id": f"fb_{uuid4().hex}",
            "created_at": utcnow(),
            **payload,
        }
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO api_feedback(feedback_id, session_id, message_id, rating, comment, created_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    item["feedback_id"],
                    item["session_id"],
                    item["message_id"],
                    item["rating"],
                    item.get("comment"),
                    item["created_at"],
                ),
            )
        return item


def _build_store() -> StoreProtocol:
    load_dotenv()
    database_url = (os.getenv("API_DATABASE_URL", "") or os.getenv("DATABASE_URL", "")).strip()
    if not database_url:
        return InMemoryStore()
    try:
        return PostgresStore(database_url=database_url)
    except Exception:
        return InMemoryStore()


store: StoreProtocol = _build_store()
