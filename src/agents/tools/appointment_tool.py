from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from dotenv import load_dotenv

from google.oauth2 import service_account
from googleapiclient.discovery import build

from src.agents.llm import OptionalLLM

SCOPES = ["https://www.googleapis.com/auth/calendar"]
try:
    VIETNAM_TZ = ZoneInfo("Asia/Ho_Chi_Minh")
except ZoneInfoNotFoundError:
    VIETNAM_TZ = timezone(timedelta(hours=7))


@dataclass
class AppointmentPayload:
    full_name: Optional[str] = None
    phone: Optional[str] = None
    appointment_date: Optional[str] = None  # YYYY-MM-DD
    appointment_time: Optional[str] = None  # HH:MM
    reason: Optional[str] = None
    confirm: Optional[bool] = None
    event_id: Optional[str] = None

load_dotenv()

def _credentials_path() -> str:
    raw = (os.getenv("GOOGLE_CREDENTIALS_PATH", "credentials.json") or "").strip().strip("\"'")
    normalized = os.path.expanduser(raw)
    return os.path.normpath(normalized)


def _calendar_id() -> str:
    raw = (os.getenv("GOOGLE_CALENDAR_ID", "") or os.getenv("CALENDAR_ID", "")).strip()
    return raw.strip("\"'")


def _build_calendar_service():
    cred_path = _credentials_path()
    calendar_id = _calendar_id()
    if not os.path.exists(cred_path):
        raise ValueError(f"Missing credentials file: {cred_path}")
    if not calendar_id:
        raise ValueError("Missing GOOGLE_CALENDAR_ID in environment.")

    creds = service_account.Credentials.from_service_account_file(cred_path, scopes=SCOPES)
    service = build("calendar", "v3", credentials=creds)
    return service, calendar_id


def _to_utc_iso(local_dt: datetime) -> str:
    return local_dt.astimezone(timezone.utc).isoformat()


def _normalize_phone(phone: Optional[str]) -> Optional[str]:
    if not phone:
        return None
    cleaned = "".join(ch for ch in phone if ch.isdigit() or ch == "+")
    return cleaned or None


def _parse_datetime(date_str: str, time_str: str) -> datetime:
    dt = datetime.fromisoformat(f"{date_str}T{time_str}:00")
    return dt.replace(tzinfo=VIETNAM_TZ)


def _extract_payload_with_llm(query: str) -> Dict[str, Any]:
    llm = OptionalLLM()
    if not llm.enabled:
        return {
            "intent": "book",
            "full_name": None,
            "phone": None,
            "appointment_date": None,
            "appointment_time": None,
            "reason": None,
            "confirm": None,
            "event_id": None,
        }

    system_prompt = (
        "Bạn là bộ trích xuất thông tin đặt lịch khám.\n"
        "Trích xuất JSON duy nhất, không giải thích, theo schema:\n"
        "{"
        '"intent":"book|check|delete|unknown",'
        '"full_name":string|null,'
        '"phone":string|null,'
        '"appointment_date":"YYYY-MM-DD"|null,'
        '"appointment_time":"HH:MM"|null,'
        '"reason":string|null,'
        '"confirm":true|false|null,'
        '"event_id":string|null'
        "}\n"
        "Quy tắc:\n"
        "- confirm=true khi người dùng thể hiện xác nhận rõ ràng (ví dụ: đồng ý đặt lịch, xác nhận).\n"
        "- confirm=false khi người dùng từ chối.\n"
        "- intent=check khi hỏi còn lịch trống/rảnh.\n"
        "- intent=delete khi muốn hủy lịch.\n"
        "- intent=book khi muốn đặt lịch.\n"
        "- Nếu không chắc chắn thì để null hoặc unknown."
    )
    raw = llm.chat(system_prompt=system_prompt, user_prompt=f"Query: {query}", temperature=0.0) or "{}"
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def _payload_from_query(query: str) -> Dict[str, Any]:
    data = _extract_payload_with_llm(query=query)
    payload = AppointmentPayload(
        full_name=data.get("full_name"),
        phone=_normalize_phone(data.get("phone")),
        appointment_date=data.get("appointment_date"),
        appointment_time=data.get("appointment_time"),
        reason=data.get("reason"),
        confirm=data.get("confirm"),
        event_id=data.get("event_id"),
    )
    return {
        "intent": data.get("intent", "unknown"),
        "payload": payload,
    }


def check_apointment_availability(
    appointment_date: str,
    appointment_time: str,
    duration_minutes: int = 30,
) -> Dict[str, Any]:
    service, calendar_id = _build_calendar_service()
    start_local = _parse_datetime(appointment_date, appointment_time)
    end_local = start_local + timedelta(minutes=duration_minutes)

    result = service.freebusy().query(
        body={
            "timeMin": _to_utc_iso(start_local),
            "timeMax": _to_utc_iso(end_local),
            "timeZone": "Asia/Ho_Chi_Minh",
            "items": [{"id": calendar_id}],
        }
    ).execute()
    busy = (result.get("calendars", {}).get(calendar_id, {}) or {}).get("busy", [])
    return {
        "available": len(busy) == 0,
        "busy": busy,
        "start": start_local.isoformat(),
        "end": end_local.isoformat(),
    }


def create_apointment(
    full_name: str,
    phone: str,
    appointment_date: str,
    appointment_time: str,
    reason: str,
    duration_minutes: int = 30,
) -> Dict[str, Any]:
    service, calendar_id = _build_calendar_service()
    start_local = _parse_datetime(appointment_date, appointment_time)
    end_local = start_local + timedelta(minutes=duration_minutes)
    event = {
        "summary": f"Lich khám - {full_name}",
        "description": (
            f"Họ tên: {full_name}\n"
            f"Số điện thoại: {phone}\n"
            f"Lý do khám/Triệu chứng: {reason}"
        ),
        "start": {
            "dateTime": start_local.isoformat(),
            "timeZone": "Asia/Ho_Chi_Minh",
        },
        "end": {
            "dateTime": end_local.isoformat(),
            "timeZone": "Asia/Ho_Chi_Minh",
        },
    }
    created = service.events().insert(calendarId=calendar_id, body=event).execute()
    return {
        "event_id": created.get("id"),
        "html_link": created.get("htmlLink"),
        "status": created.get("status"),
        "start": created.get("start", {}).get("dateTime"),
        "end": created.get("end", {}).get("dateTime"),
    }


def delete_apointment(event_id: str) -> Dict[str, Any]:
    service, calendar_id = _build_calendar_service()
    service.events().delete(calendarId=calendar_id, eventId=event_id).execute()
    return {"deleted": True, "event_id": event_id}


def _missing_fields(payload: AppointmentPayload) -> List[str]:
    missing: List[str] = []
    if not payload.full_name:
        missing.append("Họ tên")
    if not payload.phone:
        missing.append("SĐT")
    if not payload.appointment_date:
        missing.append("Ngày đặt lịch (YYYY-MM-DD)")
    if not payload.appointment_time:
        missing.append("Giờ đặt lịch (HH:MM)")
    if not payload.reason:
        missing.append("Lý do khám/triệu chứng")
    if payload.confirm is not True:
        missing.append("Xác nhận đặt lịch")
    return missing


def handle_appointment_request(query: str) -> Dict[str, Any]:
    extracted = _payload_from_query(query=query)
    intent = extracted.get("intent", "unknown")
    payload: AppointmentPayload = extracted["payload"]

    if intent == "check":
        if not payload.appointment_date or not payload.appointment_time:
            return {
                "status": "need_more_info",
                "message": "Để kiểm tra lịch trống, bạn vui lòng cung cấp ngày và giờ mong muốn (ví dụ 2026-04-21, 14:00).",
            }
        try:
            checked = check_apointment_availability(payload.appointment_date, payload.appointment_time)
            if checked["available"]:
                return {
                    "status": "ok",
                    "message": (
                        f"Khung giờ {payload.appointment_date} {payload.appointment_time} đang trống. "
                        "Nếu bạn muốn đặt lịch, hãy gửi đầy đủ: Họ tên, SĐT, ngày, giờ, lý do khám/triệu chứng và xác nhận."
                    ),
                    "data": checked,
                }
            return {
                "status": "conflict",
                "message": (
                    f"Khung giờ {payload.appointment_date} {payload.appointment_time} hiện đã có lịch. "
                    "Bạn vui lòng chọn giờ khác."
                ),
                "data": checked,
            }
        except Exception as exc:
            return {"status": "error", "message": f"Không thể kiểm tra lịch: {exc}"}

    if intent == "delete":
        if not payload.event_id:
            return {
                "status": "need_more_info",
                "message": "Bạn muốn hủy lịch nào? Vui lòng gửi `event_id` để mình xóa lịch chính xác.",
            }
        try:
            deleted = delete_apointment(payload.event_id)
            return {"status": "ok", "message": "Đã hủy lịch thành công.", "data": deleted}
        except Exception as exc:
            return {"status": "error", "message": f"Không thể hủy lịch: {exc}"}

    # Default flow: booking.
    missing = _missing_fields(payload)
    if missing:
        return {
            "status": "need_more_info",
            "message": "Để đặt lịch, bạn vui lòng cung cấp thêm: " + ", ".join(missing) + ".",
        }

    try:
        checked = check_apointment_availability(payload.appointment_date or "", payload.appointment_time or "")
        if not checked["available"]:
            return {
                "status": "conflict",
                "message": (
                    f"Khung giờ {payload.appointment_date} {payload.appointment_time} đã có lịch. "
                    "Bạn chọn giờ khác giúp mình nhé."
                ),
                "data": checked,
            }

        created = create_apointment(
            full_name=payload.full_name or "",
            phone=payload.phone or "",
            appointment_date=payload.appointment_date or "",
            appointment_time=payload.appointment_time or "",
            reason=payload.reason or "",
        )
        return {
            "status": "ok",
            "message": (
                "Đặt lịch thành công. "
                f"Mã lịch: {created.get('event_id')}. "
                f"Thời gian: {payload.appointment_date} {payload.appointment_time}."
            ),
            "data": created,
        }
    except Exception as exc:
        return {"status": "error", "message": f"Không thể tạo lịch: {exc}"}
