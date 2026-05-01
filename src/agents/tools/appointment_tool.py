from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from dotenv import load_dotenv

from google.oauth2 import service_account
from googleapiclient.discovery import build
from langchain_core import messages as lc_messages

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
    digits = "".join(ch for ch in phone if ch.isdigit())
    if not digits:
        return None
    if digits.startswith("84"):
        digits = "0" + digits[2:]
    return digits


def _mask_phone(phone: Optional[str]) -> Optional[str]:
    if not phone:
        return None
    normalized = _normalize_phone(phone)
    if not normalized:
        return None
    if len(normalized) <= 4:
        return "*" * len(normalized)
    return f"{normalized[:2]}{'*' * (len(normalized) - 4)}{normalized[-2:]}"


def _is_valid_vn_phone(phone: Optional[str]) -> bool:
    normalized = _normalize_phone(phone)
    if not normalized:
        return False
    return bool(re.fullmatch(r"0\d{9,10}", normalized))


def _parse_datetime(date_str: str, time_str: str) -> datetime:
    dt = datetime.fromisoformat(f"{date_str}T{time_str}:00")
    return dt.replace(tzinfo=VIETNAM_TZ)


def _validate_future_datetime(date_str: Optional[str], time_str: Optional[str]) -> Optional[str]:
    if not date_str or not time_str:
        return None
    try:
        target = _parse_datetime(date_str, time_str)
    except Exception:
        return "Thời gian đặt lịch chưa đúng định dạng. Vui lòng dùng YYYY-MM-DD cho ngày và HH:MM cho giờ."

    now_local = datetime.now(VIETNAM_TZ)
    if target <= now_local:
        return "Thời gian đặt lịch phải ở tương lai. Bạn vui lòng chọn ngày/giờ khác."
    return None


def _history_to_text(history: Optional[List[lc_messages.BaseMessage]] = None, keep_last: int = 8) -> str:
    lines: List[str] = []
    for msg in (history or [])[-keep_last:]:
        role = "user" if isinstance(msg, lc_messages.HumanMessage) else "assistant"
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _extract_payload_with_llm(
    query: str,
    history: Optional[List[lc_messages.BaseMessage]] = None,
    draft: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
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
        "- Nếu không chắc chắn thì để null hoặc unknown.\n"
        "- Có thể suy luận từ hội thoại gần nhất và draft hiện có để điền các trường còn thiếu."
    )
    history_text = _history_to_text(history=history)
    draft_text = json.dumps(draft or {}, ensure_ascii=False)
    user_prompt = (
        f"Draft hiện có:\n{draft_text}\n\n"
        f"Lịch sử hội thoại gần nhất:\n{history_text}\n\n"
        f"Query hiện tại: {query}"
    )
    raw = llm.chat(system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.0) or "{}"
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def _payload_from_draft(draft: Optional[Dict[str, Any]]) -> AppointmentPayload:
    source = draft or {}
    return AppointmentPayload(
        full_name=source.get("full_name"),
        phone=_normalize_phone(source.get("phone")),
        appointment_date=source.get("appointment_date"),
        appointment_time=source.get("appointment_time"),
        reason=source.get("reason"),
        confirm=source.get("confirm"),
        event_id=source.get("event_id"),
    )


def _merge_payload(base: AppointmentPayload, incoming: AppointmentPayload) -> AppointmentPayload:
    return AppointmentPayload(
        full_name=incoming.full_name or base.full_name,
        phone=_normalize_phone(incoming.phone) or _normalize_phone(base.phone),
        appointment_date=incoming.appointment_date or base.appointment_date,
        appointment_time=incoming.appointment_time or base.appointment_time,
        reason=incoming.reason or base.reason,
        confirm=incoming.confirm if incoming.confirm is not None else base.confirm,
        event_id=incoming.event_id or base.event_id,
    )


def _payload_to_draft(payload: AppointmentPayload, *, status: str, intent: str) -> Dict[str, Any]:
    return {
        "intent": intent,
        "status": status,
        "full_name": payload.full_name,
        "phone": payload.phone,
        "phone_masked": _mask_phone(payload.phone),
        "appointment_date": payload.appointment_date,
        "appointment_time": payload.appointment_time,
        "reason": payload.reason,
        "confirm": payload.confirm,
        "event_id": payload.event_id,
    }


def _payload_from_query(
    query: str,
    history: Optional[List[lc_messages.BaseMessage]] = None,
    draft: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    data = _extract_payload_with_llm(query=query, history=history, draft=draft)
    incoming_payload = AppointmentPayload(
        full_name=data.get("full_name"),
        phone=_normalize_phone(data.get("phone")),
        appointment_date=data.get("appointment_date"),
        appointment_time=data.get("appointment_time"),
        reason=data.get("reason"),
        confirm=data.get("confirm"),
        event_id=data.get("event_id"),
    )
    merged_payload = _merge_payload(_payload_from_draft(draft), incoming_payload)
    return {
        "intent": data.get("intent", "unknown"),
        "payload": merged_payload,
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
    return missing


def handle_appointment_request(
    query: str,
    history: Optional[List[lc_messages.BaseMessage]] = None,
    draft: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    extracted = _payload_from_query(query=query, history=history, draft=draft)
    intent = str(extracted.get("intent", "unknown") or "unknown").lower()
    payload: AppointmentPayload = extracted["payload"]
    previous_intent = str((draft or {}).get("intent", "book") or "book").lower()
    if intent == "unknown" and previous_intent in ("book", "check", "delete"):
        intent = previous_intent

    if intent == "check":
        if not payload.appointment_date or not payload.appointment_time:
            return {
                "status": "need_more_info",
                "message": "Để kiểm tra lịch trống, bạn vui lòng cung cấp ngày và giờ mong muốn (ví dụ 2026-04-21, 14:00).",
                "draft": _payload_to_draft(payload, status="need_more_info", intent="check"),
            }
        time_error = _validate_future_datetime(payload.appointment_date, payload.appointment_time)
        if time_error:
            return {
                "status": "need_more_info",
                "message": time_error,
                "draft": _payload_to_draft(payload, status="need_more_info", intent="check"),
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
                    "draft": _payload_to_draft(payload, status="ok", intent="check"),
                }
            return {
                "status": "conflict",
                "message": (
                    f"Khung giờ {payload.appointment_date} {payload.appointment_time} hiện đã có lịch. "
                    "Bạn vui lòng chọn giờ khác."
                ),
                "data": checked,
                "draft": _payload_to_draft(payload, status="conflict", intent="check"),
            }
        except Exception as exc:
            return {
                "status": "error",
                "message": f"Không thể kiểm tra lịch: {exc}",
                "draft": _payload_to_draft(payload, status="error", intent="check"),
            }

    if intent == "delete":
        if not payload.event_id:
            return {
                "status": "need_more_info",
                "message": "Bạn muốn hủy lịch nào? Vui lòng gửi `event_id` để mình xóa lịch chính xác.",
                "draft": _payload_to_draft(payload, status="need_more_info", intent="delete"),
            }
        try:
            deleted = delete_apointment(payload.event_id)
            return {
                "status": "ok",
                "message": "Đã hủy lịch thành công.",
                "data": deleted,
                "draft": _payload_to_draft(payload, status="ok", intent="delete"),
            }
        except Exception as exc:
            return {
                "status": "error",
                "message": f"Không thể hủy lịch: {exc}",
                "draft": _payload_to_draft(payload, status="error", intent="delete"),
            }

    # Default flow: booking.
    if payload.phone and not _is_valid_vn_phone(payload.phone):
        return {
            "status": "need_more_info",
            "message": "SĐT chưa hợp lệ. Vui lòng cung cấp số điện thoại Việt Nam hợp lệ (ví dụ 09xxxxxxxx hoặc +84xxxxxxxxx).",
            "draft": _payload_to_draft(payload, status="need_more_info", intent="book"),
        }

    time_error = _validate_future_datetime(payload.appointment_date, payload.appointment_time)
    if time_error:
        return {
            "status": "need_more_info",
            "message": time_error,
            "draft": _payload_to_draft(payload, status="need_more_info", intent="book"),
        }

    missing = _missing_fields(payload)
    if missing:
        return {
            "status": "need_more_info",
            "message": "Để đặt lịch, bạn vui lòng cung cấp thêm: " + ", ".join(missing) + ".",
            "draft": _payload_to_draft(payload, status="need_more_info", intent="book"),
        }

    if payload.confirm is False:
        return {
            "status": "cancelled",
            "message": "Đã hủy yêu cầu đặt lịch theo xác nhận của bạn.",
            "draft": _payload_to_draft(payload, status="cancelled", intent="book"),
        }

    if payload.confirm is not True:
        summary = (
            "Mình đã thu thập đủ thông tin đặt lịch:\n"
            f"- Họ tên: {payload.full_name}\n"
            f"- SĐT: {_mask_phone(payload.phone) or 'N/A'}\n"
            f"- Thời gian: {payload.appointment_date} {payload.appointment_time}\n"
            f"- Lý do khám/triệu chứng: {payload.reason}\n\n"
            "Bạn vui lòng xác nhận lại bằng cách trả lời 'đồng ý đặt lịch' để mình tạo lịch."
        )
        return {
            "status": "awaiting_confirmation",
            "message": summary,
            "draft": _payload_to_draft(payload, status="awaiting_confirmation", intent="book"),
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
                "draft": _payload_to_draft(payload, status="conflict", intent="book"),
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
            "draft": _payload_to_draft(payload, status="ok", intent="book"),
        }
    except Exception as exc:
        return {
            "status": "error",
            "message": f"Không thể tạo lịch: {exc}",
            "draft": _payload_to_draft(payload, status="error", intent="book"),
        }
