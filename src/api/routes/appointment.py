from __future__ import annotations

from fastapi import APIRouter, HTTPException
from langchain_core import messages as lc_messages

from src.agents.tools import handle_appointment_request
from src.api.schemas import (
    AppointmentConfirmRequest,
    AppointmentResponse,
    AppointmentSubmitRequest,
    FeedbackRequest,
    FeedbackResponse,
)
from src.api.store import store

router = APIRouter(tags=["appointments"])


def _session_history_as_lc_messages(session_id: str) -> list[lc_messages.BaseMessage]:
    session = store.get_session(session_id)
    if session is None:
        return []
    history: list[lc_messages.BaseMessage] = []
    for item in session.messages:
        content = str(item.get("content") or "")
        if item.get("role") == "assistant":
            history.append(lc_messages.AIMessage(content=content))
        else:
            history.append(lc_messages.HumanMessage(content=content))
    return history


@router.post("/appointments", response_model=AppointmentResponse)
def submit_appointment(payload: AppointmentSubmitRequest) -> AppointmentResponse:
    session = store.get_session(payload.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    latest = store.get_latest_appointment_for_session(payload.session_id)

    result = handle_appointment_request(
        query=payload.message,
        history=_session_history_as_lc_messages(payload.session_id),
        draft=(latest.draft if latest else {}),
    )
    record = store.upsert_appointment(
        request_id=None,
        session_id=payload.session_id,
        status=str(result.get("status") or "need_more_info"),
        message=str(result.get("message") or ""),
        draft=result.get("draft"),
        data=result.get("data"),
    )
    return AppointmentResponse(
        request_id=record.request_id,
        session_id=record.session_id,
        status=record.status,
        message=record.message,
        draft=record.draft,
        data=record.data,
        updated_at=record.updated_at,
    )


@router.get("/appointments/{request_id}", response_model=AppointmentResponse)
def get_appointment(request_id: str) -> AppointmentResponse:
    record = store.get_appointment(request_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Appointment request not found")
    return AppointmentResponse(
        request_id=record.request_id,
        session_id=record.session_id,
        status=record.status,
        message=record.message,
        draft=record.draft,
        data=record.data,
        updated_at=record.updated_at,
    )


@router.post("/appointments/{request_id}/confirm", response_model=AppointmentResponse)
def confirm_appointment(request_id: str, payload: AppointmentConfirmRequest) -> AppointmentResponse:
    record = store.get_appointment(request_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Appointment request not found")

    updated_draft = dict(record.draft or {})
    updated_draft["confirm"] = payload.confirm
    query = "Tôi xác nhận đặt lịch." if payload.confirm else "Tôi không xác nhận đặt lịch."
    result = handle_appointment_request(
        query=query,
        history=_session_history_as_lc_messages(record.session_id),
        draft=updated_draft,
    )
    next_record = store.upsert_appointment(
        request_id=request_id,
        session_id=record.session_id,
        status=str(result.get("status") or "need_more_info"),
        message=str(result.get("message") or ""),
        draft=result.get("draft"),
        data=result.get("data"),
    )
    return AppointmentResponse(
        request_id=next_record.request_id,
        session_id=next_record.session_id,
        status=next_record.status,
        message=next_record.message,
        draft=next_record.draft,
        data=next_record.data,
        updated_at=next_record.updated_at,
    )


@router.post("/feedback", response_model=FeedbackResponse, tags=["feedback"])
def create_feedback(payload: FeedbackRequest) -> FeedbackResponse:
    session = store.get_session(payload.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    feedback = store.add_feedback(payload.dict())
    return FeedbackResponse(
        feedback_id=feedback["feedback_id"],
        accepted=True,
        created_at=feedback["created_at"],
    )
