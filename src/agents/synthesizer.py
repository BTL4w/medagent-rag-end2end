from __future__ import annotations

from typing import Dict, List, Optional

from langchain_core import messages as lc_messages
from src.agents.llm import OptionalLLM


def _build_chitchat_response_with_llm(
    query: str,
    llm: OptionalLLM,
    history: Optional[List[lc_messages.BaseMessage]] = None,
) -> Optional[str]:
    if not llm.enabled:
        return None

    history_lines: List[str] = []
    for msg in history or []:
        role = "user" if isinstance(msg, lc_messages.HumanMessage) else "assistant"
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        history_lines.append(f"{role}: {content}")
    history_text = "\n".join(history_lines[-10:])

    system_prompt = (
        "Bạn là trợ lý y khoa. "
        "Nhiệm vụ: xử lý hội thoại xã giao và luôn điều hướng về domain chính gồm bệnh lý, thuốc, thông tin sức khỏe.\n"
        "Chỉ xét các nhóm: greeting, identify(bạn là ai), capability(những khả năng của bạn), thanks(lời cảm ơn của user), farewell(lời tạm biệt).\n"
        "Nếu user hỏi lại thông tin đã nói trong hội thoại (ví dụ tên), được phép dùng lịch sử để trả lời chính xác.\n"
        "Nếu query thuộc một trong 5 nhóm trên: trả lời ngắn gọn 1-2 câu, thân thiện, và kết bằng gợi ý hỏi về bệnh lý/thuốc/sức khỏe.\n"
        "Nếu query không thuộc 5 nhóm trên: chỉ trả về đúng chuỗi __NOT_CHITCHAT__."
    )
    user_prompt = f"Lịch sử hội thoại gần nhất:\n{history_text}\n\nQuery hiện tại: {query}"

    try:
        response = llm.chat(system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.0)
    except Exception:
        return None

    cleaned = (response or "").strip()
    if not cleaned or cleaned == "__NOT_CHITCHAT__":
        return None
    return cleaned


def _format_citation(item: Dict) -> str:
    metadata = item.get("metadata") or {}
    source = metadata.get("meta_url") or metadata.get("url")
    section = item.get("section") or metadata.get("section")
    chunk_id = item.get("id") or "unknown"
    parts = [f"id={chunk_id}"]
    if section:
        parts.append(f"section={section}")
    if source:
        parts.append(f"source={source}")
    return ", ".join(parts)


def _fallback_answer(query: str, contexts: List[Dict]) -> str:
    if not contexts:
        return (
            f"Chưa tìm thấy context phù hợp cho câu hỏi: '{query}'. "
            "Hãy thử lại câu hỏi cụ thể hơn."
        )

    lines: List[str] = [f"Câu hỏi: {query}", "", "Thông tin liên quan:"]
    for i, item in enumerate(contexts[:3], start=1):
        text = (item.get("text") or "").strip().replace("\n", " ")
        if len(text) > 280:
            text = f"{text[:280]}..."
        lines.append(f"- [{i}] {text}")

    lines.append("")
    lines.append("Nguồn trích dẫn:")
    for i, item in enumerate(contexts[:3], start=1):
        lines.append(f"- [{i}] {_format_citation(item)}")

    return "\n".join(lines)


def _build_llm_prompt(query: str, contexts: List[Dict]) -> str:
    context_blocks: List[str] = []
    for i, item in enumerate(contexts[:5], start=1):
        text = (item.get("text") or "").strip()
        citation = _format_citation(item)
        context_blocks.append(f"[Context {i}]\n{text}\n[Citation]\n{citation}")
    return (
        f"Câu hỏi người dùng: {query}\n\n"
        "Dưới đây là các context truy xuất được:\n\n"
        f"{chr(10).join(context_blocks)}\n\n"
        "Hãy trả lời ngắn gọn, bám sát context, không được bịa. "
        "Nếu không đủ thông tin, nói rõ là chưa đủ bằng chứng. "
        "Cuối câu trả lời, thêm mục 'Nguồn trích' với citation bao gồm id và source(url) từ context."
    )


def synthesize_answer(
    query: str,
    contexts: List[Dict],
    history: Optional[List[lc_messages.BaseMessage]] = None,
) -> str:
    """
    Generate grounded answer from retrieved contexts.
    Uses optional LLM when configured, otherwise falls back to deterministic synthesis.
    """
    llm = OptionalLLM()
    if not contexts:
        chitchat_response = _build_chitchat_response_with_llm(query=query, llm=llm, history=history)
        if chitchat_response:
            return chitchat_response

    if llm.enabled and contexts:
        system_prompt = (
            "Bạn là trợ lý y khoa sử dụng RAG. "
            "Chỉ được trả lời dựa trên context cung cấp. "
            "Không được đưa ra chân đoạn thay thế bác sĩ."
        )
        user_prompt = _build_llm_prompt(query=query, contexts=contexts)
        try:
            response = llm.chat(system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.1)
            if response:
                return response
        except Exception:
            raise Exception("Error synthesizing answer")

    return _fallback_answer(query=query, contexts=contexts)
