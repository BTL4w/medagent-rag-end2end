from __future__ import annotations

import os
from typing import Dict


def route_query(query: str) -> Dict[str, object]:
    """
    LLM-based query router.
    Output contract: only returns a single label in `route`.
    """
    from src.agents.llm import OptionalLLM

    allowed_labels = {
        "simple_qa",
        "complex_qa",
        "appointment",
        "chitchat",
        "unsupported",
        "clarify",
    }

    llm = OptionalLLM(model_name=os.getenv("ROUTER_MODEL", None))
    if not llm.enabled:
        raise ValueError("Missing OPENAI_API_KEY. Set it in environment or .env for router LLM.")

    system_prompt = (
        "Bạn là bộ định tuyến truy vấn cho hệ thống trợ lý y khoa.\n\n"
        "Nhiệm vụ:\n"
        "Phân loại mỗi truy vấn người dùng thành đúng một nhãn phù hợp nhất.\n\n"
        "Các nhãn:\n\n"
        "1. simple_qa\n"
        "- Câu hỏi kiến thức y khoa đơn giản về một chủ thể duy nhất.\n"
        "- Có thể trả lời trực tiếp bằng một lần truy xuất thông tin.\n"
        "- Không cần so sánh, không cần kết hợp nhiều yếu tố, không cần suy luận nhiều bước.\n\n"
        "2. complex_qa\n"
        "- Câu hỏi y khoa cần tổng hợp nhiều thông tin, so sánh, suy luận nhiều bước hoặc nhiều ý.\n"
        "- Liên quan từ 2 thực thể trở lên như nhiều thuốc, nhiều bệnh, thuốc và bệnh, thuốc và đối tượng đặc biệt.\n"
        "- Bao gồm các tình huống cần cân nhắc như mang thai, trẻ em, người cao tuổi, bệnh nền, đang dùng thuốc khác.\n"
        "- Bao gồm các câu hỏi như có dùng chung được không, thuốc nào tốt hơn, nên chọn gì, nguy cơ gì.\n\n"
        "3. appointment\n"
        "- Yêu cầu đặt lịch, đổi lịch, hủy lịch, tra cứu lịch hẹn, đăng ký khám.\n"
        "- Bất kỳ truy vấn nào có mục tiêu chính là thực hiện hành động liên quan lịch hẹn.\n\n"
        "4. chitchat\n"
        "- Chào hỏi, cảm ơn, xã giao, hỏi về hệ thống, hội thoại thông thường.\n\n"
        "5. unsupported\n"
        "- Ngoài phạm vi hỗ trợ của trợ lý y khoa.\n"
        "- Nội dung không liên quan y tế hoặc lịch hẹn.\n"
        "- Yêu cầu thuộc lĩnh vực khác như tài chính, lập trình, giải trí, chính trị, pháp lý.\n\n"
        "6. clarify\n"
        "- Truy vấn quá mơ hồ, thiếu thông tin, chưa xác định được mục tiêu.\n"
        "- Không đủ dữ liệu để quyết định người dùng muốn hỏi kiến thức hay muốn thực hiện hành động.\n\n"
        "Quy tắc ưu tiên:\n"
        "- Nếu truy vấn có yêu cầu hành động đặt lịch → appointment.\n"
        "- Nếu vừa hỏi kiến thức vừa đặt lịch → appointment.\n"
        "- Nếu truy vấn liên quan từ 2 yếu tố trở lên (nhiều thuốc, nhiều bệnh, mang thai + thuốc, bệnh nền + thuốc, so sánh lựa chọn) → complex_qa.\n"
        "- Nếu không chắc chắn do thiếu thông tin → clarify.\n"
        "- Nếu ngoài phạm vi hỗ trợ → unsupported.\n\n"
        "Output:\n"
        "- Chỉ trả về đúng một nhãn duy nhất từ danh sách trên.\n"
        "- Không giải thích."
    )
    user_prompt = query.strip()

    raw = llm.chat(system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.0)
    label = (raw or "").strip()

    # Post-process chỉ để đảm bảo label đúng tập cho phép
    label_lower = label.lower()
    for candidate in allowed_labels:
        if candidate == label_lower:
            return {"route": candidate}

    # Trường hợp model trả thêm ký tự/dòng, lấy label xuất hiện trong chuỗi
    for candidate in allowed_labels:
        if candidate in label_lower:
            return {"route": candidate}

    return {"route": "clarify"}
