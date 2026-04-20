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
        "- Chào hỏi, cảm ơn, xã giao, hội thoại thông thường.\n"
        "- Câu hỏi về chính trợ lý hoặc hệ thống.\n"
        "- Bao gồm: bạn là ai, bạn tên gì, bạn làm được gì, bạn hỗ trợ gì, có thể giúp gì.\n"
        "- Bao gồm: hệ thống này dùng như nào, sử dụng ra sao, cách dùng chatbot, hướng dẫn sử dụng.\n"
        "- Bao gồm: hello, hi, xin chào, cảm ơn, bye.\n"
        "- MỌI câu hỏi meta về chatbot/assistant/system đều là chitchat, KHÔNG phải unsupported.\n\n"
        "5. unsupported\n"
        "- Yêu cầu ngoài phạm vi hỗ trợ và KHÔNG phải hội thoại xã giao.\n"
        "- Không liên quan y tế, lịch hẹn, chatbot hoặc hệ thống.\n"
        "- Thuộc lĩnh vực khác như tài chính, lập trình, game, giải trí, chính trị, pháp lý.\n"
        "- Ví dụ: viết code Python, dự đoán giá vàng, ai thắng World Cup.\n\n"
        "6. clarify\n"
        "- Truy vấn quá mơ hồ, thiếu thông tin, chưa xác định được mục tiêu.\n"
        "- Không đủ dữ liệu để quyết định người dùng muốn hỏi kiến thức hay muốn thực hiện hành động.\n\n"
        "Quy tắc ưu tiên:\n"
        "- Nếu truy vấn hỏi về chatbot, trợ lý hoặc cách dùng hệ thống -> chitchat.\n"
        "- Nếu truy vấn có yêu cầu hành động đặt lịch -> appointment.\n"
        "- Nếu vừa hỏi kiến thức vừa đặt lịch -> appointment.\n"
        "- Nếu truy vấn liên quan từ 2 yếu tố trở lên (nhiều thuốc, nhiều bệnh, mang thai + thuốc, bệnh nền + thuốc, so sánh lựa chọn) -> complex_qa.\n"
        "- Nếu là câu hỏi y khoa đơn giản một chủ thể -> simple_qa.\n"
        "- Nếu không chắc chắn do thiếu thông tin -> clarify.\n"
        "- Chỉ khi hoàn toàn ngoài y tế/lịch hẹn/chatbot mới là unsupported.\n\n"
        "Ví dụ:\n"
        "- Bạn là ai -> chitchat\n"
        "- Bạn làm được những gì? -> chitchat\n"
        "- Hệ thống này sử dụng như nào -> chitchat\n"
        "- Hello -> chitchat\n"
        "- Viết code Python -> unsupported\n"
        "- Paracetamol là thuốc gì -> simple_qa\n"
        "- Paracetamol và Ibuprofen dùng chung được không -> complex_qa\n"
        "- Đặt lịch khám -> appointment\n\n"
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
