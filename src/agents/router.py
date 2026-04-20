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
        "NGUYÊN TẮC QUAN TRỌNG:\n"
        "- Nếu câu hỏi y khoa thiếu thông tin cốt lõi khiến không thể xác định chính xác đối tượng đang hỏi, ưu tiên clarify.\n"
        "- Thông tin cốt lõi gồm: tên thuốc, tên bệnh, triệu chứng chính, đối tượng sử dụng, hành động cần thực hiện.\n"
        "- Chỉ chọn complex_qa khi đã có đủ thực thể hoặc đủ ngữ nghĩa để hiểu rõ câu hỏi.\n\n"
        "Các nhãn:\n\n"
        "1. clarify\n"
        "- Truy vấn thiếu thông tin quan trọng nên chưa thể xác định chính xác ý định hoặc đối tượng.\n"
        "- Dùng đại từ mơ hồ như: thuốc này, thuốc kia, loại đó, bệnh đó, cái này, chỉ số này.\n"
        "- Thiếu tên thuốc, thiếu tên bệnh, thiếu triệu chứng chính, thiếu mục tiêu.\n"
        "- Câu hỏi có vẻ là complex_qa nhưng thiếu thực thể cốt lõi vẫn phải chọn clarify.\n"
        "- Ví dụ: Thuốc này uống chung thuốc kia được không?\n"
        "- Ví dụ: Mang thai dùng loại đó được không?\n"
        "- Ví dụ: Chỉ số này tăng có sao không?\n\n"
        "2. simple_qa\n"
        "- Câu hỏi y khoa đơn giản về một chủ thể rõ ràng.\n"
        "- Hỏi 1 thuốc, 1 bệnh, 1 triệu chứng, 1 xét nghiệm cụ thể.\n\n"
        "3. complex_qa\n"   
        "- Câu hỏi y khoa có nhiều yếu tố cần kết hợp.\n"
        "- Có từ 2 thực thể rõ ràng trở lên: nhiều thuốc, nhiều bệnh, thuốc + bệnh, thuốc + đối tượng đặc biệt.\n"
        "- Có so sánh, tương tác, lựa chọn, nguy cơ, dùng chung.\n"
        "- Các thực thể phải đủ rõ để xử lý.\n"
        "- Ví dụ: Paracetamol và Ibuprofen dùng chung được không?\n"
        "- Ví dụ: Người tiểu đường uống Metformin có dùng Prednisolone được không?\n\n"
        "4. appointment\n"
        "- Đặt lịch, đổi lịch, hủy lịch, tra cứu lịch hẹn, đăng ký khám.\n\n"
        "5. chitchat\n"
        "- Chào hỏi, cảm ơn, xã giao.\n"
        "- Hỏi về chatbot, trợ lý, hệ thống.\n"
        "- Ví dụ: bạn là ai, bạn làm được gì, hệ thống dùng sao.\n\n"
        "6. unsupported\n"
        "- Ngoài phạm vi y tế/lịch hẹn/chatbot.\n"
        "- Ví dụ: code Python, giá vàng, bóng đá.\n\n"
        "Quy tắc ưu tiên bắt buộc:\n"
        "- Nếu có đặt lịch -> appointment.\n"
        "- Nếu hỏi chatbot/hệ thống -> chitchat.\n"
        "- Nếu ngoài domain -> unsupported.\n"
        "- Nếu có đại từ mơ hồ hoặc thiếu thực thể quan trọng -> clarify.\n"
        "- Nếu nhiều yếu tố rõ ràng, đủ thông tin -> complex_qa.\n"
        "- Nếu một chủ thể rõ ràng -> simple_qa.\n\n"
        "Ví dụ:\n"
        "- Thuốc này uống chung với thuốc kia được không nếu tôi đang bị bệnh nền? -> clarify\n"
        "- Mang thai dùng loại đó được không -> clarify\n"
        "- Paracetamol và Ibuprofen dùng chung được không -> complex_qa\n"
        "- Metformin là thuốc gì -> simple_qa\n"
        "- Bạn làm được gì -> chitchat\n"
        "- Đặt lịch khám ngày mai -> appointment\n"
        "- Viết code Python -> unsupported\n\n"
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
