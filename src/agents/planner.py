from __future__ import annotations

import os
import re
from typing import List, Optional

from src.agents.llm import OptionalLLM


def _parse_subquery_lines(text: str, *, max_sub: int) -> List[str]:
    lines: List[str] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        for prefix in ("-", "*", "•"):
            if line.startswith(prefix):
                line = line[1:].strip()
        line = re.sub(r"^\d+[\).\s]+", "", line).strip()
        if len(line) >= 3:
            lines.append(line)
        if len(lines) >= max_sub:
            break
    return lines


def decompose_to_subqueries(
    query: str,
    *,
    max_sub: Optional[int] = None,
) -> List[str]:
    """
    Split a complex medical question into short retrieval-friendly sub-queries.

    Uses the planner LLM when `OPENAI_API_KEY` is set; otherwise returns the
    original query as a single step so retrieval can still run.
    """
    q = (query or "").strip()
    if not q:
        return []

    limit = max_sub if max_sub is not None else int(os.getenv("COMPLEX_QA_MAX_SUB", "5"))
    limit = max(1, min(limit, 12))

    llm = OptionalLLM(model_name=os.getenv("PLANNER_MODEL", None))
    if not llm.enabled:
        return [q]

    system_prompt = (
        "Bạn là module lập kế hoạch cho hệ thống hỏi đáp y khoa sử dụng RAG.\n"
        "Nhiệm vụ: tách một câu hỏi phức tạp thành các câu hỏi con ngắn, đủ cụ thể để truy xuất tài liệu độc lập.\n"
        f"Tối đa {limit} câu hỏi con.\n"
        "Quy tắc:\n"
        "- Mỗi dòng chỉ một câu hỏi con, không đánh số, không bullet, không giải thích.\n"
        "- Giữ ngôn ngữ giống câu hỏi gốc (Tiếng Việt).\n"
        "- Chỉ tách theo các ý xuất hiện trực tiếp trong câu hỏi gốc.\n"
        "- Mỗi câu hỏi con chỉ chứa một ý chính.\n"
        "- Ưu tiên câu ngắn, đơn giản, sát cách hỏi tự nhiên của người dùng.\n"
        "- Không thêm chi tiết mới không có trong câu hỏi gốc.\n"
        "- Không suy diễn nhu cầu ẩn.\n"
        "- Không mở rộng sang cơ chế bệnh, phác đồ, liều dùng, chống chỉ định, tác dụng phụ, thuốc thay thế, hướng dẫn điều trị... nếu người dùng không hỏi.\n"
        "- Nếu câu hỏi nhắc đến thuốc, chỉ hỏi lại đúng ý về công dụng/chỉ định/khả năng điều trị như câu gốc.\n"
        "- Nếu câu gốc chỉ có một ý thì giữ thành một câu hỏi con.\n"
        "- Nếu có nhiều ý nối bằng 'và', 'hoặc', 'hay', 'có ... không' thì tách thành nhiều câu hỏi con tương ứng.\n"
        "Ví dụ:\n"
        "Câu hỏi: Nguyên nhân gây ra bệnh bạch hầu là gì và thuốc Erythromycin có chữa được không?\n"
        "Kết quả:\n"
        "Nguyên nhân gây ra bệnh bạch hầu là gì?\n"
        "Thuốc Erythromycin có chữa được bệnh bạch hầu không?\n"

        "Câu hỏi: Triệu chứng của bệnh sốt xuất huyết và cách điều trị?"
        "Kết quả:\n"
        "Triệu chứng của bệnh sốt xuất huyết là gì?\n"
        "Cách điều trị bệnh sốt xuất huyết là gì?"
    )
    user_prompt = f"Câu hỏi gốc:\n{q}\n"
    raw = llm.chat(system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.0)
    subs = _parse_subquery_lines(raw or "", max_sub=limit)
    if not subs:
        return [q]
    return subs
