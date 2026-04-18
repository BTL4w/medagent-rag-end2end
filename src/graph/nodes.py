from __future__ import annotations

from src.agents.planner import build_plan
from src.agents.router import route_query
from src.agents.synthesizer import synthesize_answer
from src.graph.state import GraphState
from src.retrieval.hybrid_search import hybrid_retrieve


def router_node(state: GraphState) -> GraphState:
    query = (state.get("query") or "").strip()
    result = route_query(query)
    return {
        "route": result["route"],
        "route_reason": str(result.get("reason", "")),
    }


def orchestrator_entry_node(state: GraphState) -> GraphState:
    route = state.get("route", "unsupported")
    query = (state.get("query") or "").strip()
    top_k = int(state.get("top_k", 5))

    if route != "simple_qa":
        if route == "clarify":
            msg = "Truy vấn của bạn có vẻ chưa đủ rõ để xác định mục tiêu. Bạn có thể cung cấp thêm chi tiết (triệu chứng, thời gian, độ tuổi, bệnh nền, và/hoặc câu hỏi cụ thể) không?"
        elif route == "chitchat":
            msg = "Mình đã hiểu. Hiện tại hệ thống end-to-end này tập trung vào luồng `simple_qa` (hỏi đáp kiến thức y khoa)."
        elif route == "appointment":
            msg = "Chưa triển khai chức năng đặt lịch trong workflow v1 này. Bạn có thể cho mình nội dung câu hỏi y khoa cụ thể để mình hỗ trợ."
        elif route == "complex_qa":
            msg = "Luồng `complex_qa` chưa được triển khai trong workflow v1. Hiện tại bạn có thể thử lại với câu hỏi đơn giản hơn."
        else:
            msg = "Truy vấn nằm ngoài phạm vi hỗ trợ hiện tại. Nếu bạn có câu hỏi y khoa cụ thể, hãy gửi lại rõ hơn."
        return {
            "plan": build_plan(route=route, query=query),
            "answer": msg,
        }

    return {
        "top_k": top_k,
        "plan": build_plan(route=route, query=query),
    }


def retriever_node(state: GraphState) -> GraphState:
    query = (state.get("query") or "").strip()
    top_k = int(state.get("top_k", 5))
    retrieval_filter = state.get("retrieval_filter")
    if not query:
        return {"contexts": [], "answer": "Empty query."}

    contexts = hybrid_retrieve(
        query=query,
        top_k=top_k,
        filter=retrieval_filter,
    )
    citations = [
        {
            "id": item.get("id"),
            "doc_id": item.get("doc_id"),
            "section": item.get("section"),
            "subsection": item.get("subsection"),
        }
        for item in contexts[:top_k]
    ]
    return {"contexts": contexts, "citations": citations}


def synthesizer_node(state: GraphState) -> GraphState:
    query = (state.get("query") or "").strip()
    contexts = state.get("contexts") or []
    answer = synthesize_answer(query=query, contexts=contexts)
    return {"answer": answer}


def finalize_node(state: GraphState) -> GraphState:
    return {
        "final_response": {
            "query": state.get("query"),
            "route": state.get("route"),
            "route_reason": state.get("route_reason"),
            "plan": state.get("plan"),
            "answer": state.get("answer"),
            "citations": state.get("citations", []),
            "contexts": state.get("contexts", []),
            "error": state.get("error"),
        }
    }


def build_nodes() -> dict:
    """Return nodes for router -> orchestrator -> retrieve -> synthesize pipeline."""
    return {
        "router_node": router_node,
        "orchestrator_entry_node": orchestrator_entry_node,
        "retriever_node": retriever_node,
        "synthesizer_node": synthesizer_node,
        "finalize_node": finalize_node,
    }
