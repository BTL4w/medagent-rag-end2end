from __future__ import annotations

from typing import Any, Dict, List

from src.agents.planner import decompose_to_subqueries
from src.agents.router import route_query
from src.agents.synthesizer import synthesize_answer
from src.agents.tools import handle_appointment_request
from src.graph.state import GraphState
from src.retrieval.hybrid_search import hybrid_retrieve


def _dedupe_contexts(items: List[Dict[str, Any]], max_items: int) -> List[Dict[str, Any]]:
    seen: set[Any] = set()
    out: List[Dict[str, Any]] = []
    for item in items:
        text = (item.get("text") or "")[:400]
        key = item.get("id") if item.get("id") is not None else hash(text)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
        if len(out) >= max_items:
            break
    return out


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

    if route == "complex_qa":
        sub_queries = decompose_to_subqueries(query)
        return {
            "top_k": top_k,
            "sub_queries": sub_queries,
        }

    if route == "chitchat":
        # Let synthesizer handle social intents directly via LLM.
        return {}

    if route != "simple_qa":
        if route == "clarify":
            msg = "Câu hỏi của bạn có vẻ chưa đủ rõ để xác định mục tiêu. Bạn có thể cung cấp thêm chi tiết (triệu chứng, thời gian, độ tuổi, bệnh nền, và/hoặc câu hỏi cụ thể) không?"
        elif route == "appointment":
            booking_result = handle_appointment_request(query=query)
            return {
                "answer": booking_result.get("message", "Đã xử lý yêu cầu đặt lịch."),
                "appointment_result": booking_result,
            }
        else:
            msg = "Truy vấn nằm ngoài phạm vi hỗ trợ hiện tại. Nếu bạn có câu hỏi y khoa cụ thể, hãy gửi lại rõ hơn."
        return {"answer": msg}

    return {"top_k": top_k}


def retriever_node(state: GraphState) -> GraphState:
    query = (state.get("query") or "").strip()
    route = state.get("route", "unsupported")
    top_k = int(state.get("top_k", 5))
    retrieval_filter = state.get("retrieval_filter")
    if not query:
        return {"contexts": [], "answer": "Empty query."}

    if route == "complex_qa":
        subs = state.get("sub_queries") or [query]
        merged: List[Dict[str, Any]] = []
        per_sub_k = max(1, top_k)
        max_merged = min(40, max(top_k * max(2, len(subs)), top_k))
        for sq in subs:
            sq = (sq or "").strip()
            if not sq:
                continue
            part = hybrid_retrieve(
                query=sq,
                top_k=per_sub_k,
                filter=retrieval_filter,
            )
            merged.extend(part)
        contexts = _dedupe_contexts(merged, max_items=max_merged)
    else:
        contexts = hybrid_retrieve(
            query=query,
            top_k=top_k,
            filter=retrieval_filter,
        )

    cite_limit = top_k if route != "complex_qa" else min(len(contexts), max(top_k * 3, 15))
    cite_limit = max(1, cite_limit)
    citations = [
        {
            "id": item.get("id"),
            "doc_id": item.get("doc_id"),
            "section": item.get("section"),
            "subsection": item.get("subsection"),
        }
        for item in contexts[:cite_limit]
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
            "sub_queries": state.get("sub_queries", []),
            "answer": state.get("answer"),
            "citations": state.get("citations", []),
            "contexts": state.get("contexts", []),
            "appointment_result": state.get("appointment_result"),
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
