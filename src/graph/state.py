from __future__ import annotations

from typing import Any, Dict, List, Literal, TypedDict

from langchain_core import messages


class GraphState(TypedDict, total=False):
    query: str
    route: Literal["simple_qa", "complex_qa", "appointment", "chitchat", "unsupported", "clarify"]
    route_reason: str
    top_k: int
    messages: List[messages.BaseMessage]
    retrieval_filter: Dict[str, Any] | None
    plan: Dict[str, Any] | None
    contexts: List[Dict[str, Any]]
    citations: List[Dict[str, Any]]
    answer: str
    final_response: Dict[str, Any]
    error: str
