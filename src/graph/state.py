from __future__ import annotations

from typing import Annotated, Any, Dict, List, Literal, TypedDict

from langchain_core import messages
from langgraph.graph.message import add_messages


class GraphState(TypedDict, total=False):
    query: str
    route: Literal["simple_qa", "complex_qa", "appointment", "chitchat", "unsupported", "clarify"]
    route_reason: str
    top_k: int
    messages: Annotated[List[messages.BaseMessage], add_messages]
    retrieval_filter: Dict[str, Any] | None
    # Sub-queries produced by planner for complex_qa (each used for hybrid_retrieve).
    sub_queries: List[str]
    contexts: List[Dict[str, Any]]
    citations: List[Dict[str, Any]]
    answer: str
    appointment_result: Dict[str, Any]
    final_response: Dict[str, Any]
    error: str
