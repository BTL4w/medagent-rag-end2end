from __future__ import annotations

import os
import atexit
from typing import Optional

from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph

from src.graph.nodes import build_nodes
from src.graph.state import GraphState

load_dotenv()

_checkpointer: Optional[PostgresSaver] = None
_memory_url: Optional[str] = None
_checkpointer_cm = None
_in_memory_saver: Optional[InMemorySaver] = None


def _route_after_orchestrator(state: GraphState) -> str:
    route = state.get("route")
    if route in ("simple_qa", "complex_qa"):
        return "retriever_node"
    if route == "chitchat":
        return "synthesizer_node"
    return "finalize_node"


def _get_checkpointer():
    global _checkpointer, _memory_url, _checkpointer_cm, _in_memory_saver

    backend = os.getenv("CHECKPOINTER_BACKEND", "memory").strip().lower()
    if backend != "postgres":
        if _in_memory_saver is None:
            _in_memory_saver = InMemorySaver()
        return _in_memory_saver

    memory_url = os.getenv("MEMORY_URL")
    if not memory_url:
        if _in_memory_saver is None:
            _in_memory_saver = InMemorySaver()
        return _in_memory_saver

    if _checkpointer is None or _memory_url != memory_url:
        if _checkpointer_cm is not None:
            _checkpointer_cm.__exit__(None, None, None)
            _checkpointer_cm = None

        try:
            saver_or_cm = PostgresSaver.from_conn_string(memory_url)
            if hasattr(saver_or_cm, "__enter__") and hasattr(saver_or_cm, "__exit__"):
                _checkpointer_cm = saver_or_cm
                _checkpointer = saver_or_cm.__enter__()
                atexit.register(lambda: _checkpointer_cm and _checkpointer_cm.__exit__(None, None, None))
            else:
                _checkpointer = saver_or_cm
            _memory_url = memory_url
            _in_memory_saver = None
        except Exception:
            # Keep conversational memory working even if Postgres checkpointer fails.
            if _in_memory_saver is None:
                _in_memory_saver = InMemorySaver()
            return _in_memory_saver

    return _checkpointer


def build_graph():
    """
    Build workflow:
    query -> router_node -> orchestrator_entry_node -> retriever_node -> synthesizer_node -> finalize_node
                                                    -> synthesizer_node (for chitchat) -> finalize_node

    - `simple_qa`: single hybrid retrieve -> synthesize.
    - `complex_qa`: planner decomposes into sub-queries -> retrieve each -> merge -> synthesize.
    - `chitchat`: skip retrieval and let synthesizer answer directly.
    Other routes short-circuit to a direct answer in orchestrator_entry_node.
    """
    nodes = build_nodes()

    graph = StateGraph(GraphState)
    graph.add_node("router_node", nodes["router_node"])
    graph.add_node("orchestrator_entry_node", nodes["orchestrator_entry_node"])
    graph.add_node("retriever_node", nodes["retriever_node"])
    graph.add_node("synthesizer_node", nodes["synthesizer_node"])
    graph.add_node("finalize_node", nodes["finalize_node"])

    graph.add_edge(START, "router_node")
    graph.add_edge("router_node", "orchestrator_entry_node")
    graph.add_conditional_edges(
        "orchestrator_entry_node",
        _route_after_orchestrator,
        {
            "retriever_node": "retriever_node",
            "synthesizer_node": "synthesizer_node",
            "finalize_node": "finalize_node",
        },
    )
    graph.add_edge("retriever_node", "synthesizer_node")
    graph.add_edge("synthesizer_node", "finalize_node")
    graph.add_edge("finalize_node", END)

    checkpointer = _get_checkpointer()
    return graph.compile(checkpointer=checkpointer)
