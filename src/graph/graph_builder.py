from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from src.graph.nodes import build_nodes
from src.graph.state import GraphState


def _route_after_orchestrator(state: GraphState) -> str:
    if state.get("route") in ("simple_qa", "complex_qa"):
        return "retriever_node"
    return "finalize_node"


def build_graph():
    """
    Build workflow:
    query -> router_node -> orchestrator_entry_node -> retriever_node -> synthesizer_node -> finalize_node

    - `simple_qa`: single hybrid retrieve -> synthesize.
    - `complex_qa`: planner decomposes into sub-queries -> retrieve each -> merge -> synthesize.
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
            "finalize_node": "finalize_node",
        },
    )
    graph.add_edge("retriever_node", "synthesizer_node")
    graph.add_edge("synthesizer_node", "finalize_node")
    graph.add_edge("finalize_node", END)

    return graph.compile()
