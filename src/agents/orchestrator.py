from __future__ import annotations

from typing import Any, Dict, Optional

from langchain_core import messages

from src.graph.graph_builder import build_graph


def run_agent(
    query: str,
    *,
    top_k: int = 5,
    retrieval_filter: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run end-to-end workflow for current query.

    Flow:
    - route query
    - if route is `simple_qa`: retrieve contexts -> synthesize answer
    - finalize and return response for user
    """
    graph = build_graph()
    state = {
        "query": query,
        "top_k": top_k,
        "retrieval_filter": retrieval_filter,
        "messages": [messages.HumanMessage(content=query)],
    }
    result = graph.invoke(state)
    return result.get("final_response", result)
