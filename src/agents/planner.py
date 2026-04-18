from __future__ import annotations

from typing import Dict, List


def build_plan(route: str, query: str) -> Dict[str, object]:
    """Build structured execution plan, extensible for future complex_qa."""
    if route == "simple_qa":
        steps: List[str] = [
            "retrieve medical contexts",
            "synthesize grounded answer",
        ]
    elif route == "complex_qa":
        steps = [
            "decompose question",
            "multi-step retrieve evidence",
            "aggregate evidence",
            "synthesize final answer",
        ]
    elif route == "appointment":
        steps = [
            "extract booking information",
            "call appointment tool",
            "format tool response",
        ]
    else:
        steps = ["return direct response"]

    return {
        "route": route,
        "query": query,
        "steps": steps,
    }
