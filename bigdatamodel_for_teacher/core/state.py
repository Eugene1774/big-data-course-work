from __future__ import annotations

from typing import Annotated, Any, Literal

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import NotRequired, TypedDict


class AgentState(TypedDict):
    """Shared state contract for the multi-agent coaching workflow."""

    messages: Annotated[list[AnyMessage], add_messages]
    user_role: Literal["student", "teacher", "admin"]
    active_agent: str
    next_agent: str
    context_data: dict
    next_step: str
    kb_context: NotRequired[Any]
    probing_strategy: NotRequired[str]
    retrieved_heterogeneous_subgraph: NotRequired[dict]
    rubric_score_results: NotRequired[list[dict[str, Any]]]
    active_rules: NotRequired[list[str]]
