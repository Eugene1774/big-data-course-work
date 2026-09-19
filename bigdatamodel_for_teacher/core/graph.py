from __future__ import annotations

from functools import lru_cache
from typing import Any, AsyncIterator, Dict, List

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from core.nodes import (
    assessment_assistant_node,
    competition_advisor_node,
    error_node,
    ERROR_NODE_NAME,
    instructor_assistant_node,
    project_coach_node,
    student_learning_tutor_node,
)
from core.router import route_supervisor, supervisor_node
from core.state import AgentState


def _coerce_message(message: Any) -> AnyMessage:
    if isinstance(message, (AIMessage, HumanMessage, SystemMessage)):
        return message

    if isinstance(message, dict):
        role = str(message.get("role", "")).strip()
        content = str(message.get("content", "")).strip()
        if role == "assistant":
            return AIMessage(content=content)
        if role == "system":
            return SystemMessage(content=content)
        return HumanMessage(content=content)

    return HumanMessage(content=str(message))


def build_agent_state(
    portal_role: str,
    messages: List[Any],
    context_data: Dict[str, Any] | None = None,
    active_agent: str = "",
    next_agent: str = "",
    next_step: str = "",
) -> AgentState:
    return AgentState(
        messages=[_coerce_message(message) for message in messages],
        user_role=portal_role if portal_role in {"student", "teacher", "admin"} else "student",
        active_agent=active_agent,
        next_agent=next_agent,
        context_data=dict(context_data or {}),
        next_step=next_step,
    )


@lru_cache(maxsize=1)
def get_agent_app():
    workflow = StateGraph(AgentState)
    workflow.add_node("Supervisor", supervisor_node)
    workflow.add_node("Student Learning Tutor", student_learning_tutor_node)
    workflow.add_node("Project Coach", project_coach_node)
    workflow.add_node("Competition Advisor", competition_advisor_node)
    workflow.add_node("Assessment Assistant", assessment_assistant_node)
    workflow.add_node("Instructor Assistant", instructor_assistant_node)
    workflow.add_node(ERROR_NODE_NAME, error_node)

    workflow.add_edge(START, "Supervisor")
    workflow.add_conditional_edges(
        "Supervisor",
        route_supervisor,
        {
            "Student Learning Tutor": "Student Learning Tutor",
            "Project Coach": "Project Coach",
            "Competition Advisor": "Competition Advisor",
            "Assessment Assistant": "Assessment Assistant",
            "Instructor Assistant": "Instructor Assistant",
            ERROR_NODE_NAME: ERROR_NODE_NAME,
        },
    )

    workflow.add_edge("Student Learning Tutor", END)
    workflow.add_edge("Project Coach", END)
    workflow.add_edge("Competition Advisor", END)
    workflow.add_edge("Assessment Assistant", END)
    workflow.add_edge("Instructor Assistant", END)
    workflow.add_edge(ERROR_NODE_NAME, END)
    return workflow.compile()


async def stream_agent_app(state: AgentState) -> AsyncIterator[Dict[str, Any]]:
    app = get_agent_app()
    async for event in app.astream(state, stream_mode="values"):
        yield event
