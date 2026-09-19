from __future__ import annotations

from functools import lru_cache
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.store.base import BaseStore

from core.nodes import (
    assessment_assistant_node,
    competition_advisor_node,
    error_node,
    ERROR_NODE_NAME,
    force_conclusion_node,
    instructor_assistant_node,
    project_coach_node,
    project_reviewer_node,
    student_learning_tutor_node,
)
from core.router import route_supervisor, supervisor_node
from core.state import AgentState

# 决策 3：工具循环安全阀阈值
MAX_TOOL_INVOCATIONS = 5
FORCE_CONCLUSION_NODE = "force_conclusion"


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
        tool_invoke_count=0,
    )


class TrackedToolNode(ToolNode):
    """继承原生 ToolNode，安全注入计数器递增逻辑。

    注意：LangGraph 通过 RunnableCallable 直接执行 _func/_afunc，
    重写 __call__ 不会被触发，因此必须重写这两个内部方法。
    """

    def _func(
        self,
        input: Union[list[AnyMessage], dict[str, Any]],
        config: RunnableConfig,
        *,
        store: Optional[BaseStore],
    ) -> Any:
        result = super()._func(input, config, store=store)
        return self._merge_tracking(input, result)

    async def _afunc(
        self,
        input: Union[list[AnyMessage], dict[str, Any]],
        config: RunnableConfig,
        *,
        store: Optional[BaseStore],
    ) -> Any:
        result = await super()._afunc(input, config, store=store)
        return self._merge_tracking(input, result)

    @staticmethod
    def _merge_tracking(input: Any, result: Any) -> Any:
        if not isinstance(result, dict) or not isinstance(input, dict):
            return result
        # context_data 无 reducer：必须整体深合并后返回，避免丢键
        merged_context = dict(input.get("context_data") or {})
        previous_count = int(input.get("tool_invoke_count", 0) or 0)
        return {
            **result,
            # 每次成功执行一轮工具，计数器 +1
            "tool_invoke_count": previous_count + 1,
            "context_data": merged_context,
        }


def custom_tools_condition(state: AgentState) -> str:
    """自定义条件边：ReAct 循环的方向盘 + 死循环安全阀。"""

    messages = state.get("messages", [])
    last_message = messages[-1] if messages else None

    # 如果大模型决定调用工具
    if isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None):
        # 决策 3：拦截 5 次以上的无意义循环
        if int(state.get("tool_invoke_count", 0) or 0) >= MAX_TOOL_INVOCATIONS:
            return "force_conclusion"
        return "tools"

    # 如果大模型给出最终文字回答，结束当前流
    return END


def _agent_toolsets() -> Dict[str, list]:
    from logic_adapter import (
        get_ability_report,
        get_competition_score,
        search_hypergraph,
        search_knowledge_cards,
    )

    return {
        "Student Learning Tutor": [search_knowledge_cards],
        "Project Coach": [search_hypergraph, search_knowledge_cards],
        "Competition Advisor": [get_competition_score, search_knowledge_cards],
        "Assessment Assistant": [get_ability_report],
        "Instructor Assistant": [get_ability_report],
        "Project Reviewer": [search_knowledge_cards],
    }


@lru_cache(maxsize=1)
def get_agent_app():
    workflow = StateGraph(AgentState)
    workflow.add_node("Supervisor", supervisor_node)
    workflow.add_node("Student Learning Tutor", student_learning_tutor_node)
    workflow.add_node("Project Coach", project_coach_node)
    workflow.add_node("Competition Advisor", competition_advisor_node)
    workflow.add_node("Assessment Assistant", assessment_assistant_node)
    workflow.add_node("Instructor Assistant", instructor_assistant_node)
    workflow.add_node("Project Reviewer", project_reviewer_node)
    workflow.add_node(ERROR_NODE_NAME, error_node)
    workflow.add_node(FORCE_CONCLUSION_NODE, force_conclusion_node)

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
            "Project Reviewer": "Project Reviewer",
            ERROR_NODE_NAME: ERROR_NODE_NAME,
        },
    )

    # 为每个智能体织入 ReAct 循环：专属工具节点 + 安全阀条件边
    for agent_name, tools in _agent_toolsets().items():
        tools_node_name = f"{agent_name} Tools"
        workflow.add_node(tools_node_name, TrackedToolNode(tools, name=tools_node_name))
        workflow.add_conditional_edges(
            agent_name,
            custom_tools_condition,
            {
                "tools": tools_node_name,
                "force_conclusion": FORCE_CONCLUSION_NODE,
                END: END,
            },
        )
        # 工具执行完毕后，回旋到大模型进行评估（闭环）
        workflow.add_edge(tools_node_name, agent_name)

    workflow.add_edge(ERROR_NODE_NAME, END)
    workflow.add_edge(FORCE_CONCLUSION_NODE, END)
    return workflow.compile()


async def stream_agent_app(state: AgentState) -> AsyncIterator[Dict[str, Any]]:
    app = get_agent_app()
    async for event in app.astream(state, stream_mode="values"):
        yield event
