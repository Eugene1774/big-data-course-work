from __future__ import annotations

import logging
from typing import Dict, Literal

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

import logic_adapter
from config.api_config import get_llm_config
from core.nodes import ERROR_NODE_NAME
from core.state import AgentState


LOGGER = logging.getLogger(__name__)

SUPERVISOR_NAME = "Supervisor"
STUDENT_ALLOWED_ROUTES = (
    "Student Learning Tutor",
    "Project Coach",
    "Competition Advisor",
    "Project Reviewer",
)
TEACHER_ALLOWED_ROUTES = (
    "Instructor Assistant",
    "Assessment Assistant",
)

# 路由出口严格对齐现有图节点名
AgentName = Literal[
    "Student Learning Tutor",
    "Project Coach",
    "Competition Advisor",
    "Assessment Assistant",
    "Instructor Assistant",
    "Project Reviewer",
]

ROLE_DEFAULT_AGENT: Dict[str, str] = {
    "student": "Student Learning Tutor",
    "teacher": "Instructor Assistant",
    "admin": "Assessment Assistant",
}

ROUTER_SYSTEM_PROMPT = """你是一个高精度的流量分发中枢。请根据对话上下文与用户最后一条消息的语义，选择最匹配的领域专家：
- 商业模式设计、痛点诊断、风险评估、BP撰写、答辩模拟、财务/市场分析协作 -> Project Coach
- 项目评审、评审反馈、按维度分项评价、给出证据缺口和下一步建议 -> Project Reviewer
- 比赛要求、赛事规则、评分标准、竞赛准备 -> Competition Advisor
- 概念解释、术语定义、通用学习辅导、追问澄清 -> Student Learning Tutor
- 学生能力评价、评估报告、结构化诊断 -> Assessment Assistant
- 教学干预计划、教师教学建议、班级辅导策略 -> Instructor Assistant

判定原则：
1. 只依据语义意图判断，不要臆测用户未表达的需求。
2. 当用户在追问或要求澄清上一轮回答时，优先 Student Learning Tutor。
3. 涉及"答辩、路演、评委、投资人提问、帮我写/润色某部分"等协同意图时，一律 Project Coach。
4. 出现"评审/评审反馈/分项评价/按……维度评价"等评审意图时，优先分给 Project Reviewer，不要分给 Project Coach；即使同一消息中还提到风险、证据缺口等词，只要意图是"对项目做分项评审"，就选 Project Reviewer。
"""


class RouteDecision(BaseModel):
    """LLM 结构化路由决策，确保 100% 对接现有节点。"""

    next_agent: AgentName = Field(
        description="根据用户最后一条消息的语义，选择最匹配的领域专家"
    )


def _is_route_allowed_for_role(user_role: str, route_name: str) -> bool:
    if route_name == ERROR_NODE_NAME:
        return True
    if user_role == "student":
        return route_name in STUDENT_ALLOWED_ROUTES
    if user_role == "teacher":
        return route_name in TEACHER_ALLOWED_ROUTES
    if user_role == "admin":
        return route_name == "Assessment Assistant"
    return False


def _role_default_agent(user_role: str) -> str:
    return ROLE_DEFAULT_AGENT.get(user_role, "Student Learning Tutor")


def _classify_intent(state: AgentState) -> str:
    """LLM 语义分类；任何失败都回退到角色默认智能体。"""

    user_role = str(state.get("user_role", "student"))
    fallback = _role_default_agent(user_role)
    try:
        llm_config = get_llm_config()
        llm = ChatOpenAI(
            model=llm_config["model"],
            api_key=llm_config["api_key"],
            base_url=llm_config["base_url"],
            temperature=0.1,
            timeout=30.0,
            max_retries=1,
        )
        router = llm.with_structured_output(RouteDecision, method="function_calling")
        decision = router.invoke(
            [("system", ROUTER_SYSTEM_PROMPT)] + list(state.get("messages", []))
        )
        candidate = str(getattr(decision, "next_agent", "") or "").strip()
        if candidate and _is_route_allowed_for_role(user_role, candidate):
            return candidate
        LOGGER.warning("LLM route %r not allowed for role %r, fallback.", candidate, user_role)
        return fallback
    except Exception:
        LOGGER.exception("LLM routing failed, falling back to role default.")
        return fallback


def route_supervisor(state: AgentState) -> str:
    """条件边读取器：返回 supervisor_node 已写入的路由结果。"""

    user_role = str(state.get("user_role", "student"))
    target = str(state.get("next_agent", "") or "").strip()
    if target and (target == ERROR_NODE_NAME or _is_route_allowed_for_role(user_role, target)):
        return target
    return _role_default_agent(user_role)


async def supervisor_node(state: AgentState) -> Dict[str, object]:
    """守卫拦截 -> 显式 route_hint/RBAC -> LLM 语义路由，并重置工具循环计数器。"""

    user_role = str(state.get("user_role", "student"))
    context_data = dict(state.get("context_data") or {})

    messages = list(state.get("messages", []))
    user_text = ""
    for message in reversed(messages):
        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            content = message.get("content", "")
        if isinstance(content, list):
            content = " ".join(str(item) for item in content if item is not None)
        if getattr(message, "type", "") == "human" or (
            isinstance(message, dict) and message.get("role") == "user"
        ):
            user_text = str(content or "").strip()
            if user_text:
                break

    guardrail_response = logic_adapter.apply_input_guardrails(user_text, context_data)

    extra_messages: list = []
    if guardrail_response:
        target_agent = ERROR_NODE_NAME
        context_data["route_hint"] = ERROR_NODE_NAME
    else:
        route_hint = str(context_data.get("route_hint", "")).strip()
        if route_hint == ERROR_NODE_NAME:
            target_agent = ERROR_NODE_NAME
        elif route_hint and _is_route_allowed_for_role(user_role, route_hint):
            target_agent = route_hint
        elif route_hint:
            target_agent = ERROR_NODE_NAME
        else:
            target_agent = _classify_intent({**state, "context_data": context_data})

        previous_agent = str(state.get("next_agent", "") or "").strip()
        role_map = {
            "Student Learning Tutor": "学习辅导智能体",
            "Project Coach": "项目教练智能体",
            "Competition Advisor": "竞赛顾问智能体",
            "Project Reviewer": "评审反馈智能体",
        }
        if (
            previous_agent
            and previous_agent != target_agent
            and target_agent in role_map
        ):
            action_map = {
                "Student Learning Tutor": "深入学习基础概念",
                "Project Coach": "分析项目细节",
                "Competition Advisor": "准备竞赛相关事宜",
                "Project Reviewer": "获取项目分项评审反馈",
            }
            extra_messages.append(
                AIMessage(
                    content=f"识别到你正在{action_map[target_agent]}，已为你转接{role_map[target_agent]}..."
                )
            )

    context_data["route_target"] = target_agent
    context_data["student_allowed_routes"] = list(STUDENT_ALLOWED_ROUTES)
    context_data["teacher_allowed_routes"] = list(TEACHER_ALLOWED_ROUTES)

    return {
        "active_agent": SUPERVISOR_NAME,
        "next_agent": target_agent,
        "context_data": context_data,
        "next_step": f"路由到 {target_agent}。",
        # 决策 3：每轮路由后将后续 Agent 的循环计数器初始化为 0
        "tool_invoke_count": 0,
        "messages": extra_messages,
    }
