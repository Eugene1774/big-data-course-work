from __future__ import annotations

from copy import deepcopy
from typing import Callable, Dict, Sequence

from langchain_core.messages import AnyMessage, BaseMessage

import logic_adapter
from core.nodes import (
    assessment_assistant_node,
    competition_advisor_node,
    error_node,
    ERROR_NODE_NAME,
    instructor_assistant_node,
    project_coach_node,
    student_learning_tutor_node,
)
from core.state import AgentState


SUPERVISOR_NAME = "Supervisor"
STUDENT_ALLOWED_ROUTES = (
    "Student Learning Tutor",
    "Project Coach",
    "Competition Advisor",
)
TEACHER_ALLOWED_ROUTES = (
    "Instructor Assistant",
    "Assessment Assistant",
)
AGENT_NODE_REGISTRY: Dict[str, Callable[[AgentState], object]] = {
    "Student Learning Tutor": student_learning_tutor_node,
    "Project Coach": project_coach_node,
    "Competition Advisor": competition_advisor_node,
    "Assessment Assistant": assessment_assistant_node,
    "Instructor Assistant": instructor_assistant_node,
    ERROR_NODE_NAME: error_node,
}

TUTOR_KEYWORDS = (
    "是什么",
    "什么意思",
    "定义",
    "概念",
    "区别",
    "怎么理解",
)
CLARIFICATION_KEYWORDS = (
    "没懂",
    "不懂",
    "什么意思",
    "啥意思",
    "再说一遍",
    "解释一下",
    "讲白话",
    "clarify",
    "explain",
)
PROJECT_SUBMISSION_KEYWORDS = (
    "提交项目",
    "项目书",
    "bp",
    "商业计划书",
    "计划书",
    "路演稿",
    "压力测试",
    "诊断",
    "校验",
)
COMPETITION_KEYWORDS = (
    "模拟比赛",
    "模拟答辩",
    "比赛",
    "赛事",
    "挑战杯",
    "互联网+",
    "rubric",
    "评分",
)
ASSESSMENT_KEYWORDS = (
    "评估",
    "评价",
    "报告",
    "引用原文",
    "结构化",
    "rubric",
    "评分表",
    "诊断",
    "风险",
    "财务",
    "cac",
    "ltv",
    "tam",
    "sam",
    "som",
)
INSTRUCTOR_KEYWORDS = (
    "干预",
    "干预计划",
    "教学建议",
    "教学计划",
    "教师建议",
    "辅导",
    "改进",
    "提升",
)


def _clone_state(state: AgentState) -> AgentState:
    return AgentState(
        messages=list(state.get("messages", [])),
        user_role=state.get("user_role", "student"),
        active_agent=state.get("active_agent", ""),
        context_data=deepcopy(state.get("context_data", {})),
        next_step=state.get("next_step", ""),
    )


def _message_text(message: AnyMessage) -> str:
    if isinstance(message, BaseMessage):
        content = message.content
        if isinstance(content, list):
            return " ".join(str(item) for item in content if item is not None).strip()
        return str(content or "").strip()
    if isinstance(message, dict):
        return str(message.get("content", "")).strip()
    return str(message).strip()


def _message_role(message: AnyMessage) -> str:
    if isinstance(message, BaseMessage):
        message_type = getattr(message, "type", "")
        if message_type == "human":
            return "user"
        if message_type == "ai":
            return "assistant"
        return message_type or "unknown"
    if isinstance(message, dict):
        return str(message.get("role", "")).strip()
    return ""


def _latest_user_text(messages: Sequence[AnyMessage]) -> str:
    for message in reversed(messages):
        if _message_role(message) == "user":
            text = _message_text(message)
            if text:
                return text
    return ""


def _contains_any(text: str, keywords: Sequence[str]) -> bool:
    normalized = text.strip().lower()
    return any(keyword.lower() in normalized for keyword in keywords)


def _is_clarification_intent(text: str) -> bool:
    return _contains_any(text, CLARIFICATION_KEYWORDS)


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


def route_supervisor(state: AgentState) -> str:
    """Return the next agent name based on RBAC role, explicit hints, and user intent."""

    user_role = state.get("user_role", "student")
    user_text = _latest_user_text(state.get("messages", []))
    context_data = state.get("context_data", {})

    # 👑 绝对最高优先级：即使前端硬传了 route_hint 或 competition_mode，只要有答辩/协同意图，立刻劫持给 Project Coach！
    if user_role == "student":
        DEFENSE_KEYWORDS = ["答辩", "路演", "评委", "vc", "投资人", "模拟", "提问"]
        COWORK_KEYWORDS = ["协作", "一起写", "帮我写", "财务分析", "市场分析", "润色"]
        if _contains_any(user_text.lower(), DEFENSE_KEYWORDS + COWORK_KEYWORDS):
            return "Project Coach"

    # 原有的 route_hint 逻辑往下放
    route_hint = str(context_data.get("route_hint", "")).strip()
    if route_hint in AGENT_NODE_REGISTRY and _is_route_allowed_for_role(user_role, route_hint):
        return route_hint
    if route_hint in AGENT_NODE_REGISTRY and not _is_route_allowed_for_role(user_role, route_hint):
        return ERROR_NODE_NAME

    if user_role == "student":
        # 1. 常规学习意图
        if _is_clarification_intent(user_text) or _contains_any(user_text, TUTOR_KEYWORDS):
            return "Student Learning Tutor"
            
        # 2. 常规打分/竞赛意图 (被前置条件过滤后，这里的才是真正的要求打分)
        if _contains_any(user_text, COMPETITION_KEYWORDS) or context_data.get("competition_mode"):
            return "Competition Advisor"
            
        # 3. 常规项目教练
        if _contains_any(user_text, PROJECT_SUBMISSION_KEYWORDS) or context_data.get("project_full_text"):
            return "Project Coach"
            
        return "Student Learning Tutor"

    if user_role == "teacher":
        teacher_mode = str(context_data.get("teacher_mode", "")).strip()
        if teacher_mode == "intervention_plan":
            return "Instructor Assistant"
        if teacher_mode in {"assessment_report", "teacher_chat"}:
            return "Assessment Assistant"
        if _contains_any(user_text, INSTRUCTOR_KEYWORDS):
            return "Instructor Assistant"
        if _contains_any(user_text, ASSESSMENT_KEYWORDS):
            return "Assessment Assistant"
        return "Instructor Assistant"

    if user_role == "admin":
        return "Assessment Assistant"

    return "Student Learning Tutor"


async def supervisor_node(state: AgentState) -> AgentState:
    """Inspect RBAC role and intent, then write the chosen downstream route."""

    new_state = _clone_state(state)
    user_text = _latest_user_text(new_state.get("messages", []))
    guardrail_response = logic_adapter.apply_input_guardrails(
        user_text,
        new_state["context_data"],
    )

    if guardrail_response:
        target_agent = ERROR_NODE_NAME
        new_state["context_data"]["route_hint"] = ERROR_NODE_NAME
    else:
        target_agent = route_supervisor(new_state)

    new_state["active_agent"] = SUPERVISOR_NAME
    new_state["next_agent"] = target_agent
    new_state["context_data"]["route_target"] = target_agent
    new_state["context_data"]["student_allowed_routes"] = list(STUDENT_ALLOWED_ROUTES)
    new_state["context_data"]["teacher_allowed_routes"] = list(TEACHER_ALLOWED_ROUTES)
    new_state["next_step"] = f"路由到 {target_agent}。"
    
    # 添加角色切换说明
    previous_agent = state.get("next_agent", "")
    if previous_agent and previous_agent != target_agent and target_agent != ERROR_NODE_NAME:
        role_map = {
            "Student Learning Tutor": "学习辅导智能体",
            "Project Coach": "项目教练智能体",
            "Competition Advisor": "竞赛顾问智能体"
        }
        if target_agent in role_map:
            transition_message = f"识别到你正在{'深入学习基础概念' if target_agent == 'Student Learning Tutor' else '分析项目细节' if target_agent == 'Project Coach' else '准备竞赛相关事宜'}，已为你转接{role_map[target_agent]}..."
            from langchain_core.messages import AIMessage
            new_state["messages"].append(AIMessage(content=transition_message))
    
    return new_state
