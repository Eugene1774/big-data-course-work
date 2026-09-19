from __future__ import annotations

import asyncio
import json
import logging
import re
from copy import deepcopy
from functools import lru_cache, wraps
from typing import Any, Dict, List, Mapping, Sequence

from langchain_core.messages import AIMessage, AnyMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

import logic_adapter
from core.schemas import AuditTrail, NextTask, ProjectCoachOutput, RubricScoreResult
from core.state import AgentState
from config.api_config import get_llm_config
from db_init import fetch_pending_intervention, init_db, mark_intervention_executed
from scoring_engine import calculate_scores, generate_ability_report, generate_rubric_table

try:
    from kg_hypergraph import build_driver, get_risk_pattern
    from neo4j_resilience import close_managed_driver
except Exception:  # pragma: no cover
    build_driver = None
    get_risk_pattern = None
    close_managed_driver = None


LOGGER = logging.getLogger(__name__)

ERROR_NODE_NAME = "Error"
ERROR_HANDLER_NAME = "Error Handler"

RULE_ID_RE = re.compile(r"\b(H1[0-5]|H[1-9])\b", flags=re.IGNORECASE)

RUBRIC_ITEM_NAME_MAP: Dict[str, str] = {
    "R1": "Problem Definition",
    "R2": "User Evidence Strength",
    "R3": "Solution Feasibility",
    "R4": "Business Model Consistency",
    "R5": "Market & Competition",
    "R6": "Financial Logic",
    "R7": "Innovation & Differentiation",
    "R8": "Team & Execution",
    "R9": "Presentation Quality",
}

H_RULE_NAME_MAP: Dict[str, str] = {
    "H1": "BusinessModelConsistency",
    "H4": "MarketSizingConsistency",
    "H5": "EvidenceStrength",
    "H6": "CompetitionCoverage",
    "H8": "UnitEconomics",
    "AUDIT": "EvidenceTrace",
}

# 系统提示模板
COACH_SYSTEM_PROMPT = """你是一位顶级的创新创业教练。根据用户的诉求，你可能需要在三种模式中无缝切换：【严厉的压测导师】、【专业的BP协同智囊】或【毒舌的答辩评委（如激进VC、保守银行家）】。

【工具使用准则】：
- 当需要客观证据（商业模式风险、渠道错位、痛点验证、风险评估）时，调用 search_hypergraph 工具，从用户话语中提取核心业务词汇作为 keywords。
- 拿到工具返回的证据后，将其消化为人类视角的商业质问，绝不直接复述原始数据。
- 证据充分后立即给出最终文字回答，不要反复检索。

【🚨 核心强制护栏（Guardrails）- 违规将导致系统崩溃】：
1. 绝对禁止向学生提及"超图"、"逻辑风险"、"证据库"、"H1-H15规则"、"fallback"、"工具"、"JSON"、"检索"等后台工程词汇！你需要将这些抽象变量转化为人类视角的商业质问。
2. 绝对禁止以"评分预览：X分"作为回复开头，你是来对话的，不是来报分数的。
3. 绝对禁止使用Markdown格式（无加粗、无列表、无标题）。

【回复结构要求】（必须分成三个自然段，段落间空一行）：
第一段（状态判定与角色代入）：根据学生的输入，带入相应的角色语气（如协同队友、严厉VC）。点出他们项目目前最核心的商业或逻辑漏洞（基于已获取的证据）。
第二段（深度质问或协同建议）：如果不自洽，给出具体的商业逻辑反问；如果学生要求协作撰写，给出你负责的那部分（如财务结构）的专业建议和所需数据。
第三段（下一步行动）：只提出唯一一个具体、可执行的任务或关键问题，强迫学生思考。
"""

TUTOR_SYSTEM_PROMPT = """你是一位专业的创新创业学习导师，擅长苏格拉底式教学法与概念讲解。

【工具使用准则】：需要真实商业案例或知识卡片素材时，调用 search_knowledge_cards 工具；拿到素材后自然融入讲解，绝不提及"工具"、"检索"等工程词汇。素材充分后直接作答，不要反复检索。

【回复要求】使用自然语言段落，禁止Markdown格式。讲解概念时分三段：
第一段（原理解释）：清晰解答学生的疑惑。
第二段（具象案例）：结合检索到的案例或家喻户晓的真实商业案例（如美团、闲鱼、大疆），说明概念在真实商业世界中如何运作，绝不空谈理论。
第三段（苏格拉底反问）：针对学生的具体项目抛出一个开放性问题，引导其用"用户-痛点-价值-证据"框架思考。

若用户只是打招呼或提出与创业无关的常识问题，用自然语言友好、简洁地直接回答即可。
"""

COMPETITION_SYSTEM_PROMPT = """你是一位创新创业竞赛顾问，熟悉"互联网+"、"挑战杯"等赛事规则与评分标准。

【工具使用准则】：当用户要求评分、打分预览或评估竞赛竞争力，且上下文已有项目材料时，调用 get_competition_score 工具；需要案例素材时调用 search_knowledge_cards。拿到结果后直接作答，不要反复调用。

【回复要求】基于评分结果指出低分维度与证据缺口，给出可执行的备赛建议；使用自然语言段落，禁止Markdown格式；禁止暴露"工具"、"JSON"等后台工程词汇。若缺少项目材料，先请用户提供项目摘要与关键指标。
"""

ASSESSMENT_SYSTEM_PROMPT = """你是一位教务评估助手，负责生成学生能力评价与结构化诊断结论。

【工具使用准则】：需要能力画像或评估数据时，调用 get_ability_report 工具；拿到结果后直接作答，不要反复调用。

【回复要求】基于报告用自然语言段落给出简洁总结：说明评估覆盖范围、最低能力维度与证据缺口；禁止Markdown格式；禁止暴露后台工程词汇。
"""

INSTRUCTOR_SYSTEM_PROMPT = """你是一位面向教师的教学辅助智能体，负责给出干预建议与教学改进方案。

【工具使用准则】：需要学生能力数据时，调用 get_ability_report 工具；拿到结果后直接作答，不要反复调用。

【回复要求】给出可执行的教学干预建议（聚焦一个主要问题与下一步跟踪方式），使用自然语言段落，禁止Markdown格式；禁止暴露后台工程词汇。
"""

REVIEWER_SYSTEM_PROMPT = """你是一位严谨的创新创业项目评审专家，负责依据明确标准对项目做分项评审反馈。

【工具使用准则】：需要案例或方法素材辅助判断时，可调用 search_knowledge_cards 工具；拿到素材后直接作答，不要反复检索，绝不提及"工具"、"检索"等后台工程词汇。

【评审维度】必须逐项覆盖以下五个维度，缺一不可：
一、社会价值；二、实践依据；三、创新意义；四、发展前景；五、团队协作。

【输出结构】用"一、二、三、四、五"编号分节，每个维度一节，每节包含两部分：先给出该维度的评价（依据材料中实际写明的内容），再单独用"证据缺口："指出该维度缺少哪些客观支撑；五个维度之后另起一节"下一步建议"，只给1-2条最优先、可执行的修改建议。除编号分节和"证据缺口："标记外，正文使用自然语言段落，禁止加粗、标题、项目符号等Markdown格式。

【证据纪律】严格区分事实与假设：材料中有数据或来源支撑的按事实评价；没有数据支撑的说法必须标注"缺乏资料支持"，禁止写成"确实存在""已经验证""真实可靠"等确定性表述；材料未提及的维度，如实说明材料未提供相关信息，不要替项目编造内容。

【边界】若材料不足以评审某个维度，明确指出还需要补充什么信息；禁止暴露"超图"、"H1-H15规则"、"fallback"、"JSON"等后台工程词汇。
"""

FORCE_CONCLUSION_PROMPT = (
    "[System 防御指令] 图谱检索深度已达上限。请立即停止调用工具，"
    "直接基于当前上下文已获取的证据给出最终回答。"
    "回答使用自然语言段落，禁止Markdown格式，禁止暴露任何后台工程词汇。"
)


@lru_cache(maxsize=1)
def get_agent_llm() -> ChatOpenAI:
    llm_config = get_llm_config()
    return ChatOpenAI(
        model=llm_config["model"],
        api_key=llm_config["api_key"],
        base_url=llm_config["base_url"],
        temperature=0.3,
        timeout=60.0,
        max_retries=1,
    )


def _agent_tools(agent_name: str):
    """延迟绑定，避免与 logic_adapter 的循环导入在模块加载期触发。"""
    from logic_adapter import (
        get_ability_report,
        get_competition_score,
        search_hypergraph,
        search_knowledge_cards,
    )

    mapping = {
        "Student Learning Tutor": [search_knowledge_cards],
        "Project Coach": [search_hypergraph, search_knowledge_cards],
        "Competition Advisor": [get_competition_score, search_knowledge_cards],
        "Assessment Assistant": [get_ability_report],
        "Instructor Assistant": [get_ability_report],
        "Project Reviewer": [search_knowledge_cards],
    }
    return mapping.get(agent_name, [])


AGENT_SYSTEM_PROMPTS: Dict[str, str] = {
    "Student Learning Tutor": TUTOR_SYSTEM_PROMPT,
    "Project Coach": COACH_SYSTEM_PROMPT,
    "Competition Advisor": COMPETITION_SYSTEM_PROMPT,
    "Assessment Assistant": ASSESSMENT_SYSTEM_PROMPT,
    "Instructor Assistant": INSTRUCTOR_SYSTEM_PROMPT,
    "Project Reviewer": REVIEWER_SYSTEM_PROMPT,
}


def _parse_json_like(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None
    return value


def _coerce_rubric_rows(ability_report: Mapping[str, Any], context_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    candidates: List[Any] = []

    report_rows = ability_report.get("rubric_score_results")
    report_rows = _parse_json_like(report_rows)
    if isinstance(report_rows, list):
        candidates.extend(report_rows)

    context_rows = context_data.get("rubric_score_results")
    context_rows = _parse_json_like(context_rows)
    if isinstance(context_rows, list):
        candidates.extend(context_rows)

    score_table = ability_report.get("score_table")
    if isinstance(score_table, list) and not candidates:
        candidates.extend(score_table)

    normalized_rows: List[Dict[str, Any]] = []
    for row in candidates:
        parsed_row = _parse_json_like(row)
        if isinstance(parsed_row, dict):
            normalized_rows.append(parsed_row)
    return normalized_rows


def _format_h_rule_id(raw_rule_id: str) -> str:
    cleaned = str(raw_rule_id or "").strip()
    if not cleaned:
        return "Rule-H_UNKNOWN (UnknownRule)"
    if cleaned.lower().startswith("rule-"):
        return cleaned

    normalized = cleaned.upper()
    match = re.search(r"\b(H1[0-5]|H[1-9])\b", normalized)
    if match:
        normalized = match.group(1)
    rule_label = H_RULE_NAME_MAP.get(normalized, "UnknownRule")
    return f"Rule-{normalized} ({rule_label})"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clone_state(state: AgentState) -> AgentState:
    cloned: AgentState = AgentState(
        messages=list(state.get("messages", [])),
        user_role=state.get("user_role", "student"),
        active_agent=state.get("active_agent", ""),
        next_agent=state.get("next_agent", ""),
        context_data=deepcopy(state.get("context_data", {})),
        next_step=state.get("next_step", ""),
    )
    for key in ("kb_context", "probing_strategy", "retrieved_heterogeneous_subgraph", "rubric_score_results"):
        if key in state:
            cloned[key] = state.get(key)
    return cloned


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
        kind = str(getattr(message, "type", "") or "")
        if kind == "human":
            return "user"
        if kind == "ai":
            return "assistant"
        return kind or "unknown"
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


def _latest_assistant_text(messages: Sequence[AnyMessage]) -> str:
    for message in reversed(messages):
        if _message_role(message) == "assistant" and not getattr(message, "tool_calls", None):
            text = _message_text(message)
            if text:
                return text
    return ""


def _append_ai_message(state: AgentState, content: str) -> None:
    metadata = {
        "agent_role": state.get("active_agent", ""),
        "model_used": get_llm_config()["model"]
    }
    state["messages"] = list(state.get("messages", [])) + [AIMessage(content=content, metadata=metadata)]


def _strip_rule_ids(text: str) -> str:
    return RULE_ID_RE.sub("", str(text or "")).replace("  ", " ").strip()


def _extract_card_ids(value: Any) -> List[str]:
    card_ids: List[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            lowered = str(key).strip().lower()
            if lowered in {"card_id", "knowledge_card_id", "kg_card_id"}:
                candidate = str(item or "").strip()
                if candidate:
                    card_ids.append(candidate)
            elif lowered == "knowledge_card_ids" and isinstance(item, (list, tuple, set)):
                card_ids.extend(str(x).strip() for x in item if str(x).strip())
            else:
                card_ids.extend(_extract_card_ids(item))
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            card_ids.extend(_extract_card_ids(item))
    return list(dict.fromkeys(card_ids))


def _safe_context_list(context_data: Dict[str, Any], key: str) -> List[Any]:
    existing = context_data.get(key)
    if isinstance(existing, list):
        return existing
    context_data[key] = []
    return context_data[key]


def _extract_business_nodes(context_data: Dict[str, Any]) -> Dict[str, str]:
    target_user = str(context_data.get("target_user") or context_data.get("customer") or "").strip()
    value_proposition = str(context_data.get("value_proposition") or context_data.get("value") or "").strip()
    channel = str(context_data.get("channel") or context_data.get("primary_channels") or "").strip()

    text = " ".join(
        [
            str(context_data.get("project_full_text", "")),
            str(context_data.get("project_summary", "")),
            str(context_data.get("bp_text", "")),
        ]
    ).lower()

    if not target_user and "farmer" in text:
        target_user = "farmer"
    if not value_proposition and ("delivery" in text or "agri" in text):
        value_proposition = "efficient agri-delivery"
    if not channel and any(token in text for token in ("xiaohongshu", "douyin", "tiktok", "wechat")):
        channel = "social media"

    return {
        "target_user": target_user,
        "value_proposition": value_proposition,
        "channel": channel,
    }


def handle_agent_error(func):
    @wraps(func)
    async def wrapper(state: AgentState) -> Dict[str, Any]:
        try:
            return await func(state)
        except Exception as exc:  # pragma: no cover
            LOGGER.exception("Node execution failed: %s", func.__name__)
            context_data = dict(state.get("context_data") or {})
            context_data["error"] = {"node": func.__name__, "message": str(exc)}
            return {
                "messages": [
                    AIMessage(
                        content="The system hit an error and switched to fallback response.",
                        metadata={"agent_role": ERROR_HANDLER_NAME, "model_used": get_llm_config()["model"]},
                    )
                ],
                "active_agent": ERROR_HANDLER_NAME,
                "context_data": context_data,
                "next_step": "System error occurred. Please retry.",
            }

    return wrapper


def query_hypergraph_for_risk(extracted_nodes: Dict[str, Any] | None) -> Dict[str, Any] | None:
    """Pure function: only detect risk and return payload, no state mutation."""
    if not isinstance(extracted_nodes, dict):
        return None

    target_user = str(extracted_nodes.get("target_user") or "").strip()
    channel = str(extracted_nodes.get("channel") or "").strip()
    value = str(extracted_nodes.get("value_proposition") or "").strip()
    if not target_user or not channel:
        return None

    mismatch = (
        ("farmer" in target_user.lower())
        and any(token in channel.lower() for token in ("xiaohongshu", "douyin", "tiktok", "wechat"))
    )
    if not mismatch:
        return None

    result = {
        "retrieved_heterogeneous_subgraph": True,
        "edge_type": "Risk_Pattern_Edge",
        "rule_id": "H1",
        "rule_triggered": "H1_BusinessModelConsistency",
        "nodes_involved": {
            "customer": target_user,
            "value_proposition": value or "unknown",
            "channel": channel,
        },
        "logical_evaluation": "channel.cannot_reach(customer)",
        "impact": "Business model mismatch: channel likely cannot reach customer segment.",
        "strategy_selected": "precision-acquisition",
        "kg_card_id": "CARD_BIZ_001",
    }

    LOGGER.info("========== HYPERGRAPH RETRIEVAL ASSERTION ==========")
    LOGGER.info("Agent Name: Project Coach")
    LOGGER.info("Rule Triggered: %s", result["rule_triggered"])
    LOGGER.info("%s", json.dumps(result, ensure_ascii=False, indent=2))
    LOGGER.info("====================================================")
    return result


def check_business_logic(state: AgentState, agent_name: str = "Project Coach") -> Dict[str, Any] | None:
    context_data = state.get("context_data", {})
    if not isinstance(context_data, dict):
        context_data = {}
        state["context_data"] = context_data

    extracted_nodes = _extract_business_nodes(context_data)
    subgraph = query_hypergraph_for_risk(extracted_nodes)
    if subgraph is None:
        context_data.setdefault("kb_context", "fallback")
        return None

    strategy = str(subgraph.get("strategy_selected", "")).strip()
    context_data["kb_context"] = subgraph
    context_data["probing_strategy"] = strategy
    context_data["retrieved_heterogeneous_subgraph"] = subgraph
    context_data["selected_strategy"] = strategy
    context_data["detected_fallacies"] = [str(subgraph.get("rule_id", "unknown"))]

    state["kb_context"] = subgraph
    state["probing_strategy"] = strategy
    state["retrieved_heterogeneous_subgraph"] = subgraph

    LOGGER.info(
        "========== HYPERGRAPH RETRIEVAL ASSERTION ==========\nAgent Name: %s\nRule Triggered: %s\n%s\n====================================================",
        agent_name,
        str(subgraph.get("rule_triggered", "")),
        json.dumps(subgraph, ensure_ascii=False, indent=2),
    )
    return subgraph


async def query_hypergraph(state: AgentState) -> Dict[str, Any]:
    context_data = state.get("context_data", {})
    if not isinstance(context_data, dict):
        context_data = {}

    extracted_nodes = _extract_business_nodes(context_data)
    driver = None
    try:
        database = str(context_data.get("neo4j_database") or "neo4j").strip()
        if build_driver and get_risk_pattern:
            uri = str(context_data.get("neo4j_uri") or "").strip()
            user = str(context_data.get("neo4j_user") or context_data.get("neo4j_username") or "").strip()
            password = str(context_data.get("neo4j_password") or "").strip()
            if uri and user and password:
                try:
                    driver = build_driver(uri, user, password, database)
                    keywords = logic_adapter.extract_keywords(_latest_user_text(state.get("messages", [])))
                    try:
                        risk_patterns = await asyncio.wait_for(
                            asyncio.to_thread(get_risk_pattern, driver, database, keywords, 5),
                            timeout=5.0
                        )
                        normalized = logic_adapter.normalize_hyperedge_records(
                            [
                                {
                                    "hyperedge_id": item.get("edge_id"),
                                    "edge_type": "Risk_Pattern_Edge",
                                    "rule_triggered": "H1_BusinessModelConsistency",
                                    "participants": item.get("participants", []),
                                    "logical_evaluation": item.get("logical_evaluation", ""),
                                    "impact": item.get("edge_description") or item.get("impact") or "",
                                }
                                for item in risk_patterns
                                if isinstance(item, dict)
                            ]
                        )
                        retrieved = {"database": database, "risk_patterns": risk_patterns, "hyperedge_records": normalized}
                        return {
                            "kb_context": retrieved,
                            "retrieved_heterogeneous_subgraph": retrieved,
                            "risk_patterns": risk_patterns,
                            "hyperedge_records": normalized,
                        }
                    except asyncio.TimeoutError:
                        LOGGER.warning("Neo4j query timeout, using fallback.")
                except Exception as e:
                    LOGGER.warning(f"Neo4j connection failed: {e}, using fallback.")

        fallback = query_hypergraph_for_risk(extracted_nodes)
        if fallback:
            return {
                "kb_context": fallback,
                "retrieved_heterogeneous_subgraph": fallback,
                "risk_patterns": [fallback],
                "hyperedge_records": [fallback],
            }

        return {
            "kb_context": "fallback",
            "retrieved_heterogeneous_subgraph": {},
            "risk_patterns": [],
            "hyperedge_records": [],
        }
    except Exception:
        LOGGER.exception("Hypergraph query degraded to fallback path.")
        return {
            "kb_context": "fallback",
            "retrieved_heterogeneous_subgraph": {},
            "risk_patterns": [],
            "hyperedge_records": [],
        }
    finally:
        if driver is not None and close_managed_driver is not None:
            try:
                close_managed_driver(uri, user, password)
            except Exception:
                pass


def _scoring_input_from_context(context_data: Dict[str, Any]) -> Dict[str, Any]:
    cached = context_data.get("scoring_input")
    if isinstance(cached, dict) and cached:
        return cached

    project_text = str(
        context_data.get("project_full_text")
        or context_data.get("bp_text")
        or context_data.get("project_summary")
        or ""
    ).strip()
    if not project_text:
        return {}

    return {
        "project_name": str(context_data.get("project_name") or "Unnamed Project"),
        "project_summary": project_text,
        "target_user": context_data.get("target_user") or context_data.get("customer") or "",
        "value_proposition": context_data.get("value_proposition") or context_data.get("value") or "",
        "primary_channels": context_data.get("primary_channels") or context_data.get("channels") or [],
        "metrics": context_data.get("metrics", {}),
        "student_quotes": context_data.get("student_quotes", []),
        "behavior_logs": context_data.get("behavior_logs", []),
    }


def _inject_pending_intervention(context_data: Dict[str, Any]) -> None:
    init_db()
    student_id = str(context_data.get("student_id") or context_data.get("user_id") or "").strip()
    if not student_id:
        return
    pending = fetch_pending_intervention(student_id)
    if pending:
        context_data["pending_teacher_intervention"] = pending


def _ack_pending_intervention(context_data: Dict[str, Any]) -> None:
    pending = context_data.get("pending_teacher_intervention")
    if not isinstance(pending, dict):
        return
    intervention_id = pending.get("id")
    if intervention_id is None:
        return
    try:
        mark_intervention_executed(int(intervention_id))
    except Exception:
        LOGGER.exception("Failed to mark intervention executed.")
    context_data["pending_teacher_intervention"] = {}


def _latest_tool_payload(messages: Sequence[AnyMessage], tool_name: str) -> Dict[str, Any] | None:
    """从最近的工具回执中提取 JSON 载荷（用于最终轮的副作用与前端轨迹面板）。"""
    for message in reversed(messages):
        if isinstance(message, ToolMessage) and getattr(message, "name", "") == tool_name:
            payload = _parse_json_like(_message_text(message))
            if isinstance(payload, dict):
                return payload
    return None


async def _run_react_turn(state: AgentState, agent_name: str, system_prompt: str) -> AIMessage:
    """ReAct 单轮：绑定工具后由大模型自主决定直接作答还是发起 tool_calls。"""
    llm = get_agent_llm()
    tools = _agent_tools(agent_name)
    agent_llm = llm.bind_tools(tools) if tools else llm

    messages = [SystemMessage(content=system_prompt)] + list(state.get("messages", []))
    response = await agent_llm.ainvoke(messages)
    if not isinstance(response, AIMessage):
        response = AIMessage(content=str(response))

    llm_config = get_llm_config()
    existing_metadata = getattr(response, "metadata", None) or {}
    response.metadata = {
        **existing_metadata,
        "agent_role": agent_name,
        "model_used": llm_config["model"],
    }
    return response


def _has_pending_tool_calls(response: AIMessage) -> bool:
    return bool(getattr(response, "tool_calls", None))


@handle_agent_error
async def student_learning_tutor_node(state: AgentState) -> Dict[str, Any]:
    context_data = dict(state.get("context_data") or {})
    response = await _run_react_turn(state, "Student Learning Tutor", TUTOR_SYSTEM_PROMPT)

    update: Dict[str, Any] = {
        "messages": [response],
        "active_agent": "Student Learning Tutor",
        "context_data": context_data,
    }
    if not _has_pending_tool_calls(response):
        _ack_pending_intervention(context_data)
        update["next_step"] = "Tutor answered with Socratic guidance."
    return update


@handle_agent_error
async def project_coach_node(state: AgentState) -> Dict[str, Any]:
    context_data = dict(state.get("context_data") or {})
    _inject_pending_intervention(context_data)

    system_prompt = COACH_SYSTEM_PROMPT
    pending_intervention = context_data.get("pending_teacher_intervention")
    if pending_intervention:
        intervention_content = pending_intervention.get("content", "")
        system_prompt = (
            f"{COACH_SYSTEM_PROMPT}\n\n"
            f"====== 🚨 导师最高优先级指令 🚨 ======\n"
            f"【紧急任务】：{intervention_content}\n"
            f"【执行要求】：在你的下一次回复中，必须立即改变当前话题，极其严厉地执行上述导师指令进行追问。不要提导师的名字，直接以 AI 教练的身份发难。\n"
            f"====================================="
        )

    response = await _run_react_turn(state, "Project Coach", system_prompt)

    update: Dict[str, Any] = {
        "messages": [response],
        "active_agent": "Project Coach",
        "context_data": context_data,
    }

    if _has_pending_tool_calls(response):
        return update

    # 最终文字回答：执行副作用（结构化输出、证据采集、轨迹面板数据）
    student_input = _latest_user_text(state.get("messages", []))

    graph_payload = _latest_tool_payload(state.get("messages", []), "search_hypergraph")
    if graph_payload and graph_payload.get("status") == "ok" and graph_payload.get("risk_patterns"):
        retrieved = {
            "risk_patterns": graph_payload.get("risk_patterns", []),
            "hyperedge_records": graph_payload.get("risk_patterns", []),
        }
        context_data["kb_context"] = retrieved
        context_data["retrieved_heterogeneous_subgraph"] = retrieved
        context_data["hyperedge_records"] = retrieved["hyperedge_records"]
        update["kb_context"] = retrieved
        update["retrieved_heterogeneous_subgraph"] = retrieved

    temp_state: AgentState = {**state, "context_data": context_data}
    subgraph = check_business_logic(temp_state, agent_name="Project Coach")
    strategy = str((subgraph or {}).get("strategy_selected") or "evidence-first")
    context_data["selected_strategy"] = strategy
    context_data["detected_fallacies"] = [str((subgraph or {}).get("rule_id") or "unknown")]

    project_stage = str(context_data.get("current_stage") or "diagnosis")
    next_task_text = "根据诊断结果提供具体的改进方案"
    structured = ProjectCoachOutput(
        project_stage=project_stage,
        current_diagnosis="根据超图分析生成的诊断",
        evidence_used=student_input or "No quote provided",
        impact_if_unfixed="Market and financial reasoning will stay unreliable.",
        next_task=NextTask(
            task_description=next_task_text,
            template_guideline="Use user-pain-value-channel-evidence format.",
            acceptance_criteria="At least one quote and one channel reach proof.",
        ),
    )
    context_data["project_coach_structured_output"] = structured.model_dump()

    collected = _safe_context_list(context_data, "collected_evidence")
    collected.append(
        {
            "quote": student_input or "No quote provided",
            "h_rule_id": str((subgraph or {}).get("rule_id") or "H_UNKNOWN"),
            "kg_card_id": str((subgraph or {}).get("kg_card_id") or "KG_CARD_UNBOUND"),
            "explanation": f"Triggered hypergraph pattern: {str((subgraph or {}).get('edge_type') or 'Risk_Pattern_Edge')}",
        }
    )

    update["next_step"] = structured.next_task.task_description
    _ack_pending_intervention(context_data)
    return update


@handle_agent_error
async def competition_advisor_node(state: AgentState) -> Dict[str, Any]:
    context_data = dict(state.get("context_data") or {})
    response = await _run_react_turn(state, "Competition Advisor", COMPETITION_SYSTEM_PROMPT)

    update: Dict[str, Any] = {
        "messages": [response],
        "active_agent": "Competition Advisor",
        "context_data": context_data,
    }

    if _has_pending_tool_calls(response):
        return update

    # 最终轮：优先采用工具返回的评分报告；若模型未调用工具但材料齐全，则按原有确定性路径兜底
    tool_payload = _latest_tool_payload(state.get("messages", []), "get_competition_score")
    score_report = None
    if isinstance(tool_payload, dict) and tool_payload.get("status") == "ok":
        score_report = tool_payload.get("score_report")

    if not isinstance(score_report, dict):
        scoring_input = _scoring_input_from_context(context_data)
        if scoring_input:
            score_report = await asyncio.to_thread(calculate_scores, scoring_input)

    if isinstance(score_report, dict):
        rubric_table = await asyncio.to_thread(generate_rubric_table, score_report)
        context_data["competition_score_report"] = score_report
        context_data["competition_rubric_preview"] = rubric_table.to_dict("records")
        low_items = score_report.get("low_score_items", [])
        update["next_step"] = f"优先修复 {low_items[0]} 的证据不足问题。" if low_items else "继续加强证据支撑。"
    else:
        update["next_step"] = "在评分前请提供项目摘要。"
    return update


def _normalize_rubric_results_for_state(
    rubric_rows: Sequence[Dict[str, Any]],
    context_data: Dict[str, Any],
    quotes: Sequence[str],
) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    fallback_quote = quotes[0] if quotes else "No quote captured"
    fallback_card = (_extract_card_ids(context_data.get("retrieved_knowledge_cards", [])) or ["KG_CARD_UNBOUND"])[0]

    for row in rubric_rows:
        if not isinstance(row, dict):
            continue

        audit_raw = _parse_json_like(row.get("audit_trail", {}))
        audit_dict = dict(audit_raw) if isinstance(audit_raw, dict) else {}

        h_rule_display = _format_h_rule_id(str(audit_dict.get("h_rule_id") or row.get("rule_id") or "H_UNKNOWN"))

        audit = AuditTrail(
            quote=str(audit_dict.get("quote") or fallback_quote),
            h_rule_id=h_rule_display,
            kg_card_id=str(audit_dict.get("kg_card_id") or fallback_card),
            explanation=str(audit_dict.get("explanation") or row.get("reason") or "No explanation"),
        )

        raw_score = _safe_float(row.get("score", row.get("score_5", 0.0)), 0.0)
        clipped_score = max(0.0, min(5.0, round(raw_score, 2)))
        max_score = max(0.1, _safe_float(row.get("max_score", 5.0), 5.0))

        # NOTE: Validate with Pydantic first, then add contract fields name/max_score.
        score_result = RubricScoreResult(
            item_id=str(row.get("item_id") or "R0"),
            score=clipped_score,
            audit_trail=audit,
        )
        validated = score_result.model_dump()
        item_id = str(validated.get("item_id") or "R0")

        normalized.append(
            {
                "item_id": item_id,
                "name": str(row.get("name") or RUBRIC_ITEM_NAME_MAP.get(item_id, f"{item_id} (Unknown Item)")),
                "score": float(validated.get("score", 0.0)),
                "max_score": float(round(max_score, 2)),
                # Contract: audit_trail keeps exactly 4 keys.
                "audit_trail": {
                    "quote": str(validated.get("audit_trail", {}).get("quote", "")),
                    "h_rule_id": str(validated.get("audit_trail", {}).get("h_rule_id", "")),
                    "kg_card_id": str(validated.get("audit_trail", {}).get("kg_card_id", "")),
                    "explanation": str(validated.get("audit_trail", {}).get("explanation", "")),
                },
            }
        )

    return normalized


async def _finalize_assessment(update: Dict[str, Any], state: AgentState, context_data: Dict[str, Any]) -> None:
    diagnostics = context_data.get("diagnostic_results", [])
    diagnostics = diagnostics if isinstance(diagnostics, list) else []

    # NOTE: Reuse scoring_engine as the single scoring source of truth.
    ability_report = await asyncio.to_thread(
        generate_ability_report,
        state.get("messages", []),
        diagnostics,
        context_data,
    )
    context_data["ability_report"] = dict(ability_report)

    raw_rubric_rows = _coerce_rubric_rows(ability_report, context_data)
    quotes = [text for text in (_latest_user_text(state.get("messages", [])),) if text]
    normalized_results = _normalize_rubric_results_for_state(
        raw_rubric_rows,
        context_data,
        quotes,
    )

    evidence_list = _safe_context_list(context_data, "collected_evidence")
    for row in normalized_results:
        audit = row.get("audit_trail", {})
        if not isinstance(audit, dict):
            continue
        evidence_list.append(
            {
                "quote": audit.get("quote", ""),
                "h_rule_id": audit.get("h_rule_id", "Rule-H_UNKNOWN (UnknownRule)"),
                "kg_card_id": audit.get("kg_card_id", "KG_CARD_UNBOUND"),
                "explanation": audit.get("explanation", ""),
            }
        )

    # Contract output: write both into context_data and global state.
    context_data["rubric_score_results"] = normalized_results
    update["rubric_score_results"] = normalized_results

    low_items: List[str] = []
    score_report = context_data.get("competition_score_report", {})
    if isinstance(score_report, dict):
        low_items = list(score_report.get("low_score_items", []) or [])

    update["next_step"] = "Fill evidence gaps for the lowest rubric dimension."


@handle_agent_error
async def assessment_assistant_node(state: AgentState) -> Dict[str, Any]:
    context_data = dict(state.get("context_data") or {})
    response = await _run_react_turn(state, "Assessment Assistant", ASSESSMENT_SYSTEM_PROMPT)

    update: Dict[str, Any] = {
        "messages": [response],
        "active_agent": "Assessment Assistant",
        "context_data": context_data,
    }
    if not _has_pending_tool_calls(response):
        await _finalize_assessment(update, state, context_data)
    return update


@handle_agent_error
async def instructor_assistant_node(state: AgentState) -> Dict[str, Any]:
    context_data = dict(state.get("context_data") or {})
    response = await _run_react_turn(state, "Instructor Assistant", INSTRUCTOR_SYSTEM_PROMPT)

    update: Dict[str, Any] = {
        "messages": [response],
        "active_agent": "Instructor Assistant",
        "context_data": context_data,
    }

    if _has_pending_tool_calls(response):
        return update

    diagnostics = context_data.get("diagnostic_results", [])
    diagnostics = diagnostics if isinstance(diagnostics, list) else []
    ability_report = await asyncio.to_thread(
        generate_ability_report,
        state.get("messages", []),
        diagnostics,
        context_data,
    )
    context_data["ability_report"] = dict(ability_report)
    update["next_step"] = "Schedule one focused intervention and track the next re-test."
    return update


@handle_agent_error
async def project_reviewer_node(state: AgentState) -> Dict[str, Any]:
    context_data = dict(state.get("context_data") or {})
    response = await _run_react_turn(state, "Project Reviewer", REVIEWER_SYSTEM_PROMPT)

    update: Dict[str, Any] = {
        "messages": [response],
        "active_agent": "Project Reviewer",
        "context_data": context_data,
    }
    if not _has_pending_tool_calls(response):
        update["next_step"] = "按五个评审维度给出分项评价、证据缺口与下一步建议。"
    return update


@handle_agent_error
async def force_conclusion_node(state: AgentState) -> Dict[str, Any]:
    """安全阀熔断节点：计数器超载时勒令模型停止检索并基于已有证据作答。"""

    messages = list(state.get("messages", []))
    appended: List[Any] = []

    last = messages[-1] if messages else None
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        # 为悬挂的 tool_calls 合成错误回执，避免 OpenAI 兼容端点拒绝非法消息序列
        for call in last.tool_calls:
            synthetic = ToolMessage(
                content="工具调用次数已达上限，本次调用被系统安全阀拦截。",
                tool_call_id=str(call.get("id") or ""),
                name=str(call.get("name") or ""),
                status="error",
            )
            messages.append(synthetic)
            appended.append(synthetic)

    llm = get_agent_llm()
    response = await llm.ainvoke(messages + [SystemMessage(content=FORCE_CONCLUSION_PROMPT)])
    if not isinstance(response, AIMessage):
        response = AIMessage(content=str(response))

    llm_config = get_llm_config()
    existing_metadata = getattr(response, "metadata", None) or {}
    response.metadata = {
        **existing_metadata,
        "agent_role": str(state.get("active_agent") or "Force Conclusion"),
        "model_used": llm_config["model"],
    }
    appended.append(response)

    return {
        "messages": appended,
        "next_step": "工具循环达到安全上限，已强制生成最终结论。",
    }


@handle_agent_error
async def error_node(state: AgentState) -> Dict[str, Any]:
    context_data = dict(state.get("context_data") or {})
    return {
        "messages": [
            AIMessage(
                content="Fallback handler activated due to input/system error.",
                metadata={"agent_role": ERROR_NODE_NAME, "model_used": get_llm_config()["model"]},
            )
        ],
        "active_agent": ERROR_NODE_NAME,
        "context_data": context_data,
        "next_step": "Adjust input and retry.",
    }


__all__ = [
    "ERROR_NODE_NAME",
    "ERROR_HANDLER_NAME",
    "query_hypergraph_for_risk",
    "check_business_logic",
    "query_hypergraph",
    "student_learning_tutor_node",
    "project_coach_node",
    "competition_advisor_node",
    "assessment_assistant_node",
    "instructor_assistant_node",
    "project_reviewer_node",
    "force_conclusion_node",
    "error_node",
]
