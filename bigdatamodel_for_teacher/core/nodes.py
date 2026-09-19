from __future__ import annotations

import asyncio
import json
import logging
import re
from copy import deepcopy
from functools import wraps
from typing import Any, Dict, List, Mapping, Sequence

from langchain_core.messages import AIMessage, AnyMessage, BaseMessage, SystemMessage
from openai import OpenAI

import logic_adapter
from core.schemas import AuditTrail, NextTask, ProjectCoachOutput, RubricScoreResult
from core.state import AgentState
from config.api_config import API_KEY, BASE_URL, DEFAULT_MODEL
from db_init import fetch_pending_intervention, init_db, mark_intervention_executed
from scoring_engine import calculate_scores, generate_ability_report, generate_rubric_table

try:
    from kg_hypergraph import build_driver, get_risk_pattern
except Exception:  # pragma: no cover
    build_driver = None
    get_risk_pattern = None


LOGGER = logging.getLogger(__name__)

ERROR_NODE_NAME = "Error"
ERROR_HANDLER_NAME = "Error Handler"

RULE_ID_RE = re.compile(r"\b(H1[0-5]|H[1-9])\b", flags=re.IGNORECASE)
CLARIFICATION_RE = re.compile(r"(clarify|explain|没懂|不懂|再说一遍|解释)", flags=re.IGNORECASE)

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

【后台逻辑诊断结果（仅供你参考，绝不可直接暴露给学生）】：
{kb_context}

【🚨 核心强制护栏（Guardrails）- 违规将导致系统崩溃】：
1. 绝对禁止向学生提及"超图"、"逻辑风险"、"证据库"、"H1-H15规则"、"fallback"等后台工程词汇！你需要将这些抽象变量转化为人类视角的商业质问。
2. 绝对禁止以"评分预览：X分"作为回复开头，你是来对话的，不是来报分数的。
3. 绝对禁止使用Markdown格式（无加粗、无列表、无标题）。

【回复结构要求】（必须分成三个自然段，段落间空一行）：
第一段（状态判定与角色代入）：根据学生的输入，带入相应的角色语气（如协同队友、严厉VC）。点出他们项目目前最核心的商业或逻辑漏洞（基于后台诊断结果）。
第二段（深度质问或协同建议）：如果不自洽，给出具体的商业逻辑反问；如果学生要求协作撰写，给出你负责的那部分（如财务结构）的专业建议和所需数据。
第三段（下一步行动）：只提出唯一一个具体、可执行的任务或关键问题，强迫学生思考。
"""

# 初始化 OpenAI 客户端（使用 SiliconFlow 配置）
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)


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
        if _message_role(message) == "assistant":
            text = _message_text(message)
            if text:
                return text
    return ""


def _append_ai_message(state: AgentState, content: str) -> None:
    metadata = {
        "agent_role": state.get("active_agent", ""),
        "model_used": DEFAULT_MODEL
    }
    state["messages"] = list(state.get("messages", [])) + [AIMessage(content=content, metadata=metadata)]


def _is_clarification_intent(text: str) -> bool:
    return bool(CLARIFICATION_RE.search(str(text or "").strip()))


def _strip_rule_ids(text: str) -> str:
    return RULE_ID_RE.sub("", str(text or "")).replace("  ", " ").strip()


def _extract_target_index(text: str) -> int:
    digit = re.search(r"[1-6]", str(text or ""))
    if digit:
        return int(digit.group(0))
    cn_map = {
        "一": 1,
        "二": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
    }
    raw = str(text or "")
    for token, index in cn_map.items():
        if token in raw:
            return index
    return 0


def _extract_tutor_section(text: str, field_name: str) -> str:
    fields = ("Definition", "Example", "Common Mistakes", "Practice Task", "Expected Artifact", "Evaluation Criteria")
    pattern = rf"{re.escape(field_name)}:\s*(.*?)(?=\n(?:{'|'.join(re.escape(name) for name in fields)}):|\Z)"
    match = re.search(pattern, text, flags=re.DOTALL)
    if not match:
        return ""
    return " ".join(match.group(1).split()).strip()


def _build_clarification_reply(user_text: str, messages: Sequence[AnyMessage]) -> str:
    previous_reply = _latest_assistant_text(messages)
    if not previous_reply:
        return "Simple explanation: define the user first, then prove pain with evidence."

    idx = _extract_target_index(user_text)
    fields = ("Definition", "Example", "Common Mistakes", "Practice Task", "Expected Artifact", "Evaluation Criteria")
    if 1 <= idx <= 6:
        section = _extract_tutor_section(previous_reply, fields[idx - 1])
        if section:
            return f"第{idx}点: {section}"

    compact = " ".join(previous_reply.split())
    if len(compact) > 120:
        compact = compact[:117].rstrip() + "..."
    return f"Plain version: {compact}"


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
    async def wrapper(state: AgentState) -> AgentState:
        try:
            return await func(state)
        except Exception as exc:  # pragma: no cover
            LOGGER.exception("Node execution failed: %s", func.__name__)
            new_state = _clone_state(state)
            context_data = new_state["context_data"]
            context_data["error"] = {"node": func.__name__, "message": str(exc)}
            new_state["active_agent"] = ERROR_HANDLER_NAME
            new_state["next_step"] = "System error occurred. Please retry."
            _append_ai_message(new_state, "The system hit an error and switched to fallback response.")
            return new_state

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
                    driver = build_driver(uri, user, password)
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
        if driver is not None:
            try:
                driver.close()
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


@handle_agent_error
async def student_learning_tutor_node(state: AgentState) -> AgentState:
    new_state = _clone_state(state)
    new_state["active_agent"] = "Student Learning Tutor"

    context_data = new_state["context_data"]
    user_text = _latest_user_text(new_state.get("messages", []))

    # 检测简单问候语
    greeting_patterns = ["hi", "hello", "你好", "嗨", "hey", "hallo", "hiya"]
    normalized_user_text = user_text.strip().lower()
    is_greeting = any(greeting in normalized_user_text for greeting in greeting_patterns)
    is_short_greeting = len(normalized_user_text) <= 10 and not any(keyword in normalized_user_text for keyword in ["什么", "怎么", "如何", "为什么"])

    if _is_clarification_intent(user_text):
        reply = _build_clarification_reply(user_text, new_state.get("messages", []))
        new_state["next_step"] = "Clarification provided."
    elif is_greeting or is_short_greeting:
        # 友好的问候回复
        reply = "你好！我是你的创新创业学习导师。很高兴能帮助你完善创业项目。\n\n如果你有任何关于创业概念、项目诊断或竞赛准备的问题，随时告诉我。例如：\n- 什么是 TAM/SAM/SOM？\n- 如何分析竞品？\n- 我的项目逻辑有什么问题？\n- 如何准备互联网+比赛？"
        new_state["next_step"] = "Ask your first question about entrepreneurship or your project."
    elif not any(keyword in normalized_user_text for keyword in ["创业", "项目", "竞赛", "商业", "计划书", "bp", "互联网+", "挑战杯", "tam", "sam", "som", "竞品", "用户", "痛点", "价值", "渠道", "证据"]):
        # 与项目无关的问题，调用大模型回答
        system_prompt = "你是一位友好、专业的智能助手，能够回答各种问题。请直接回答用户的问题，不要添加任何额外的解释或引导。回复时请使用自然语言和段落形式，不要使用任何Markdown格式。如果内容较长，请合理分段，让回复更易读。"
        openai_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text}
        ]
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=openai_messages,
            temperature=0.7,
            max_tokens=500,
            timeout=30.0
        )
        reply = response.choices[0].message.content
        new_state["next_step"] = "回答了与项目无关的问题。"
    elif any(keyword in normalized_user_text for keyword in ["是什么", "什么意思", "定义", "概念", "区别", "怎么理解"]):
        # 概念性问题，调用大模型回答
        system_prompt = "你是一位专业的创新创业导师，擅长解释创业相关的概念和术语。请直接、清晰地回答用户的问题，提供专业且易于理解的解释。回复时请使用自然语言和段落形式，不要使用任何Markdown格式。如果内容较多，请合理分段，让回复更易读。"
        openai_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text}
        ]
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=openai_messages,
            temperature=0.3,
            max_tokens=500,
            timeout=30.0
        )
        reply = response.choices[0].message.content
        new_state["next_step"] = "回答了概念性问题。"
    else:
        # 项目相关问题，调用大模型回答并使用苏格拉底式提问引导学生
        
        # RAG 知识卡片召回逻辑
        try:
            from core.logic_adapter import retrieve_knowledge_cards
            import jieba
            
            keywords = list(jieba.cut(user_text))[:5]
            cards = retrieve_knowledge_cards(keywords, limit=1)
            
            if cards:
                card_context = "【系统检索到的关联案例】：项目名：{0}，目标客户：{1}，核心策略：{2}。".format(
                    cards[0].get('title', '未知'),
                    cards[0].get('target', '未知'),
                    cards[0].get('summary', '未知')
                )
            else:
                card_context = "【系统暂无直接匹配案例，请自行引用美团、闲鱼、大疆等家喻户晓的真实商业案例进行类比】。"
        except Exception as e:
            card_context = "【请自行引用真实的知名商业案例进行说明】。"

        # 将案例上下文编织进 System Prompt
        system_prompt = """你是一位专业的创新创业导师，擅长使用苏格拉底式教学法。
        
        {0}
        
        请严格按照以下三步处理学生的提问（分三段，段落间空行，绝不使用任何Markdown格式）：
        第一段（原理解释）：清晰解答学生的疑惑。
        第二段（具象案例）：必须结合上述【关联案例】（或你自带的经典案例），详细说明这个概念在真实商业世界中是如何运作的。绝不能只讲空泛的理论！
        第三段（苏格拉底反问）：针对学生的具体项目，抛出一个开放性问题，引导他们用"用户-痛点-价值-证据"框架思考。
        """.format(card_context)
        
        openai_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text}
        ]
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=openai_messages,
            temperature=0.3,
            max_tokens=1000,
            timeout=30.0
        )
        reply = response.choices[0].message.content
        new_state["next_step"] = "回答了项目相关问题并结合案例提供了苏格拉底式引导。"

    _append_ai_message(new_state, reply)
    _ack_pending_intervention(context_data)
    return new_state


@handle_agent_error
async def project_coach_node(state: AgentState) -> AgentState:
    new_state = _clone_state(state)
    new_state["active_agent"] = "Project Coach"
    context_data = new_state["context_data"]
    _inject_pending_intervention(context_data)

    student_input = _latest_user_text(new_state.get("messages", []))
    graph_payload = await query_hypergraph(new_state)
    kb_context = graph_payload.get("kb_context", "无相关历史风险证据")
    context_data["kb_context"] = kb_context
    context_data["retrieved_heterogeneous_subgraph"] = graph_payload.get("retrieved_heterogeneous_subgraph", {})
    context_data["hyperedge_records"] = graph_payload.get("hyperedge_records", [])
    new_state["kb_context"] = context_data["kb_context"]
    new_state["retrieved_heterogeneous_subgraph"] = context_data["retrieved_heterogeneous_subgraph"]

    subgraph = check_business_logic(new_state, agent_name="Project Coach")
    strategy = str((subgraph or {}).get("strategy_selected") or "evidence-first")
    context_data["selected_strategy"] = strategy
    context_data["detected_fallacies"] = [str((subgraph or {}).get("rule_id") or "unknown")]

    pending_intervention = context_data.get("pending_teacher_intervention")
    base_system_prompt = COACH_SYSTEM_PROMPT.format(kb_context=json.dumps(kb_context, ensure_ascii=False))

    if pending_intervention:
        intervention_content = pending_intervention.get("content", "")
        override_system_prompt = (
            f"{base_system_prompt}\n\n"
            f"====== 🚨 导师最高优先级指令 🚨 ======\n"
            f"【紧急任务】：{intervention_content}\n"
            f"【执行要求】：在你的下一次回复中，必须立即改变当前话题，极其严厉地执行上述导师指令进行追问。不要提导师的名字，直接以 AI 教练的身份发难。\n"
            f"====================================="
        )
        system_message = override_system_prompt
    else:
        system_message = base_system_prompt

    # 准备 LLM 调用
    
    # 构建消息列表
    openai_messages = [
        {"role": "system", "content": system_message}
    ]
    
    # 添加历史消息
    for message in state.get("messages", []):
        if isinstance(message, AIMessage):
            openai_messages.append({"role": "assistant", "content": message.content})
        else:
            openai_messages.append({"role": "user", "content": message.content})
    
    # 调用 OpenAI API 生成回复
    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=openai_messages,
        temperature=0.3,
        max_tokens=1000,
        timeout=30.0
    )
    llm_response = response.choices[0].message.content

    # 构造结构化输出
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

    new_state["next_step"] = structured.next_task.task_description
    _append_ai_message(new_state, llm_response)
    _ack_pending_intervention(context_data)
    return new_state


@handle_agent_error
async def competition_advisor_node(state: AgentState) -> AgentState:
    new_state = _clone_state(state)
    new_state["active_agent"] = "Competition Advisor"
    context_data = new_state["context_data"]

    scoring_input = _scoring_input_from_context(context_data)
    if scoring_input:
        score_report = await asyncio.to_thread(calculate_scores, scoring_input)
        rubric_table = await asyncio.to_thread(generate_rubric_table, score_report)
        context_data["competition_score_report"] = score_report
        context_data["competition_rubric_preview"] = rubric_table.to_dict("records")
        low_items = score_report.get("low_score_items", [])
        new_state["next_step"] = f"优先修复 {low_items[0]} 的证据不足问题。" if low_items else "继续加强证据支撑。"
        reply = f"竞赛评分预览：{score_report.get('overall_score', 0)}/5；需要改进的项目：{', '.join(low_items[:3]) or '无'}。"
    else:
        new_state["next_step"] = "在评分前请提供项目摘要。"
        reply = "缺少评分所需信息，请提供项目摘要和关键指标。"

    _append_ai_message(new_state, reply)
    return new_state

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


@handle_agent_error
async def assessment_assistant_node(state: AgentState) -> AgentState:
    new_state = _clone_state(state)
    new_state["active_agent"] = "Assessment Assistant"
    context_data = new_state["context_data"]

    diagnostics = context_data.get("diagnostic_results", [])
    diagnostics = diagnostics if isinstance(diagnostics, list) else []

    # NOTE: Reuse scoring_engine as the single scoring source of truth.
    ability_report = await asyncio.to_thread(
        generate_ability_report,
        new_state.get("messages", []),
        diagnostics,
        context_data,
    )
    context_data["ability_report"] = dict(ability_report)

    raw_rubric_rows = _coerce_rubric_rows(ability_report, context_data)
    quotes = [text for text in (_latest_user_text(new_state.get("messages", [])),) if text]
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
    new_state["rubric_score_results"] = normalized_results

    low_items: List[str] = []
    score_report = context_data.get("competition_score_report", {})
    if isinstance(score_report, dict):
        low_items = list(score_report.get("low_score_items", []) or [])

    new_state["next_step"] = "Fill evidence gaps for the lowest rubric dimension."
    _append_ai_message(
        new_state,
        "\n".join(
            [
                "Assessment report generated.",
                f"- Rubric rows: {len(normalized_results)}",
                f"- Low dimensions: {', '.join(low_items[:3]) or 'none'}",
                f"- Bound AuditTrail rows: {len(evidence_list)}",
            ]
        ),
    )
    return new_state


@handle_agent_error
async def instructor_assistant_node(state: AgentState) -> AgentState:
    new_state = _clone_state(state)
    new_state["active_agent"] = "Instructor Assistant"
    context_data = new_state["context_data"]

    diagnostics = context_data.get("diagnostic_results", [])
    diagnostics = diagnostics if isinstance(diagnostics, list) else []

    ability_report = await asyncio.to_thread(
        generate_ability_report,
        new_state.get("messages", []),
        diagnostics,
        context_data,
    )
    context_data["ability_report"] = dict(ability_report)
    avg_score = ability_report.get("average_score_5", 0)

    new_state["next_step"] = "Schedule one focused intervention and track the next re-test."
    _append_ai_message(new_state, f"Instructor summary: current average ability score is {avg_score}/5.")
    return new_state


@handle_agent_error
async def error_node(state: AgentState) -> AgentState:
    new_state = _clone_state(state)
    new_state["active_agent"] = ERROR_NODE_NAME
    new_state["next_step"] = "Adjust input and retry."
    _append_ai_message(new_state, "Fallback handler activated due to input/system error.")
    return new_state


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
    "error_node",
]
