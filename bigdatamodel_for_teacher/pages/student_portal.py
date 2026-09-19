from __future__ import annotations

import asyncio
import html
import hashlib
import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st
import plotly.graph_objects as go
from langchain_core.messages import BaseMessage

from core.rbac import enforce_rbac
from core.graph import build_agent_state, get_agent_app
from core.db_init import init_interaction_log_db, log_interaction_to_db_async
from core.project_data_service import load_project_data, save_project_data, create_project
from core.contest_engine import calculate_contest_scores, CONTEST_MODELS
from config.api_config import DEFAULT_MODEL
import logic_adapter
from ingest import load_document_bytes
from scoring_engine import (
    calculate_scores,
    generate_ability_report,
    generate_rubric_table,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROJECT_UPLOAD_TYPES = ["pdf", "doc", "docx"]
PROJECT_TEXT_PREVIEW_LENGTH = 1200
STAGE_FLOW = [
    "核心价值探测",
    "逻辑压力测试",
    "落地可行性校验",
]
DEFAULT_ASSISTANT_MESSAGE = (
    "我是你的项目陪跑教练。先上传项目计划书并完成初步诊断，"
    "然后我们按“核心价值探测 -> 逻辑压力测试 -> 落地可行性校验”的顺序推进。"
)
SHOW_AGENT_DEBUG = str(os.getenv("SHOW_AGENT_DEBUG", "")).strip() == "1"

# 默认学生画像
DEFAULT_PROFILE = {
    "user_id": "",
    "name": "",
    "role": "student",
    "scores": [0, 0, 0, 0, 0],  # 五力模型初始值
    "projects": [],
    "diagnostic_history": [],
    "grade": "",
    "major": "",
    "interests": []
}

# 五力模型维度
FIVE_FORCES = ["痛点发现", "方案策划", "商业建模", "资源杠杆", "逻辑表达"]

enforce_rbac("student", page_name="student_portal")


def load_student_profile(user_id: str) -> dict:
    """加载学生画像，支持冷启动"""
    profile_path = PROJECT_ROOT / "data" / "profiles" / f"{user_id}.json"
    if profile_path.exists():
        try:
            return json.loads(profile_path.read_text())
        except Exception:
            return DEFAULT_PROFILE.copy()
    return DEFAULT_PROFILE.copy()


def save_student_profile(user_id: str, profile: dict) -> None:
    """保存学生画像"""
    profile_dir = PROJECT_ROOT / "data" / "profiles"
    profile_dir.mkdir(exist_ok=True, parents=True)
    profile_path = profile_dir / f"{user_id}.json"
    try:
        profile_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2))
    except Exception:
        pass


def render_radar_chart(ability_report: dict):
    """渲染五力模型雷达图"""
    if not ability_report:
        # 冷启动状态，显示默认雷达图
        fig = go.Figure(data=go.Scatterpolar(
            r=[0, 0, 0, 0, 0],
            theta=FIVE_FORCES,
            fill='toself'
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
            showlegend=False,
            title="能力画像（开始对话以解锁）"
        )
        st.plotly_chart(fig)
        st.info("开始对话以解锁画像")
        return

    # 从能力报告中提取数据
    radar_data = ability_report.get("radar_chart_data", {})
    r_values = radar_data.get("r", [0, 0, 0, 0, 0])
    theta_values = radar_data.get("theta", FIVE_FORCES)

    fig = go.Figure(data=go.Scatterpolar(
        r=r_values,
        theta=theta_values,
        fill='toself'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
        showlegend=False,
        title="五力模型能力画像"
    )
    st.plotly_chart(fig)


def init_student_state() -> None:
    """Initialize student portal state while preserving shared multipage context."""
    defaults: Dict[str, Any] = {
        "role": "student",
        "user_id": "",
        "project_id": None,
        "chat_history": [],
        "messages": [],
        "current_project": None,
        "diag_results": [],
        "current_stage": STAGE_FLOW[0],
        "diagnostic_stage_index": 0,
        "project_full_text": "",
        "project_upload_name": "",
        "project_upload_signature": "",
        "project_upload_parser": "",
        "project_upload_warnings": [],
        "project_upload_error": "",
        "project_source_locator": "",
        "retrieved_cards": [],
        "student_ability_report": None,
        "current_project_id": None,
        "agent_next_step": "",
        "agent_active_agent": "",
        "agent_route_target": "",
        "agent_next_agent": "",
        "agent_path_evidence": [],
        "agent_response_severity": "",
        "task_tracker_cards": [],
        "student_session_id": "",
        "student_profile": {},
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # 加载学生画像
    user_id = st.session_state.get("user_id", "") or st.session_state.get("student_session_id", "")
    if user_id:
        st.session_state.student_profile = load_student_profile(user_id)
    else:
        st.session_state.student_profile = DEFAULT_PROFILE.copy()

    if not st.session_state.messages and st.session_state.chat_history:
        st.session_state.messages = list(st.session_state.chat_history)

    if not st.session_state.messages:
        st.session_state.messages = [
            {"role": "assistant", "content": DEFAULT_ASSISTANT_MESSAGE}
        ]

    init_interaction_log_db()
    _ensure_student_session_id()
    sync_chat_history()
    sync_stage_progress()


def sync_chat_history() -> None:
    """Keep the shared chat transcript aligned across pages."""
    st.session_state.chat_history = list(st.session_state.messages)


def build_project_upload_signature(file_name: str, raw_bytes: bytes) -> str:
    digest = hashlib.sha1(raw_bytes).hexdigest()
    return f"{file_name}:{len(raw_bytes)}:{digest}"


def build_project_id(seed_text: str) -> str:
    digest = hashlib.sha1(seed_text.encode("utf-8")).hexdigest()
    return digest[:12]


def set_active_project_id(seed_text: str) -> None:
    project_id = build_project_id(seed_text)
    st.session_state.project_id = project_id
    st.session_state.current_project_id = project_id


def _ensure_student_session_id() -> str:
    session_id = str(st.session_state.get("student_session_id", "")).strip()
    if session_id:
        return session_id
    session_id = uuid.uuid4().hex
    st.session_state.student_session_id = session_id
    return session_id


def _extract_triggered_rules(event_type: str, details: Dict[str, Any]) -> List[Dict[str, Any]]:
    if event_type not in {"chat_turn", "diagnostic_run"}:
        return []
    raw_rules = details.get("diagnostic_results")
    if isinstance(raw_rules, dict):
        raw_rules = [raw_rules]
    if not isinstance(raw_rules, list):
        return []

    normalized: List[Dict[str, Any]] = []
    for item in raw_rules:
        if isinstance(item, dict):
            normalized.append(dict(item))
        elif isinstance(item, str):
            rule_id = item.strip().upper()
            if rule_id:
                normalized.append({"rule_id": rule_id})
    return normalized


def append_behavior_log(event_type: str, details: Dict[str, Any]) -> None:
    """Persist one interaction event into SQLite-backed structured logs."""
    default_agent_role = "Student Learning Tutor" if event_type == "chat_turn" else f"student_portal:{event_type}"
    student_id = str(
        st.session_state.get("user_id")
        or st.session_state.get("project_id")
        or ""
    ).strip()
    project_id = str(st.session_state.get("project_id") or "").strip()
    agent_role = str(
        details.get("agent_role")
        or st.session_state.get("agent_active_agent")
        or default_agent_role
    ).strip()
    user_input = str(details.get("user_prompt") or details.get("file_name") or event_type).strip()
    agent_response = str(
        details.get("assistant_reply")
        or details.get("error")
        or details.get("source_locator")
        or details.get("parser")
        or ""
    ).strip()
    next_step = str(details.get("next_step") or st.session_state.get("agent_next_step") or "").strip()
    session_id = str(details.get("session_id") or _ensure_student_session_id()).strip()
    triggered_rules = _extract_triggered_rules(event_type, details)

    log_interaction_to_db_async(
        student_id=student_id,
        project_id=project_id,
        agent_role=agent_role,
        user_input=user_input,
        agent_response=agent_response,
        triggered_rules=triggered_rules,
        next_step=next_step,
        session_id=session_id,
    )


def build_diagnostic_assistant_reply(
    diagnostic_results: List[Dict[str, Any]],
) -> str:
    """Convert diagnostic hits into a short coaching handoff message."""
    if not diagnostic_results:
        return ""

    guidance_lines = ["我先完成了一轮诊断，当前最值得优先处理的问题是："]
    for item in diagnostic_results:
        message = str(
            item.get("assistant_message")
            or item.get("trigger_message")
            or item.get("raw_message")
            or ""
        ).strip()
        if message:
            guidance_lines.append(f"- {message}")
    return "\n".join(guidance_lines)


def clear_uploaded_project_context(reset_identity: bool = False) -> None:
    """Clear upload-derived context while keeping unrelated sidebar state intact."""
    st.session_state.current_project = None
    st.session_state.project_full_text = ""
    st.session_state.project_upload_name = ""
    st.session_state.project_upload_signature = ""
    st.session_state.project_upload_parser = ""
    st.session_state.project_upload_warnings = []
    st.session_state.project_upload_error = ""
    st.session_state.project_source_locator = ""
    st.session_state.diag_results = []
    st.session_state.retrieved_cards = []
    st.session_state.student_ability_report = None
    st.session_state.diagnostic_stage_index = 0
    st.session_state.current_stage = STAGE_FLOW[0]

    if reset_identity:
        st.session_state.project_id = None
        st.session_state.current_project_id = None
        st.session_state.student_session_id = uuid.uuid4().hex


def reset_dialogue_for_new_project() -> None:
    """Reset dialogue when the user explicitly uploads a different project."""
    st.session_state.messages = [
        {"role": "assistant", "content": DEFAULT_ASSISTANT_MESSAGE}
    ]
    st.session_state.student_session_id = uuid.uuid4().hex
    sync_chat_history()


def sync_uploaded_project_file(uploaded_file: Any) -> None:
    """Parse the uploaded project document and cache the extracted text."""
    if uploaded_file is None:
        if st.session_state.project_upload_signature:
            clear_uploaded_project_context(reset_identity=True)
            reset_dialogue_for_new_project()
        return

    raw_bytes = uploaded_file.getvalue()
    upload_signature = build_project_upload_signature(uploaded_file.name, raw_bytes)

    if (
        upload_signature == st.session_state.project_upload_signature
        and st.session_state.project_full_text
    ):
        return

    try:
        document = load_document_bytes(
            source_name=uploaded_file.name,
            raw_bytes=raw_bytes,
            source_path=f"uploaded://{uploaded_file.name}",
        )
    except Exception as exc:
        clear_uploaded_project_context(reset_identity=False)
        st.session_state.project_upload_error = str(exc)
        append_behavior_log(
            "project_upload_failed",
            {"file_name": uploaded_file.name, "error": str(exc)},
        )
        return

    is_new_project = upload_signature != st.session_state.project_upload_signature
    st.session_state.current_project = document.cleaned_text
    st.session_state.project_full_text = document.cleaned_text
    st.session_state.project_upload_name = uploaded_file.name
    st.session_state.project_upload_signature = upload_signature
    st.session_state.project_upload_parser = document.parser
    st.session_state.project_upload_warnings = list(document.warnings)
    st.session_state.project_upload_error = ""
    st.session_state.project_source_locator = document.source_locator or ""
    set_active_project_id(upload_signature)

    if is_new_project:
        st.session_state.diag_results = []
        st.session_state.retrieved_cards = []
        st.session_state.student_ability_report = None
        st.session_state.diagnostic_stage_index = 0
        st.session_state.current_stage = STAGE_FLOW[0]
        reset_dialogue_for_new_project()

    append_behavior_log(
        "project_upload_parsed",
        {
            "file_name": uploaded_file.name,
            "parser": document.parser,
            "text_length": len(document.cleaned_text),
            "warning_count": len(document.warnings),
            "source_locator": document.source_locator,
        },
    )


def sync_stage_progress() -> None:
    """Advance the three-stage diagnostic flow without regressing on reruns."""
    user_turns = sum(
        1
        for message in st.session_state.messages
        if isinstance(message, dict) and message.get("role") == "user"
    )

    derived_stage_index = 0
    if st.session_state.diag_results or user_turns >= 2:
        derived_stage_index = 1
    if user_turns >= 3:
        derived_stage_index = 2

    previous_stage_index = int(st.session_state.get("diagnostic_stage_index", 0) or 0)
    stage_index = min(
        len(STAGE_FLOW) - 1,
        max(previous_stage_index, derived_stage_index),
    )

    st.session_state.diagnostic_stage_index = stage_index
    st.session_state.current_stage = STAGE_FLOW[stage_index]


def build_scoring_input() -> Dict[str, Any]:
    """Build a lightweight score payload from uploaded text and recent dialogue."""
    project_text = (st.session_state.project_full_text or "").strip()
    if not project_text:
        return {}

    project_name = Path(
        st.session_state.project_upload_name or "uploaded_project"
    ).stem
    return {
        "project_name": project_name,
        "title": project_name,
        "description": project_text[:500],
        "narrative": project_text,
        "project_full_text": project_text,
        "metadata": {
            "project_full_text": project_text,
            "project_upload_name": st.session_state.project_upload_name,
            "project_upload_parser": st.session_state.project_upload_parser,
            "project_source_locator": st.session_state.project_source_locator,
        },
        "chat_transcript": [
            f"{item['role']}: {item['content']}"
            for item in st.session_state.messages[-8:]
            if isinstance(item, dict) and item.get("role") and item.get("content")
        ],
    }


def build_score_snapshot() -> Dict[str, Any]:
    """Return a compact score snapshot for logging only."""
    scoring_input = build_scoring_input()
    if not scoring_input:
        return {}

    score_report = calculate_scores(scoring_input)
    return {
        "overall_score": score_report.get("overall_score"),
        "weighted_percentage": score_report.get("weighted_percentage"),
        "low_score_items": score_report.get("low_score_items", []),
    }


def _message_role(message: Any) -> str:
    if isinstance(message, BaseMessage):
        message_type = getattr(message, "type", "")
        if message_type == "human":
            return "user"
        if message_type == "ai":
            return "assistant"
        return message_type or "assistant"
    if isinstance(message, dict):
        return str(message.get("role", "assistant"))
    return "assistant"


def _message_content(message: Any) -> str:
    if isinstance(message, BaseMessage):
        content = message.content
        if isinstance(content, list):
            return " ".join(str(item) for item in content if item is not None).strip()
        return str(content or "").strip()
    if isinstance(message, dict):
        return str(message.get("content", "")).strip()
    return str(message).strip()


def _stream_text_chunks(text: str):
    for line in text.splitlines(keepends=True):
        yield line
    if "\n" not in text:
        yield ""


def _render_assistant_text(text: str, severity: str = "", target: Any | None = None) -> None:
    """Render assistant content with severity-aware border colors."""
    cleaned_text = logic_adapter.strip_evidence_tag(text)
    color_map = {
        "high": "#d14343",
        "warning": "#f08c00",
    }
    border_color = color_map.get(str(severity).strip().lower())
    escaped_text = html.escape(cleaned_text).replace("\n", "<br>")

    if border_color:
        markup = (
            "<div style='border-left: 4px solid {color}; padding: 0.8rem 1rem; "
            "background: rgba(0,0,0,0.02); border-radius: 0.4rem; margin: 0.1rem 0 0.4rem 0;'>"
            "{text}</div>"
        ).format(color=border_color, text=escaped_text)
        if target is None:
            st.markdown(markup, unsafe_allow_html=True)
        else:
            target.markdown(markup, unsafe_allow_html=True)
        return

    if target is None:
        st.markdown(cleaned_text)
    else:
        target.markdown(cleaned_text)


def _assistant_message_severity(message: Dict[str, Any]) -> str:
    metadata = message.get("metadata", {}) if isinstance(message, dict) else {}
    if isinstance(metadata, dict):
        severity = str(metadata.get("severity", "")).strip()
        if severity:
            return severity

    evidence_payload = logic_adapter.extract_evidence_payload(
        message.get("content", "") if isinstance(message, dict) else ""
    )
    return str(evidence_payload.get("response_severity", "")).strip()


def _render_hypergraph_trace_panel(
    retrieved_subgraph: Any,
    detected_fallacies: Any,
    selected_strategy: Any,
) -> None:
    if not retrieved_subgraph:
        return

    with st.expander(" [验收专用] 底层超图推理追踪 (Hypergraph Trace)", expanded=False):
        activated_rules = set()
        activated_rules_list = []
        hyperedges = []
        edges = []
        nodes = []

        if isinstance(retrieved_subgraph, dict):
            hyperedges = retrieved_subgraph.get("hyperedges", [])
            edges = retrieved_subgraph.get("edges", [])
            nodes = retrieved_subgraph.get("nodes", [])

            edge_type_to_rules = {}
            for hyperedge in hyperedges:
                if isinstance(hyperedge, dict):
                    logic_desc = hyperedge.get("logic_description", "")
                    edge_type = hyperedge.get("type", "")
                    rules = re.findall(r'H\d+', logic_desc + edge_type)
                    activated_rules.update(rules)
                    for rule in rules:
                        if rule not in edge_type_to_rules:
                            edge_type_to_rules[rule] = []
                        edge_type_to_rules[rule].append(hyperedge)

            activated_rules_list = list(activated_rules)[:5]

        st.markdown("### 激活的逻辑规则")
        if activated_rules_list:
            rule_colors = {
                "H1": "#007bff", "H4": "#28a745", "H6": "#ffc107",
                "H8": "#dc3545", "H9": "#6f42c1", "H10": "#fd7e14", "H15": "#17a2b8"
            }
            cols = st.columns(min(len(activated_rules_list), 4))
            for idx, rule in enumerate(activated_rules_list):
                color = rule_colors.get(rule, "#6c757d")
                cols[idx % 4].markdown(
                    f"<span style='color: {color}; font-weight: bold; font-size: 1.1em;'>{rule}</span>",
                    unsafe_allow_html=True
                )
        else:
            st.markdown("- 无")

        st.markdown("### 决策策略关联")
        st.markdown(f"**选择的策略**: {selected_strategy if selected_strategy else '未提供'}")

        st.markdown("### 逻辑谬误追踪")
        if detected_fallacies:
            for fallacy in detected_fallacies[:3]:
                st.markdown(f"- **{fallacy}**")
        else:
            st.markdown("- 无")

        with st.expander("查看激活规则详情", expanded=False):
            for rule in activated_rules_list:
                with st.expander(f"为什么触发 {rule}", expanded=False):
                    related_hyperedges = [
                        h for h in hyperedges
                        if isinstance(h, dict) and rule in (h.get("logic_description", "") + h.get("type", ""))
                    ]
                    if related_hyperedges:
                        st.markdown("**相关超边**:")
                        for h in related_hyperedges[:2]:
                            st.markdown(f"- **{h.get('edge_id', 'N/A')}**: {h.get('logic_description', '')[:100]}")
                    else:
                        st.markdown("**相关超边**: 无")

                    related_edges = [
                        e for e in edges
                        if isinstance(e, dict) and rule in e.get("logic", "")
                    ]
                    if related_edges:
                        st.markdown("**相关路径**:")
                        for e in related_edges[:2]:
                            st.markdown(f"- {e.get('source', '?')} --[{e.get('type', '')}]--> {e.get('target', '?')}")

        with st.expander("查看完整路径列表", expanded=False):
            st.markdown("### 超图路径证据")
            if edges:
                for i, edge in enumerate(edges[:10]):
                    if isinstance(edge, dict):
                        source = edge.get("source", "")
                        target = edge.get("target", "")
                        edge_type = edge.get("type", "")
                        source_name = source
                        target_name = target
                        for node in nodes:
                            if isinstance(node, dict):
                                if node.get("id") == source:
                                    source_name = node.get("name", source)
                                if node.get("id") == target:
                                    target_name = node.get("name", target)
                        st.markdown(f"**路径 {i+1}**: {source_name} --[{edge_type}]--> {target_name}")
            else:
                st.markdown("- 无")

        with st.expander("查看原始数据（JSON）", expanded=False):
            st.json(retrieved_subgraph, expanded=False)


def build_agent_context_data() -> Dict[str, Any]:
    """Build the LangGraph context payload from current portal state."""
    scoring_input = build_scoring_input()
    student_id = str(
        st.session_state.get("user_id")
        or st.session_state.get("project_id")
        or ""
    ).strip()
    return {
        "portal_role": st.session_state.role,
        "student_id": student_id,
        "user_id": str(st.session_state.get("user_id", "")).strip(),
        "diagnostic_results": st.session_state.diag_results,
        "current_stage": st.session_state.current_stage,
        "project_full_text": st.session_state.project_full_text,
        "project_name": Path(st.session_state.project_upload_name or "uploaded_project").stem,
        "project_id": st.session_state.project_id,
        "project_source_locator": st.session_state.project_source_locator,
        "project_upload_name": st.session_state.project_upload_name,
        "retrieved_knowledge_cards": st.session_state.retrieved_cards,
        "path_evidence": st.session_state.agent_path_evidence,
        "scoring_input": scoring_input,
    }


def update_agent_tracker(final_state: Dict[str, Any]) -> None:
    """Persist graph outputs into Streamlit session state."""
    context_data = final_state.get("context_data", {})
    st.session_state.agent_next_step = str(final_state.get("next_step", "")).strip()
    st.session_state.agent_active_agent = str(
        final_state.get("active_agent") or context_data.get("route_target") or ""
    ).strip()
    st.session_state.agent_route_target = str(context_data.get("route_target", "")).strip()
    st.session_state.agent_next_agent = str(final_state.get("next_agent", "")).strip()
    st.session_state.agent_path_evidence = list(context_data.get("path_evidence", []))
    st.session_state.agent_response_severity = str(
        context_data.get("response_severity")
        or logic_adapter.infer_response_severity(context_data.get("diagnostic_results", []))
        or ""
    ).strip()

    retrieved_cards = context_data.get("retrieved_knowledge_cards")
    if isinstance(retrieved_cards, list):
        st.session_state.retrieved_cards = retrieved_cards


def _severity_rank(severity: str) -> int:
    normalized = str(severity or "").strip().lower()
    if normalized == "high":
        return 2
    if normalized == "warning":
        return 1
    return 0


def _active_diagnostic_items() -> List[Dict[str, Any]]:
    items = []
    for item in st.session_state.get("diag_results", []):
        if not isinstance(item, dict):
            continue
        status = str(item.get("status", "")).strip().lower()
        if status in {"passed", "pass", "success"}:
            continue
        items.append(item)
    return items


def _infer_task_rule_id(next_step: str, diag_items: List[Dict[str, Any]]) -> str:
    match = re.search(r"\b(H1[0-5]|H[1-9])\b", str(next_step or ""), flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()

    normalized_next_step = str(next_step or "").strip().lower()
    for item in diag_items:
        for key in ("fix_suggestion", "assistant_message", "trigger_message", "raw_message"):
            value = str(item.get(key, "")).strip().lower()
            if value and (value in normalized_next_step or normalized_next_step in value):
                return str(item.get("rule_id", "")).strip().upper()

    sorted_items = sorted(
        diag_items,
        key=lambda item: (
            -_severity_rank(str(item.get("severity", ""))),
            str(item.get("rule_id", "")),
        ),
    )
    return str(sorted_items[0].get("rule_id", "")).strip().upper() if sorted_items else ""


def _severity_for_rule(rule_id: str, diag_items: List[Dict[str, Any]], fallback: str = "") -> str:
    for item in diag_items:
        if str(item.get("rule_id", "")).strip().upper() == rule_id:
            return str(item.get("severity", "")).strip() or fallback
    return fallback


def sync_task_tracker_cards() -> None:
    """Parse next_step into task cards and auto-complete resolved rule tasks."""
    diag_items = _active_diagnostic_items()
    active_rule_ids = {
        str(item.get("rule_id", "")).strip().upper()
        for item in diag_items
        if str(item.get("rule_id", "")).strip()
    }
    cards = [
        dict(card)
        for card in st.session_state.get("task_tracker_cards", [])
        if isinstance(card, dict)
    ]

    for card in cards:
        rule_id = str(card.get("rule_id", "")).strip().upper()
        if rule_id:
            card["completed"] = rule_id not in active_rule_ids
        card["is_current"] = False

    next_step = str(st.session_state.get("agent_next_step", "") or "").strip()
    if next_step:
        rule_id = _infer_task_rule_id(next_step, diag_items)
        severity = _severity_for_rule(
            rule_id,
            diag_items,
            st.session_state.get("agent_response_severity", ""),
        )
        card_id = rule_id or hashlib.sha1(next_step.encode("utf-8")).hexdigest()[:12]
        completed = bool(rule_id and rule_id not in active_rule_ids)
        urgent = str(severity).strip().lower() == "high" or rule_id in {"H4", "H8"}
        updated_card = {
            "card_id": card_id,
            "rule_id": rule_id,
            "task_text": next_step,
            "severity": severity,
            "completed": completed,
            "is_current": True,
            "updated_at": datetime.now().strftime("%H:%M:%S"),
            "urgent": urgent,
        }

        existing_index = next(
            (index for index, card in enumerate(cards) if card.get("card_id") == card_id),
            None,
        )
        if existing_index is None:
            cards.insert(0, updated_card)
        else:
            cards[existing_index] = updated_card

    st.session_state.task_tracker_cards = cards[:5]


def _render_task_tracker_styles() -> None:
    st.markdown(
        """
        <style>
        .task-card {
            border: 1px solid rgba(49, 51, 63, 0.18);
            border-left: 4px solid #6c757d;
            border-radius: 0.6rem;
            padding: 0.7rem 0.9rem 0.45rem 0.9rem;
            margin: 0.45rem 0 0.35rem 0;
            background: rgba(255, 255, 255, 0.02);
        }
        .task-card.high {
            border-left-color: #d14343;
            background: rgba(209, 67, 67, 0.08);
        }
        .task-card.warning {
            border-left-color: #f08c00;
            background: rgba(240, 140, 0, 0.06);
        }
        .task-card.done {
            opacity: 0.75;
        }
        .task-badge {
            display: inline-block;
            font-size: 0.72rem;
            font-weight: 700;
            padding: 0.08rem 0.45rem;
            border-radius: 999px;
            margin-right: 0.35rem;
        }
        .task-badge.high {
            color: #fff;
            background: #d14343;
        }
        .task-badge.warning {
            color: #fff;
            background: #f08c00;
        }
        .task-badge.urgent {
            color: #fff;
            background: #b42318;
            animation: taskPulse 1.2s infinite;
        }
        .task-title {
            font-size: 0.78rem;
            font-weight: 600;
            margin-bottom: 0.35rem;
            line-height: 1.45;
        }
        .task-meta {
            font-size: 0.72rem;
            color: rgba(49, 51, 63, 0.72);
            margin-bottom: 0.25rem;
        }
        @keyframes taskPulse {
            0% { opacity: 1; box-shadow: 0 0 0 0 rgba(180, 35, 24, 0.45); }
            70% { opacity: 0.78; box-shadow: 0 0 0 6px rgba(180, 35, 24, 0); }
            100% { opacity: 1; box-shadow: 0 0 0 0 rgba(180, 35, 24, 0); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_task_tracker() -> None:
    """Render the sidebar task tracker driven by LangGraph next_step."""
    sync_task_tracker_cards()
    _render_task_tracker_styles()

    st.subheader("任务追踪器")
    if SHOW_AGENT_DEBUG and st.session_state.agent_active_agent:
        st.caption(f"当前智能体：{st.session_state.agent_active_agent}")
    if SHOW_AGENT_DEBUG and st.session_state.agent_route_target:
        st.caption(f"当前路由：{st.session_state.agent_route_target}")
    cards = st.session_state.get("task_tracker_cards", [])
    if not cards:
        st.caption("每轮对话后，这里会显示唯一下一步任务。")
    else:
        for card in cards:
            severity = str(card.get("severity", "")).strip().lower()
            rule_id = str(card.get("rule_id", "")).strip()
            urgent = bool(card.get("urgent"))
            completed = bool(card.get("completed"))
            badge_markup = ""
            if urgent:
                badge_markup += "<span class='task-badge urgent'>紧急任务</span>"
            if severity in {"high", "warning"}:
                severity_label = "高风险" if severity == "high" else "预警"
                badge_markup += f"<span class='task-badge {severity}'>{severity_label}</span>"
            if SHOW_AGENT_DEBUG and rule_id:
                badge_markup += f"<span class='task-badge'>{rule_id}</span>"

            card_classes = ["task-card"]
            if severity in {"high", "warning"}:
                card_classes.append(severity)
            if completed:
                card_classes.append("done")

            st.markdown(
                (
                    f"<div class='{' '.join(card_classes)}'>"
                    f"<div>{badge_markup}</div>"
                    f"<div class='task-title'>{html.escape(str(card.get('task_text', '')))}</div>"
                    f"<div class='task-meta'>更新时间: {html.escape(str(card.get('updated_at', '')))}</div>"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )
            st.checkbox(
                "已达成" if completed else "待完成",
                value=completed,
                disabled=True,
                key=f"task_tracker_checkbox_{card.get('card_id')}",
            )

    if SHOW_AGENT_DEBUG and st.session_state.agent_path_evidence:
        with st.expander("图谱溯源路径", expanded=False):
            for item in st.session_state.agent_path_evidence:
                if isinstance(item, dict):
                    path_string = str(item.get("path_string", "")).strip()
                    source_excerpt = str(item.get("source_excerpt", "")).strip()
                    if path_string:
                        st.code(path_string)
                    if source_excerpt:
                        st.caption(source_excerpt)
                else:
                    st.code(item)
    
    # 添加路由决策日志
    if SHOW_AGENT_DEBUG and (st.session_state.agent_next_agent or st.session_state.agent_route_target):
        with st.expander("[决策日志] 智能体路由", expanded=False):
            if st.session_state.agent_next_agent:
                st.markdown(f"**当前智能体**: {st.session_state.agent_next_agent}")
            if st.session_state.agent_route_target:
                st.markdown(f"**路由目标**: {st.session_state.agent_route_target}")
            if st.session_state.agent_active_agent:
                st.markdown(f"**活跃智能体**: {st.session_state.agent_active_agent}")
            st.markdown(f"**下一步任务**: {st.session_state.agent_next_step or '无'}")


def render_knowledge_sidebar() -> None:
    """Render the latest retrieved knowledge cards inside the sidebar."""
    cards = st.session_state.get("retrieved_cards", [])
    # 获取retrieved_subgraph，用于关联知识卡片与超图节点
    retrieved_subgraph = None
    # 从最近的助手消息中提取retrieved_subgraph
    for msg in reversed(st.session_state.messages):
        if msg.get("role") == "assistant" and isinstance(msg.get("metadata"), dict):
            retrieved_subgraph = msg["metadata"].get("retrieved_heterogeneous_subgraph")
            if retrieved_subgraph:
                break
    
    with st.sidebar.expander("相关知识储备", expanded=False):
        if not cards:
            st.write("暂无匹配知识点。")
            return

        for card in cards:
            title = card.get("title", "未命名卡片")
            summary = (
                card.get("summary")
                or card.get("snippet")
                or card.get("content", "")
            )
            matched = ", ".join(card.get("matched_keywords", []))
            st.markdown(f"**{title}**")
            if summary:
                st.caption(str(summary)[:140] + ("..." if len(str(summary)) > 140 else ""))
            if matched:
                st.caption(f"关键词：{matched}")
            
            # 标注卡片在超图中的位置
            if retrieved_subgraph and isinstance(retrieved_subgraph, dict):
                nodes = retrieved_subgraph.get("nodes", [])
                # 尝试匹配卡片标题与节点名称
                matched_nodes = []
                for node in nodes:
                    if isinstance(node, dict) and node.get("name"):
                        node_name = node.get("name", "")
                        node_id = node.get("id", "")
                        node_label = node.get("label", "")
                        # 简单的字符串匹配
                        if title.lower() in node_name.lower() or node_name.lower() in title.lower():
                            matched_nodes.append(f"{node_label}: {node_name} (ID: {node_id})")
                if matched_nodes:
                    st.caption(f"来自超图：{', '.join(matched_nodes[:2])}")  # 最多显示两个匹配节点
                else:
                    st.caption("来自超图：未明确关联")


def handle_project_upload() -> None:
    """Handle project upload, preview, and first-pass diagnostics."""
    uploaded_file = st.file_uploader(
        "上传你的项目计划书（PDF/Docx）",
        type=PROJECT_UPLOAD_TYPES,
        key="student_project_uploader",
    )
    sync_uploaded_project_file(uploaded_file)

    if st.session_state.project_upload_error:
        st.error(f"项目书解析失败：{st.session_state.project_upload_error}")
        return

    if st.session_state.project_full_text:
        st.success(
            f"已加载项目书《{st.session_state.project_upload_name}》，"
            f"提取文本 {len(st.session_state.project_full_text)} 字。"
        )
        if st.session_state.project_upload_warnings:
            for warning in st.session_state.project_upload_warnings:
                st.warning(warning)

        preview_text = st.session_state.project_full_text[:PROJECT_TEXT_PREVIEW_LENGTH]
        if len(st.session_state.project_full_text) > PROJECT_TEXT_PREVIEW_LENGTH:
            preview_text += "\n\n...[已截断预览]"
        with st.expander("查看项目书纯文本预览", expanded=False):
            st.text_area(
                "项目书纯文本",
                value=preview_text,
                height=220,
                disabled=True,
            )
    else:
        st.info("上传项目计划书后，这里会显示解析预览，并作为后续诊断上下文。")

    if st.button("开始深度诊断", use_container_width=True):
        if not st.session_state.project_full_text:
            st.warning("请先上传项目书，再开始诊断。")
            return

        with st.spinner("正在解析项目并生成诊断，请稍候..."):
            diagnostic_input = {
                "text": st.session_state.project_full_text,
                "project_full_text": st.session_state.project_full_text,
                "title": Path(st.session_state.project_upload_name or "uploaded_project").stem,
                "project_name": Path(
                    st.session_state.project_upload_name or "uploaded_project"
                ).stem,
                "metadata": {
                    "project_upload_name": st.session_state.project_upload_name,
                    "project_upload_parser": st.session_state.project_upload_parser,
                    "project_source_locator": st.session_state.project_source_locator,
                },
            }
            results = logic_adapter.run_diagnostics(diagnostic_input)
            st.session_state.diag_results = results
            sync_stage_progress()

            diagnostic_reply = build_diagnostic_assistant_reply(results)
            if diagnostic_reply:
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": diagnostic_reply,
                        "metadata": {
                            "severity": logic_adapter.infer_response_severity(results),
                        },
                    }
                )
                sync_chat_history()

            append_behavior_log(
                "diagnostic_run",
                {
                    "file_name": st.session_state.project_upload_name,
                    "diagnostic_results": results,
                    "score_snapshot": build_score_snapshot(),
                },
            )

        st.success("诊断完成。现在继续通过对话推进下一轮校验。")


def render_stage_status() -> None:
    """Render a compact three-stage progress summary."""
    current_index = st.session_state.diagnostic_stage_index
    stage_text = " -> ".join(
        f"[{stage}]" if index == current_index else stage
        for index, stage in enumerate(STAGE_FLOW)
    )
    st.caption(f"当前阶段：{st.session_state.current_stage}")
    st.progress((current_index + 1) / len(STAGE_FLOW))
    st.caption(stage_text)


def render_team_members() -> None:
    """在侧边栏渲染当前项目的团队成员列表，并提供邀请功能"""
    from core.project_data_service import get_project_members, load_project_data, save_project_data

    current_project_id = st.session_state.get("project_id")

    if not current_project_id:
        st.info("📌 请先上传计划书或创建项目，以解锁团队协作。")
        return

    members = get_project_members(current_project_id)

    with st.expander(f"👥 团队成员 ({len(members)}/5)", expanded=True):
        if not members:
            st.write("暂无成员信息")
        else:
            for member in members:
                role_icon = "👑" if member.get("team_role") in ["队长", "组长", "创始人"] else "👤"
                st.markdown(f"{role_icon} **{member.get('user_name', '未知用户')}** \n<span style='color:gray; font-size:12px;'>角色: {member.get('team_role', '队员')} | ID: {member.get('user_id', '')}</span>", unsafe_allow_html=True)

        st.divider()

        with st.form(f"add_member_form_{current_project_id}", border=False):
            st.write("➕ 邀请新成员")
            new_user_id = st.text_input("成员学号 / User ID", placeholder="例如：2023001")
            new_role = st.selectbox("分配团队角色", ["队员", "技术研发", "市场运营", "财务分析", "外部顾问"])

            submitted = st.form_submit_button("绑定到本项目", use_container_width=True)

            if submitted:
                if not new_user_id:
                    st.warning("请输入有效的用户 ID")
                else:
                    proj_data = load_project_data(current_project_id)
                    if proj_data:
                        if "members" not in proj_data:
                            proj_data["members"] = []

                        if any(m.get("user_id") == new_user_id for m in proj_data["members"]):
                            st.warning("该成员已在团队中。")
                        else:
                            proj_data["members"].append({
                                "user_id": new_user_id,
                                "user_name": "待系统同步",
                                "team_role": new_role
                            })
                            save_project_data(current_project_id, proj_data)
                            st.success(f"已发送绑定邀请！(需刷新生效)")


def _render_ability_report(report: Any) -> None:
    if not isinstance(report, dict):
        st.markdown(str(report))
        return

    report_markdown = str(report.get("report_markdown", "") or "").strip()
    if report_markdown:
        st.markdown(report_markdown)

    score_rows = report.get("score_table", [])
    audit_rows = report.get("audit_trail", [])
    audit_by_dimension = {}
    if isinstance(audit_rows, list):
        for row in audit_rows:
            if not isinstance(row, dict):
                continue
            dim = str(row.get("dimension", "")).strip()
            if dim:
                audit_by_dimension[dim] = row

    if isinstance(score_rows, list) and score_rows:
        # 👑 使用 HTML details 标签完美平替 st.expander，绕过嵌套限制
        html_str = """
        <details style='margin-bottom: 1rem; border: 1px solid #e0e0e0; border-radius: 8px; padding: 0.5rem 1rem;'>
            <summary style='cursor: pointer; font-weight: bold; outline: none;'>证据链溯源（点击展开）</summary>
            <div style='margin-top: 1rem; line-height: 1.6;'>
        """

        for row in score_rows:
            if not isinstance(row, dict):
                continue
            dimension = str(row.get("dimension", "未命名维度")).strip()
            score = row.get("score_5", "-")
            reason = str(row.get("reason", "") or "").strip()
            audit = audit_by_dimension.get(dimension, {})
            quote = str(
                audit.get("student_quote_snippet")
                or row.get("student_quote_snippet")
                or ""
            ).strip()

            html_str += f"<p><strong>{dimension}</strong> · {score}/5<br>"
            if reason:
                html_str += f"诊断解释：{reason}<br>"
            if quote:
                html_str += f"<span style='color:gray; font-size:0.85em;'>证据摘录：\"{quote}\"</span></p>"
            else:
                html_str += f"<span style='color:gray; font-size:0.85em;'>证据摘录：暂无直接原文，请补充学生原话或项目原文。</span></p>"
            
            retrieved_nodes = audit.get("retrieved_nodes", [])
            retrieved_hyperedges = audit.get("retrieved_hyperedges", [])
            evidence_nodes = audit.get("evidence_nodes", [])
            kg_card_ids = audit.get("kg_knowledge_card_ids", [])
            
            if retrieved_nodes or retrieved_hyperedges or evidence_nodes or kg_card_ids:
                html_str += "<p><strong>审计追踪详情：</strong></p><ul style='margin-top: 0;'>"
                if evidence_nodes:
                    html_str += "<li><strong>证据节点：</strong><ul>"
                    for node_id in evidence_nodes[:5]:
                        html_str += f"<li><code>{node_id}</code></li>"
                    html_str += "</ul></li>"
                if kg_card_ids:
                    html_str += "<li><strong>知识卡 ID：</strong><ul>"
                    for card_id in kg_card_ids[:3]:
                        html_str += f"<li><code>{card_id}</code></li>"
                    html_str += "</ul></li>"
                if retrieved_nodes:
                    html_str += "<li><strong>检索到的超图节点：</strong><ul>"
                    for node in retrieved_nodes[:3]:
                        if isinstance(node, dict):
                            html_str += f"<li><strong>{node.get('name', 'N/A')}</strong> (ID: <code>{node.get('id', 'N/A')}</code>, 标签: {node.get('labels', [])})</li>"
                    html_str += "</ul></li>"
                if retrieved_hyperedges:
                    html_str += "<li><strong>检索到的超边：</strong><ul>"
                    for hyperedge in retrieved_hyperedges[:2]:
                        if isinstance(hyperedge, dict):
                            html_str += f"<li><strong>{hyperedge.get('edge_type', 'N/A')}</strong> (ID: <code>{hyperedge.get('edge_id', 'N/A')}</code>)<br><span style='font-size:0.9em; color:gray;'>包含节点: {hyperedge.get('contained_nodes', [])}</span></li>"
                    html_str += "</ul></li>"
                html_str += "</ul><hr style='border: none; border-top: 1px solid #eee;' />"

        html_str += "</div></details>"
        
        # 统一输出渲染
        st.markdown(html_str, unsafe_allow_html=True)


def render_student_page() -> None:
    """Render the student portal with persistent stage flow and logs."""
    init_student_state()
    sync_stage_progress()

    st.title("创业项目陪跑教练（学生端）")

    # 顶部布局：基础资料和能力画像
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("基础资料管理")
        profile = st.session_state.student_profile
        
        # 编辑基础资料
        profile["grade"] = st.text_input("年级", value=profile.get("grade", ""))
        profile["major"] = st.text_input("专业", value=profile.get("major", ""))
        interests = st.text_input("兴趣领域（如 AI、乡村振兴等）", value=", ".join(profile.get("interests", [])))
        profile["interests"] = [i.strip() for i in interests.split(",") if i.strip()]
        
        # 保存资料
        if st.button("保存资料"):
            user_id = st.session_state.get("user_id", "") or st.session_state.get("student_session_id", "")
            if user_id:
                save_student_profile(user_id, profile)
                st.success("资料保存成功！")
        
        # 显示项目列表
        st.subheader("我的项目")
        projects = profile.get("projects", [])
        
        # 从上传的项目书提取项目名
        extracted_project_name = ""
        if st.session_state.project_upload_name:
            extracted_project_name = Path(st.session_state.project_upload_name).stem
        
        # 如果有提取到项目名，显示确认界面
        if extracted_project_name:
            existing_index = None
            for i, proj in enumerate(projects):
                if isinstance(proj, dict) and proj.get("name") == extracted_project_name:
                    existing_index = i
                    break
                elif proj == extracted_project_name:
                    existing_index = i
                    break
            
            if existing_index is not None:
                st.warning(f"检测到同名项目：**{extracted_project_name}**")
                col_confirm1, col_confirm2 = st.columns(2)
                with col_confirm1:
                    if st.button("✅ 确认更新", key="confirm_update_project"):
                        projects[existing_index] = {
                            "name": extracted_project_name,
                            "upload_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "role": "组长" if i == 0 else "组员"
                        }
                        profile["projects"] = projects
                        save_student_profile(
                            st.session_state.get("user_id", "") or st.session_state.get("student_session_id", ""),
                            profile
                        )
                        st.success(f"已更新项目：{extracted_project_name}")
                        st.rerun()
                with col_confirm2:
                    if st.button("❌ 取消", key="cancel_update_project"):
                        st.info(f"已取消更新：{extracted_project_name}")
            else:
                st.info(f"检测到新项目：**{extracted_project_name}**")
                col_confirm1, col_confirm2 = st.columns(2)
                with col_confirm1:
                    if st.button("✅ 添加到我的项目", key="add_new_project"):
                        projects.append({
                            "name": extracted_project_name,
                            "upload_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "role": "组长"
                        })
                        profile["projects"] = projects
                        save_student_profile(
                            st.session_state.get("user_id", "") or st.session_state.get("student_session_id", ""),
                            profile
                        )
                        st.success(f"已添加项目：{extracted_project_name}")
                        st.rerun()
                with col_confirm2:
                    if st.button("❌ 暂不添加", key="skip_add_project"):
                        st.info("已跳过添加")
        
        # 显示已有项目列表
        if projects:
            for i, project in enumerate(projects):
                if isinstance(project, dict):
                    st.markdown(f"- **{project.get('name', '未命名项目')}** ({project.get('role', '组员')})")
                else:
                    st.markdown(f"- {project}")
        else:
            if not extracted_project_name:
                st.info("暂无项目记录，请上传项目计划书")
    
    with col2:
        st.subheader("能力画像")
        # 生成能力报告
        if st.session_state.project_full_text:
            scoring_input = build_scoring_input()
            if scoring_input:
                ability_report = generate_ability_report(scoring_input)
                st.session_state.student_ability_report = ability_report
                
                # 1. 赛事视角选择器
                view_mode = st.radio(
                    "选择评估视角",
                    ["通用五力模型"] + list(CONTEST_MODELS.keys()),
                    horizontal=True
                )
                
                if view_mode == "通用五力模型":
                    # 使用原始五力模型数据
                    render_radar_chart(ability_report)
                else:
                    # 2. 调用赛事转换引擎
                    # 提取五力模型分数
                    radar_data = ability_report.get("radar_chart_data", {})
                    five_forces_scores = radar_data.get("r", [0, 0, 0, 0, 0])
                    
                    contest_data = calculate_contest_scores(five_forces_scores, view_mode)
                    
                    # 3. 构造赛事视角的能力报告
                    contest_ability_report = {
                        "radar_chart_data": {
                            "r": contest_data["scores"],
                            "theta": contest_data["dimensions"]
                        }
                    }
                    
                    # 显示赛事综合评估分
                    st.info(f"🏆 当前赛事综合评估分：{contest_data['overall']}")
                    
                    # 4. 渲染赛事视角雷达图
                    render_radar_chart(contest_ability_report)
                
                # 显示详细报告
                if st.expander("查看详细能力报告"):
                    _render_ability_report(ability_report)
        else:
            render_radar_chart(None)

    with st.sidebar:
        st.header("项目工作区")

        render_team_members()
        st.markdown("<br>", unsafe_allow_html=True)

        render_stage_status()
        render_task_tracker()
        handle_project_upload()

        if st.session_state.diag_results:
            st.divider()
            st.subheader("诊断雷达")
            score_df = generate_rubric_table(st.session_state.diag_results)
            st.dataframe(score_df, use_container_width=True)

    render_knowledge_sidebar()

    chat_container = st.container(height=500)
    with chat_container:
        for msg in st.session_state.messages:
            # 对于助手消息，使用智能体角色作为显示名称
            if msg["role"] == "assistant":
                agent_role = msg.get("agent_role", "assistant")
                metadata = msg.get("metadata", {}) if isinstance(msg, dict) else {}
                model_used = metadata.get("model_used", "未知模型")
                with st.chat_message(agent_role):
                    # 显示智能体角色和使用的大模型
                    st.markdown(f"**智能体**: {agent_role} | **模型**: {model_used}")
                    _render_assistant_text(
                        msg.get("content", ""),
                        _assistant_message_severity(msg),
                    )
                    if isinstance(metadata, dict):
                        _render_hypergraph_trace_panel(
                            metadata.get("retrieved_heterogeneous_subgraph"),
                            metadata.get("detected_fallacies"),
                            metadata.get("selected_strategy"),
                        )
            else:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

    prompt = st.chat_input("向教练请教或完善你的方案...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        sync_chat_history()
        sync_stage_progress()

        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)

        with chat_container:
            with st.chat_message("assistant"):
                with st.spinner("正在思考，请稍候..."):
                    agent_app = get_agent_app()
                    agent_state = build_agent_state(
                        portal_role=st.session_state.role,
                        messages=st.session_state.messages,
                        context_data=build_agent_context_data(),
                        active_agent=st.session_state.agent_active_agent,
                        next_agent=st.session_state.agent_next_agent,
                        next_step=st.session_state.agent_next_step,
                    )
                    try:
                        final_state = agent_app.invoke(agent_state)
                    except Exception:
                        final_state = asyncio.run(agent_app.ainvoke(agent_state))
                    update_agent_tracker(final_state)

                    graph_messages = final_state.get("messages", [])
                    assistant_message = next(
                        (
                            message
                            for message in reversed(graph_messages)
                            if _message_role(message) == "assistant"
                        ),
                        None,
                    )
                    active_agent = final_state.get("active_agent", "未知智能体")
                    st.markdown(f"**智能体**: {active_agent} | **模型**: {DEFAULT_MODEL}")
                    response = _message_content(assistant_message)
                    evidence_payload = logic_adapter.extract_evidence_payload(response)
                    response_severity = str(
                        evidence_payload.get("response_severity")
                        or st.session_state.agent_response_severity
                        or logic_adapter.infer_response_severity(st.session_state.diag_results)
                        or ""
                    ).strip()
                    context_payload = final_state.get("context_data", {}) if isinstance(final_state, dict) else {}
                    if not isinstance(context_payload, dict):
                        context_payload = {}
                    retrieved_subgraph = (
                        final_state.get("retrieved_heterogeneous_subgraph")
                        if isinstance(final_state, dict)
                        else None
                    ) or context_payload.get("retrieved_heterogeneous_subgraph")
                    detected_fallacies = (
                        final_state.get("detected_fallacies")
                        if isinstance(final_state, dict)
                        else None
                    ) or context_payload.get("detected_fallacies")
                    selected_strategy = ""
                    if isinstance(retrieved_subgraph, dict):
                        selected_strategy = str(
                            retrieved_subgraph.get("selected_strategy")
                            or context_payload.get("selected_strategy")
                            or ""
                        ).strip()
                    visible_response = logic_adapter.strip_evidence_tag(response)
                    placeholder = st.empty()
                    streamed_response = ""
                    for chunk in _stream_text_chunks(visible_response):
                        streamed_response += chunk
                        _render_assistant_text(
                            streamed_response,
                            response_severity,
                            target=placeholder,
                        )
                    response = visible_response
                    _render_hypergraph_trace_panel(
                        retrieved_subgraph,
                        detected_fallacies,
                        selected_strategy,
                    )

        # 添加智能体角色信息到消息中
        agent_role_map = {
            "Student Learning Tutor": "📘 学习辅导智能体",
            "Project Coach": "🧠 项目教练智能体",
            "Competition Advisor": "🏆 竞赛顾问智能体"
        }
        current_agent = st.session_state.agent_next_agent
        agent_display_name = agent_role_map.get(current_agent, current_agent)
        
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response,
                "agent_role": agent_display_name,
                "metadata": {
                    "severity": response_severity,
                    "retrieved_heterogeneous_subgraph": retrieved_subgraph,
                    "detected_fallacies": detected_fallacies,
                    "selected_strategy": selected_strategy,
                    "agent_role": current_agent,
                    "model_used": DEFAULT_MODEL,
                },
            }
        )
        sync_chat_history()

        keywords = logic_adapter.extract_keywords(prompt)
        if st.session_state.retrieved_cards:
            retrieved_cards = st.session_state.retrieved_cards
        else:
            retrieved_cards = logic_adapter.retrieve_knowledge_cards(keywords)
            st.session_state.retrieved_cards = retrieved_cards
        sync_stage_progress()

        append_behavior_log(
            "chat_turn",
            {
                "turn_index": sum(
                    1
                    for item in st.session_state.messages
                    if isinstance(item, dict) and item.get("role") == "user"
                ),
                "user_prompt": prompt,
                "assistant_reply": response,
                "diagnostic_results": st.session_state.diag_results,
                "retrieved_cards": [card.get("path") for card in retrieved_cards],
                "retrieved_card_titles": [
                    card.get("title", "未命名卡片") for card in retrieved_cards
                ],
                "score_snapshot": build_score_snapshot(),
            },
        )
        st.rerun()

    if st.session_state.current_project:
        st.divider()
        if st.button("生成阶段性能力画像报告"):
            retrieved_nodes = []
            retrieved_hyperedges = []
            knowledge_card_ids = []
            
            for msg in st.session_state.messages:
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    metadata = msg.get("metadata", {})
                    if isinstance(metadata, dict):
                        subgraph = metadata.get("retrieved_heterogeneous_subgraph", {})
                        if isinstance(subgraph, dict):
                            retrieved_nodes.extend(subgraph.get("nodes", []))
                            retrieved_hyperedges.extend(subgraph.get("hyperedges", []))
            
            for diag in st.session_state.diag_results:
                if isinstance(diag, dict):
                    evidence_nodes = diag.get("evidence_nodes", [])
                    if isinstance(evidence_nodes, list):
                        for node_id in evidence_nodes:
                            if node_id.startswith("KC_"):
                                knowledge_card_ids.append(node_id)
            
            ability_source = {
                "title": Path(st.session_state.project_upload_name or "uploaded_project").stem,
                "project_name": Path(st.session_state.project_upload_name or "uploaded_project").stem,
                "project_full_text": st.session_state.project_full_text,
                "text": st.session_state.project_full_text,
                "student_quotes": [
                    str(item.get("content", "")).strip()
                    for item in st.session_state.messages
                    if isinstance(item, dict) and item.get("role") == "user"
                ],
                "behavior_logs": st.session_state.get("behavior_logs", []),
                "retrieved_nodes": retrieved_nodes,
                "retrieved_hyperedges": retrieved_hyperedges,
                "knowledge_card_ids": list(dict.fromkeys(knowledge_card_ids)),
            }
            st.session_state.student_ability_report = generate_ability_report(
                st.session_state.messages,
                st.session_state.diag_results,
                ability_source,
            )

        if st.session_state.student_ability_report:
            with st.expander("你的能力画像报告", expanded=True):
                _render_ability_report(st.session_state.student_ability_report)

render_student_page()

