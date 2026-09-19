from __future__ import annotations

import asyncio
from datetime import datetime
import html
import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from core.rbac import enforce_rbac
from core.graph import build_agent_state, get_agent_app
from core.db_init import get_interaction_log_db_path, init_interaction_log_db
from core.teacher_data_service import TeacherDataService
from core.contest_engine import calculate_contest_scores, CONTEST_MODELS
from db_init import init_db, insert_intervention, list_recent_interventions


try:
    from student_portal import render_radar_chart, save_student_profile
except ImportError:
    def render_radar_chart(ability_report: dict):
        """Fallback radar chart renderer"""
        if not ability_report:
            st.info("开始对话以解锁画像")
            return
        scores = ability_report.get("scores", [0, 0, 0, 0, 0])
        dimensions = ability_report.get("dimensions", ["痛点发现", "方案策划", "商业建模", "资源杠杆", "逻辑表达"])
        fig = go.Figure(data=go.Scatterpolar(
            r=scores + [scores[0]],
            theta=dimensions + [dimensions[0]],
            fill='toself'
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
            showlegend=False,
            title="五力模型能力画像"
        )
        st.plotly_chart(fig)

    def save_student_profile(user_id: str, profile: dict) -> None:
        pass


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_FILE = get_interaction_log_db_path()
TEACHER_UI_CACHE_VERSION = 4
RULE_TO_DIMENSION = {
    "H1": "商业",
    "H4": "商业",
    "H8": "商业",
    "H2": "同理",
    "H5": "同理",
    "H3": "创意",
    "H7": "创意",
    "H6": "逻辑",
    "H14": "逻辑",
    "H15": "逻辑",
    "H9": "执行",
    "H10": "执行",
    "H11": "执行",
}

RULE_ID_TO_NAME = {
    "H1": "客户-价值主张错位",
    "H2": "用户画像模糊",
    "H3": "创新点不可验证",
    "H4": "TAM/SAM/SOM混乱",
    "H5": "竞品分析浅薄",
    "H6": "商业逻辑不自洽",
    "H7": "差异化不明确",
    "H8": "单位经济不成立",
    "H9": "团队能力存疑",
    "H10": "里程碑不现实",
    "H11": "资源杠杆不足",
    "H12": "技术路线不匹配",
    "H13": "合规性风险",
    "H14": "渠道策略矛盾",
    "H15": "定价逻辑错误",
}

enforce_rbac("teacher", page_name="teacher_portal")

KNOWLEDGE_CARDS_PATH = PROJECT_ROOT / "data" / "knowledge_cards.json"


def load_knowledge_cards() -> List[Dict[str, Any]]:
    if not KNOWLEDGE_CARDS_PATH.exists():
        return []
    try:
        return json.loads(KNOWLEDGE_CARDS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_knowledge_cards(cards: List[Dict[str, Any]]) -> bool:
    try:
        KNOWLEDGE_CARDS_PATH.parent.mkdir(parents=True, exist_ok=True)
        KNOWLEDGE_CARDS_PATH.write_text(json.dumps(cards, ensure_ascii=False, indent=2), encoding="utf-8")
        return True
    except Exception:
        return False


DISPLAY_LABEL_MAP = {
    "Assessment Assistant": "评估助手",
    "Instructor Assistant": "教学干预助手",
    "Project Coach": "项目教练",
    "Student Learning Tutor": "学习辅导助手",
    "Competition Advisor": "竞赛顾问",
    "chat_updated": "已更新对话",
    "assessment_updated": "已更新评估",
    "intervention_updated": "已更新干预",
    "report_ready": "报告已生成",
    "queued": "已下发",
    "pending": "待执行",
    "executed": "已执行",
    "db_write_failed": "入库失败",
    "not_started": "未跟进",
}

TEACHER_DATA_SERVICE = TeacherDataService(rule_to_dimension=RULE_TO_DIMENSION)


def _zh_label(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    exact = DISPLAY_LABEL_MAP.get(text)
    if exact:
        return exact
    lowered = DISPLAY_LABEL_MAP.get(text.lower())
    return lowered if lowered else text


def _clean_display_text(value: Any, default_text: str = "") -> str:
    text = str(value or "").strip()
    if not text:
        return default_text

    # Handle legacy mojibake / English placeholders from old logs.
    text = text.replace("瀛︾敓", "学生")
    if text.lower() == "unnamed project":
        return "未命名项目"
    if text.lower() == "not_started":
        return "未跟进"
    return text


def _inject_teacher_portal_styles() -> None:
    # 仅 UI 调整，不影响逻辑
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.15rem; padding-bottom: 1.3rem;}
        .tp-section-title {font-size: 1.02rem; font-weight: 700; margin: 0.2rem 0 0.65rem 0;}
        .tp-card {
            border: 1px solid rgba(49, 51, 63, 0.16);
            border-radius: 0.7rem;
            padding: 0.75rem 0.9rem;
            background: linear-gradient(180deg, rgba(248,249,252,0.65), rgba(255,255,255,0.88));
            margin-bottom: 0.75rem;
        }
        .tp-subtle {color: rgba(49, 51, 63, 0.72); font-size: 0.9rem;}
        .evidence-quote {
            border-left: 3px solid #0f766e;
            background: rgba(15, 118, 110, 0.07);
            padding: 0.55rem 0.7rem;
            border-radius: 0.35rem;
            margin: 0.38rem 0 0.38rem 0;
            font-size: 0.92rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_teacher_state() -> None:
    init_db()
    init_interaction_log_db()
    defaults: Dict[str, Any] = {
        "role": "teacher",
        "teacher_selected_class": "全部班级",
        "teacher_selected_student_key": "",
        "teacher_selected_student_detail": {},
        "teacher_reports": {},
        "teacher_intervention_status": {},
        "teacher_intervention_history": [],
        "teacher_chat_messages": [],
        "teacher_agent_next_step": "",
        "teacher_agent_active_agent": "",
        "teacher_agent_route_target": "",
        "teacher_class_teaching_suggestion": {},
        "teacher_use_mock_data": False,
        "teacher_ui_cache_version": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Clear cached profile/log views when UI cache version changes.
    if int(st.session_state.get("teacher_ui_cache_version", 0)) != TEACHER_UI_CACHE_VERSION:
        load_log_records.clear()
        load_and_process_logs.clear()
        build_student_profiles.clear()
        list_class_options.clear()
        get_class_top_mistakes.clear()
        st.session_state.teacher_ui_cache_version = TEACHER_UI_CACHE_VERSION

    # Normalize legacy status values already stored in session_state.
    normalized_status: Dict[str, Any] = {}
    for student_key, payload in dict(st.session_state.teacher_intervention_status).items():
        if not isinstance(payload, dict):
            continue
        status_text = _clean_display_text(_zh_label(payload.get("status", "")), "未跟进")
        next_step = _clean_display_text(payload.get("next_step", ""), "")
        updated_at = str(payload.get("updated_at", "")).strip()
        normalized_status[str(student_key)] = {
            "status": status_text,
            "next_step": next_step,
            "updated_at": updated_at,
        }
    st.session_state.teacher_intervention_status = normalized_status

    st.session_state.role = "teacher"


@st.cache_data
def load_log_records(log_path: str, use_mock: bool = False) -> List[Dict[str, Any]]:
    return TEACHER_DATA_SERVICE.load_records(log_path=log_path, use_mock=use_mock)


@st.cache_data
def list_class_options(log_path: str, use_mock: bool = False) -> List[str]:
    records = load_log_records(log_path, use_mock=use_mock)
    return TEACHER_DATA_SERVICE.get_class_options(records)


@st.cache_data
def load_and_process_logs(
    log_path: str,
    selected_class: str = "全部班级",
    use_mock: bool = False,
) -> pd.DataFrame:
    records = load_log_records(log_path, use_mock=use_mock)
    return TEACHER_DATA_SERVICE.build_rule_dataframe(records=records, selected_class=selected_class)


@st.cache_data
def build_student_profiles(
    log_path: str,
    selected_class: str = "全部班级",
    use_mock: bool = False,
) -> List[Dict[str, Any]]:
    records = load_log_records(log_path, use_mock=use_mock)
    return TEACHER_DATA_SERVICE.build_student_profiles(records=records, selected_class=selected_class)


@st.cache_data
def get_class_top_mistakes(log_path: str, top_n: int = 5) -> List[Tuple[str, int]]:
    db_path = str(log_path or "").strip()
    if not db_path.lower().endswith(".db"):
        return []

    try:
        with sqlite3.connect(db_path, timeout=10.0) as conn:
            df = pd.read_sql_query(
                """
                SELECT triggered_rules
                FROM interaction_logs
                WHERE triggered_rules IS NOT NULL
                  AND TRIM(triggered_rules) != ''
                  AND triggered_rules != '[]'
                """,
                conn,
            )
    except sqlite3.Error:
        return []

    if df.empty or "triggered_rules" not in df.columns:
        return []

    rule_counts: Dict[str, int] = {}
    for rules_str in df["triggered_rules"]:
        try:
            rules = json.loads(str(rules_str or "[]"))
        except (TypeError, json.JSONDecodeError):
            continue

        if isinstance(rules, dict):
            rules = [rules]
        if not isinstance(rules, list):
            continue

        for rule in rules:
            if isinstance(rule, dict):
                rule_id = str(rule.get("rule_id", "")).strip().upper()
            elif isinstance(rule, str):
                rule_id = rule.strip().upper()
            else:
                rule_id = ""
            if rule_id:
                rule_counts[rule_id] = rule_counts.get(rule_id, 0) + 1

    if not rule_counts:
        return []
    return sorted(rule_counts.items(), key=lambda item: item[1], reverse=True)[: max(1, int(top_n))]


def _reset_teacher_student_context() -> None:
    st.session_state.teacher_selected_student_key = ""
    st.session_state.teacher_selected_student_detail = {}
    st.session_state.teacher_reports = {}
    st.session_state.teacher_chat_messages = []
    st.session_state.teacher_agent_next_step = ""
    st.session_state.teacher_agent_active_agent = ""
    st.session_state.teacher_agent_route_target = ""


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


def _is_tool_activity_message(message: Any) -> bool:
    """决策 2：隐藏图谱检索的原始 JSON 与 AI 的工具调用请求。"""
    if isinstance(message, ToolMessage):
        return True
    if isinstance(message, AIMessage) and getattr(message, "tool_calls", None):
        return True
    if isinstance(message, dict):
        role = str(message.get("role", ""))
        if role == "tool":
            return True
        if role == "assistant" and message.get("tool_calls"):
            return True
    return False


def _message_content(message: Any) -> str:
    if isinstance(message, BaseMessage):
        content = message.content
        if isinstance(content, list):
            return " ".join(str(item) for item in content if item is not None).strip()
        return str(content or "").strip()
    if isinstance(message, dict):
        return str(message.get("content", "")).strip()
    return str(message).strip()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _collect_rubric_results(
    report_payload: Dict[str, Any] | None,
    profile: Dict[str, Any],
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    state = (report_payload or {}).get("state", {})
    context_data = state.get("context_data", {}) if isinstance(state, dict) else {}

    raw_results = []
    if isinstance(state, dict):
        raw_results = state.get("rubric_score_results", []) or []
    if not raw_results and isinstance(context_data, dict):
        raw_results = context_data.get("rubric_score_results", []) or []

    if isinstance(raw_results, str):
        try:
            raw_results = json.loads(raw_results)
        except json.JSONDecodeError:
            raw_results = []

    if isinstance(raw_results, list):
        for idx, row in enumerate(raw_results):
            if not isinstance(row, dict):
                continue
            item_id = str(row.get("item_id") or f"R{idx + 1}").strip() or f"R{idx + 1}"
            audit = row.get("audit_trail", {})
            audit = audit if isinstance(audit, dict) else {}
            results.append(
                {
                    "item_id": item_id,
                    "name": str(row.get("name") or "未命名维度").strip() or "未命名维度",
                    "score": _safe_float(row.get("score"), 0.0),
                    "max_score": max(0.1, _safe_float(row.get("max_score"), 5.0)),
                    "audit_trail": {
                        "quote": str(audit.get("quote") or "（未捕获原文）").strip(),
                        "h_rule_id": str(audit.get("h_rule_id") or "Rule-H_UNKNOWN").strip(),
                        "kg_card_id": str(audit.get("kg_card_id") or "KG_CARD_UNBOUND").strip(),
                        "explanation": str(audit.get("explanation") or "暂无解释").strip(),
                    },
                }
            )
    if results:
        return results

    # Fallback: adapt legacy rubric_rows for compatibility.
    legacy_rows = profile.get("rubric_rows", [])
    if not isinstance(legacy_rows, list):
        return []
    quotes = [str(q).strip() for q in profile.get("student_quotes", []) if str(q).strip()]
    fallback_quote = quotes[0] if quotes else "（未捕获原文）"
    for idx, row in enumerate(legacy_rows):
        if not isinstance(row, dict):
            continue
        results.append(
            {
                "item_id": str(row.get("item_id") or f"R{idx + 1}").strip() or f"R{idx + 1}",
                "name": str(row.get("criterion") or row.get("name") or "未命名维度").strip() or "未命名维度",
                "score": _safe_float(row.get("score"), 0.0),
                "max_score": max(0.1, _safe_float(row.get("max_score"), 5.0)),
                "audit_trail": {
                    "quote": fallback_quote,
                    "h_rule_id": "Rule-H_UNKNOWN",
                    "kg_card_id": "KG_CARD_UNBOUND",
                    "explanation": str(row.get("comment") or "暂无解释").strip(),
                },
            }
        )
    return results


def render_audit_trail_dashboard(rubric_results: List[Dict[str, Any]]) -> None:
    st.subheader("形成性评价与溯源证据链 (Evidence-Driven Assessment)")
    st.caption("系统基于底层超图逻辑自动提取评分依据，避免大模型幻觉评分。")

    if not rubric_results:
        st.caption("暂无结构化评分数据。")
        return

    for index, res in enumerate(rubric_results):
        if not isinstance(res, dict):
            continue

        item_id = str(res.get("item_id") or f"R{index + 1}").strip() or f"R{index + 1}"
        name = str(res.get("name") or "未命名维度").strip() or "未命名维度"
        score = _safe_float(res.get("score"), 0.0)
        max_score = max(0.1, _safe_float(res.get("max_score"), 5.0))
        audit = res.get("audit_trail", {})
        audit = audit if isinstance(audit, dict) else {}
        is_high_score = score >= (max_score * 0.7)
        status_icon = "✅" if is_high_score else "⚠️ 高风险指标"

        with st.expander(
            f"{status_icon} | 维度 {item_id}: {name} ({score}/{max_score} 分)",
            expanded=not is_high_score,
        ):
            st.markdown("#### 底层溯源证据 (Audit Trail)")
            quote = str(audit.get("quote") or "（未捕获原文）").strip()
            st.markdown(f"> **原文定点捕获：**\n> *\"{quote}\"*")

            col1, col2 = st.columns(2)
            with col1:
                st.warning(f"**触发诊断规则：**\n\n`{str(audit.get('h_rule_id') or 'Rule-H_UNKNOWN').strip()}`")
            with col2:
                st.info(f"**关联知识图谱卡片：**\n\n`{str(audit.get('kg_card_id') or 'KG_CARD_UNBOUND').strip()}`")

            st.markdown("---")
            explanation = str(audit.get("explanation") or "暂无解释").strip()
            st.markdown(f"**逻辑断层分析：** {explanation}")

            if not is_high_score:
                button_key = f"btn_push_kg_{item_id}_{index}"
                if st.button("将此知识卡片推送给学生", key=button_key):
                    st.toast(
                        f"已将 {str(audit.get('kg_card_id') or 'KG_CARD_UNBOUND').strip()} 压入下一次学生对话上下文！",
                        icon="✅",
                    )


def _build_latest_risk_distribution(profiles: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    students_with_diagnosis = 0

    for profile in profiles:
        latest_diagnosis = [
            item
            for item in profile.get("diagnostic_results", [])
            if isinstance(item, dict) and str(item.get("rule_id", "")).strip()
        ]
        if not latest_diagnosis:
            continue
        students_with_diagnosis += 1
        seen_rule_ids = set()
        for item in latest_diagnosis:
            rule_id = str(item.get("rule_id", "")).strip().upper()
            if not rule_id or rule_id in seen_rule_ids:
                continue
            seen_rule_ids.add(rule_id)
            rows.append(
                {
                    "rule_id": rule_id,
                    "student_key": profile.get("student_key", ""),
                    "student_name": profile.get("student_label", ""),
                }
            )

    if not rows or students_with_diagnosis == 0:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    summary = (
        df.groupby("rule_id")
        .agg(
            hit_count=("student_key", "count"),
            impacted_students=("student_name", lambda values: ", ".join(sorted(dict.fromkeys(values))[:6])),
        )
        .reset_index()
    )
    summary["hit_rate"] = (summary["hit_count"] / students_with_diagnosis).round(4)
    summary["hit_rate_pct"] = (summary["hit_rate"] * 100).round(1)
    summary = summary.sort_values(["hit_count", "rule_id"], ascending=[False, True]).reset_index(drop=True)
    return summary


def _build_class_profile(profiles: List[Dict[str, Any]], risk_df: pd.DataFrame) -> Dict[str, Any]:
    latest_hits: List[Dict[str, Any]] = []
    triggered_history: List[Dict[str, Any]] = []
    for profile in profiles:
        latest_diagnosis = [
            item for item in profile.get("diagnostic_results", []) if isinstance(item, dict)
        ]
        latest_hits.extend(latest_diagnosis)
        triggered_history.extend(profile.get("triggered_rules_history", []))

    summary_text = ""
    if not risk_df.empty:
        top_rows = risk_df.head(5)
        summary_text = "; ".join(
            f"{row.rule_id}: {int(row.hit_count)}次 {row.hit_rate_pct}%"
            for row in top_rows.itertuples()
        )

    return {
        "student_key": "class_overview",
        "student_name": "全班",
        "student_label": "全班",
        "user_id": "class_overview",
        "project_id": "class_overview",
        "project_name": "班级逻辑风险分布",
        "project_summary": summary_text,
        "project_full_text": summary_text,
        "diagnostic_results": latest_hits,
        "triggered_rules_history": triggered_history,
        "behavior_logs": [],
        "student_quotes": [],
        "score_snapshot": {},
    }


def _get_class_teaching_suggestion(
    profiles: List[Dict[str, Any]],
    risk_df: pd.DataFrame,
) -> Dict[str, Any]:
    if risk_df.empty:
        return {}

    high_frequency = risk_df[risk_df["hit_rate"] > 0.5].copy()
    if high_frequency.empty:
        return {}

    dominant_row = high_frequency.iloc[0]
    signature = f"{dominant_row['rule_id']}|{dominant_row['hit_count']}|{len(profiles)}"
    cached = st.session_state.teacher_class_teaching_suggestion
    if cached.get("signature") == signature:
        return cached

    class_profile = _build_class_profile(profiles, risk_df)
    prompt = (
        f"当前班级最新诊断中，{dominant_row['rule_id']} 命中率为 "
        f"{dominant_row['hit_rate_pct']}%。请生成一条班级层面的补救教学建议。"
    )
    final_state, response = _invoke_teacher_agent(
        class_profile,
        prompt=prompt,
        teacher_mode="intervention_plan",
        route_hint="Instructor Assistant",
    )
    payload = {
        "signature": signature,
        "rule_id": str(dominant_row["rule_id"]),
        "hit_rate_pct": float(dominant_row["hit_rate_pct"]),
        "response": response,
        "state": final_state,
    }
    st.session_state.teacher_class_teaching_suggestion = payload
    return payload


def _build_student_roster(profiles: List[Dict[str, Any]]) -> pd.DataFrame:
    """构建学生花名册数据（优化版：聚合风险标签，降噪设计）"""
    rows = []
    for profile in profiles:
        student_label = _clean_display_text(profile.get("student_label"), "学生")
        project_name = _clean_display_text(profile.get("project_name"), "未命名项目")
        overall_score = profile.get("score_snapshot", {}).get("overall_score", 0)

        rule_counts = {f"H{i}": 0 for i in range(1, 16)}
        for item in profile.get("diagnostic_results", []):
            if isinstance(item, dict):
                rule_id = str(item.get("rule_id", "")).strip().upper()
                if rule_id in rule_counts:
                    rule_counts[rule_id] += 1

        for item in profile.get("triggered_rules_history", []):
            if isinstance(item, dict):
                rule_id = str(item.get("rule_id", "")).strip().upper()
                if rule_id in rule_counts:
                    rule_counts[rule_id] += 1

        top_rules = sorted(rule_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        top_tags = []
        for rule_id, count in top_rules:
            if count > 0:
                rule_name = RULE_ID_TO_NAME.get(rule_id, rule_id)
                top_tags.append(f"{rule_name}({count}次)")
        risk_summary = " | ".join(top_tags) if top_tags else "✅ 健康"

        risk_level = "🔴 高危" if overall_score < 3 or rule_counts.get("H1", 0) > 2 or rule_counts.get("H8", 0) > 2 else \
                      "🟡 中危" if overall_score < 4 or any(rule_counts.get(f"H{i}", 0) > 1 for i in range(1, 16)) else \
                      "🟢 低危"

        rows.append({
            "学生姓名": student_label,
            "项目名称": project_name,
            "平均分": overall_score,
            "风险等级": risk_level,
            "核心风险诊断": risk_summary,
            "风险指数": sum(rule_counts.values()),
        })

    df = pd.DataFrame(rows)
    if not df.empty and "风险指数" in df.columns:
        df = df.sort_values(by="风险指数", ascending=False)
    return df


def _render_dashboard(df: pd.DataFrame, profiles: List[Dict[str, Any]], log_path: str) -> None:
    # 仅 UI 调整，不影响逻辑
    st.markdown("### 教师班级看板")
    st.markdown("<div class='tp-subtle'>聚焦班级风险信号与干预就绪度。</div>", unsafe_allow_html=True)

    if df.empty:
        st.warning("未找到学生日志数据。")
        return

    latest_risk_df = _build_latest_risk_distribution(profiles)
    total_hits = int(len(df["rule_id"].dropna()))
    high_risk_count = (
        int(len(df[df["severity"].astype(str).str.lower() == "high"]))
        if "severity" in df.columns
        else 0
    )
    active_projects = int(df["project_id"].nunique()) if "project_id" in df.columns else 0
    students_total = max(len(profiles), 1)
    students_with_diagnosis = sum(
        1
        for profile in profiles
        if isinstance(profile.get("diagnostic_results"), list) and profile.get("diagnostic_results")
    )
    coverage_pct = round((students_with_diagnosis / students_total) * 100, 1)
    high_ratio_pct = round((high_risk_count / total_hits) * 100, 1) if total_hits else 0.0

    # 仅 UI 调整，不影响逻辑
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    with kpi_col1:
        st.metric("规则命中总数", total_hits, delta=f"{high_risk_count} 条高风险")
    with kpi_col2:
        st.metric("高风险占比", f"{high_ratio_pct}%", delta="风险集中度")
    with kpi_col3:
        st.metric("知识覆盖率", f"{coverage_pct}%", delta=f"{students_with_diagnosis}/{students_total} 位学生")
    with kpi_col4:
        st.metric("活跃项目数", active_projects, delta="当前班级范围")

    # 添加班级全景看板
    st.markdown("#### 班级全景看板")
    
    # 添加视图切换开关
    view_mode = st.radio(
        "👀 看板视图", 
        ["按项目聚合 (团队视角)", "按学生展开 (个体视角)"], 
        horizontal=True
    )
    
    roster_df = _build_student_roster(profiles)
    if not roster_df.empty:
        if view_mode == "按项目聚合 (团队视角)":
            # 按项目名称分组，聚合团队数据
            project_df = roster_df.groupby("项目名称").agg(
                包含成员=pd.NamedAgg(column="学生姓名", aggfunc=lambda x: ", ".join(x)),
                团队平均分=pd.NamedAgg(column="平均分", aggfunc="mean"),
                风险总指数=pd.NamedAgg(column="风险指数", aggfunc="sum"),
                核心风险诊断=pd.NamedAgg(
                    column="核心风险诊断", 
                    aggfunc=lambda x: " | ".join(set([tag for tags in x for tag in tags.split(" | ") if tag != "✅ 健康"]))
                )
            ).reset_index()
            
            # 处理空的风险诊断
            project_df["核心风险诊断"] = project_df["核心风险诊断"].apply(
                lambda x: x if x else "✅ 健康"
            )
            
            # 重新排序
            project_df = project_df.sort_values(by="风险总指数", ascending=False)
            
            # 渲染项目视图
            def highlight_project_risk(row):
                risk_index = row.get("风险总指数", 0)
                if risk_index >= 10:
                    return ["background-color: #fee2e2"] * len(row)
                elif risk_index >= 5:
                    return ["background-color: #fef9c3"] * len(row)
                return [""] * len(row)
            
            st.dataframe(
                project_df.style.apply(highlight_project_risk, axis=1),
                use_container_width=True,
                column_config={
                    "项目名称": st.column_config.TextColumn(width="medium"),
                    "包含成员": st.column_config.TextColumn(width="large"),
                    "团队平均分": st.column_config.NumberColumn(format="%.2f"),
                    "风险总指数": st.column_config.NumberColumn(),
                    "核心风险诊断": st.column_config.TextColumn(width="large"),
                }
            )
        else:
            # 渲染个体视角
            def highlight_risk_level(row):
                if "🔴 高危" in str(row.get("风险等级", "")):
                    return ["background-color: #fee2e2"] * len(row)
                elif "🟡 中危" in str(row.get("风险等级", "")):
                    return ["background-color: #fef9c3"] * len(row)
                return [""] * len(row)

            st.dataframe(
                roster_df.style.apply(highlight_risk_level, axis=1),
                use_container_width=True,
                column_config={
                    "学生姓名": st.column_config.TextColumn(width="medium"),
                    "项目名称": st.column_config.TextColumn(width="medium"),
                    "平均分": st.column_config.NumberColumn(format="%.2f"),
                    "风险等级": st.column_config.TextColumn(width="small"),
                    "核心风险诊断": st.column_config.TextColumn(width="large"),
                }
            )
    else:
        st.info("暂无学生花名册数据。")

    # 仅 UI 调整，不影响逻辑
    chart_col1, chart_col2 = st.columns([1, 1.2], gap="large")
    with chart_col1:
        st.markdown("<div class='tp-section-title'>维度覆盖雷达图</div>", unsafe_allow_html=True)
        dim_counts = (
            df.groupby("dimension").size().reset_index(name="count")
            if "dimension" in df.columns
            else pd.DataFrame(columns=["dimension", "count"])
        )
        if dim_counts.empty:
            st.info("暂无维度数据。")
        else:
            max_count = max(int(dim_counts["count"].max()), 1)
            dim_counts["coverage_pct"] = (dim_counts["count"] / max_count * 100).round(1)
            radar_fig = go.Figure(
                data=go.Scatterpolar(
                    r=dim_counts["coverage_pct"],
                    theta=dim_counts["dimension"],
                    fill="toself",
                    name="覆盖率",
                    line=dict(color="#0f766e", width=2),
                    marker=dict(size=6, color="#0f766e"),
                    hovertemplate="<b>%{theta}</b><br>覆盖率: %{r:.1f}%<extra></extra>",
                )
            )
            radar_fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=False,
                margin=dict(l=16, r=16, t=10, b=10),
                height=370,
            )
            st.plotly_chart(radar_fig, use_container_width=True)

    with chart_col2:
        st.markdown("<div class='tp-section-title'>共性规则错误（前 8）</div>", unsafe_allow_html=True)
        if latest_risk_df.empty:
            st.info("暂无可用于班级风险分布的最新诊断数据。")
        else:
            top_rules = latest_risk_df.head(8).sort_values("hit_count", ascending=True)
            error_fig = px.bar(
                top_rules,
                x="hit_count",
                y="rule_id",
                orientation="h",
                color="hit_rate_pct",
                color_continuous_scale=["#94a3b8", "#334155", "#b91c1c"],
                custom_data=["hit_rate_pct", "impacted_students"],
            )
            error_fig.update_traces(
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "命中次数: %{x}<br>"
                    "命中率: %{customdata[0]}%<br>"
                    "涉及学生: %{customdata[1]}<extra></extra>"
                )
            )
            error_fig.update_layout(
                margin=dict(l=12, r=12, t=10, b=10),
                height=370,
                coloraxis_colorbar=dict(title="命中率 %"),
                xaxis_title="命中次数",
                yaxis_title="规则编号",
            )
            st.plotly_chart(error_fig, use_container_width=True)

    st.markdown("#### 班级共性错误 Top 5 (A6-2)")
    top_mistakes = get_class_top_mistakes(log_path, top_n=5)
    if top_mistakes:
        for rule_id, count in top_mistakes:
            st.error(f"频次: {count} | 规则: {rule_id}")
    else:
        st.caption("暂无可用于 Top Mistakes 的触发规则数据。")

    # 仅 UI 调整，不影响逻辑
    st.markdown("#### 风险明细表")
    if not latest_risk_df.empty:
        st.dataframe(
            latest_risk_df.rename(
                columns={
                    "rule_id": "规则编号",
                    "hit_count": "命中次数",
                    "hit_rate_pct": "命中率 (%)",
                    "impacted_students": "涉及学生",
                }
            )[["规则编号", "命中次数", "命中率 (%)", "涉及学生"]],
            use_container_width=True,
        )

    suggestion = _get_class_teaching_suggestion(profiles, latest_risk_df)
    if suggestion:
        # 仅 UI 调整，不影响逻辑
        st.info(
            f"班级重点风险：{suggestion['rule_id']}（命中率 {suggestion['hit_rate_pct']}%）\n\n"
            f"{suggestion.get('response', '')}"
        )


def _build_teacher_context(profile: Dict[str, Any], teacher_mode: str, route_hint: str) -> Dict[str, Any]:
    return {
        "portal_role": "teacher",
        "teacher_mode": teacher_mode,
        "route_hint": route_hint,
        "student_key": profile["student_key"],
        "student_name": profile["student_name"],
        "student_label": profile["student_label"],
        "student_id": profile.get("user_id") or profile["student_key"],
        "project_id": profile.get("project_id"),
        "project_name": profile.get("project_name"),
        "project_summary": profile.get("project_summary", ""),
        "project_full_text": profile.get("project_full_text", ""),
        "diagnostic_results": profile.get("diagnostic_results", []),
        "triggered_rules_history": profile.get("triggered_rules_history", []),
        "behavior_logs": profile.get("behavior_logs", []),
        "student_quotes": profile.get("student_quotes", []),
        "competition_score_report": profile.get("score_snapshot", {}),
    }


def _update_teacher_agent_tracker(final_state: Dict[str, Any]) -> None:
    context_data = final_state.get("context_data", {})
    st.session_state.teacher_agent_next_step = str(final_state.get("next_step", "")).strip()
    st.session_state.teacher_agent_active_agent = str(
        final_state.get("active_agent") or context_data.get("route_target") or ""
    ).strip()
    st.session_state.teacher_agent_route_target = str(context_data.get("route_target", "")).strip()


def _invoke_teacher_agent(
    profile: Dict[str, Any],
    prompt: str,
    teacher_mode: str,
    route_hint: str,
    messages: List[Dict[str, str]] | None = None,
) -> Tuple[Dict[str, Any], str]:
    agent_app = get_agent_app()
    agent_state = build_agent_state(
        portal_role="teacher",
        messages=messages or [{"role": "user", "content": prompt}],
        context_data=_build_teacher_context(profile, teacher_mode, route_hint),
        active_agent=st.session_state.teacher_agent_active_agent,
        next_step=st.session_state.teacher_agent_next_step,
    )
    final_state = asyncio.run(agent_app.ainvoke(agent_state))

    _update_teacher_agent_tracker(final_state)
    graph_messages = final_state.get("messages", [])
    # 决策 2：跳过工具回执与带 tool_calls 的中间消息，只取最终文字回答
    assistant_message = next(
        (
            message
            for message in reversed(graph_messages)
            if not _is_tool_activity_message(message)
            and _message_role(message) == "assistant"
            and _message_content(message)
        ),
        None,
    )
    return final_state, _message_content(assistant_message)


def _set_intervention_status(student_key: str, status: str, next_step: str) -> None:
    updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.teacher_intervention_status[student_key] = {
        "status": status,
        "next_step": next_step,
        "updated_at": updated_at,
    }
    history = list(st.session_state.get("teacher_intervention_history", []))
    history.insert(
        0,
        {
            "student_key": student_key,
            "status": status,
            "next_step": next_step,
            "updated_at": updated_at,
        },
    )
    st.session_state.teacher_intervention_history = history[:80]


def _student_db_id(profile: Dict[str, Any]) -> str:
    return str(profile.get("user_id") or profile.get("student_key") or "").strip()


def _enqueue_intervention(student_id: str, content: str) -> int:
    normalized_student_id = str(student_id or "").strip()
    normalized_content = str(content or "").strip()
    if not normalized_student_id or not normalized_content:
        return 0
    try:
        return insert_intervention(normalized_student_id, 'user', normalized_content)
    except (sqlite3.Error, ValueError):
        return 0


def _resolve_student_from_prompt(
    prompt: str,
    profiles: List[Dict[str, Any]],
    selected_key: str,
) -> Dict[str, Any] | None:
    normalized_prompt = prompt.strip()
    if normalized_prompt:
        for profile in profiles:
            candidates = [
                profile.get("student_label", ""),
                profile.get("student_name", ""),
                profile.get("project_name", ""),
                profile.get("user_id", ""),
            ]
            if any(candidate and candidate in normalized_prompt for candidate in candidates):
                return profile

    if selected_key:
        for profile in profiles:
            if profile["student_key"] == selected_key:
                return profile

    return profiles[0] if profiles else None


@st.dialog("🎯 学生透视与干预中心", width="large")
def _student_drilldown_dialog(profile: Dict[str, Any]) -> None:
    # ==========================================
    # 🍳 第一步：在渲染前，动态计算和组装数据
    # ==========================================
    
    # 1. 动态计算 rule_count (从触发历史或诊断结果中提取)
    triggered_history = profile.get("triggered_rules_history", [])
    diag_results = profile.get("diagnostic_results", [])
    if isinstance(triggered_history, list) and len(triggered_history) > 0:
        rule_count = len(triggered_history)
    else:
        # 兜底逻辑：遍历 diagnostic_results 计算包含 rule_id 的项
        rule_count = len([item for item in diag_results if isinstance(item, dict) and item.get("rule_id")])
        
    # 2. 动态提取 latest_stage (从行为日志的最后一条提取)
    behavior_logs = profile.get("behavior_logs", [])
    latest_stage = "初期探索"  # 默认冷启动阶段
    if isinstance(behavior_logs, list) and len(behavior_logs) > 0:
        last_log = behavior_logs[-1]
        if isinstance(last_log, dict):
            # 根据你日志的具体结构获取，可能是 'stage', 'action' 或 'status'
            latest_stage = last_log.get("stage") or last_log.get("action", "项目推进中")
            
    # 3. 处理五力模型 scores (防空载兜底)
    # 检查是否存在 score_snapshot，否则退回默认的冷启动全0数组
    scores = profile.get("scores") or profile.get("score_snapshot", {}).get("scores", [0, 0, 0, 0, 0])

    # ==========================================
    # 🎨 第二步：开始渲染 UI
    # ==========================================
    student_label = profile.get("student_label") or profile.get("student_name", "未知学生")
    
    tab1, tab2, tab3 = st.tabs(["📊 能力画像", "🔍 证据溯源", "⚡ 导师干预与改分"])
    
    with tab1:
        st.markdown(f"### {student_label} 的学习档案")
        
        # 修复指标卡显示 0 和 未知 的问题
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🚩 规则命中总数", rule_count)
        col2.metric("📌 最新阶段", latest_stage)
        col3.metric("📊 综合评估得分", round(sum(scores)/5, 1) if sum(scores) > 0 else "未评估")
        
        st.markdown("---")
        
        # 引入赛事选择，帮助导师判断项目适合报哪个赛道
        st.write("### 赛事竞争力模拟")
        selected_contest = st.selectbox(
            "模拟参赛视角",
            list(CONTEST_MODELS.keys()),
            help="改变赛事视角将按不同权重重新计算画像"
        )
        
        # 计算赛事得分
        contest_report = calculate_contest_scores(scores, selected_contest)
        
        # 布局调整：左侧显示分数，右侧显示雷达图
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("赛事预估总分", contest_report["overall"])
            # 展示具体的四维得分
            for d, s in zip(contest_report["dimensions"], contest_report["scores"]):
                st.write(f"**{d}**: {s}")
        
        with col2:
            # 构造雷达图所需的数据结构
            ability_report = {
                "radar_chart_data": {
                    "r": contest_report["scores"],
                    "theta": contest_report["dimensions"]
                }
            }
            
            # 渲染雷达图
            try:
                render_radar_chart(ability_report)
            except Exception as e:
                st.error(f"雷达图渲染失败: 请检查 render_radar_chart 的参数格式。详细报错: {str(e)}")
                st.write("当前传入的数据为:", ability_report)

    with tab2:
        st.markdown("### 关键逻辑漏洞追踪")
        # 这里可以循环渲染 triggered_history 或 diag_results
        if triggered_history:
            for rule in triggered_history:
                st.warning(f"**触发规则**: {rule}")
        else:
            st.success("✅ 该团队目前表现良好，未触发高危逻辑漏洞。")
            
    with tab3:
        st.markdown("### 👩‍🏫 人工复核与评分干预")

        dimensions = ["痛点发现", "方案策划", "商业建模", "资源杠杆", "逻辑表达"]
        ai_scores = profile.get("scores", [0, 0, 0, 0, 0])
        if len(ai_scores) != 5:
            ai_scores = [0, 0, 0, 0, 0]

        with st.form("manual_override_form"):
            st.write("覆盖 AI 预评分 (提交后将同步更新学生雷达图，并发送通知卡片)")

            new_scores = []
            cols = st.columns(5)
            for i, dim in enumerate(dimensions):
                with cols[i]:
                    new_val = st.number_input(dim, min_value=0.0, max_value=5.0, value=float(ai_scores[i]), step=0.5)
                    new_scores.append(new_val)

            teacher_comment = st.text_area("教师复核评语", placeholder="例如：AI 忽略了你们在线下渠道的优势，我将你的资源杠杆分从 2 提高到了 4。")

            force_stage = st.selectbox(
                "强制修改项目阶段",
                ["不修改", "阶段一：痛点探测", "阶段二：方案策划", "阶段三：商业模型"],
                index=0
            )

            intervention_type = st.radio(
                "选择压测维度",
                ["⚖️ 法律合规性核查", "💰 财务模型极限承压", "👥 竞品防御战"],
                horizontal=True
            )

            submitted = st.form_submit_button("💾 保存并下发卡片", type="primary")

            if submitted:
                profile["scores"] = new_scores
                if force_stage != "不修改":
                    profile["current_stage"] = force_stage.split("：")[1]

                user_id = profile.get("user_id") or profile.get("student_key", "")
                if user_id:
                    save_student_profile(user_id, profile)

                intervention_card = f"""
**🚨 导师复核通知**

导师已重新评估你的项目。

- **调整后评分**: {dict(zip(dimensions, new_scores))}
- **导师评语**: {teacher_comment}
- **项目阶段**: {force_stage if force_stage != "不修改" else "保持不变"}
"""
                _enqueue_intervention(
                    student_id=profile.get("user_id") or profile.get("student_key", ""),
                    content=intervention_card
                )

                st.success("✅ 评分已覆写，通知卡片已推送到学生端对话流！")


def _render_case_library() -> None:
    st.markdown("### 📚 优质商业案例库建设 (知识干预)")
    st.caption("上传优秀案例，案例将作为知识卡片供学生端 AI 在诊断时引用。")

    uploaded_file = st.file_uploader("上传案例文件 (PDF/TXT)", type=["pdf", "txt"])

    if uploaded_file is not None:
        try:
            text_content = uploaded_file.getvalue()
            try:
                text_content = text_content.decode("utf-8", errors="ignore")
            except Exception:
                text_content = str(text_content)
        except Exception as e:
            st.error(f"读取文件失败: {str(e)}")
            return

        match_name = re.search(r"项目名称[：:]\s*(.+?)(?:\n|$)", text_content)
        match_target = re.search(r"目标客户[：:]\s*(.+?)(?:\n|$)", text_content)

        extracted_name = match_name.group(1).strip() if match_name else Path(uploaded_file.name).stem
        extracted_target = match_target.group(1).strip() if match_target else "未能自动识别"

        col1, col2 = st.columns(2)
        with col1:
            st.text_input("自动抽取的项目名称", value=extracted_name, key="case_name_input")
        with col2:
            st.text_input("自动抽取的目标客户", value=extracted_target, key="case_target_input")

        case_summary = st.text_area(
            "提炼卡片摘要 (将推送给相似的学生)",
            placeholder="该案例展示了极低成本的 MVP 验证方法...",
            key="case_summary_input"
        )

        if st.button("💾 将此案例存入知识库", type="primary"):
            cards = load_knowledge_cards()
            new_card = {
                "card_id": f"card_{len(cards) + 1}_{int(datetime.now().timestamp())}",
                "title": st.session_state.get("case_name_input", extracted_name),
                "target": st.session_state.get("case_target_input", extracted_target),
                "summary": st.session_state.get("case_summary_input", ""),
                "source_file": uploaded_file.name,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            cards.append(new_card)
            if save_knowledge_cards(cards):
                st.success("✅ 案例卡片已生成！智能体将在后续诊断中引用此案例。")
            else:
                st.error("❌ 保存失败，请检查文件权限。")

    with st.expander("📖 查看已有案例库", expanded=False):
        existing_cards = load_knowledge_cards()
        if existing_cards:
            for card in existing_cards:
                st.markdown(f"**{card.get('title', '未命名')}** - 目标客户: {card.get('target', '未知')}")
                st.caption(f"摘要: {card.get('summary', '')[:100]}...")
                st.caption(f"来源: {card.get('source_file', '未知')} | 创建时间: {card.get('created_at', '未知')}")
                st.divider()
        else:
            st.info("暂无案例库，请上传第一个案例。")


def _render_student_workspace(profiles: List[Dict[str, Any]]) -> None:
    """个体评估与干预（Dialog 版）"""
    st.markdown("### 个体评估与干预")
    if not profiles:
        st.info("当前日志中暂无可用的学生记录。")
        return

    normalized_profiles: List[Dict[str, Any]] = []
    for raw_profile in profiles:
        profile = dict(raw_profile)
        profile["student_label"] = _clean_display_text(profile.get("student_label"), "学生")
        profile["project_name"] = _clean_display_text(profile.get("project_name"), "未命名项目")
        normalized_profiles.append(profile)
    profiles = normalized_profiles

    roster_df = _build_student_roster(profiles)

    st.markdown("#### 👆 点击学生姓名查看深度透视")
    for idx, row in roster_df.iterrows():
        student_name = row["学生姓名"]
        project_name = row["项目名称"]
        risk_level = row.get("风险等级", "🟢 低危")
        risk_summary = row.get("核心风险诊断", "")

        col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
        col1.markdown(f"**{student_name}**")
        col2.markdown(project_name)
        col3.markdown(f"{risk_level} | {risk_summary}")

        target_profile = next(
            (p for p in profiles if p.get("student_label") == student_name), None
        )
        if target_profile and col4.button("🔍 洞察", key=f"view_{student_name}_{idx}"):
            _student_drilldown_dialog(target_profile)


def _render_teacher_chat(profiles: List[Dict[str, Any]]) -> None:
    # 仅 UI 调整，不影响逻辑
    st.markdown("### 教师智能助理对话")
    if st.session_state.teacher_agent_active_agent:
        st.caption(f"当前智能体：{_zh_label(st.session_state.teacher_agent_active_agent)}")
    if st.session_state.teacher_agent_route_target:
        st.caption(f"当前路由：{_zh_label(st.session_state.teacher_agent_route_target)}")
    if st.session_state.teacher_agent_next_step:
        st.info(f"任务追踪：{st.session_state.teacher_agent_next_step}")

    for message in st.session_state.teacher_chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("例如：帮我看看这名学生的财务风险")
    if not prompt:
        return

    profile = _resolve_student_from_prompt(
        prompt,
        profiles,
        st.session_state.teacher_selected_student_key,
    )
    if profile is None:
        st.warning("当前没有可用的学生上下文。")
        return

    st.session_state.teacher_selected_student_key = profile["student_key"]
    st.session_state.teacher_chat_messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("🧠 助理正在查询知识图谱与学生数据，请稍候..."):
            try:
                st.toast("DEBUG: 开始调用 LangGraph...")

                final_state, response = _invoke_teacher_agent(
                    profile,
                    prompt=prompt,
                    teacher_mode="teacher_chat",
                    route_hint="",
                    messages=st.session_state.teacher_chat_messages,
                )

                if response:
                    st.markdown(response)
                    st.session_state.teacher_chat_messages.append({"role": "assistant", "content": response})
                else:
                    st.warning("智能体执行完毕，但返回了空响应。请检查 System Prompt 或图谱检索结果。")

                route_target = str(final_state.get("context_data", {}).get("route_target", "")).strip()
                status_label = "已更新对话"
                if route_target == "Assessment Assistant":
                    status_label = "已更新评估"
                elif route_target == "Instructor Assistant":
                    status_label = "已更新干预"

                _set_intervention_status(
                    profile["student_key"],
                    status_label,
                    str(final_state.get("next_step", "")).strip(),
                )

            except Exception as e:
                st.error("执行中断！捕获到以下异常：")
                st.exception(e)
                st.session_state.teacher_chat_messages.append({"role": "assistant", "content": f"❌ 执行出错: {str(e)}"})

    st.rerun()


def render_teacher_page() -> None:
    init_teacher_state()
    # 仅 UI 调整，不影响逻辑
    _inject_teacher_portal_styles()

    with st.sidebar:
        st.header("日志配置")
        log_file = st.text_input("日志数据库路径", value=str(DEFAULT_LOG_FILE))
        st.caption("教师端按钮流与对话流均使用该数据库构建学生上下文。")
        use_mock_data = st.toggle(
            "使用高保真模拟数据",
            value=bool(st.session_state.get("teacher_use_mock_data", False)),
            help="当日志不足或接口波动时，可切换到稳定的创新创业评估模拟数据。",
        )
        st.session_state.teacher_use_mock_data = bool(use_mock_data)
        class_options = list_class_options(log_file, use_mock=st.session_state.teacher_use_mock_data)
        selected_class = st.selectbox(
            "当前班级",
            options=class_options,
            index=(
                class_options.index(st.session_state.teacher_selected_class)
                if st.session_state.teacher_selected_class in class_options
                else 0
            ),
        )
        if selected_class != st.session_state.teacher_selected_class:
            st.session_state.teacher_selected_class = selected_class
            _reset_teacher_student_context()
            st.rerun()

        history_rows = st.session_state.get("teacher_intervention_history", [])
        if history_rows:
            st.divider()
            st.markdown("#### 最近干预记录")
            for row in history_rows[:5]:
                status_text = _clean_display_text(_zh_label(row.get("status", "")), "未知")
                st.caption(
                    f"{row.get('updated_at', '')} | {row.get('student_key', '')} | {status_text}"
                )

    selected_class = st.session_state.teacher_selected_class
    use_mock = st.session_state.teacher_use_mock_data
    df = load_and_process_logs(log_file, selected_class=selected_class, use_mock=use_mock)
    profiles = build_student_profiles(log_file, selected_class=selected_class, use_mock=use_mock)
    if not profiles and selected_class != "全部班级":
        st.info("当前班级暂无记录，已自动回退到全部班级视图。")
        st.session_state.teacher_selected_class = "全部班级"
        _reset_teacher_student_context()
        st.rerun()

    _render_dashboard(df, profiles, log_file)
    _render_student_workspace(profiles)
    _render_teacher_chat(profiles)
    _render_case_library()


render_teacher_page()


