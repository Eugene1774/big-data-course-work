from __future__ import annotations

import json
import logging
import os
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from core.rbac import enforce_rbac
from neo4j_resilience import get_managed_driver
from rules.rule_switches import RULE_IDS, load_rule_switches, save_rule_switches


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")
LOGGER = logging.getLogger(__name__)

ALLOWED_GRAPH_LABELS = [
    "Project",
    "Case",
    "Concept",
    "KnowledgeCard",
    "Method",
    "Risk_Pattern_Edge",
    "Value_Loop_Edge",
    "Risk_Pattern",
    "Value_Loop",
    "Hyperedge",
]

# 图谱渲染调试大盘在 session_state 中的暂存键（跨函数传递埋点指标）
GRAPH_DEBUG_STATE_KEY = "graph_render_debug"

HEALTH_PLACEHOLDER = pd.DataFrame(
    {
        "health_band": ["优秀", "良好", "关注", "高风险"],
        "project_count": [18, 32, 14, 6],
    }
)

# 数据存储降级：使用本地 JSON 存储用户信息与能力分数
USERS_JSON_PATH = PROJECT_ROOT / "data" / "users.json"
CLASSES_JSON_PATH = PROJECT_ROOT / "data" / "classes.json"
USER_PROFILES_DIR = PROJECT_ROOT / "data" / "profiles"
INTERACTION_LOGS_DIR = PROJECT_ROOT / "data" / "interaction_logs"

USER_PROFILES_DIR.mkdir(parents=True, exist_ok=True)
INTERACTION_LOGS_DIR.mkdir(parents=True, exist_ok=True)


def load_classes_data() -> Dict[str, Any]:
    if not CLASSES_JSON_PATH.exists():
        return {}
    try:
        return json.loads(CLASSES_JSON_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_classes_data(classes: Dict[str, Any]) -> bool:
    try:
        CLASSES_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
        CLASSES_JSON_PATH.write_text(json.dumps(classes, ensure_ascii=False, indent=2), encoding="utf-8")
        return True
    except Exception:
        return False


def load_users_data() -> List[Dict[str, Any]]:
    """加载用户数据"""
    if not USERS_JSON_PATH.exists():
        return []
    try:
        return json.loads(USERS_JSON_PATH.read_text())
    except Exception:
        return []


def save_users_data(users: List[Dict[str, Any]]) -> None:
    """保存用户数据"""
    data_dir = USERS_JSON_PATH.parent
    data_dir.mkdir(exist_ok=True, parents=True)
    try:
        USERS_JSON_PATH.write_text(json.dumps(users, ensure_ascii=False, indent=2))
    except Exception:
        pass


TOP_VULNERABILITIES = [
    {"rule_id": "H1", "name": "客户-价值主张错位", "count": 28},
    {"rule_id": "H8", "name": "单位经济不成立", "count": 22},
    {"rule_id": "H4", "name": "解决方案不可行", "count": 18},
]


def build_business_vulnerabilities_chart():
    """构建业务漏洞 Top 3 图表"""
    df = pd.DataFrame(TOP_VULNERABILITIES)
    fig = px.bar(
        df,
        x="name",
        y="count",
        color="rule_id",
        title="业务漏洞 Top 3 统计",
        labels={"name": "漏洞类型", "count": "触发次数"},
    )
    fig.update_layout(height=400, showlegend=False)
    return fig


def _get_runtime_config(key: str, default: str = "") -> str:
    value = str(os.getenv(key) or "").strip()
    if value:
        return value
    try:
        secret_value = st.secrets.get(key)
        if secret_value:
            return str(secret_value).strip()
    except Exception:
        pass
    return default


def _is_hyperedge_label(labels: List[str]) -> bool:
    normalized = {str(label or "").strip() for label in labels}
    if (
        "Risk_Pattern" in normalized
        or "Value_Loop" in normalized
        or "Hyperedge" in normalized
        or "Risk_Pattern_Edge" in normalized
        or "Value_Loop_Edge" in normalized
    ):
        return True
    return any(label.endswith("_Edge") for label in normalized)


def _node_kind(labels: List[str]) -> str:
    normalized = {str(label or "").strip() for label in labels}
    if _is_hyperedge_label(list(normalized)):
        return "Hyperedge"
    if "Project" in normalized or "Case" in normalized:
        return "Project"
    return "Concept"


def _node_style(kind: str, labels: List[str]) -> Dict[str, Any]:
    if kind == "Hyperedge":
        # Middle ring: hyperedges in red family.
        color = "#d62728" if "Risk_Pattern_Edge" in labels or "Risk_Pattern" in labels else "#ef4444"
        return {"size": 34, "color": color, "shape": "triangle", "symbolType": "triangle", "level": 2, "mass": 2.0}
    if kind == "Project":
        # Outer ring: projects in blue.
        return {"size": 30, "color": "#2563eb", "shape": "dot", "symbolType": "circle", "level": 3, "mass": 2.4}
    # Core ring: concepts/cards/methods in green.
    return {"size": 22, "color": "#16a34a", "shape": "dot", "symbolType": "circle", "level": 1, "mass": 1.4}


def _node_id(node: Any) -> str:
    element_id = getattr(node, "element_id", None)
    if element_id:
        return str(element_id)
    legacy_id = getattr(node, "id", None)
    return str(legacy_id) if legacy_id is not None else str(id(node))


def _node_label(node: Any, labels: List[str]) -> str:
    payload = dict(node)
    for key in ("name", "title", "project_name", "id", "rule_id", "case_id"):
        text = str(payload.get(key, "")).strip()
        if text:
            return text
    return labels[0] if labels else "Node"


def _has_allowed_label(raw_labels: List[str]) -> bool:
    return any(label in ALLOWED_GRAPH_LABELS or label.endswith("_Edge") for label in raw_labels)


def _fetch_graph_records(limit: int = 800) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    uri = _get_runtime_config("NEO4J_URI", "")
    user = _get_runtime_config("NEO4J_USER", "") or _get_runtime_config("NEO4J_USERNAME", "")
    password = _get_runtime_config("NEO4J_PASSWORD", "")
    db_name = _get_runtime_config("NEO4J_DATABASE", "")
    if not uri or not user or not password:
        raise RuntimeError("Neo4j 未配置：请设置 NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD。")

    print(f"[DEBUG] 准备向 Neo4j 发送查询，当前 Limit 值为 -> {limit}")
    print(f"[DEBUG] URI: {uri}, Database: {db_name}, User: {user}")

    driver = get_managed_driver(uri, user, password, db_name, validate_schema=False)

    with driver.session(database=db_name) as session:
        project_count = session.run("MATCH (p:Project) RETURN count(p) AS cnt").single()
        concept_count = session.run("MATCH (c:Concept) RETURN count(c) AS cnt").single()
        print(f"[DEBUG] 数据库统计 - Project数量: {project_count['cnt'] if project_count else 0}, Concept数量: {concept_count['cnt'] if concept_count else 0}")

    cypher = """
    MATCH (n)
    WHERE any(label IN labels(n) WHERE label IN $allowed_labels OR label ENDS WITH '_Edge')
    OPTIONAL MATCH (n)-[r]->(m)
    WHERE m IS NULL OR any(label IN labels(m) WHERE label IN $allowed_labels OR label ENDS WITH '_Edge')
    RETURN n, r, m
    LIMIT $limit
    """
    print(f"[DEBUG] Cypher查询: {cypher[:100]}...")
    connectivity_audit_cypher = """
    MATCH (c)
    WHERE any(label IN labels(c) WHERE label IN ['Concept', 'KnowledgeCard', 'Method'])
    WITH c,
         size([(c)--() | 1]) AS degree,
         size([(c)-[:PREREQ]-() | 1]) AS prereq_degree,
         size([(:Risk_Pattern_Edge)-[:INVOLVES]->(c) | 1]) + size([(:Value_Loop_Edge)-[:INVOLVES]->(c) | 1]) AS hyperedge_degree
    RETURN
      count(c) AS concept_total,
      sum(CASE WHEN degree = 0 THEN 1 ELSE 0 END) AS isolated_total,
      sum(CASE WHEN prereq_degree = 0 AND hyperedge_degree = 0 THEN 1 ELSE 0 END) AS no_prereq_or_hyperedge_total
    """
    stats = {"concept_total": 0, "isolated_total": 0, "no_prereq_or_hyperedge_total": 0}
    with driver.session(database=db_name) as session:
        records = list(
            session.run(
                cypher,
                {"limit": int(limit), "allowed_labels": list(ALLOWED_GRAPH_LABELS)},
            )
        )
        print(f"[DEBUG] Neo4j原始返回记录数: {len(records)}")
        
        audit_record = session.run(connectivity_audit_cypher).single()
        if audit_record:
            stats = {
                "concept_total": int(audit_record.get("concept_total") or 0),
                "isolated_total": int(audit_record.get("isolated_total") or 0),
                "no_prereq_or_hyperedge_total": int(audit_record.get("no_prereq_or_hyperedge_total") or 0),
            }

    if stats["isolated_total"] > 0:
        LOGGER.warning(
            "Concept connectivity warning: %s isolated concepts/cards detected (total=%s).",
            stats["isolated_total"],
            stats["concept_total"],
        )
    if stats["no_prereq_or_hyperedge_total"] > 0:
        LOGGER.warning(
            "Concept connectivity warning: %s concepts/cards are missing both PREREQ and hyperedge INVOLVES links.",
            stats["no_prereq_or_hyperedge_total"],
        )

    return records, stats


def _build_agraph_payload(records: List[Dict[str, Any]]) -> Tuple[List[Any], List[Any], Dict[str, Dict[str, Any]], Dict[str, int]]:
    from streamlit_agraph import Edge, Node

    node_map: Dict[str, Any] = {}
    edge_map: Dict[str, Any] = {}
    node_details: Dict[str, Dict[str, Any]] = {}
    dirty_edge_count = 0
    # 调试埋点：统计标签漂移导致的静默过滤规模
    total_node_occurrences = 0
    filtered_node_count = 0

    CORE_NODE_LABELS = {"Concept", "KnowledgeCard", "Method", "Task", "Metric", "Mistake", "Evidence"}

    for record in records:
        n = record.get("n")
        m = record.get("m")
        rel = record.get("r")
        for raw_node in (n, m):
            if raw_node is None:
                continue
            labels = [str(label) for label in list(getattr(raw_node, "labels", []))]
            total_node_occurrences += 1
            if not _has_allowed_label(labels):
                filtered_node_count += 1
                LOGGER.warning(
                    "Graph node filtered out by label whitelist | labels=%s | element_id=%s",
                    labels,
                    _node_id(raw_node),
                )
                continue
            kind = _node_kind(labels)
            element_id = _node_id(raw_node)
            node_name = _node_label(raw_node, labels)

            if kind == "Concept" or any(label in ["Concept", "KnowledgeCard", "Method", "Tag", "Rule"] for label in labels):
                dedup_key = f"core:{node_name}"
            elif kind == "Hyperedge":
                dedup_key = f"hyperedge:{node_name}"
            else:
                dedup_key = f"project:{element_id}"

            if dedup_key not in node_map:
                style = _node_style(kind, labels)
                node_map[dedup_key] = Node(
                    id=dedup_key,
                    label=node_name,
                    title=f"{kind} | {', '.join(labels) if labels else 'NoLabel'}",
                    size=style["size"],
                    color=style["color"],
                    shape=style["shape"],
                    symbolType=style["symbolType"],
                    group=kind,
                    level=style.get("level", 1),
                    mass=style.get("mass", 1.0),
                )
                node_details[dedup_key] = {
                    "kind": kind,
                    "labels": labels,
                    "properties": dict(raw_node),
                    "original_element_id": element_id,
                }

        if n is None or m is None or rel is None:
            continue

        n_labels = [str(label) for label in list(getattr(n, "labels", []))]
        m_labels = [str(label) for label in list(getattr(m, "labels", []))]

        source_name = _node_label(n, n_labels)
        target_name = _node_label(m, m_labels)

        if any(label in ["Concept", "KnowledgeCard", "Method", "Tag", "Rule"] for label in n_labels):
            source_id = f"core:{source_name}"
        elif "Hyperedge" in str(n_labels):
            source_id = f"hyperedge:{source_name}"
        else:
            source_id = f"project:{_node_id(n)}"

        if any(label in ["Concept", "KnowledgeCard", "Method", "Tag", "Rule"] for label in m_labels):
            target_id = f"core:{target_name}"
        elif "Hyperedge" in str(m_labels):
            target_id = f"hyperedge:{target_name}"
        else:
            target_id = f"project:{_node_id(m)}"

        if source_id not in node_map or target_id not in node_map:
            dirty_edge_count += 1
            continue
        rel_type = str(getattr(rel, "type", "") or "RELATED_TO")
        rel_id = str(getattr(rel, "element_id", "") or f"{source_id}:{rel_type}:{target_id}")
        if rel_id not in edge_map:
            edge_map[rel_id] = Edge(
                source=source_id,
                target=target_id,
                label=rel_type,
                length=220,
            )

    if dirty_edge_count > 0:
        LOGGER.warning("Filtered %s dirty edges with missing source/target nodes.", dirty_edge_count)
    if filtered_node_count > 0:
        LOGGER.warning(
            "Filtered %s nodes outside ALLOWED_GRAPH_LABELS whitelist (total occurrences=%s).",
            filtered_node_count,
            total_node_occurrences,
        )

    payload_metrics = {
        "total_node_occurrences": total_node_occurrences,
        "filtered_node_count": filtered_node_count,
        "dirty_edge_count": dirty_edge_count,
        "unique_node_count": len(node_map),
        "edge_count": len(edge_map),
    }
    return list(node_map.values()), list(edge_map.values()), node_details, payload_metrics


def _render_graph_debug_panel() -> None:
    """在侧边栏渲染开发者调试大盘，读取 session_state 中暂存的图谱管线埋点指标。"""
    debug = st.session_state.get(GRAPH_DEBUG_STATE_KEY)
    if not debug:
        return

    with st.sidebar.expander("🛠️ 图谱渲染调试大盘", expanded=False):
        st.caption("本轮图谱渲染管线的遥测指标（仅供开发者排查标签漂移/静默过滤）")

        m1, m2 = st.columns(2)
        m1.metric("Neo4j 返回原始记录数", int(debug.get("records_count", 0)))
        m2.metric("处理节点总数(含重复)", int(debug.get("total_node_occurrences", 0)))

        m3, m4 = st.columns(2)
        m3.metric("被白名单过滤节点数", int(debug.get("filtered_node_count", 0)))
        m4.metric("被丢弃脏边数", int(debug.get("dirty_edge_count", 0)))

        m5, m6 = st.columns(2)
        m5.metric("最终渲染节点数", int(debug.get("unique_node_count", 0)))
        m6.metric("最终渲染边数", int(debug.get("edge_count", 0)))

        status = str(debug.get("status", "unknown"))
        if status == "ok":
            st.success("管线状态：渲染成功，无过滤。")
        elif status == "filtered":
            st.warning("管线状态：渲染成功，但检测到节点/边被过滤（疑似标签漂移）。")
        elif status == "empty":
            st.info("管线状态：数据库连通但返回空数据。")
        else:
            st.error(f"管线状态：异常（{status}）")

        if debug.get("error"):
            st.text(f"最近错误：{debug['error']}")
        if int(debug.get("filtered_node_count", 0)) > 0 or int(debug.get("dirty_edge_count", 0)) > 0:
            st.caption("⚠️ 完整被过滤节点的标签明细已写入服务端日志（streamlit.log），请检索关键字 'label whitelist'。")


def _render_hypergraph_dashboard() -> None:
    st.subheader("全校双创项目与知识图谱拓扑大盘 (Hypergraph Visualization)")
    st.caption("展示 Project(外圈) -> Hyperedge(中层) -> Concept/Method(核心圈) 的全量图谱链路。")

    # 初始化本轮渲染的遥测暂存区：_build_agraph_payload 写入指标，侧边栏调试大盘读取。
    st.session_state[GRAPH_DEBUG_STATE_KEY] = {
        "status": "init",
        "records_count": 0,
        "total_node_occurrences": 0,
        "filtered_node_count": 0,
        "dirty_edge_count": 0,
        "unique_node_count": 0,
        "edge_count": 0,
        "error": "",
    }

    try:
        from streamlit_agraph import Config, agraph
    except Exception:
        LOGGER.exception("streamlit-agraph 组件导入失败")
        st.error("未检测到 `streamlit-agraph`，请先安装后重启。")
        st.code("pip install streamlit-agraph", language="bash")
        return

    limit = st.slider("图谱拉取上限", min_value=500, max_value=2000, value=800, step=100)

    try:
        records, stats = _fetch_graph_records(limit=limit)
    except Exception as exc:
        # 文件日志：记录完整 Traceback，便于事后在 streamlit.log 中复盘
        LOGGER.exception("图谱查询失败（Neo4j 连通性/查询异常）")
        st.session_state[GRAPH_DEBUG_STATE_KEY].update({"status": "query_failed", "error": str(exc)})
        _render_graph_debug_panel()
        # 优雅降级：渲染显眼的"维护中"占位面板，替代页面底部干瘪的 st.error
        with st.container(border=True):
            st.warning(
                "🛌 **知识图谱数据库当前处于休眠或维护状态，图谱可视化暂不可用。**\n\n"
                "Neo4j 实例可能因长时间无访问被自动暂停。请联系管理员前往 Neo4j Aura Console "
                "唤醒（Resume）实例后刷新本页面。",
                icon="🚧",
            )
            with st.expander("查看错误详情"):
                st.code(traceback.format_exc(), language="text")
        st.toast("⚠️ 图谱数据库连接失败，请查看维护提示", icon="🚨")
        return

    st.session_state[GRAPH_DEBUG_STATE_KEY]["records_count"] = len(records)

    if not records:
        st.session_state[GRAPH_DEBUG_STATE_KEY].update({"status": "empty"})
        _render_graph_debug_panel()
        # 空数据占位：约 400px 高的空白容器 + 居中的友好提示
        placeholder = st.container(height=400, border=True)
        placeholder.markdown(
            "<div style='display:flex;flex-direction:column;align-items:center;"
            "justify-content:center;height:100%;color:#8a8f98;'>"
            "<h3 style='margin-bottom:8px;'>📭 暂无图谱数据</h3>"
            "<p style='margin:0;'>Neo4j 当前无可视化关系数据，请先运行图谱构建流水线（kg_pipeline.py）。</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Concept/Card 总数", int(stats.get("concept_total", 0)))
    c2.metric("孤立 Concept 数", int(stats.get("isolated_total", 0)))
    c3.metric("缺 PREREQ/INVOLVES 数", int(stats.get("no_prereq_or_hyperedge_total", 0)))
    if int(stats.get("isolated_total", 0)) > 0 or int(stats.get("no_prereq_or_hyperedge_total", 0)) > 0:
        st.warning("检测到知识卡片连通性风险：存在孤立节点或未被 PREREQ/超边贯穿的节点。")

    try:
        nodes, edges, node_details, payload_metrics = _build_agraph_payload(records)
        print(f"[DEBUG] 准备渲染，最终生成的节点数: {len(nodes)}，边数: {len(edges)}")
        # 埋点回写：将 payload 组装指标暂存到 session_state，供侧边栏调试大盘读取
        debug_state = st.session_state[GRAPH_DEBUG_STATE_KEY]
        debug_state.update(payload_metrics)
        if payload_metrics["filtered_node_count"] > 0 or payload_metrics["dirty_edge_count"] > 0:
            debug_state["status"] = "filtered"
            st.toast("⚠️ 触发图谱节点/边过滤，请查看调试面板", icon="🚨")
        else:
            debug_state["status"] = "ok"
        _render_graph_debug_panel()
        config = Config(
            width="100%",
            height=820,
            directed=True,
            physics=True,
            hierarchical=False,
            direction="LR",
            levelSeparation=240,
            nodeSpacing=220,
            treeSpacing=320,
            sortMethod="directed",
            solver="forceAtlas2Based",
            minVelocity=0.75,
            maxVelocity=80,
            timestep=0.45,
            forceAtlas2Based={
                "gravitationalConstant": -150,
                "centralGravity": 0.01,
                "springLength": 200,
                "springConstant": 0.08,
                "damping": 0.4,
                "avoidOverlap": 0.5,
            },
            nodeHighlightBehavior=True,
            highlightColor="#ffd166",
            edges={
                "color": {"inherit": False, "color": "#94a3b8", "highlight": "#334155"},
                "smooth": {"enabled": True, "type": "dynamic", "roundness": 0.2},
                "length": 240,
            },
            nodes={
                "font": {"size": 11, "strokeWidth": 3, "strokeColor": "#ffffff"},
                "shadow": True,
            },
        )
        return_value = agraph(nodes=nodes, edges=edges, config=config)
    except Exception as exc:
        LOGGER.exception("图谱渲染失败（payload 组装或 agraph 组件异常）")
        st.session_state[GRAPH_DEBUG_STATE_KEY].update({"status": "render_failed", "error": str(exc)})
        _render_graph_debug_panel()
        with st.container(border=True):
            st.warning(
                "🛌 **知识图谱可视化组件渲染失败，图谱暂不可用。**\n\n"
                "完整错误已记录到服务端日志，请联系管理员排查。",
                icon="🚧",
            )
            with st.expander("查看错误详情"):
                st.code(traceback.format_exc(), language="text")
        return

    if not return_value:
        return

    selected_id = str(return_value if not isinstance(return_value, dict) else return_value.get("id", "")).strip()
    if not selected_id:
        return

    details = node_details.get(selected_id, {})
    if not details:
        st.info(f"已选节点: {selected_id}")
        return

    st.info(f"你点击了节点: {selected_id}")
    if str(details.get("kind", "")).strip() == "Hyperedge":
        properties = details.get("properties", {}) if isinstance(details.get("properties"), dict) else {}
        detected_fallacies = properties.get("detected_fallacies") or properties.get("rule_id") or []
        selected_strategy = properties.get("selected_strategy") or properties.get("strategy") or "未配置"
        historical_failure = properties.get("historical_failure") or properties.get("failure_case") or "未提供"
        with st.expander("风险超边详情", expanded=False):
            st.markdown(f"**detected_fallacies**: `{detected_fallacies}`")
            st.markdown(f"**selected_strategy**: `{selected_strategy}`")
            st.markdown(f"**historical_failure**: `{historical_failure}`")
            st.json(details)
    else:
        with st.expander("节点详情", expanded=False):
            st.json(details)

def init_admin_state() -> None:
    if "admin_rule_switches" not in st.session_state:
        st.session_state.admin_rule_switches = load_rule_switches()


def build_health_distribution_chart():
    fig = px.bar(
        HEALTH_PLACEHOLDER,
        x="health_band",
        y="project_count",
        color="health_band",
        title="全校项目健康度分布",
        labels={"health_band": "健康分层", "project_count": "项目数"},
    )
    fig.update_layout(height=420, showlegend=False)
    return fig


def render_rule_switch_panel() -> None:
    st.subheader("规则开关")
    st.caption("支持按规则维度启用/关闭，实时写入本地配置。")

    before = dict(st.session_state.admin_rule_switches)
    all_enabled = st.checkbox("全部启用", value=all(st.session_state.admin_rule_switches.values()))
    if all_enabled != all(st.session_state.admin_rule_switches.values()):
        for rule_id in RULE_IDS:
            st.session_state.admin_rule_switches[rule_id] = all_enabled

    columns = st.columns(3)
    for index, rule_id in enumerate(RULE_IDS):
        column = columns[index % 3]
        with column:
            st.session_state.admin_rule_switches[rule_id] = st.toggle(
                rule_id,
                value=st.session_state.admin_rule_switches[rule_id],
                key=f"toggle_{rule_id}",
            )

    if before != st.session_state.admin_rule_switches:
        st.session_state.admin_rule_switches = save_rule_switches(st.session_state.admin_rule_switches)

    switch_df = pd.DataFrame(
        [
            {"rule_id": rule_id, "enabled": st.session_state.admin_rule_switches[rule_id]}
            for rule_id in RULE_IDS
        ]
    )
    st.dataframe(switch_df, use_container_width=True)


def render_admin_page() -> None:
    enforce_rbac("admin", page_name="admin_portal")
    init_admin_state()

    st.title("全局预警")
    st.caption("教务端全局监控页面：展示项目健康分布与规则开关状态。")

    # 教务宏观看板
    st.markdown("## 教务宏观看板")
    
    # 关键指标
    col1, col2, col3 = st.columns(3)
    col1.metric("全校项目数", "70")
    col2.metric("高风险项目数", "6")
    col3.metric("平均健康度", "78%")
    
    # 健康度分布图
    st.plotly_chart(build_health_distribution_chart(), use_container_width=True)
    
    # 业务漏洞 Top 3 统计
    st.plotly_chart(build_business_vulnerabilities_chart(), use_container_width=True)
    
    # 数据存储降级管理
    st.markdown("## 数据存储管理")
    st.caption("使用本地 JSON 存储用户信息与能力分数，Neo4j 仅作为证据库")

    # 查看用户数据
    users = load_users_data()
    if users:
        st.dataframe(pd.DataFrame(users), use_container_width=True)
    else:
        st.info("暂无用户数据")

    # CSV 批量导入功能
    st.markdown("### 📥 批量导入师生名单")
    st.markdown("上传 CSV 格式：`账号(uid), 姓名(name), 角色(role), 班级编号(class_id)`")

    csv_file = st.file_uploader("选择 CSV 文件", type=["csv"])
    if csv_file is not None:
        try:
            df = pd.read_csv(csv_file)
            st.dataframe(df.head())

            if st.button("确认导入并初始化账号", type="primary"):
                classes_data = load_classes_data()

                success_count = 0
                for index, row in df.iterrows():
                    uid = str(row.get("uid", "")).strip()
                    name = str(row.get("name", "")).strip()
                    role = str(row.get("role", "student")).strip()
                    class_id = str(row.get("class_id", "")).strip()

                    if not uid:
                        continue

                    if class_id:
                        if class_id not in classes_data:
                            classes_data[class_id] = {"class_id": class_id, "students": []}
                        if uid not in classes_data[class_id]["students"]:
                            classes_data[class_id]["students"].append(uid)

                    initial_log = [
                        {
                            "role": "assistant",
                            "content": f"你好，{name}！我是你的特色大创智能教练。目前你的项目档案还是空白的，请问你已经有初步的创业想法了吗？",
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                        }
                    ]
                    log_path = INTERACTION_LOGS_DIR / f"{uid}_logs.json"
                    try:
                        log_path.write_text(json.dumps(initial_log, ensure_ascii=False), encoding="utf-8")
                    except Exception:
                        pass

                    profile_data = {
                        "user_id": uid,
                        "name": name,
                        "role": role,
                        "class_id": class_id,
                        "scores": [0, 0, 0, 0, 0],
                        "projects": []
                    }
                    profile_path = USER_PROFILES_DIR / f"{uid}.json"
                    try:
                        profile_path.write_text(json.dumps(profile_data, ensure_ascii=False, indent=2), encoding="utf-8")
                    except Exception:
                        pass

                    success_count += 1

                save_classes_data(classes_data)
                st.success(f"✅ 成功导入并创建了 {success_count} 个用户账号！")

        except Exception as e:
            st.error(f"读取 CSV 失败: {str(e)}")

    # 查看和编辑班级数据
    st.markdown("### 🏢 班级管理")
    classes_data = load_classes_data()
    if classes_data:
        for class_id, class_info in classes_data.items():
            with st.expander(f"班级: {class_info.get('class_name', class_id)} ({len(class_info.get('students', []))} 人)"):
                st.write(f"班级ID: {class_id}")
                st.write(f"学生列表: {', '.join(class_info.get('students', []))}")
    else:
        st.info("暂无班级数据，请先上传 CSV 文件导入师生名单。")

    # 规则开关
    render_rule_switch_panel()
    
    # 超图可视化
    _render_hypergraph_dashboard()


render_admin_page()

