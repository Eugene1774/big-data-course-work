from __future__ import annotations

import json
import logging
import os
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Sequence

from langchain_core.tools import BaseTool
from openai import OpenAI
from pydantic import BaseModel, Field

from neo4j_resilience import ensure_external_driver_schema, get_managed_driver
from prompts import SYSTEM_PROMPT
from rules.checker import GraphRuleChecker, RuleResult, RuleStatus
from schema.case import KnowledgeGraphCase


TARGET_RULE_IDS: tuple[str, ...] = ("H1", "H4", "H6", "H8")
METRIC_KEYS: tuple[str, ...] = ("TAM", "SAM", "SOM", "CAC", "LTV")
PROJECT_ROOT = Path(__file__).resolve().parent
KNOWLEDGE_CARD_DIR = PROJECT_ROOT / "knowledge_base" / "cards"
DEFAULT_SILICONFLOW_BASE_URL = "https://api.siliconflow.com/v1"
DEFAULT_SILICONFLOW_MODEL = "Qwen/Qwen3-8B"
DEFAULT_GPUSTACK_BASE_URL = "https://maas.bit.edu.cn/v1"
DEFAULT_GPUSTACK_MODEL = "deepseek-v4-flash"
DEFAULT_NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
DEFAULT_NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
DEFAULT_NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4j")
DEFAULT_NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
LOGGER = logging.getLogger(__name__)
MAX_GUARDED_INPUT_LENGTH = 2000
EMOJI_ONLY_RE = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]")
SUSPICIOUS_MOJIBAKE_RE = re.compile(r"(?:�|Ã.|Â.|Ð.|Ñ.|Ê.|Õ.|æ.|å.|ç.|þ.|¤)")
PROMPT_INJECTION_RE = re.compile(
    r"(ignore\s+(all|any|previous)\s+(instructions|rules|messages)|"
    r"system\s+prompt|developer\s+message|reveal\s+prompt|jailbreak|do\s+anything\s+now|"
    r"忽略(之前|以上|所有).{0,20}(指令|规则|消息)|系统提示词|开发者消息|越狱|绕过)",
    re.IGNORECASE,
)
STOPWORDS = {
    "我们",
    "你们",
    "他们",
    "这个",
    "那个",
    "现在",
    "目前",
    "以及",
    "如果",
    "因为",
    "所以",
    "然后",
    "需要",
    "希望",
    "怎么",
    "什么",
    "请问",
    "一个",
    "项目",
    "学生",
    "老师",
    "用户",
    "产品",
    "方案",
}
BUSINESS_CONCEPTS = (
    "商业模式",
    "痛点",
    "用户画像",
    "价值主张",
    "核心竞争力",
    "技术壁垒",
    "获客成本",
    "留存率",
    "复购率",
    "转化率",
    "渠道策略",
    "销售渠道",
    "团队分工",
    "里程碑",
    "最小可行产品",
    "mvp",
    "tam",
    "sam",
    "som",
    "cac",
    "ltv",
)


def _ensure_list(value: Any) -> List[Any]:
    """Normalize a scalar or tuple-like value into a list."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _get_config_value(key: str, default: str | None = None) -> str | None:
    """Read configuration from env first, then Streamlit secrets."""
    value = os.getenv(key)
    if value:
        return value
    try:
        import streamlit as st

        secret_value = st.secrets.get(key)
        if secret_value:
            return str(secret_value).strip()
    except Exception:
        pass
    return default


class GraphSearchInput(BaseModel):
    """Input schema for the LangChain graph search tool."""

    cypher_query: str = Field(..., description="Cypher query used to retrieve graph paths.")
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Query parameters passed to Neo4j.",
    )
    limit: int = Field(default=5, description="Maximum number of graph paths to return.")


def _format_graph_node(node_payload: Dict[str, Any]) -> Dict[str, Any]:
    labels = node_payload.get("labels") or []
    return {
        "id": node_payload.get("id", ""),
        "name": node_payload.get("name", ""),
        "labels": labels,
        "label": labels[0] if labels else "Node",
        "source_excerpt": str(node_payload.get("source_excerpt", "") or "").strip(),
        "page_ref": str(node_payload.get("page_ref", "") or "").strip(),
    }


HYPEREDGE_ROLE_ALIASES = {
    "customer": "customer",
    "market": "customer",
    "persona": "customer",
    "segment": "customer",
    "user": "customer",
    "value_proposition": "value_proposition",
    "value proposition": "value_proposition",
    "value": "value_proposition",
    "technology": "value_proposition",
    "solution": "value_proposition",
    "mistake": "value_proposition",
    "channel": "channel",
    "distribution": "channel",
    "acquisition": "channel",
    "go_to_market": "channel",
    "gtm": "channel",
    "outcome": "channel",
}


def _canonical_edge_type(raw_edge_type: Any) -> str:
    text = str(raw_edge_type or "").strip()
    if text in {"Risk_Pattern_Edge", "Value_Loop_Edge"}:
        return text
    lowered = text.lower()
    if "risk" in lowered:
        return "Risk_Pattern_Edge"
    if "value" in lowered or "loop" in lowered:
        return "Value_Loop_Edge"
    return "Risk_Pattern_Edge"


def _canonical_rule_triggered(raw_rule: Any, edge_type: str) -> str:
    text = str(raw_rule or "").strip()
    if text:
        upper = text.upper()
        if upper.startswith("H1"):
            return "H1_BusinessModelConsistency"
        return text
    if edge_type == "Risk_Pattern_Edge":
        return "H1_BusinessModelConsistency"
    return "H9_ValueLoopFit"


def _canonical_role(role: Any) -> str:
    normalized = str(role or "").strip().lower().replace("-", "_").replace(" ", "_")
    return HYPEREDGE_ROLE_ALIASES.get(normalized, "")


def _normalize_nodes_involved(raw: Dict[str, Any], node_lookup: Dict[str, str] | None = None) -> Dict[str, str]:
    lookup = dict(node_lookup or {})
    nodes_involved = {
        "customer": "",
        "value_proposition": "",
        "channel": "",
    }

    direct_nodes = raw.get("nodes_involved")
    if isinstance(direct_nodes, dict):
        for key, value in direct_nodes.items():
            role = _canonical_role(key)
            if role and str(value or "").strip():
                nodes_involved[role] = str(value).strip()

    participants = raw.get("participants")
    if isinstance(participants, dict):
        participants_iter = [
            {"role": role, "node_name": lookup.get(str(node_id), str(node_id)), "node_id": str(node_id)}
            for role, node_id in participants.items()
        ]
    elif isinstance(participants, list):
        participants_iter = participants
    else:
        participants_iter = []

    for item in participants_iter:
        if not isinstance(item, dict):
            continue
        role = _canonical_role(item.get("role"))
        if not role:
            continue
        node_name = str(
            item.get("node_name")
            or item.get("node")
            or lookup.get(str(item.get("node_id", "")).strip(), "")
            or ""
        ).strip()
        if node_name and not nodes_involved[role]:
            nodes_involved[role] = node_name

    for key in ("customer", "value_proposition", "channel"):
        if not nodes_involved[key]:
            nodes_involved[key] = "unknown"
    return nodes_involved


def normalize_hyperedge_record(
    raw: Dict[str, Any],
    node_lookup: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    edge_type = _canonical_edge_type(raw.get("edge_type") or raw.get("label"))
    hyperedge_id = str(
        raw.get("hyperedge_id")
        or raw.get("edge_id")
        or raw.get("id")
        or ""
    ).strip()
    if not hyperedge_id:
        hyperedge_id = f"he_{slugify(str(raw.get('name') or edge_type))}"

    nodes_involved = _normalize_nodes_involved(raw, node_lookup=node_lookup)
    logical_evaluation = str(raw.get("logical_evaluation") or "").strip()
    if not logical_evaluation:
        logical_evaluation = "channel.cannot_reach(customer)"
    impact = str(raw.get("impact") or raw.get("edge_description") or raw.get("description") or "").strip()
    if not impact:
        impact = "unknown"

    return {
        "hyperedge_id": hyperedge_id,
        "edge_type": edge_type,
        "rule_triggered": _canonical_rule_triggered(raw.get("rule_triggered"), edge_type),
        "nodes_involved": nodes_involved,
        "logical_evaluation": logical_evaluation,
        "impact": impact,
    }


def normalize_hyperedge_records(
    records: Sequence[Dict[str, Any]],
    node_lookup: Dict[str, str] | None = None,
) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for item in records:
        if not isinstance(item, dict):
            continue
        normalized.append(normalize_hyperedge_record(item, node_lookup=node_lookup))
    return normalized


def _build_default_graph_query(keywords: Sequence[str], limit: int = 5) -> tuple[str, Dict[str, Any]]:
    filtered_keywords = [
        _normalize_text(keyword)
        for keyword in keywords
        if _normalize_text(keyword)
    ]
    query = """
    WITH $keywords AS keywords
    // 首先找到包含关键词的种子节点
    MATCH (seed)
    WHERE any(kw IN keywords WHERE
        toLower(coalesce(seed.name, '')) CONTAINS kw OR
        toLower(coalesce(seed.description, '')) CONTAINS kw OR
        toLower(coalesce(seed.tags_text, '')) CONTAINS kw)
    // 然后从种子节点进行1-2步跳跃检索
    MATCH path=(seed)-[*1..2]-(related)
    WITH DISTINCT path
    LIMIT $limit
    RETURN
        [node IN nodes(path) | {
            id: coalesce(node.id, node.rule_id, node.case_id, ''),
            name: coalesce(node.name, node.title, node.rule_id, node.case_id, ''),
            labels: labels(node),
            source_excerpt: coalesce(node.source_excerpt, node.evidence_source, node.description, ''),
            page_ref: coalesce(node.page_ref, node.page, node.page_number, '')
        }] AS path_nodes,
        [rel IN relationships(path) | {
            type: type(rel)
        }] AS path_relationships,
        [node IN nodes(path)
            WHERE any(label IN labels(node) WHERE label ENDS WITH '_Edge' OR label IN ['Hyperedge', 'Risk_Pattern', 'Value_Loop'])
            | {
                id: coalesce(node.id, ''),
                name: coalesce(node.name, node.id, ''),
                edge_type: coalesce(node.edge_type, head(labels(node)), ''),
                labels: labels(node),
                source_excerpt: coalesce(node.source_excerpt, node.description, ''),
                page_ref: coalesce(node.page_ref, node.page, node.page_number, '')
            }
        ] AS hyperedges
        ,
        [edge IN nodes(path)
            WHERE any(label IN labels(edge) WHERE label ENDS WITH '_Edge' OR label IN ['Hyperedge', 'Risk_Pattern', 'Value_Loop'])
            | {
                hyperedge_id: coalesce(edge.id, ''),
                edge_type: coalesce(edge.edge_type, head(labels(edge)), ''),
                rule_triggered: head([(edge)-[:EVALUATED_BY]->(rule:Rule) | coalesce(rule.rule_id, '')]),
                participants: [(edge)-[part_rel]->(participant)
                    WHERE type(part_rel) IN ['HAS_PARTICIPANT', 'INVOLVES']
                    | {
                        role: coalesce(part_rel.role, toLower(type(part_rel))),
                        node_name: coalesce(participant.name, participant.title, participant.id, ''),
                        node_id: coalesce(participant.id, '')
                    }
                ],
                logical_evaluation: coalesce(edge.logical_evaluation, edge.evaluation, edge.logic_result, ''),
                impact: coalesce(edge.impact, edge.description, ''),
                description: coalesce(edge.description, ''),
                source_excerpt: coalesce(edge.source_excerpt, edge.description, ''),
                page_ref: coalesce(edge.page_ref, edge.page, edge.page_number, '')
            }
        ] AS hyperedge_records
    """
    return query, {"keywords": filtered_keywords, "limit": limit}


def _resolve_graph_driver(context_data: Dict[str, Any]) -> tuple[Any, str, bool]:
    database = str(
        context_data.get("neo4j_database")
        or _get_config_value("NEO4J_DATABASE", DEFAULT_NEO4J_DATABASE)
        or DEFAULT_NEO4J_DATABASE
    )

    driver = context_data.get("neo4j_driver")
    if driver is not None:
        uri = str(context_data.get("neo4j_uri") or _get_config_value("NEO4J_URI") or "").strip()
        user = str(context_data.get("neo4j_user") or _get_config_value("NEO4J_USER") or "").strip()
        password = str(context_data.get("neo4j_password") or _get_config_value("NEO4J_PASSWORD") or "").strip()
        ensure_external_driver_schema(
            driver,
            database,
            uri=uri or "<external-driver>",
            user=user,
            password=password,
        )
        return driver, database, False

    uri = str(
        context_data.get("neo4j_uri")
        or _get_config_value("NEO4J_URI", DEFAULT_NEO4J_URI)
        or DEFAULT_NEO4J_URI
    ).strip()
    user = str(
        context_data.get("neo4j_user")
        or _get_config_value("NEO4J_USER", DEFAULT_NEO4J_USER)
        or DEFAULT_NEO4J_USER
    ).strip()
    password = str(
        context_data.get("neo4j_password")
        or _get_config_value("NEO4J_PASSWORD", DEFAULT_NEO4J_PASSWORD)
        or DEFAULT_NEO4J_PASSWORD
    ).strip()

    return (
        get_managed_driver(
            uri,
            user,
            password,
            database,
            validate_schema=True,
        ),
        database,
        False,
    )


def _close_graph_driver(driver: Any, should_close: bool) -> None:
    return None


def _stringify_graph_path(
    nodes: Sequence[Dict[str, Any]],
    relationships: Sequence[Dict[str, Any]],
) -> str:
    if not nodes:
        return ""

    segments: List[str] = []
    for index, node in enumerate(nodes):
        formatted = _format_graph_node(node)
        segments.append(f"{formatted['label']}:{formatted['name']}")
        if index < len(relationships):
            rel_type = str(relationships[index].get("type", "RELATED_TO"))
            segments.append(f"-[{rel_type}]->")
    return " ".join(segments)


def _empty_graph_payload() -> Dict[str, Any]:
    return {
        "retrieved_nodes": [],
        "retrieved_hyperedges": [],
        "path_evidence": [],
        "paths": [],
        "source_excerpt": "",
    }


def _execute_graph_search(
    context_data: Dict[str, Any],
    cypher_query: str,
    parameters: Dict[str, Any] | None = None,
    limit: int = 5,
) -> Dict[str, Any]:
    if not isinstance(context_data, dict):
        return _empty_graph_payload()

    driver, database, owns_driver = _resolve_graph_driver(context_data)

    retrieved_nodes: List[Dict[str, Any]] = []
    retrieved_edges: List[Dict[str, Any]] = []
    retrieved_hyperedges: List[Dict[str, Any]] = []
    path_evidence: List[Dict[str, Any]] = []
    paths: List[Dict[str, Any]] = []
    parameters = dict(parameters or {})
    parameters.setdefault("limit", limit)

    try:
        with driver.session(database=database) as session:
            result = session.run(cypher_query, parameters)
            for record in result:
                nodes = [_format_graph_node(item) for item in record.get("path_nodes", [])]
                relationships = list(record.get("path_relationships", []))
                legacy_hyperedges = [_format_graph_node(item) for item in record.get("hyperedges", [])]
                node_lookup = {
                    str(node.get("id", "")).strip(): str(node.get("name", "")).strip()
                    for node in [*nodes, *legacy_hyperedges]
                    if str(node.get("id", "")).strip()
                }
                raw_hyperedge_records = [
                    item for item in record.get("hyperedge_records", [])
                    if isinstance(item, dict)
                ]
                hyperedges = normalize_hyperedge_records(
                    raw_hyperedge_records if raw_hyperedge_records else legacy_hyperedges,
                    node_lookup=node_lookup,
                )
                
                # 构建edges列表
                edges = []
                for i, rel in enumerate(relationships):
                    if i < len(nodes) - 1:
                        edges.append({
                            "source": nodes[i].get("id", ""),
                            "target": nodes[i+1].get("id", ""),
                            "type": str(rel.get("type", "RELATED_TO")),
                            "logic": ""
                        })
                
                path_string = _stringify_graph_path(nodes, relationships)
                source_excerpt = next(
                    (
                        str(node.get("source_excerpt", "")).strip()
                        for node in [*nodes, *legacy_hyperedges, *raw_hyperedge_records]
                        if str(node.get("source_excerpt", "")).strip()
                    ),
                    "",
                )
                page_ref = next(
                    (
                        str(node.get("page_ref", "")).strip()
                        for node in [*nodes, *legacy_hyperedges, *raw_hyperedge_records]
                        if str(node.get("page_ref", "")).strip()
                    ),
                    "",
                )

                if path_string:
                    path_evidence.append(
                        {
                            "path_string": path_string,
                            "source_excerpt": source_excerpt,
                            "page_ref": page_ref,
                        }
                    )
                retrieved_nodes.extend(nodes)
                retrieved_edges.extend(edges)
                retrieved_hyperedges.extend(hyperedges)
                paths.append(
                    {
                        "nodes": nodes,
                        "relationships": relationships,
                        "edges": edges,
                        "hyperedges": hyperedges,
                        "hyperedge_nodes": legacy_hyperedges,
                        "path_string": path_string,
                        "source_excerpt": source_excerpt,
                        "page_ref": page_ref,
                    }
                )
            
            # 多跳超图遍历：对于每个节点，获取其1-2跳邻居
            from kg_hypergraph import get_weighted_neighbors_khop
            
            # 提取所有节点ID
            node_ids = [str(node.get("id", "")).strip() for node in retrieved_nodes if str(node.get("id", "")).strip()]
            if node_ids:
                # 对每个节点执行k-hop查询
                for node_id in node_ids[:5]:  # 限制处理的节点数量，避免性能问题
                    neighbors = get_weighted_neighbors_khop(
                        driver, database, node_id, k_hop=2, top_k=10
                    )
                    
                    # 为每个邻居获取详细信息
                    for neighbor in neighbors:
                        neighbor_id = neighbor.get("neighbor_id")
                        if neighbor_id and neighbor_id not in [n.get("id") for n in retrieved_nodes]:
                            # 查询邻居节点的详细信息
                            neighbor_query = """
                            MATCH (n:Node {id: $node_id})
                            RETURN {
                                id: coalesce(n.id, n.rule_id, n.case_id, ''),
                                name: coalesce(n.name, n.title, n.rule_id, n.case_id, ''),
                                labels: labels(n),
                                source_excerpt: coalesce(n.source_excerpt, n.evidence_source, n.description, ''),
                                page_ref: coalesce(n.page_ref, n.page, n.page_number, '')
                            } AS node
                            """
                            neighbor_result = session.run(neighbor_query, {"node_id": neighbor_id})
                            for neighbor_record in neighbor_result:
                                neighbor_node = neighbor_record.get("node")
                                if neighbor_node:
                                    formatted_node = _format_graph_node(neighbor_node)
                                    retrieved_nodes.append(formatted_node)
                                    
                                    # 添加从原节点到邻居的边
                                    retrieved_edges.append({
                                        "source": node_id,
                                        "target": neighbor_id,
                                        "type": "RELATED_TO",
                                        "logic": f"从节点 {node_id} 到邻居节点的关联"
                                    })
            
            # 超边显式提取：查询与规则相关的超边
            triggered_rules = _extract_triggered_rule_ids(context_data.get("diagnostic_results", []))
            if triggered_rules:
                for rule_id in triggered_rules:
                    # 构建查询，查找与规则相关的超边
                    rule_query = """
                    MATCH (rule:Rule {rule_id: $rule_id})<-[:EVALUATED_BY]-(hyperedge)
                    MATCH (hyperedge)-[r:HAS_PARTICIPANT]->(participant)
                    RETURN 
                        {
                            hyperedge_id: coalesce(hyperedge.id, ''),
                            edge_type: coalesce(hyperedge.edge_type, head(labels(hyperedge)), ''),
                            rule_triggered: $rule_id,
                            participants: [(hyperedge)-[part_rel:HAS_PARTICIPANT]->(p) 
                                | {
                                    role: coalesce(part_rel.role, toLower(type(part_rel))),
                                    node_name: coalesce(p.name, p.title, p.id, ''),
                                    node_id: coalesce(p.id, '')
                                }
                            ],
                            logical_evaluation: coalesce(hyperedge.logical_evaluation, hyperedge.evaluation, hyperedge.logic_result, ''),
                            impact: coalesce(hyperedge.impact, hyperedge.description, ''),
                            description: coalesce(hyperedge.description, ''),
                            source_excerpt: coalesce(hyperedge.source_excerpt, hyperedge.description, ''),
                            page_ref: coalesce(hyperedge.page_ref, hyperedge.page, hyperedge.page_number, '')
                        } AS hyperedge_record,
                        [p IN nodes((hyperedge)-[*1]->()) WHERE NOT p = hyperedge | {
                            id: coalesce(p.id, p.rule_id, p.case_id, ''),
                            name: coalesce(p.name, p.title, p.rule_id, p.case_id, ''),
                            labels: labels(p),
                            source_excerpt: coalesce(p.source_excerpt, p.evidence_source, p.description, ''),
                            page_ref: coalesce(p.page_ref, p.page, p.page_number, '')
                        }] AS participant_nodes
                    """
                    rule_result = session.run(rule_query, {"rule_id": rule_id})
                    for record in rule_result:
                        hyperedge_record = record.get("hyperedge_record")
                        participant_nodes = record.get("participant_nodes", [])
                        
                        if hyperedge_record:
                            # 规范化超边记录
                            node_lookup = {
                                str(node.get("id", "")).strip(): str(node.get("name", "")).strip()
                                for node in participant_nodes
                                if str(node.get("id", "")).strip()
                            }
                            normalized_hyperedge = normalize_hyperedge_record(hyperedge_record, node_lookup=node_lookup)
                            
                            # 添加到结果中
                            if normalized_hyperedge not in retrieved_hyperedges:
                                retrieved_hyperedges.append(normalized_hyperedge)
                            
                            # 添加参与者节点
                            for node in participant_nodes:
                                formatted_node = _format_graph_node(node)
                                if formatted_node not in retrieved_nodes:
                                    retrieved_nodes.append(formatted_node)
    except Exception as exc:
        LOGGER.exception(
            "Neo4j graph search failed | database=%s | query=%s | parameters=%s",
            database,
            cypher_query.strip(),
            parameters,
        )
        raise RuntimeError(f"Neo4j graph search failed for database '{database}'.") from exc
    finally:
        _close_graph_driver(driver, owns_driver)

    deduped_nodes = {
        (node["id"], node["name"], tuple(node.get("labels", []))): node
        for node in retrieved_nodes
    }
    deduped_edges = {
        (edge["source"], edge["target"], edge["type"]): edge
        for edge in retrieved_edges
    }
    deduped_hyperedges = {
        (
            edge.get("hyperedge_id", ""),
            edge.get("edge_type", ""),
            edge.get("rule_triggered", ""),
            edge.get("logical_evaluation", ""),
        ): edge
        for edge in retrieved_hyperedges
    }
    deduped_paths_map: Dict[str, Dict[str, Any]] = {}
    for item in path_evidence:
        path_string = str(item.get("path_string", "")).strip()
        if not path_string:
            continue
        if path_string not in deduped_paths_map:
            deduped_paths_map[path_string] = item
    deduped_paths = list(deduped_paths_map.values())
    top_excerpt = next(
        (str(item.get("source_excerpt", "")).strip() for item in deduped_paths if str(item.get("source_excerpt", "")).strip()),
        "",
    )

    # 构建符合要求的retrieved_subgraph结构
    retrieved_subgraph = {
        "nodes": [
            {
                "id": node.get("id", ""),
                "label": node.get("label", "Node"),
                "name": node.get("name", ""),
                "properties": {k: v for k, v in node.items() if k not in ["id", "label", "name"]}
            }
            for node in deduped_nodes.values()
        ],
        "edges": [
            {
                "source": edge["source"],
                "target": edge["target"],
                "type": edge["type"],
                "logic": edge.get("logic", "")
            }
            for edge in deduped_edges.values()
        ],
        "hyperedges": [
            {
                "edge_id": edge.get("hyperedge_id", ""),
                "type": edge.get("edge_type", ""),
                "contained_nodes": list(edge.get("nodes_involved", {}).values()),
                "logic_description": edge.get("impact", "") or edge.get("logical_evaluation", "")
            }
            for edge in deduped_hyperedges.values()
        ]
    }

    # 注入逻辑谬误追踪，当匹配到Risk_Pattern_Edge时返回具体谬误标签
    detected_fallacies = []
    for hyperedge in retrieved_hyperedges:
        if hyperedge.get("edge_type") == "Risk_Pattern_Edge":
            # 提取谬误标签
            rule_triggered = hyperedge.get("rule_triggered", "")
            impact = hyperedge.get("impact", "")
            logical_evaluation = hyperedge.get("logical_evaluation", "")
            
            # 根据规则ID和内容提取具体谬误标签
            fallacy_label = ""
            if "H1" in rule_triggered:
                fallacy_label = "商业模式一致性问题"
            elif "H4" in rule_triggered:
                fallacy_label = "市场规模估算错误"
            elif "H6" in rule_triggered:
                fallacy_label = "竞品分析不足"
            elif "H8" in rule_triggered:
                fallacy_label = "财务假设风险"
            elif "大数" in impact or "大数" in logical_evaluation:
                fallacy_label = "大数幻觉"
            elif "渠道" in impact or "渠道" in logical_evaluation:
                fallacy_label = "渠道错位"
            elif "用户" in impact or "用户" in logical_evaluation:
                fallacy_label = "用户需求误判"
            else:
                fallacy_label = "业务逻辑风险"
            
            if fallacy_label and fallacy_label not in detected_fallacies:
                detected_fallacies.append(fallacy_label)

    result = {
        "retrieved_nodes": list(deduped_nodes.values()),
        "retrieved_edges": list(deduped_edges.values()),
        "retrieved_hyperedges": list(deduped_hyperedges.values()),
        "retrieved_subgraph": retrieved_subgraph,
        "path_evidence": deduped_paths,
        "paths": paths,
        "source_excerpt": top_excerpt,
        "detected_fallacies": detected_fallacies,
    }

    # 明文打印检索结果，证明这是真实的超图推理，而非向量检索 RAG
    LOGGER.info("=" * 80)
    LOGGER.info("【超图推理证据链】检索结果明文日志")
    LOGGER.info("=" * 80)
    LOGGER.info(f"检索到的节点数量: {len(result['retrieved_nodes'])}")
    for i, node in enumerate(result['retrieved_nodes'][:5], 1):
        LOGGER.info(f"  节点 {i}: ID={node.get('id')}, 名称={node.get('name')}, 标签={node.get('labels')}")
    LOGGER.info(f"检索到的边数量: {len(result['retrieved_edges'])}")
    for i, edge in enumerate(result['retrieved_edges'][:5], 1):
        LOGGER.info(f"  边 {i}: {edge.get('source')} --[{edge.get('type')}]--> {edge.get('target')}")
    LOGGER.info(f"检索到的超边数量: {len(result['retrieved_hyperedges'])}")
    for i, hyperedge in enumerate(result['retrieved_hyperedges'][:5], 1):
        LOGGER.info(f"  超边 {i}: ID={hyperedge.get('edge_id')}, 类型={hyperedge.get('edge_type')}")
        LOGGER.info(f"    包含节点: {hyperedge.get('contained_nodes', [])}")
        LOGGER.info(f"    逻辑描述: {hyperedge.get('logic_description', '')[:100]}")
    LOGGER.info(f"路径证据数量: {len(result['path_evidence'])}")
    for i, path in enumerate(result['path_evidence'][:3], 1):
        LOGGER.info(f"  路径 {i}: {path.get('path_string', '')[:150]}")
    LOGGER.info(f"检测到的逻辑谬误: {result['detected_fallacies']}")
    LOGGER.info("=" * 80)

    return result


def _apply_graph_payload_to_context(
    context_data: Dict[str, Any],
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    context_data["retrieved_nodes"] = payload.get("retrieved_nodes", [])
    context_data["retrieved_hyperedges"] = payload.get("retrieved_hyperedges", [])
    context_data["path_evidence"] = payload.get("path_evidence", [])
    context_data["graph_paths"] = payload.get("paths", [])
    context_data["retrieved_subgraph"] = payload.get("retrieved_subgraph", {})
    context_data["detected_fallacies"] = payload.get("detected_fallacies", [])
    if payload.get("source_excerpt"):
        context_data["source_excerpt"] = payload.get("source_excerpt", "")
    return context_data


class GraphSearchTool(BaseTool):
    """LangChain tool that executes Cypher and persists graph evidence."""

    name: str = "GraphSearchTool"
    description: str = (
        "Execute a Cypher query against Neo4j and return retrieved nodes, "
        "hyperedges, and full path evidence."
    )
    args_schema: type[BaseModel] = GraphSearchInput
    context_data: Dict[str, Any] = Field(default_factory=dict)

    def _run(
        self,
        cypher_query: str,
        parameters: Dict[str, Any] | None = None,
        limit: int = 5,
    ) -> Dict[str, Any]:
        payload = _execute_graph_search(
            self.context_data,
            cypher_query,
            parameters=parameters,
            limit=limit,
        )
        _apply_graph_payload_to_context(self.context_data, payload)
        return payload

    async def _arun(
        self,
        cypher_query: str,
        parameters: Dict[str, Any] | None = None,
        limit: int = 5,
    ) -> Dict[str, Any]:
        return self._run(cypher_query, parameters=parameters, limit=limit)


def _normalize_text(value: Any) -> str:
    """Convert mixed text into a normalized searchable string."""
    lowered = str(value or "").lower().replace("\ufeff", " ")
    lowered = re.sub(r"[^\w\u4e00-\u9fff]+", " ", lowered)
    return " ".join(lowered.split())


def slugify(value: str) -> str:
    text = _normalize_text(value).replace(" ", "_")
    return text[:80] or "node"


def _build_metric_node(metric_key: str, value: Any) -> Dict[str, Any] | None:
    """Build a Metric node payload from scalar or dict input."""
    metric_id = f"metric_{metric_key.lower()}"
    metric_name = metric_key.upper()

    if isinstance(value, dict):
        metric_value = value.get("value")
        unit = value.get("unit")
        description = value.get("description", "")
        tags = _ensure_list(value.get("tags"))
        metadata = value.get("metadata", {})
    else:
        metric_value = value
        unit = None
        description = ""
        tags = []
        metadata = {}

    if metric_value is None:
        return None

    return {
        "id": metric_id,
        "node_type": "Metric",
        "name": metric_name,
        "description": description,
        "metric_key": metric_name,
        "value": metric_value,
        "unit": unit,
        "tags": tags,
        "metadata": metadata,
    }


def _build_knowledge_card_node(index: int, raw_card: Dict[str, Any]) -> Dict[str, Any]:
    """Build a KnowledgeCard node payload from free-form card input."""
    card_id = raw_card.get("id") or raw_card.get("card_id") or f"knowledge_card_{index}"
    evidence_source = raw_card.get("evidence_source") or "manual_input"

    return {
        "id": card_id,
        "node_type": "KnowledgeCard",
        "name": raw_card.get("name") or raw_card.get("title") or card_id,
        "description": raw_card.get("description", ""),
        "tags": _ensure_list(raw_card.get("tags")),
        "metadata": raw_card.get("metadata", {}),
        "card_id": raw_card.get("card_id"),
        "type": raw_card.get("type", "行业基准"),
        "labels": _ensure_list(raw_card.get("labels")),
        "applicable_scenarios": _ensure_list(raw_card.get("applicable_scenarios")),
        "industry": raw_card.get("industry"),
        "evidence_source": evidence_source,
        "industry_benchmarks": raw_card.get("industry_benchmarks", {}),
    }


def _build_case_from_draft(input_data: Dict[str, Any]) -> KnowledgeGraphCase:
    """Coerce raw draft data into a KnowledgeGraphCase."""
    if {"case_id", "title", "nodes", "relations", "hyperedges"} <= set(input_data.keys()):
        return KnowledgeGraphCase.from_dict(input_data)

    project_metadata = dict(input_data.get("metadata", {}))
    passthrough_keys = (
        "customer_profiles",
        "target_customers",
        "customer_segments",
        "primary_customers",
        "served_segments",
        "customer_needs",
        "pain_points",
        "target_scenarios",
        "applicable_scenarios",
        "use_cases",
        "value_propositions",
        "core_values",
        "solution_benefits",
        "benefits",
        "channels",
        "primary_channels",
        "go_to_market_channels",
        "distribution_channels",
        "sales_channels",
        "marketing_channels",
        "competitor_comparisons",
        "competitor_analysis",
        "benchmark_comparisons",
        "competitor_analysis_required",
        "customer_node_ids",
        "problem_node_ids",
        "value_node_ids",
        "evidence_node_ids",
        "metric_node_ids",
        "narrative",
        "target_customer",
        "problem_statement",
        "value_proposition",
    )
    for key in passthrough_keys:
        if key in input_data and key not in project_metadata:
            project_metadata[key] = input_data[key]

    nodes: List[Dict[str, Any]] = [
        {
            "id": input_data.get("project_id", "project_draft"),
            "node_type": "Project",
            "name": input_data.get("project_name")
            or input_data.get("name")
            or input_data.get("title")
            or "项目草案",
            "description": input_data.get("description", ""),
            "stage": input_data.get("stage"),
            "team_name": input_data.get("team_name"),
            "tags": _ensure_list(input_data.get("tags")),
            "metadata": project_metadata,
        }
    ]

    metrics = dict(input_data.get("metrics", {}))
    for metric_key in METRIC_KEYS:
        if metric_key not in metrics and metric_key.lower() in input_data:
            metrics[metric_key] = input_data[metric_key.lower()]
        if metric_key not in metrics and metric_key in input_data:
            metrics[metric_key] = input_data[metric_key]

    for metric_key in METRIC_KEYS:
        metric_node = _build_metric_node(metric_key, metrics.get(metric_key))
        if metric_node is not None:
            nodes.append(metric_node)

    for index, raw_card in enumerate(_ensure_list(input_data.get("knowledge_cards")), start=1):
        if isinstance(raw_card, dict):
            nodes.append(_build_knowledge_card_node(index, raw_card))

    for index, raw_node in enumerate(_ensure_list(input_data.get("nodes")), start=1):
        if not isinstance(raw_node, dict):
            continue
        if raw_node.get("node_type") == "KnowledgeCard":
            nodes.append(_build_knowledge_card_node(index, raw_node))
            continue
        nodes.append(raw_node)

    relations = [item for item in _ensure_list(input_data.get("relations")) if isinstance(item, dict)]
    hyperedges = [item for item in _ensure_list(input_data.get("hyperedges")) if isinstance(item, dict)]
    node_lookup = {
        str(node.get("id", "")).strip(): str(node.get("name", "")).strip()
        for node in nodes
        if isinstance(node, dict) and str(node.get("id", "")).strip()
    }
    project_metadata["standard_hyperedge_records"] = normalize_hyperedge_records(
        hyperedges,
        node_lookup=node_lookup,
    )

    raw_case = {
        "case_id": input_data.get("case_id", "DRAFT"),
        "title": input_data.get("title") or input_data.get("name") or "项目草案诊断",
        "description": input_data.get("description", ""),
        "nodes": nodes,
        "relations": relations,
        "hyperedges": hyperedges,
    }
    return KnowledgeGraphCase.from_dict(raw_case)


def _run_selected_rules(
    case: KnowledgeGraphCase,
    rule_ids: Sequence[str] = TARGET_RULE_IDS,
) -> List[RuleResult]:
    """Run the selected graph rules against a case."""
    checker = GraphRuleChecker()
    results: List[RuleResult] = []

    for rule_id in rule_ids:
        definition = checker.rule_catalog[rule_id]
        handler = getattr(checker, f"_check_{rule_id.lower()}")
        results.append(handler(case, definition))

    return results


def _build_guided_message(result: RuleResult) -> str:
    """Convert a rule result into a coaching-style diagnostic message."""
    if result.rule_id == "H4" and result.status == RuleStatus.SKIPPED:
        missing_metrics = result.details.get("missing_metrics", [])
        if missing_metrics:
            return (
                "你还没有补齐市场规模数据（TAM/SAM/SOM）。"
                "先从目标城市、目标人群和可触达场景开始估算。"
            )

    if result.rule_id == "H6" and result.details.get("competitor_analysis_required"):
        return (
            "不要默认自己没有对手。请先回答："
            "如果用户不用你的方案，他们现在用什么替代办法解决问题？"
        )

    if result.rule_id == "H8" and result.status == RuleStatus.WARNING:
        if result.details.get("CAC") == 0:
            return "CAC 不能默认为 0。请把渠道投放、人力、时间成本一起算进去。"

    prefixes = {
        "H1": "用户与价值主张之间还没有完全对齐。",
        "H4": "市场规模逻辑还需要再推敲一轮。",
        "H6": "竞品分析还不够扎实。",
        "H8": "财务假设里还有明显风险。",
    }
    prefix = prefixes.get(result.rule_id)
    if prefix:
        return f"{prefix} {result.message}"
    return result.message


def _format_diagnostic_item(result: RuleResult) -> Dict[str, Any]:
    """Shape a RuleResult into the UI-friendly diagnostic structure."""
    guided_message = _build_guided_message(result)
    return {
        "rule_id": result.rule_id,
        "rule_name": result.name,
        "status": result.status.value,
        "severity": result.severity.value,
        "trigger_message": guided_message,
        "assistant_message": guided_message,
        "raw_message": result.message,
        "is_triggered": result.is_triggered,
        "evidence_nodes": list(result.evidence_nodes),
        "details": dict(result.details),
        "fix_suggestion": result.fix_suggestion,
    }


def run_diagnostics(input_data: Dict[str, Any]) -> List[Dict[str, str]]:
    """Run the lightweight draft diagnostics used by the Streamlit pages."""
    case = _build_case_from_draft(input_data)
    results = _run_selected_rules(case)
    return [
        _format_diagnostic_item(result)
        for result in results
        if result.status != RuleStatus.PASSED
    ]


def _build_history_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Keep only the recent valid chat turns for LLM context."""
    history: List[Dict[str, str]] = []
    for message in messages[-8:]:
        role = str(message.get("role", "")).strip()
        content = str(message.get("content", "")).strip()
        if role not in {"system", "user", "assistant"} or not content:
            continue
        history.append({"role": role, "content": content})
    return history


def _build_diag_context(diag_results: Any) -> str:
    """Render diagnostic results into a compact system-context string."""
    if not diag_results:
        return "当前还没有诊断结果，优先澄清用户、场景、价值主张和商业闭环。"

    if isinstance(diag_results, dict):
        diag_items = [diag_results]
    elif isinstance(diag_results, list):
        diag_items = [item for item in diag_results if isinstance(item, dict)]
    else:
        diag_items = []

    if not diag_items:
        return "诊断结构暂不可解析，请继续做澄清式追问。"

    lines = ["当前诊断摘要："]
    for item in diag_items[:5]:
        message = str(
            item.get("assistant_message")
            or item.get("trigger_message")
            or item.get("raw_message")
            or ""
        ).strip()
        if not message:
            continue
        lines.append(f"- {message}")
    return "\n".join(lines)

def infer_response_severity(diag_results: Any) -> str:
    """Return the highest response severity implied by triggered rule results."""
    if isinstance(diag_results, dict):
        items = [diag_results]
    elif isinstance(diag_results, list):
        items = [item for item in diag_results if isinstance(item, dict)]
    else:
        items = []

    highest = ""
    for item in items:
        severity = str(item.get("severity", "")).strip().lower()
        if severity == "high":
            return "High"
        if severity == "warning":
            highest = "Warning"
    return highest


def _extract_triggered_rule_ids(diag_results: Any) -> List[str]:
    if isinstance(diag_results, dict):
        items = [diag_results]
    elif isinstance(diag_results, list):
        items = [item for item in diag_results if isinstance(item, dict)]
    else:
        items = []

    rule_ids: List[str] = []
    for item in items:
        rule_id = str(item.get("rule_id", "")).strip().upper()
        status = str(item.get("status", "")).strip().lower()
        if not rule_id:
            continue
        if status in {"passed", "pass", "success"}:
            continue
        rule_ids.append(rule_id)
    return list(dict.fromkeys(rule_ids))


def _path_evidence_entries(context_data: Any) -> List[Dict[str, Any]]:
    if not isinstance(context_data, dict):
        return []

    entries: List[Dict[str, Any]] = []
    for item in _ensure_list(context_data.get("path_evidence")):
        if isinstance(item, dict):
            path_string = str(item.get("path_string", "") or item.get("path", "")).strip()
            if not path_string:
                continue
            entries.append(
                {
                    "path_string": path_string,
                    "source_excerpt": str(item.get("source_excerpt", "") or "").strip(),
                    "page_ref": str(item.get("page_ref", "") or "").strip(),
                }
            )
        elif isinstance(item, str):
            path_string = item.strip()
            if path_string:
                entries.append(
                    {
                        "path_string": path_string,
                        "source_excerpt": "",
                        "page_ref": "",
                    }
                )
    return entries


def _select_relevant_path_evidence(user_input: str, context_data: Any) -> Dict[str, Any]:
    keywords = {_normalize_text(user_input)}
    keywords.update(_normalize_text(item) for item in extract_keywords(user_input))
    keywords = {item for item in keywords if item}

    best_entry: Dict[str, Any] = {}
    best_score = -1
    for entry in _path_evidence_entries(context_data):
        score = 0
        haystacks = [
            _normalize_text(entry.get("path_string", "")),
            _normalize_text(entry.get("source_excerpt", "")),
        ]
        for keyword in keywords:
            for haystack in haystacks:
                if keyword and keyword in haystack:
                    score += 1
        if str(entry.get("source_excerpt", "")).strip():
            score += 2
        if score > best_score:
            best_entry = entry
            best_score = score
    return best_entry


def select_locked_source_excerpt(user_input: str, context_data: Any) -> str:
    """Return the most relevant source_excerpt without rewriting it."""
    entry = _select_relevant_path_evidence(user_input, context_data)
    return str(entry.get("source_excerpt", "") or "").strip()


def seed_path_evidence_from_project_text(user_input: str, context_data: Any) -> None:
    """Seed path_evidence with an exact project excerpt when graph evidence is absent."""
    if not isinstance(context_data, dict):
        return
    if _path_evidence_entries(context_data):
        return

    project_text = str(context_data.get("project_full_text", "") or "").strip()
    if not project_text:
        return

    sentences = [
        segment.strip()
        for segment in re.split(r"(?<=[。！？.!?])|\n+", project_text)
        if segment and segment.strip()
    ]
    if not sentences:
        sentences = [project_text[:220].strip()]

    keywords = {_normalize_text(user_input)}
    keywords.update(_normalize_text(item) for item in extract_keywords(user_input))
    keywords = {item for item in keywords if item}

    best_excerpt = ""
    best_score = -1
    for sentence in sentences[:20]:
        normalized_sentence = _normalize_text(sentence)
        score = sum(1 for keyword in keywords if keyword and keyword in normalized_sentence)
        if score > best_score:
            best_excerpt = sentence[:220].strip()
            best_score = score

    if not best_excerpt:
        best_excerpt = sentences[0][:220].strip()
    if not best_excerpt:
        return

    context_data["path_evidence"] = [
        {
            "path_string": "ProjectExcerpt",
            "source_excerpt": best_excerpt,
            "page_ref": "",
        }
    ]
    context_data["source_excerpt"] = best_excerpt


def _coerce_context_text(context_data: Any) -> str:
    """Convert mixed context payloads into text without changing signatures."""
    if context_data is None:
        return ""
    if isinstance(context_data, str):
        return context_data.strip()
    if isinstance(context_data, (list, dict)):
        return _build_diag_context(context_data)
    return str(context_data).strip()


@lru_cache(maxsize=1)
def _load_searchable_cards() -> List[Dict[str, Any]]:
    """Load local knowledge cards into a search-friendly in-memory index."""
    cards: List[Dict[str, Any]] = []
    if not KNOWLEDGE_CARD_DIR.exists():
        return cards

    for path in KNOWLEDGE_CARD_DIR.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".json", ".md", ".markdown", ".txt"}:
            continue

        title = path.stem
        content = ""
        searchable_parts: List[str] = [path.stem, str(path)]

        try:
            raw_text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                raw_text = path.read_text(encoding="utf-8-sig")
            except UnicodeDecodeError:
                continue
        except OSError:
            continue

        if path.suffix.lower() == ".json":
            try:
                payload = json.loads(raw_text)
            except json.JSONDecodeError:
                payload = {}
            title = str(payload.get("name") or payload.get("title") or path.stem)
            content = str(
                payload.get("description")
                or payload.get("evidence_source")
                or payload.get("metadata", {}).get("source_excerpt", "")
                or ""
            ).strip()
            searchable_parts.extend(
                [
                    title,
                    content,
                    str(payload.get("industry", "")),
                    " ".join(str(item) for item in _ensure_list(payload.get("tags"))),
                    " ".join(str(item) for item in _ensure_list(payload.get("labels"))),
                    " ".join(str(item) for item in _ensure_list(payload.get("applicable_scenarios"))),
                ]
            )
        else:
            content = raw_text.strip()
            searchable_parts.extend([title, content[:1200]])

        cards.append(
            {
                "title": title,
                "content": content,
                "summary": content[:180] if content else "",
                "path": str(path),
                "searchable_text": _normalize_text(" ".join(searchable_parts)),
            }
        )

    return cards


def extract_keywords(text: str) -> List[str]:
    """Extract lightweight business keywords from free-form student input."""
    if not text or not text.strip():
        return []

    normalized_text = _normalize_text(text)
    keywords: List[str] = []

    for concept in BUSINESS_CONCEPTS:
        if _normalize_text(concept) in normalized_text:
            keywords.append(concept)

    token_counts: Dict[str, int] = {}
    for chunk in re.split(r"[\s,，。！？；:：（）)\[\]{}<>/\\|]+", text):
        token = chunk.strip().lower()
        if not token or token in STOPWORDS or token.isdigit():
            continue
        if len(token) < 2 or len(token) > 18:
            continue
        token_counts[token] = token_counts.get(token, 0) + 1

    for token, _ in sorted(token_counts.items(), key=lambda item: (-item[1], -len(item[0]), item[0])):
        keywords.append(token)

    deduped: List[str] = []
    seen = set()
    for keyword in keywords:
        normalized_keyword = _normalize_text(keyword)
        if not normalized_keyword or normalized_keyword in seen:
            continue
        seen.add(normalized_keyword)
        deduped.append(keyword)
    return deduped[:12]


@lru_cache(maxsize=128)
def _retrieve_knowledge_cards_cached(keywords_tuple: tuple[str, ...], limit: int) -> tuple[dict, ...]:
    """Cached wrapper for retrieve_knowledge_cards using hashable arguments."""
    keywords = list(keywords_tuple)
    if not keywords:
        return tuple()

    ranked_cards: List[Dict[str, Any]] = []
    for card in _load_searchable_cards():
        score = 0
        matched_keywords: List[str] = []
        title_text = _normalize_text(card["title"])
        searchable_text = card["searchable_text"]

        for keyword in keywords:
            normalized_keyword = _normalize_text(keyword)
            if not normalized_keyword:
                continue
            if normalized_keyword in title_text:
                score += 6
                matched_keywords.append(keyword)
                continue
            if normalized_keyword in searchable_text:
                score += min(4, searchable_text.count(normalized_keyword))
                matched_keywords.append(keyword)

        if score == 0:
            continue

        ranked_cards.append(
            {
                **card,
                "matched_keywords": matched_keywords[:5],
                "score": score,
            }
        )

    ranked_cards.sort(key=lambda item: (-item["score"], item["title"]))
    return tuple(ranked_cards[:limit])


def retrieve_knowledge_cards(keywords: List[str], limit: int = 3) -> List[Dict[str, Any]]:
    """Outer wrapper: converts non-hashable list to hashable tuple and triggers cache."""
    if not keywords:
        return []

    keywords_tuple = tuple(keywords)
    return list(_retrieve_knowledge_cards_cached(keywords_tuple, limit))


def _build_knowledge_context(keywords: List[str]) -> str:
    """Build a short knowledge reference block for the LLM."""
    cards = retrieve_knowledge_cards(keywords, limit=2)
    if not cards:
        return ""

    lines = ["以下知识卡片仅用于辅助追问，不得直接照抄:"]
    for card in cards:
        lines.append(f"- {card['title']}: {card.get('summary') or card.get('content', '')[:120]}")
    return "\n".join(lines)


def _build_graph_context(context_data: Any) -> str:
    """Render graph retrieval evidence into a compact LLM context block."""
    if not isinstance(context_data, dict):
        return ""

    path_evidence = _path_evidence_entries(context_data)
    retrieved_nodes = context_data.get("retrieved_nodes", [])
    retrieved_hyperedges = context_data.get("retrieved_hyperedges", [])
    if not path_evidence and not retrieved_nodes and not retrieved_hyperedges:
        return ""

    lines = ["以下是图检索返回的证据路径，请严格基于这些路径追问或回应："]
    if retrieved_nodes:
        node_names = [
            str(node.get("name", ""))
            for node in retrieved_nodes[:8]
            if str(node.get("name", "")).strip()
        ]
        if node_names:
            lines.append("- Retrieved Nodes: " + ", ".join(node_names))
    if retrieved_hyperedges:
        edge_names = []
        for edge in retrieved_hyperedges[:5]:
            if not isinstance(edge, dict):
                continue
            edge_id = str(edge.get("hyperedge_id") or edge.get("id") or "").strip()
            edge_type = str(edge.get("edge_type") or "").strip()
            rule = str(edge.get("rule_triggered") or "").strip()
            label = " / ".join(part for part in [edge_type, edge_id, rule] if part)
            if label:
                edge_names.append(label)
        if edge_names:
            lines.append("- Hyperedges: " + ", ".join(edge_names))
    for item in path_evidence[:5]:
        path_string = str(item.get("path_string", "")).strip()
        source_excerpt = str(item.get("source_excerpt", "")).strip()
        page_ref = str(item.get("page_ref", "")).strip()
        if path_string:
            lines.append(f"- Path: {path_string}")
        if source_excerpt:
            page_suffix = f" | page_ref={page_ref}" if page_ref else ""
            lines.append(f"- Locked Source Excerpt{page_suffix}: {source_excerpt}")
    return "\n".join(lines)


def _ensure_graph_evidence(
    user_input: str,
    keywords: List[str],
    context_data: Any,
) -> Dict[str, Any]:
    """Run GraphSearchTool before reply generation and persist path evidence."""
    if not isinstance(context_data, dict):
        return _empty_graph_payload()

    query = str(context_data.get("graph_cypher_query") or "").strip()
    parameters = context_data.get("graph_cypher_parameters", {})
    limit = int(context_data.get("graph_limit", 5) or 5)

    if not query:
        query, parameters = _build_default_graph_query(
            keywords or extract_keywords(user_input),
            limit=limit,
        )

    tool = GraphSearchTool(context_data=context_data)
    result = tool.invoke(
        {
            "cypher_query": query,
            "parameters": parameters,
            "limit": limit,
        }
    )

    # 检查是否有触发的H规则，如果有，提取对应的超边
    triggered_rules = _extract_triggered_rule_ids(context_data.get("diagnostic_results", []))
    if triggered_rules:
        # 对于每个触发的H规则，执行额外的查询来获取对应的超边
        driver, database, owns_driver = _resolve_graph_driver(context_data)
        try:
            with driver.session(database=database) as session:
                for rule_id in triggered_rules:
                    # 构建查询，查找与规则相关的超边
                    rule_query = """
                    MATCH (rule:Rule {rule_id: $rule_id})<-[:EVALUATED_BY]-(hyperedge)
                    MATCH (hyperedge)-[r]->(participant)
                    RETURN 
                        {
                            hyperedge_id: coalesce(hyperedge.id, ''),
                            edge_type: coalesce(hyperedge.edge_type, head(labels(hyperedge)), ''),
                            rule_triggered: $rule_id,
                            participants: [(hyperedge)-[part_rel]->(p) 
                                WHERE type(part_rel) IN ['HAS_PARTICIPANT', 'INVOLVES']
                                | {
                                    role: coalesce(part_rel.role, toLower(type(part_rel))),
                                    node_name: coalesce(p.name, p.title, p.id, ''),
                                    node_id: coalesce(p.id, '')
                                }
                            ],
                            logical_evaluation: coalesce(hyperedge.logical_evaluation, hyperedge.evaluation, hyperedge.logic_result, ''),
                            impact: coalesce(hyperedge.impact, hyperedge.description, ''),
                            description: coalesce(hyperedge.description, ''),
                            source_excerpt: coalesce(hyperedge.source_excerpt, hyperedge.description, ''),
                            page_ref: coalesce(hyperedge.page_ref, hyperedge.page, hyperedge.page_number, '')
                        } AS hyperedge_record,
                        [p IN nodes((hyperedge)-[*1]->()) WHERE NOT p = hyperedge | {
                            id: coalesce(p.id, p.rule_id, p.case_id, ''),
                            name: coalesce(p.name, p.title, p.rule_id, p.case_id, ''),
                            labels: labels(p),
                            source_excerpt: coalesce(p.source_excerpt, p.evidence_source, p.description, ''),
                            page_ref: coalesce(p.page_ref, p.page, p.page_number, '')
                        }] AS participant_nodes
                    """
                    rule_result = session.run(rule_query, {"rule_id": rule_id})
                    for record in rule_result:
                        hyperedge_record = record.get("hyperedge_record")
                        participant_nodes = record.get("participant_nodes", [])
                        
                        if hyperedge_record:
                            # 规范化超边记录
                            node_lookup = {
                                str(node.get("id", "")).strip(): str(node.get("name", "")).strip()
                                for node in participant_nodes
                                if str(node.get("id", "")).strip()
                            }
                            normalized_hyperedge = normalize_hyperedge_record(hyperedge_record, node_lookup=node_lookup)
                            
                            # 添加到结果中
                            if normalized_hyperedge not in result.get("retrieved_hyperedges", []):
                                result["retrieved_hyperedges"].append(normalized_hyperedge)
                            
                            # 添加参与者节点
                            for node in participant_nodes:
                                if node not in result.get("retrieved_nodes", []):
                                    result["retrieved_nodes"].append(node)
        except Exception as exc:
            LOGGER.exception(
                "Neo4j rule-based graph search failed | database=%s | rules=%s",
                database,
                triggered_rules,
            )
        finally:
            _close_graph_driver(driver, owns_driver)

        # 重新构建retrieved_subgraph，包含新添加的超边和节点
        if result.get("retrieved_nodes") and result.get("retrieved_edges") and result.get("retrieved_hyperedges"):
            retrieved_nodes = result.get("retrieved_nodes", [])
            retrieved_edges = result.get("retrieved_edges", [])
            retrieved_hyperedges = result.get("retrieved_hyperedges", [])
            
            # 去重
            deduped_nodes = {
                (node["id"], node["name"], tuple(node.get("labels", []))): node
                for node in retrieved_nodes
            }
            deduped_edges = {
                (edge["source"], edge["target"], edge["type"]): edge
                for edge in retrieved_edges
            }
            deduped_hyperedges = {
                (
                    edge.get("hyperedge_id", ""),
                    edge.get("edge_type", ""),
                    edge.get("rule_triggered", ""),
                    edge.get("logical_evaluation", ""),
                ): edge
                for edge in retrieved_hyperedges
            }
            
            # 重新构建retrieved_subgraph
            result["retrieved_subgraph"] = {
                "nodes": [
                    {
                        "id": node.get("id", ""),
                        "label": node.get("label", "Node"),
                        "name": node.get("name", ""),
                        "properties": {k: v for k, v in node.items() if k not in ["id", "label", "name"]}
                    }
                    for node in deduped_nodes.values()
                ],
                "edges": [
                    {
                        "source": edge["source"],
                        "target": edge["target"],
                        "type": edge["type"],
                        "logic": edge.get("logic", "")
                    }
                    for edge in deduped_edges.values()
                ],
                "hyperedges": [
                    {
                        "edge_id": edge.get("hyperedge_id", ""),
                        "type": edge.get("edge_type", ""),
                        "contained_nodes": list(edge.get("nodes_involved", {}).values()),
                        "logic_description": edge.get("impact", "") or edge.get("logical_evaluation", "")
                    }
                    for edge in deduped_hyperedges.values()
                ]
            }

    # 注入逻辑谬误追踪，当匹配到Risk_Pattern_Edge时返回具体谬误标签
    detected_fallacies = []
    for hyperedge in result.get("retrieved_hyperedges", []):
        if hyperedge.get("edge_type") == "Risk_Pattern_Edge":
            # 提取谬误标签
            rule_triggered = hyperedge.get("rule_triggered", "")
            impact = hyperedge.get("impact", "")
            logical_evaluation = hyperedge.get("logical_evaluation", "")
            
            # 根据规则ID和内容提取具体谬误标签
            fallacy_label = ""
            if "H1" in rule_triggered:
                fallacy_label = "商业模式一致性问题"
            elif "H4" in rule_triggered:
                fallacy_label = "市场规模估算错误"
            elif "H6" in rule_triggered:
                fallacy_label = "竞品分析不足"
            elif "H8" in rule_triggered:
                fallacy_label = "财务假设风险"
            elif "大数" in impact or "大数" in logical_evaluation:
                fallacy_label = "大数幻觉"
            elif "渠道" in impact or "渠道" in logical_evaluation:
                fallacy_label = "渠道错位"
            elif "用户" in impact or "用户" in logical_evaluation:
                fallacy_label = "用户需求误判"
            else:
                fallacy_label = "业务逻辑风险"
            
            if fallacy_label and fallacy_label not in detected_fallacies:
                detected_fallacies.append(fallacy_label)
    
    # 将检测到的谬误添加到结果中
    result["detected_fallacies"] = detected_fallacies
    
    # 如果retrieved_subgraph存在，也添加detected_fallacies字段
    if "retrieved_subgraph" in result:
        result["retrieved_subgraph"]["detected_fallacies"] = detected_fallacies
    
    return result


def _build_evidence_tag(context_data: Any) -> str:
    """Serialize graph evidence into the frontend trace tag."""
    if isinstance(context_data, dict):
        payload = {
            "path_evidence": _path_evidence_entries(context_data),
            "retrieved_nodes": context_data.get("retrieved_nodes", []),
            "retrieved_hyperedges": context_data.get("retrieved_hyperedges", []),
            "source_excerpt": str(context_data.get("source_excerpt", "") or "").strip(),
            "response_severity": infer_response_severity(context_data.get("diagnostic_results", [])),
        }
    else:
        payload = {
            "path_evidence": [],
            "retrieved_nodes": [],
            "retrieved_hyperedges": [],
            "source_excerpt": "",
            "response_severity": "",
        }
    return f"<evidence>{json.dumps(payload, ensure_ascii=False)}</evidence>"


def _append_evidence_tag(response_text: str, context_data: Any) -> str:
    """Append the evidence tag required by the frontend renderer."""
    stripped = str(response_text or "").strip()
    tag = _build_evidence_tag(context_data)
    if "<evidence>" in stripped:
        return stripped
    if stripped:
        return f"{stripped}\n\n{tag}"
    return tag


def _looks_like_garbled_text(text: str) -> bool:
    stripped = str(text or "").strip()
    if not stripped:
        return False
    if re.fullmatch(r"[?？]{4,}", stripped):
        return True
    if SUSPICIOUS_MOJIBAKE_RE.search(stripped):
        return True

    suspicious_chars = 0
    for char in stripped:
        category = unicodedata.category(char)
        if category in {"Cc", "Cs"} or char == "\ufffd":
            suspicious_chars += 1
    return bool(stripped) and suspicious_chars / max(len(stripped), 1) >= 0.1


def _is_emoji_only_text(text: str) -> bool:
    stripped = str(text or "").strip()
    if not stripped:
        return False
    emojis = EMOJI_ONLY_RE.findall(stripped)
    if not emojis:
        return False

    reduced = EMOJI_ONLY_RE.sub("", stripped)
    reduced = reduced.replace("\ufe0f", "").replace("\u200d", "")
    reduced = re.sub(r"[\s\.,!?;:'\"“”‘’，。！？、~`@#\$%\^&\*\(\)\[\]\{\}<>\-_=+/\\|:：…·♥❤]+", "", reduced)
    return not reduced


def _is_long_prompt_injection(text: str) -> bool:
    stripped = str(text or "").strip()
    if len(stripped) <= MAX_GUARDED_INPUT_LENGTH:
        return False
    return bool(PROMPT_INJECTION_RE.search(stripped))


def _build_guardrail_response(reason: str) -> tuple[str, str]:
    if reason == "garbled_text":
        return (
            "我先不猜测这段乱码想表达什么。请你换成一条清晰文字，再回答三个点：你的场景是什么、最想验证什么、手上已有哪一条真实事实？",
            "把问题改写成一句清晰陈述，并补一条真实事实。",
        )
    if reason == "emoji_only":
        return (
            "我先不根据表情替你补全问题。你现在最想讨论的是用户、价值、渠道还是财务？只选一个方向，并用一句话说明原因。",
            "用一句文字明确你最想讨论的一个方向。",
        )
    return (
        "我不会执行提示词注入或越过系统边界。先回到真实业务问题。请在 200 字内说明：你的对象是谁、当前结论是什么、哪条事实最能支持它？",
        "删掉注入指令，只保留真实业务问题和一条可核验事实。",
    )


def inspect_user_input_guardrails(user_input: str) -> Dict[str, Any]:
    stripped = str(user_input or "").strip()
    if not stripped:
        return {"blocked": False, "reason": "", "response": "", "next_step": ""}

    if _looks_like_garbled_text(stripped):
        response, next_step = _build_guardrail_response("garbled_text")
        return {
            "blocked": True,
            "reason": "garbled_text",
            "response": response,
            "next_step": next_step,
        }

    if _is_emoji_only_text(stripped):
        response, next_step = _build_guardrail_response("emoji_only")
        return {
            "blocked": True,
            "reason": "emoji_only",
            "response": response,
            "next_step": next_step,
        }

    if _is_long_prompt_injection(stripped):
        response, next_step = _build_guardrail_response("prompt_injection")
        return {
            "blocked": True,
            "reason": "prompt_injection",
            "response": response,
            "next_step": next_step,
        }

    return {"blocked": False, "reason": "", "response": "", "next_step": ""}


# ---------------------------------------------------------------------------
# ReAct 工具封装（Tool Binding）
# 保留 Neo4j 原生查询逻辑与强类型返回，仅包装为标准 @tool，
# 由大模型自主决定是否调用及传参。
# ---------------------------------------------------------------------------

from typing import Annotated  # noqa: E402

from langchain_core.tools import tool as _tool_decorator  # noqa: E402
from langgraph.prebuilt import InjectedState  # noqa: E402

MAX_TOOL_KEYWORDS = 8


def _neo4j_env_credentials() -> Dict[str, str]:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass
    return {
        "uri": str(os.getenv("NEO4J_URI") or "").strip(),
        "user": str(os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER") or "").strip(),
        "password": str(os.getenv("NEO4J_PASSWORD") or ""),
        "database": str(os.getenv("NEO4J_DATABASE") or "neo4j").strip(),
    }


def _clean_keywords(keywords: Sequence[str]) -> List[str]:
    cleaned: List[str] = []
    for keyword in keywords or []:
        text = str(keyword or "").strip()
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned[:MAX_TOOL_KEYWORDS]


@_tool_decorator
def search_hypergraph(keywords: List[str]) -> str:
    """当遇到商业模式设计、痛点分析、风险评估等需要客观图谱证据的问题时，必须调用此工具。
    参数 keywords: 提取用户话语中的核心业务词汇数组，如 ['ToB', '渠道错位']。
    """
    cleaned = _clean_keywords(keywords)
    if not cleaned:
        return json.dumps({"status": "empty", "risk_patterns": []}, ensure_ascii=False)

    try:
        from kg_hypergraph import get_risk_pattern

        credentials = _neo4j_env_credentials()
        if not (credentials["uri"] and credentials["user"] and credentials["password"]):
            return json.dumps(
                {"status": "degraded", "reason": "neo4j_credentials_missing", "risk_patterns": []},
                ensure_ascii=False,
            )

        driver = get_managed_driver(
            credentials["uri"],
            credentials["user"],
            credentials["password"],
            credentials["database"],
        )
        try:
            records = get_risk_pattern(driver, credentials["database"], cleaned, 5)
        finally:
            try:
                from neo4j_resilience import close_managed_driver

                close_managed_driver(
                    credentials["uri"], credentials["user"], credentials["password"]
                )
            except Exception:
                pass

        normalized = normalize_hyperedge_records(
            [
                {
                    "hyperedge_id": item.get("edge_id"),
                    "edge_type": "Risk_Pattern_Edge",
                    "rule_triggered": "H1_BusinessModelConsistency",
                    "participants": item.get("participants", []),
                    "logical_evaluation": item.get("logical_evaluation", ""),
                    "impact": item.get("edge_description") or item.get("impact") or "",
                    "case_id": item.get("case_id"),
                    "case_title": item.get("case_title"),
                    "fix_strategies": item.get("fix_strategies", []),
                }
                for item in records
                if isinstance(item, dict)
            ]
        )
        return json.dumps(
            {"status": "ok", "risk_patterns": normalized},
            ensure_ascii=False,
        )
    except Exception as exc:
        LOGGER.warning("search_hypergraph degraded: %s", exc)
        return json.dumps(
            {"status": "degraded", "reason": str(exc), "risk_patterns": []},
            ensure_ascii=False,
        )


@_tool_decorator
def search_knowledge_cards(keywords: List[str]) -> str:
    """当需要真实商业案例、知识卡片或类比素材辅助回答时调用此工具。
    参数 keywords: 用户话语中的核心主题词数组，如 ['渠道', '获客成本']。
    """
    cleaned = _clean_keywords(keywords)
    if not cleaned:
        return json.dumps({"status": "empty", "cards": []}, ensure_ascii=False)
    try:
        cards = retrieve_knowledge_cards(cleaned, limit=3)
        return json.dumps({"status": "ok", "cards": cards}, ensure_ascii=False, default=str)
    except Exception as exc:
        LOGGER.warning("search_knowledge_cards degraded: %s", exc)
        return json.dumps(
            {"status": "degraded", "reason": str(exc), "cards": []}, ensure_ascii=False
        )


def _scoring_input_from_state(state: Dict[str, Any]) -> Dict[str, Any]:
    context_data = state.get("context_data") if isinstance(state, dict) else None
    if not isinstance(context_data, dict):
        return {}

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


@_tool_decorator
def get_competition_score(state: Annotated[Dict[str, Any], InjectedState]) -> str:
    """当用户要求竞赛评分、打分预览或评估项目竞赛竞争力时调用此工具。无需用户提供参数。"""
    try:
        from scoring_engine import calculate_scores

        scoring_input = _scoring_input_from_state(state)
        if not scoring_input:
            return json.dumps(
                {"status": "missing_input", "message": "缺少项目摘要，无法评分，请让用户提供项目材料。"},
                ensure_ascii=False,
            )
        report = calculate_scores(scoring_input)
        return json.dumps({"status": "ok", "score_report": report}, ensure_ascii=False, default=str)
    except Exception as exc:
        LOGGER.warning("get_competition_score degraded: %s", exc)
        return json.dumps({"status": "degraded", "reason": str(exc)}, ensure_ascii=False)


@_tool_decorator
def get_ability_report(state: Annotated[Dict[str, Any], InjectedState]) -> str:
    """当需要生成学生能力画像、结构化评估报告或教学干预依据时调用此工具。无需用户提供参数。"""
    try:
        from scoring_engine import generate_ability_report

        context_data = state.get("context_data") if isinstance(state, dict) else {}
        if not isinstance(context_data, dict):
            context_data = {}
        diagnostics = context_data.get("diagnostic_results", [])
        diagnostics = diagnostics if isinstance(diagnostics, list) else []
        messages = state.get("messages", []) if isinstance(state, dict) else []

        report = generate_ability_report(messages, diagnostics, context_data)
        return json.dumps({"status": "ok", "ability_report": report}, ensure_ascii=False, default=str)
    except Exception as exc:
        LOGGER.warning("get_ability_report degraded: %s", exc)
        return json.dumps({"status": "degraded", "reason": str(exc)}, ensure_ascii=False)


def apply_input_guardrails(user_input: str, context_data: Any = None) -> str | None:
    decision = inspect_user_input_guardrails(user_input)
    if not decision.get("blocked"):
        return None

    if isinstance(context_data, dict):
        context_data["guardrail_blocked"] = True
        context_data["guardrail_reason"] = decision["reason"]
        context_data["guardrail_response"] = decision["response"]
        context_data["guardrail_next_step"] = decision["next_step"]
        context_data["response_severity"] = "warning"

    LOGGER.warning(
        "Input guardrail triggered | reason=%s | input_length=%s",
        decision["reason"],
        len(str(user_input or "")),
    )
    return str(decision["response"]).strip()


def extract_evidence_payload(response_text: str) -> Dict[str, Any]:
    """Parse the <evidence> JSON tag from a response string."""
    text = str(response_text or "")
    match = re.search(r"<evidence>(.*?)</evidence>", text, flags=re.DOTALL)
    if not match:
        return {}
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def strip_evidence_tag(response_text: str) -> str:
    """Remove the trailing <evidence> tag before rendering in chat."""
    text = str(response_text or "")
    cleaned = re.sub(r"\s*<evidence>.*?</evidence>\s*", "", text, flags=re.DOTALL)
    return cleaned.strip()


def _fallback_socratic_response(prompt: str, diag_results: Any) -> str:
    """Return a deterministic coaching response when no LLM is available."""
    extracted = extract_keywords(prompt)
    primary_focus = extracted[0] if extracted else "核心假设"
    context_text = _coerce_context_text(diag_results)
    citation_prefix = ""
    if isinstance(diag_results, dict):
        citation_prefix = build_rule_citation_prefix(prompt, diag_results)

    if context_text:
        return (
            (f"{citation_prefix}\n" if citation_prefix else "")
            +
            f"先不要急着下结论。当前最值得澄清的是：{context_text}\n\n"
            f"请围绕“{primary_focus}”先回答三个问题：\n"
            "1. 这个判断基于什么真实用户证据，而不是主观想象？\n"
            "2. 如果把目标用户缩小到一个最具体的人群，谁最痛、最频繁、最愿意付费？\n"
            "3. 你准备如何用一周内可执行的方式验证这件事？"
        )

    return (
        (f"{citation_prefix}\n" if citation_prefix else "")
        +
        f"围绕“{primary_focus}”，先不要直接讲完整方案，请先回答：\n"
        "1. 你锁定的具体用户是谁？\n"
        "2. 这个用户现在用什么替代方案解决问题？\n"
        "3. 你的方案为什么会比现有替代方案更强？"
    )


def _safe_fallback_socratic_response(prompt: str, context_data: Any) -> str:
    """Return a stable fallback response even if context building fails."""
    try:
        return _fallback_socratic_response(prompt, context_data)
    except Exception:
        return (
            "先不要直接给出完整方案。请先用三句话说明：\n"
            "1. 你最具体的目标用户是谁？\n"
            "2. 他们现在用什么方式解决这个问题？\n"
            "3. 你手上有什么证据能证明你的假设成立？"
        )


def _build_rule_citation_instruction(user_input: str, context_data: Any) -> str:
    """Build a hard constraint that forces exact source citation when rules are triggered."""
    triggered_rule_ids = _extract_triggered_rule_ids(
        context_data.get("diagnostic_results", []) if isinstance(context_data, dict) else []
    )
    if not triggered_rule_ids:
        return ""

    selected_entry = _select_relevant_path_evidence(user_input, context_data)
    locked_excerpt = str(selected_entry.get("source_excerpt", "") or "").strip()
    page_ref = str(selected_entry.get("page_ref", "") or "").strip()

    if locked_excerpt:
        page_hint = f"；如有 page_ref，可写成‘正如你在 BP {page_ref} 提到的……’" if page_ref else ""
        return (
            "若触发关键风险，请优先给出严格引用证据的句子。"
            "禁止输出内部规则编号、路由名称或系统状态。"
            "引用内容必须逐字使用下方原文，不可改写、扩写或伪造：\n"
            f"锁定原文：{locked_excerpt}\n"
            f"{page_hint}"
        )

    return (
        "若触发关键风险，请优先尝试引用证据；"
        "若当前没有可用 source_excerpt，请明确说明“当前无法直接引用原文证据”。"
        "禁止伪造原文，且禁止输出内部规则编号。"
    )

def build_rule_citation_prefix(user_input: str, context_data: Any) -> str:
    """Return a deterministic citation prefix for node-based replies."""
    triggered_rule_ids = _extract_triggered_rule_ids(
        context_data.get("diagnostic_results", []) if isinstance(context_data, dict) else []
    )
    if not triggered_rule_ids:
        return ""

    selected_entry = _select_relevant_path_evidence(user_input, context_data)
    locked_excerpt = str(selected_entry.get("source_excerpt", "") or "").strip()
    page_ref = str(selected_entry.get("page_ref", "") or "").strip()

    if locked_excerpt:
        if page_ref:
            return f"正如你在 BP {page_ref} 提到的“{locked_excerpt}”，这句话暴露了当前论证里最需要补证的一环。"
        return f"正如你在 BP 中提到的“{locked_excerpt}”，这句话暴露了当前论证里最需要补证的一环。"

    return "当前存在关键论证风险，但缺少可直接锁定的原文摘录。"

def _get_socratic_response_impl(
    user_input: str,
    chat_history: List[Dict[str, Any]],
    context_data: Any,
) -> str:
    """Generate a Socratic coaching response while preserving the old signature."""
    guardrail_response = apply_input_guardrails(user_input, context_data)
    if guardrail_response:
        return _append_evidence_tag(guardrail_response, context_data)

    api_key = _get_config_value("GPUSTACK_API_KEY") or _get_config_value("SILICONFLOW_API_KEY")
    keywords = extract_keywords(user_input)
    _ensure_graph_evidence(user_input, keywords, context_data)
    seed_path_evidence_from_project_text(user_input, context_data)
    if isinstance(context_data, dict) and not str(context_data.get("source_excerpt", "")).strip():
        context_data["source_excerpt"] = select_locked_source_excerpt(user_input, context_data)
    context_text = _coerce_context_text(context_data)
    graph_context = _build_graph_context(context_data)
    rule_citation_instruction = _build_rule_citation_instruction(user_input, context_data)

    if not api_key:
        return _append_evidence_tag(
            _safe_fallback_socratic_response(user_input, context_data),
            context_data,
        )

    request_messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "system",
            "content": (
                "你必须严格遵守苏格拉底式引导。"
                "只做启发、追问、澄清与反证，禁止代写商业计划书，"
                "禁止直接补全市场、财务、用户或竞品结论。"
            ),
        },
    ]

    if rule_citation_instruction:
        request_messages.append({"role": "system", "content": rule_citation_instruction})

    if context_text:
        request_messages.append({"role": "system", "content": context_text})

    if graph_context:
        request_messages.append({"role": "system", "content": graph_context})

    knowledge_context = _build_knowledge_context(keywords)
    if knowledge_context:
        request_messages.append({"role": "system", "content": knowledge_context})

    request_messages.extend(_build_history_messages(chat_history))
    if not request_messages or request_messages[-1].get("role") != "user":
        request_messages.append({"role": "user", "content": user_input})

    base_url = (
        _get_config_value("GPUSTACK_BASE_URL")
        or _get_config_value("SILICONFLOW_BASE_URL", DEFAULT_SILICONFLOW_BASE_URL)
    )
    model = (
        _get_config_value("GPUSTACK_MODEL")
        or _get_config_value("SILICONFLOW_MODEL", DEFAULT_SILICONFLOW_MODEL)
    )

    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            model=model,
            messages=request_messages,
            temperature=0.6,
        )
        content = response.choices[0].message.content
        if content and content.strip():
            return _append_evidence_tag(content.strip(), context_data)
    except Exception:
        return _append_evidence_tag(
            _safe_fallback_socratic_response(user_input, context_data),
            context_data,
        )

    return _append_evidence_tag(
        _safe_fallback_socratic_response(user_input, context_data),
        context_data,
    )


def get_socratic_response(
    user_input: str,
    chat_history: List[Dict[str, Any]],
    context_data: Any,
) -> str:
    """Generate a Socratic coaching response with a hard fallback path."""
    try:
        return _get_socratic_response_impl(user_input, chat_history, context_data)
    except Exception:
        return _append_evidence_tag(
            _safe_fallback_socratic_response(user_input, context_data),
            context_data,
        )
