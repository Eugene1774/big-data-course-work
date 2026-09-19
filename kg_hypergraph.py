from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

from neo4j import Driver

from neo4j_resilience import close_managed_driver, get_managed_driver
from rules.checker import GraphRuleChecker
from schema.case import KnowledgeGraphCase


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "processed_json"
DEFAULT_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
DEFAULT_USER = os.getenv("NEO4J_USER", "neo4j")
DEFAULT_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4j")
DEFAULT_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
RULE_IDS = [f"H{i}" for i in range(1, 16)]
PAYLOAD_KEYS = (
    "cases",
    "nodes",
    "contains",
    "validates",
    "mistakes",
    "relations",
    "hyperedges",
    "participants",
    "semantics",
    "case_rules",
)

CUSTOMER_CUES = (
    "customer",
    "persona",
    "segment",
    "user",
    "target customer",
    "customer profile",
    "customer segment",
    "client",
    "student",
)
VALUE_CUES = (
    "value",
    "benefit",
    "solution",
    "value proposition",
    "usp",
    "pmf",
    "pain point",
    "problem-solution",
)
CHANNEL_CUES = (
    "channel",
    "distribution",
    "acquisition",
    "go to market",
    "gtm",
    "media",
    "community",
    "sales",
)
COMPETITOR_CUES = ("competitor", "benchmark", "comparison", "vs", "alternative")


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))
REPORT_CUES = ("report", "interview", "research", "insight", "survey")


def build_driver(uri: str, user: str, password: str, database: str = DEFAULT_DATABASE) -> Driver:
    # 统一走 neo4j_resilience：连接池、重试、可选 resolver 与 schema 校验均由其托管，
    # 不要在此处自行 GraphDatabase.driver(...)（见 AI_CODING_ASSISTANT_PROMPT.md）。
    return get_managed_driver(uri, user, password, database, validate_schema=False)


def clear_db(driver: Driver, database: str) -> None:
    with driver.session(database=database) as session:
        session.run("MATCH (n) DETACH DELETE n")


def initialize_constraints(driver: Driver, database: str) -> None:
    statements = [
        "CREATE CONSTRAINT case_case_id_unique IF NOT EXISTS FOR (n:Case) REQUIRE n.case_id IS UNIQUE",
        "CREATE CONSTRAINT rule_rule_id_unique IF NOT EXISTS FOR (n:Rule) REQUIRE n.rule_id IS UNIQUE",
        "CREATE CONSTRAINT value_loop_edge_id_unique IF NOT EXISTS FOR (n:Value_Loop_Edge) REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT risk_pattern_edge_id_unique IF NOT EXISTS FOR (n:Risk_Pattern_Edge) REQUIRE n.id IS UNIQUE",
    ]
    node_labels = (
        "Concept",
        "Method",
        "Task",
        "Artifact",
        "Metric",
        "Mistake",
        "Evidence",
        "KnowledgeCard",
        "Project",
    )
    statements.extend(
        [
            f"CREATE CONSTRAINT {label.lower()}_id_unique IF NOT EXISTS FOR (n:{label}) REQUIRE n.id IS UNIQUE"
            for label in node_labels
        ]
    )
    statements.extend(
        [
            "CREATE INDEX node_id_index IF NOT EXISTS FOR (n:Node) ON (n.id)",
            "CREATE INDEX hyperedge_id_index IF NOT EXISTS FOR (e:Hyperedge) ON (e.id)",
            "CREATE INDEX hyperedge_rank_index IF NOT EXISTS FOR (e:Hyperedge) ON (e.rank)",
        ]
    )
    with driver.session(database=database) as session:
        for statement in statements:
            session.run(statement)


def initialize_rule_nodes(driver: Driver, database: str) -> None:
    checker = GraphRuleChecker()
    rows = []
    for rule_id in RULE_IDS:
        definition = checker.rule_catalog.get(rule_id)
        rows.append(
            {
                "rule_id": rule_id,
                "name": definition.name if definition else rule_id,
                "description": definition.description if definition else "",
                "implemented": bool(definition.implemented) if definition else False,
                "severity": definition.default_severity.value if definition else None,
            }
        )

    query = """
    UNWIND $rows AS row
    MERGE (r:Rule {rule_id: row.rule_id})
    SET r.name = row.name,
        r.description = row.description,
        r.implemented = row.implemented,
        r.default_severity = row.severity,
        r.updated_at = datetime()
    """
    with driver.session(database=database) as session:
        session.run(query, {"rows": rows})


def normalize_label(label: str | None) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_]", "", str(label or "Concept"))
    if not cleaned:
        return "Concept"
    if cleaned[0].isdigit():
        cleaned = f"Node_{cleaned}"
    return cleaned


def normalize_text(value: Any) -> str:
    lowered = str(value or "").lower().replace("\ufeff", " ")
    lowered = re.sub(r"[^\w\u4e00-\u9fff]+", " ", lowered)
    return " ".join(lowered.split())


def slugify(value: str) -> str:
    text = normalize_text(value).replace(" ", "_")
    return text[:80] or "node"


def ensure_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in re.split(r"[,;/|]", value) if item.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def iter_case_files(data_dir: Path) -> Iterable[Path]:
    for path in sorted(data_dir.glob("*.json")):
        if path.name.startswith("._"):
            continue
        if path.name.startswith("_failed_runs_"):
            continue
        yield path


def add_row_once(
    rows: List[Dict[str, Any]],
    seen: set[Tuple[Any, ...]],
    key: Tuple[Any, ...],
    row: Dict[str, Any],
) -> None:
    if key in seen:
        return
    seen.add(key)
    rows.append(row)


def matches_any(texts: Sequence[str], cues: Sequence[str]) -> bool:
    haystack = normalize_text(" ".join(str(item) for item in texts if item))
    return any(normalize_text(cue) in haystack for cue in cues)


def node_to_row(case_id: str, node: Any) -> Dict[str, Any]:
    payload = node.dict()
    return {
        "node_label": normalize_label(payload["node_type"]),
        "id": payload["id"],
        "name": payload["name"],
        "node_type": payload["node_type"],
        "description": payload.get("description", ""),
        "tags": payload.get("tags", []),
        "tags_text": " ".join(payload.get("tags", [])),
        "metadata_json": json.dumps(payload.get("metadata", {}), ensure_ascii=False),
        "aliases": payload.get("aliases"),
        "steps": payload.get("steps"),
        "expected_output": payload.get("expected_output"),
        "artifact_type": payload.get("artifact_type"),
        "metric_key": payload.get("metric_key"),
        "value": payload.get("value"),
        "unit": payload.get("unit"),
        "symptom": payload.get("symptom"),
        "source": payload.get("source"),
        "confidence": payload.get("confidence"),
        "card_id": payload.get("card_id"),
        "type": payload.get("type"),
        "labels": payload.get("labels"),
        "applicable_scenarios": payload.get("applicable_scenarios"),
        "industry": payload.get("industry"),
        "evidence_source": payload.get("evidence_source"),
        "industry_benchmarks_json": json.dumps(payload.get("industry_benchmarks", {}), ensure_ascii=False),
        "stage": payload.get("stage"),
        "team_name": payload.get("team_name"),
        "case_id": case_id,
    }


def derive_inferred_nodes(case: KnowledgeGraphCase, existing_ids: set[str]) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
    project = case.get_project()
    if project is None:
        return [], {"customer": [], "value": [], "channel": []}

    inferred_rows: List[Dict[str, Any]] = []
    grouped_ids: Dict[str, List[str]] = {"customer": [], "value": [], "channel": []}
    metadata = project.metadata or {}

    customer_names = []
    for item in metadata.get("customer_profiles", []):
        if isinstance(item, dict):
            candidate = item.get("segment") or item.get("name") or item.get("persona")
            if candidate:
                customer_names.append(str(candidate))
    customer_names.extend(ensure_list(metadata.get("target_customer")))
    customer_names.extend(ensure_list(metadata.get("customer_segments")))

    value_names = []
    for item in metadata.get("value_propositions", []):
        if isinstance(item, dict):
            candidate = item.get("name") or item.get("text") or item.get("value")
            if candidate:
                value_names.append(str(candidate))
    value_names.extend(ensure_list(metadata.get("value_proposition")))
    value_names.extend(ensure_list(metadata.get("core_values")))

    channel_names = ensure_list(metadata.get("primary_channels"))
    channel_names.extend(ensure_list(metadata.get("channels")))
    channel_names.extend(ensure_list(metadata.get("go_to_market_channels")))

    inferred_specs = [
        ("customer", customer_names, "Inferred customer concept", ["customer", "persona"]),
        ("value", value_names, "Inferred value proposition concept", ["value", "solution"]),
        ("channel", channel_names, "Inferred channel concept", ["channel", "distribution"]),
    ]

    for role, names, description, tags in inferred_specs:
        for name in names:
            node_id = f"{case.case_id.lower()}_{role}_{slugify(name)}"
            if node_id in existing_ids:
                grouped_ids[role].append(node_id)
                continue
            row = {
                "node_label": "Concept",
                "id": node_id,
                "name": name,
                "node_type": "Concept",
                "description": description,
                "tags": tags,
                "tags_text": " ".join(tags),
                "metadata_json": json.dumps({"inferred_from": "project.metadata", "role": role}, ensure_ascii=False),
                "aliases": None,
                "steps": None,
                "expected_output": None,
                "artifact_type": None,
                "metric_key": None,
                "value": None,
                "unit": None,
                "symptom": None,
                "source": None,
                "confidence": None,
                "card_id": None,
                "type": None,
                "labels": None,
                "applicable_scenarios": None,
                "industry": None,
                "evidence_source": None,
                "industry_benchmarks_json": None,
                "stage": None,
                "team_name": None,
                "case_id": case.case_id,
            }
            inferred_rows.append(row)
            existing_ids.add(node_id)
            grouped_ids[role].append(node_id)

    return inferred_rows, grouped_ids


def infer_role_node_ids(node_index: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
    role_ids: Dict[str, List[str]] = {"customer": [], "value": [], "channel": []}
    for node_id, info in node_index.items():
        texts = [info["name"], info.get("description", ""), info.get("tags_text", "")]
        if info["node_type"] == "Concept" and matches_any(texts, CUSTOMER_CUES):
            role_ids["customer"].append(node_id)
        if info["node_type"] in {"Concept", "Artifact", "Project"} and matches_any(texts, VALUE_CUES):
            role_ids["value"].append(node_id)
        if info["node_type"] in {"Concept", "Artifact", "KnowledgeCard"} and matches_any(texts, CHANNEL_CUES):
            role_ids["channel"].append(node_id)
    return role_ids


def build_case_payload(case: KnowledgeGraphCase) -> Dict[str, List[Dict[str, Any]]]:
    case_rows: List[Dict[str, Any]] = []
    node_rows: List[Dict[str, Any]] = []
    contains_rows: List[Dict[str, Any]] = []
    validates_rows: List[Dict[str, Any]] = []
    mistake_rows: List[Dict[str, Any]] = []
    relation_rows: List[Dict[str, Any]] = []
    hyperedge_rows: List[Dict[str, Any]] = []
    participant_rows: List[Dict[str, Any]] = []
    semantic_rows: List[Dict[str, Any]] = []
    case_rule_rows: List[Dict[str, Any]] = []

    seen_nodes: set[Tuple[Any, ...]] = set()
    seen_contains: set[Tuple[Any, ...]] = set()
    seen_validates: set[Tuple[Any, ...]] = set()
    seen_mistakes: set[Tuple[Any, ...]] = set()
    seen_relations: set[Tuple[Any, ...]] = set()
    seen_hyperedges: set[Tuple[Any, ...]] = set()
    seen_participants: set[Tuple[Any, ...]] = set()
    seen_semantics: set[Tuple[Any, ...]] = set()
    seen_case_rules: set[Tuple[Any, ...]] = set()

    node_index: Dict[str, Dict[str, Any]] = {}

    case_rows.append(
        {
            "case_id": case.case_id,
            "title": case.title,
            "description": case.description,
            "metadata_json": json.dumps({}, ensure_ascii=False),
        }
    )

    for node in case.nodes:
        row = node_to_row(case.case_id, node)
        add_row_once(node_rows, seen_nodes, (row["node_label"], row["id"]), row)
        node_index[row["id"]] = row
        add_row_once(
            contains_rows,
            seen_contains,
            (case.case_id, row["node_label"], row["id"]),
            {"case_id": case.case_id, "node_label": row["node_label"], "node_id": row["id"]},
        )
        if row["node_type"] == "Concept":
            add_row_once(
                validates_rows,
                seen_validates,
                (case.case_id, row["id"]),
                {"case_id": case.case_id, "node_label": row["node_label"], "node_id": row["id"]},
            )
        if row["node_type"] == "Mistake":
            add_row_once(
                mistake_rows,
                seen_mistakes,
                (case.case_id, row["id"]),
                {"case_id": case.case_id, "node_label": row["node_label"], "node_id": row["id"]},
            )

    inferred_rows, inferred_grouped_ids = derive_inferred_nodes(case, set(node_index.keys()))
    for row in inferred_rows:
        add_row_once(node_rows, seen_nodes, (row["node_label"], row["id"]), row)
        node_index[row["id"]] = row
        add_row_once(
            contains_rows,
            seen_contains,
            (case.case_id, row["node_label"], row["id"]),
            {"case_id": case.case_id, "node_label": row["node_label"], "node_id": row["id"]},
        )
        add_row_once(
            validates_rows,
            seen_validates,
            (case.case_id, row["id"]),
            {"case_id": case.case_id, "node_label": row["node_label"], "node_id": row["id"]},
        )

    inferred_roles = infer_role_node_ids(node_index)
    for role, ids in inferred_grouped_ids.items():
        inferred_roles[role].extend(ids)
        inferred_roles[role] = list(dict.fromkeys(inferred_roles[role]))

    project = case.get_project()
    if project is not None:
        for rule_id, reason in (("H12", "project_node"), ("H14", "project_narrative_anchor")):
            add_row_once(
                semantic_rows,
                seen_semantics,
                ("Project", project.id, rule_id),
                {"node_label": "Project", "node_id": project.id, "rule_id": rule_id, "reason": reason},
            )

    for relation in case.relations:
        source_info = node_index.get(relation.source_id)
        target_info = node_index.get(relation.target_id)
        if source_info is None or target_info is None:
            continue
        add_row_once(
            relation_rows,
            seen_relations,
            (relation.relation_type.value, relation.source_id, relation.target_id),
            {
                "relation_type": relation.relation_type.value,
                "source_label": source_info["node_label"],
                "source_id": relation.source_id,
                "target_label": target_info["node_label"],
                "target_id": relation.target_id,
                "relation_id": relation.id,
                "description": relation.description,
                "weight": relation.weight,
                "metadata_json": json.dumps(relation.metadata, ensure_ascii=False),
            },
        )

        relation_rule_map = {
            "PREREQ": "H2",
            "PRODUCES": "H3",
            "MEASURED_BY": "H5",
            "COMMON_MISTAKE": "H7",
        }
        direct_rule_id = relation_rule_map.get(relation.relation_type.value)
        if direct_rule_id:
            for row in (source_info, target_info):
                add_row_once(
                    semantic_rows,
                    seen_semantics,
                    (row["node_label"], row["id"], direct_rule_id),
                    {
                        "node_label": row["node_label"],
                        "node_id": row["id"],
                        "rule_id": direct_rule_id,
                        "reason": f"{relation.relation_type.value.lower()}_relation",
                    },
                )

        if relation.relation_type.value == "EVIDENCED_BY":
            for row, rule_id in (
                (source_info, "H13"),
                (target_info, "H13"),
                (source_info, "H14"),
            ):
                add_row_once(
                    semantic_rows,
                    seen_semantics,
                    (row["node_label"], row["id"], rule_id),
                    {
                        "node_label": row["node_label"],
                        "node_id": row["id"],
                        "rule_id": rule_id,
                        "reason": "evidenced_by_relation",
                    },
                )

    for node_id, row in node_index.items():
        texts = [row["name"], row.get("description", ""), row.get("tags_text", "")]
        if row["node_type"] == "Concept" and (matches_any(texts, CUSTOMER_CUES) or matches_any(texts, VALUE_CUES)):
            add_row_once(
                semantic_rows,
                seen_semantics,
                (row["node_label"], node_id, "H1"),
                {"node_label": row["node_label"], "node_id": node_id, "rule_id": "H1", "reason": "customer_value_concept"},
            )
        if row["node_type"] == "Artifact" and matches_any(texts, REPORT_CUES):
            for rule_id in ("H1", "H14"):
                add_row_once(
                    semantic_rows,
                    seen_semantics,
                    (row["node_label"], node_id, rule_id),
                    {"node_label": row["node_label"], "node_id": node_id, "rule_id": rule_id, "reason": "report_artifact"},
                )
        if row["node_type"] == "Metric":
            metric_key = str(row.get("metric_key") or row["name"]).upper()
            if metric_key in {"TAM", "SAM", "SOM"}:
                add_row_once(
                    semantic_rows,
                    seen_semantics,
                    (row["node_label"], node_id, "H4"),
                    {"node_label": row["node_label"], "node_id": node_id, "rule_id": "H4", "reason": "market_size_metric"},
                )
            if metric_key in {"CAC", "LTV"}:
                add_row_once(
                    semantic_rows,
                    seen_semantics,
                    (row["node_label"], node_id, "H8"),
                    {"node_label": row["node_label"], "node_id": node_id, "rule_id": "H8", "reason": "unit_economics_metric"},
                )
            add_row_once(
                semantic_rows,
                seen_semantics,
                (row["node_label"], node_id, "H5"),
                {"node_label": row["node_label"], "node_id": node_id, "rule_id": "H5", "reason": "metric_presence"},
            )
        if row["node_type"] == "Evidence":
            add_row_once(
                semantic_rows,
                seen_semantics,
                (row["node_label"], node_id, "H13"),
                {"node_label": row["node_label"], "node_id": node_id, "rule_id": "H13", "reason": "evidence_node"},
            )
        if row["node_type"] == "Mistake":
            for rule_id in ("H7", "H10"):
                add_row_once(
                    semantic_rows,
                    seen_semantics,
                    (row["node_label"], node_id, rule_id),
                    {"node_label": row["node_label"], "node_id": node_id, "rule_id": rule_id, "reason": "mistake_node"},
                )
        if row["node_type"] in {"Method", "Task"}:
            add_row_once(
                semantic_rows,
                seen_semantics,
                (row["node_label"], node_id, "H11"),
                {"node_label": row["node_label"], "node_id": node_id, "rule_id": "H11", "reason": "method_or_task"},
            )
        if matches_any(texts, COMPETITOR_CUES):
            add_row_once(
                semantic_rows,
                seen_semantics,
                (row["node_label"], node_id, "H6"),
                {"node_label": row["node_label"], "node_id": node_id, "rule_id": "H6", "reason": "competitor_signal"},
            )

    for hyperedge in case.hyperedges:
        edge_label = normalize_label(hyperedge.edge_type)
        fit_score = getattr(hyperedge, "fit_score", None)
        risk_score = getattr(hyperedge, "risk_score", None)
        add_row_once(
            hyperedge_rows,
            seen_hyperedges,
            (edge_label, hyperedge.id),
            {
                "edge_label": edge_label,
                "id": hyperedge.id,
                "name": hyperedge.name,
                "description": hyperedge.description,
                "case_id": case.case_id,
                "participants_json": json.dumps(hyperedge.participants, ensure_ascii=False),
                "fit_score": fit_score,
                "risk_score": risk_score,
                "metadata_json": json.dumps(hyperedge.metadata, ensure_ascii=False),
                "inferred": False,
                "edge_type": hyperedge.edge_type,
            },
        )

        edge_rule = "H9" if hyperedge.edge_type == "Value_Loop_Edge" else "H10"
        add_row_once(
            semantic_rows,
            seen_semantics,
            (edge_label, hyperedge.id, edge_rule),
            {"node_label": edge_label, "node_id": hyperedge.id, "rule_id": edge_rule, "reason": "explicit_hyperedge"},
        )
        if hyperedge.edge_type == "Risk_Pattern_Edge":
            add_row_once(
                semantic_rows,
                seen_semantics,
                (edge_label, hyperedge.id, "H15"),
                {"node_label": edge_label, "node_id": hyperedge.id, "rule_id": "H15", "reason": "risk_pattern_hyperedge"},
            )

        participant_pairs = []
        if hasattr(hyperedge, "iter_participant_pairs"):
            participant_pairs = list(hyperedge.iter_participant_pairs())
        else:
            for role, participant_ids in hyperedge.participants.items():
                for participant_id in ensure_list(participant_ids):
                    participant_pairs.append((role, participant_id))

        for role, participant_id in participant_pairs:
            participant_info = node_index.get(participant_id)
            if participant_info is None:
                continue
            add_row_once(
                participant_rows,
                seen_participants,
                (edge_label, hyperedge.id, role, participant_id),
                {
                    "edge_label": edge_label,
                    "edge_id": hyperedge.id,
                    "node_label": participant_info["node_label"],
                    "node_id": participant_id,
                    "role": role,
                },
            )

    inferred_customer_ids = inferred_roles["customer"]
    inferred_value_ids = inferred_roles["value"]
    inferred_channel_ids = inferred_roles["channel"]
    max_inferred_edges = 12
    inferred_edge_count = 0

    for customer_id in inferred_customer_ids:
        for value_id in inferred_value_ids:
            for channel_id in inferred_channel_ids:
                if inferred_edge_count >= max_inferred_edges:
                    break
                customer_node = node_index.get(customer_id)
                value_node = node_index.get(value_id)
                channel_node = node_index.get(channel_id)
                if customer_node is None or value_node is None or channel_node is None:
                    continue
                edge_id = f"{case.case_id.lower()}_value_loop_{slugify(customer_id)}_{slugify(value_id)}_{slugify(channel_id)}"
                participants = {"customer": customer_id, "value": value_id, "channel": channel_id}
                add_row_once(
                    hyperedge_rows,
                    seen_hyperedges,
                    ("Value_Loop_Edge", edge_id),
                    {
                        "edge_label": "Value_Loop_Edge",
                        "id": edge_id,
                        "name": f"Customer-Value-Channel Loop {case.case_id}",
                        "description": "Inferred customer-value-channel hyperedge",
                        "case_id": case.case_id,
                        "participants_json": json.dumps(participants, ensure_ascii=False),
                        "fit_score": None,
                        "risk_score": None,
                        "metadata_json": json.dumps({"inferred": True, "source": "heuristic"}, ensure_ascii=False),
                        "inferred": True,
                        "edge_type": "Value_Loop_Edge",
                    },
                )
                for role, participant_id, participant_info in (
                    ("customer", customer_id, customer_node),
                    ("value", value_id, value_node),
                    ("channel", channel_id, channel_node),
                ):
                    add_row_once(
                        participant_rows,
                        seen_participants,
                        ("Value_Loop_Edge", edge_id, role, participant_id),
                        {
                            "edge_label": "Value_Loop_Edge",
                            "edge_id": edge_id,
                            "node_label": participant_info["node_label"],
                            "node_id": participant_id,
                            "role": role,
                        },
                    )
                for rule_id, reason in (("H1", "inferred_customer_value_channel"), ("H9", "inferred_customer_value_channel")):
                    add_row_once(
                        semantic_rows,
                        seen_semantics,
                        ("Value_Loop_Edge", edge_id, rule_id),
                        {"node_label": "Value_Loop_Edge", "node_id": edge_id, "rule_id": rule_id, "reason": reason},
                    )
                inferred_edge_count += 1
            if inferred_edge_count >= max_inferred_edges:
                break
        if inferred_edge_count >= max_inferred_edges:
            break

    for row in semantic_rows:
        add_row_once(
            case_rule_rows,
            seen_case_rules,
            (case.case_id, row["rule_id"]),
            {"case_id": case.case_id, "rule_id": row["rule_id"]},
        )

    return {
        "cases": case_rows,
        "nodes": node_rows,
        "contains": contains_rows,
        "validates": validates_rows,
        "mistakes": mistake_rows,
        "relations": relation_rows,
        "hyperedges": hyperedge_rows,
        "participants": participant_rows,
        "semantics": semantic_rows,
        "case_rules": case_rule_rows,
    }


def merge_payloads(payloads: Sequence[Dict[str, List[Dict[str, Any]]]]) -> Dict[str, List[Dict[str, Any]]]:
    merged = {key: [] for key in PAYLOAD_KEYS}
    seen: Dict[str, set[str]] = {key: set() for key in PAYLOAD_KEYS}

    for payload in payloads:
        for key in PAYLOAD_KEYS:
            for row in payload.get(key, []):
                serialized = json.dumps(row, sort_keys=True, ensure_ascii=False)
                if serialized in seen[key]:
                    continue
                seen[key].add(serialized)
                merged[key].append(row)

    return merged


def group_by(rows: Sequence[Dict[str, Any]], *keys: str) -> DefaultDict[Tuple[Any, ...], List[Dict[str, Any]]]:
    grouped: DefaultDict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    return grouped


def batch_upsert_cases(driver: Driver, database: str, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    query = """
    UNWIND $rows AS row
    MERGE (c:Case {case_id: row.case_id})
    SET c.title = row.title,
        c.description = row.description,
        c.metadata_json = row.metadata_json,
        c.updated_at = datetime()
    """
    with driver.session(database=database) as session:
        session.run(query, {"rows": list(rows)})


def batch_upsert_nodes(driver: Driver, database: str, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    with driver.session(database=database) as session:
        for (node_label,), chunk in group_by(rows, "node_label").items():
            query = f"""
            UNWIND $rows AS row
            MERGE (n:{node_label} {{id: row.id}})
            SET n:Node,
                n.name = row.name,
                n.node_type = row.node_type,
                n.description = row.description,
                n.tags = row.tags,
                n.tags_text = row.tags_text,
                n.metadata_json = row.metadata_json,
                n.aliases = row.aliases,
                n.steps = row.steps,
                n.expected_output = row.expected_output,
                n.artifact_type = row.artifact_type,
                n.metric_key = row.metric_key,
                n.value = row.value,
                n.unit = row.unit,
                n.symptom = row.symptom,
                n.source = row.source,
                n.confidence = row.confidence,
                n.card_id = row.card_id,
                n.type = row.type,
                n.labels = row.labels,
                n.applicable_scenarios = row.applicable_scenarios,
                n.industry = row.industry,
                n.evidence_source = row.evidence_source,
                n.industry_benchmarks_json = row.industry_benchmarks_json,
                n.stage = row.stage,
                n.team_name = row.team_name,
                n.case_id = row.case_id,
                n.updated_at = datetime()
            """
            session.run(query, {"rows": chunk})


def batch_create_case_links(
    driver: Driver,
    database: str,
    rows: Sequence[Dict[str, Any]],
    relationship_type: str,
) -> None:
    if not rows:
        return
    with driver.session(database=database) as session:
        for (node_label,), chunk in group_by(rows, "node_label").items():
            query = f"""
            UNWIND $rows AS row
            MATCH (c:Case {{case_id: row.case_id}})
            MATCH (n:{node_label} {{id: row.node_id}})
            MERGE (c)-[rel:{relationship_type}]->(n)
            """
            session.run(query, {"rows": chunk})


def batch_create_relations(driver: Driver, database: str, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    with driver.session(database=database) as session:
        for (relation_type, source_label, target_label), chunk in group_by(
            rows,
            "relation_type",
            "source_label",
            "target_label",
        ).items():
            query = f"""
            UNWIND $rows AS row
            MATCH (source:{source_label} {{id: row.source_id}})
            MATCH (target:{target_label} {{id: row.target_id}})
            MERGE (source)-[rel:{relation_type}]->(target)
            SET rel.id = row.relation_id,
                rel.description = row.description,
                rel.weight = row.weight,
                rel.metadata_json = row.metadata_json
            """
            session.run(query, {"rows": chunk})


def batch_upsert_hyperedges(driver: Driver, database: str, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    with driver.session(database=database) as session:
        for (edge_label,), chunk in group_by(rows, "edge_label").items():
            query = f"""
            UNWIND $rows AS row
            MATCH (c:Case {{case_id: row.case_id}})
            MERGE (e:{edge_label} {{id: row.id}})
            SET e:Hyperedge,
                e.name = row.name,
                e.description = row.description,
                e.edge_type = row.edge_type,
                e.participants_json = row.participants_json,
                e.fit_score = row.fit_score,
                e.risk_score = row.risk_score,
                e.metadata_json = row.metadata_json,
                e.inferred = row.inferred,
                e.updated_at = datetime()
            MERGE (c)-[:HAS_HYPEREDGE]->(e)
            """
            session.run(query, {"rows": chunk})


def batch_attach_participants(driver: Driver, database: str, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    with driver.session(database=database) as session:
        for (edge_label, node_label), chunk in group_by(rows, "edge_label", "node_label").items():
            query = f"""
            UNWIND $rows AS row
            MATCH (e:{edge_label} {{id: row.edge_id}})
            MATCH (n:{node_label} {{id: row.node_id}})
            MERGE (e)-[rel:HAS_PARTICIPANT {{role: row.role}}]->(n)
            """
            session.run(query, {"rows": chunk})


def backfill_projection_labels(driver: Driver, database: str) -> None:
    """
    为历史数据补齐通用投影标签，便于统一执行 Node/Hyperedge 查询。
    """
    node_query = """
    MATCH (n)
    WHERE n:Concept OR n:Method OR n:Task OR n:Artifact OR n:Metric OR n:Mistake OR n:Evidence OR n:KnowledgeCard OR n:Project
    SET n:Node
    """
    edge_query = """
    MATCH (e)
    WHERE e:Value_Loop_Edge OR e:Risk_Pattern_Edge
    SET e:Hyperedge
    """
    with driver.session(database=database) as session:
        session.run(node_query)
        session.run(edge_query)


def refresh_hyperedge_ranks(driver: Driver, database: str, edge_ids: Optional[Sequence[str]] = None) -> None:
    """
    预计算超边度 rank 与 inv_rank，避免在线查询重复 count()。
    """
    if edge_ids:
        query = """
        UNWIND $edge_ids AS edge_id
        MATCH (e:Hyperedge {id: edge_id})
        OPTIONAL MATCH (e)-[:HAS_PARTICIPANT]->(n:Node)
        WITH e, count(DISTINCT n) AS rank
        SET e.rank = rank,
            e.inv_rank = CASE WHEN rank = 0 THEN 0.0 ELSE 1.0 / toFloat(rank) END,
            e.rank_updated_at = datetime()
        """
        params = {"edge_ids": list(dict.fromkeys(edge_ids))}
    else:
        query = """
        MATCH (e:Hyperedge)
        OPTIONAL MATCH (e)-[:HAS_PARTICIPANT]->(n:Node)
        WITH e, count(DISTINCT n) AS rank
        SET e.rank = rank,
            e.inv_rank = CASE WHEN rank = 0 THEN 0.0 ELSE 1.0 / toFloat(rank) END,
            e.rank_updated_at = datetime()
        """
        params = {}

    with driver.session(database=database) as session:
        session.run(query, params)


def batch_attach_semantics(driver: Driver, database: str, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    with driver.session(database=database) as session:
        for (node_label,), chunk in group_by(rows, "node_label").items():
            query = f"""
            UNWIND $rows AS row
            MATCH (n:{node_label} {{id: row.node_id}})
            MATCH (r:Rule {{rule_id: row.rule_id}})
            MERGE (n)-[rel:EVALUATED_BY]->(r)
            SET rel.reason = row.reason
            """
            session.run(query, {"rows": chunk})


def batch_attach_case_rules(driver: Driver, database: str, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    query = """
    UNWIND $rows AS row
    MATCH (c:Case {case_id: row.case_id})
    MATCH (r:Rule {rule_id: row.rule_id})
    MERGE (c)-[:HIT_RULE]->(r)
    """
    with driver.session(database=database) as session:
        session.run(query, {"rows": list(rows)})


def import_cases(driver: Driver, database: str, data_dir: Path) -> Dict[str, int]:
    payloads = []
    imported_files = 0

    for path in iter_case_files(data_dir):
        case = KnowledgeGraphCase.from_json_file(path)
        payloads.append(build_case_payload(case))
        imported_files += 1
        print(f"[OK] imported case {case.case_id} from {path.name}")

    if not payloads:
        return {"case_files": 0, "cases": 0, "nodes": 0, "hyperedges": 0, "relations": 0, "case_rules": 0}

    merged = merge_payloads(payloads)

    BATCH_SIZE = 20

    try:
        for chunk in chunker(merged["cases"], BATCH_SIZE):
            batch_upsert_cases(driver, database, chunk)

        for chunk in chunker(merged["nodes"], BATCH_SIZE):
            batch_upsert_nodes(driver, database, chunk)

        for chunk in chunker(merged["contains"], BATCH_SIZE):
            batch_create_case_links(driver, database, chunk, "CONTAINS")

        for chunk in chunker(merged["validates"], BATCH_SIZE):
            batch_create_case_links(driver, database, chunk, "VALIDATES")

        for chunk in chunker(merged["mistakes"], BATCH_SIZE):
            batch_create_case_links(driver, database, chunk, "HAS_MISTAKE")

        for chunk in chunker(merged["relations"], BATCH_SIZE):
            batch_create_relations(driver, database, chunk)

        for chunk in chunker(merged["hyperedges"], BATCH_SIZE):
            batch_upsert_hyperedges(driver, database, chunk)

        for chunk in chunker(merged["participants"], BATCH_SIZE):
            batch_attach_participants(driver, database, chunk)

        backfill_projection_labels(driver, database)
        refresh_hyperedge_ranks(driver, database, edge_ids=[row["id"] for row in merged["hyperedges"]])
        batch_attach_semantics(driver, database, merged["semantics"])
        batch_attach_case_rules(driver, database, merged["case_rules"])

        print(" 全量数据分批导入完成！")

    except Exception as e:
        print(f" 导入过程中发生中断: {e}")
        raise

    return {
        "case_files": imported_files,
        "cases": len(merged["cases"]),
        "nodes": len(merged["nodes"]),
        "hyperedges": len(merged["hyperedges"]),
        "relations": len(merged["relations"]),
        "case_rules": len(merged["case_rules"]),
    }


RULE_VERIFY_QUERIES: Dict[str, Tuple[str, str]] = {
    "H1": (
        "customer/value concepts and report artifacts exist",
        """
        MATCH (r:Rule {rule_id:'H1'})
        OPTIONAL MATCH (c:Concept)-[:EVALUATED_BY]->(r)
        WITH r, count(DISTINCT c) AS concept_count
        OPTIONAL MATCH (a:Artifact)-[:EVALUATED_BY]->(r)
        RETURN concept_count > 0 AND count(DISTINCT a) > 0 AS ok,
               concept_count AS concept_count,
               count(DISTINCT a) AS artifact_count
        """,
    ),
    "H2": (
        "prerequisite chain exists",
        """
        MATCH (r:Rule {rule_id:'H2'})
        OPTIONAL MATCH (:Concept)-[rel:PREREQ]-(:Task)
        WITH r, count(rel) AS prereq_count
        OPTIONAL MATCH (n)-[:EVALUATED_BY]->(r)
        RETURN prereq_count > 0 AND count(DISTINCT n) > 1 AS ok,
               prereq_count AS prereq_count,
               count(DISTINCT n) AS annotated_nodes
        """,
    ),
    "H3": (
        "task produces artifact path exists",
        """
        MATCH (r:Rule {rule_id:'H3'})
        OPTIONAL MATCH (:Task)-[rel:PRODUCES]->(:Artifact)
        WITH r, count(rel) AS produces_count
        OPTIONAL MATCH (n)-[:EVALUATED_BY]->(r)
        RETURN produces_count > 0 AND count(DISTINCT n) > 1 AS ok,
               produces_count AS produces_count,
               count(DISTINCT n) AS annotated_nodes
        """,
    ),
    "H4": (
        "TAM/SAM/SOM metrics are present",
        """
        MATCH (r:Rule {rule_id:'H4'})
        OPTIONAL MATCH (m:Metric)-[:EVALUATED_BY]->(r)
        WHERE toUpper(coalesce(m.metric_key, m.name, '')) IN ['TAM', 'SAM', 'SOM']
        RETURN count(DISTINCT m) = 3 AS ok,
               collect(DISTINCT coalesce(m.metric_key, m.name)) AS metrics
        """,
    ),
    "H5": (
        "measurable artifact-metric chain exists",
        """
        MATCH (r:Rule {rule_id:'H5'})
        OPTIONAL MATCH (:Artifact)-[rel:MEASURED_BY]->(:Metric)
        WITH r, count(rel) AS measured_count
        OPTIONAL MATCH (m:Metric)-[:EVALUATED_BY]->(r)
        RETURN measured_count > 0 AND count(DISTINCT m) > 0 AS ok,
               measured_count AS measured_count,
               count(DISTINCT m) AS metric_count
        """,
    ),
    "H6": (
        "competitor comparison signals exist",
        """
        MATCH (r:Rule {rule_id:'H6'})
        OPTIONAL MATCH (n)-[:EVALUATED_BY]->(r)
        RETURN count(DISTINCT n) > 0 AS ok,
               count(DISTINCT n) AS signal_count
        """,
    ),
    "H7": (
        "mistake links exist",
        """
        MATCH (r:Rule {rule_id:'H7'})
        OPTIONAL MATCH (:Case)-[rel:HAS_MISTAKE]->(m:Mistake)
        WITH r, count(rel) AS mistake_link_count, count(DISTINCT m) AS mistake_count
        OPTIONAL MATCH (m2:Mistake)-[:EVALUATED_BY]->(r)
        RETURN mistake_link_count > 0 AND count(DISTINCT m2) > 0 AS ok,
               mistake_link_count AS mistake_link_count,
               mistake_count AS case_mistake_count,
               count(DISTINCT m2) AS annotated_mistake_count
        """,
    ),
    "H8": (
        "CAC/LTV metrics are present",
        """
        MATCH (r:Rule {rule_id:'H8'})
        OPTIONAL MATCH (m:Metric)-[:EVALUATED_BY]->(r)
        WHERE toUpper(coalesce(m.metric_key, m.name, '')) IN ['CAC', 'LTV']
        RETURN count(DISTINCT m) = 2 AS ok,
               collect(DISTINCT coalesce(m.metric_key, m.name)) AS metrics
        """,
    ),
    "H9": (
        "value-loop hyperedges have participants",
        """
        MATCH (r:Rule {rule_id:'H9'})
        OPTIONAL MATCH (e:Value_Loop_Edge)-[:EVALUATED_BY]->(r)
        WITH collect(DISTINCT e) AS edges
        UNWIND CASE WHEN size(edges) = 0 THEN [null] ELSE edges END AS edge
        OPTIONAL MATCH (edge)-[p:HAS_PARTICIPANT]->()
        WITH size([e IN edges WHERE e IS NOT NULL]) AS edge_count,
             edge,
             count(p) AS participant_count
        RETURN edge_count > 0 AND max(participant_count) >= 3 AS ok,
               edge_count AS edge_count,
               max(participant_count) AS max_participants
        """,
    ),
    "H10": (
        "risk-pattern hyperedges have participants",
        """
        MATCH (r:Rule {rule_id:'H10'})
        OPTIONAL MATCH (e:Risk_Pattern_Edge)-[:EVALUATED_BY]->(r)
        WITH collect(DISTINCT e) AS edges
        UNWIND CASE WHEN size(edges) = 0 THEN [null] ELSE edges END AS edge
        OPTIONAL MATCH (edge)-[p:HAS_PARTICIPANT]->()
        WITH size([e IN edges WHERE e IS NOT NULL]) AS edge_count,
             edge,
             count(p) AS participant_count
        RETURN edge_count > 0 AND max(participant_count) >= 3 AS ok,
               edge_count AS edge_count,
               max(participant_count) AS max_participants
        """,
    ),
    "H11": (
        "methods/tasks are attached to the rule",
        """
        MATCH (r:Rule {rule_id:'H11'})
        OPTIONAL MATCH (n)-[:EVALUATED_BY]->(r)
        WHERE n:Method OR n:Task
        RETURN count(DISTINCT n) > 0 AS ok,
               count(DISTINCT n) AS node_count
        """,
    ),
    "H12": (
        "project nodes are attached to the rule",
        """
        MATCH (r:Rule {rule_id:'H12'})
        OPTIONAL MATCH (p:Project)-[:EVALUATED_BY]->(r)
        RETURN count(DISTINCT p) > 0 AS ok,
               count(DISTINCT p) AS project_count
        """,
    ),
    "H13": (
        "evidence nodes are attached to the rule",
        """
        MATCH (r:Rule {rule_id:'H13'})
        OPTIONAL MATCH (e:Evidence)-[:EVALUATED_BY]->(r)
        RETURN count(DISTINCT e) > 0 AS ok,
               count(DISTINCT e) AS evidence_count
        """,
    ),
    "H14": (
        "project narrative anchors have evidence and metrics",
        """
        MATCH (r:Rule {rule_id:'H14'})
        OPTIONAL MATCH (p:Project)-[:EVALUATED_BY]->(r)
        WITH r, count(DISTINCT p) AS project_count
        OPTIONAL MATCH (x)-[:EVALUATED_BY]->(r)
        WHERE x:Artifact OR x:Evidence OR x:Metric
        RETURN project_count > 0 AND count(DISTINCT x) > 0 AS ok,
               project_count AS project_count,
               count(DISTINCT x) AS support_count
        """,
    ),
    "H15": (
        "risk patterns can be connected to fix strategies",
        """
        MATCH (r:Rule {rule_id:'H15'})
        OPTIONAL MATCH (e:Risk_Pattern_Edge)-[:EVALUATED_BY]->(r)
        WITH collect(DISTINCT e) AS edges
        UNWIND CASE WHEN size(edges) = 0 THEN [null] ELSE edges END AS edge
        OPTIONAL MATCH (c:Case)-[:HAS_HYPEREDGE]->(edge)
        OPTIONAL MATCH (c)-[:CONTAINS]->(s)
        WHERE s:Task OR s:Method OR s:Artifact
        WITH size([e IN edges WHERE e IS NOT NULL]) AS edge_count,
             count(DISTINCT s) AS strategy_count
        RETURN edge_count > 0 AND strategy_count > 0 AS ok,
               edge_count AS risk_edge_count,
               strategy_count AS strategy_count
        """,
    ),
}


def verify_h_rules(driver: Driver, database: str) -> Dict[str, Dict[str, Any]]:
    report: Dict[str, Dict[str, Any]] = {}
    with driver.session(database=database) as session:
        for rule_id in RULE_IDS:
            description, query = RULE_VERIFY_QUERIES[rule_id]
            record = session.run(query).single()
            payload = record.data() if record else {"ok": False}
            payload["description"] = description
            report[rule_id] = payload
    return report


WEIGHTED_ONE_HOP_QUERY = """
MATCH (src:Node {id: $node_id})<-[:HAS_PARTICIPANT]-(seed:Hyperedge)
WITH src, collect(DISTINCT seed) AS seed_edges
UNWIND seed_edges AS e
WITH src, e, coalesce(e.rank, size([(e)-[:HAS_PARTICIPANT]->(:Node) | 1])) AS edge_rank, e.inv_rank AS inv_rank
WHERE edge_rank > 1 AND ($max_edge_rank IS NULL OR edge_rank <= $max_edge_rank)
MATCH (e)-[:HAS_PARTICIPANT]->(nbr:Node)
WHERE nbr <> src
WITH DISTINCT e, nbr, edge_rank, inv_rank
WITH nbr,
     sum(
        CASE
            WHEN $beta = 1.0 THEN coalesce(inv_rank, 1.0 / toFloat(edge_rank))
            ELSE 1.0 / pow(toFloat(edge_rank), $beta)
        END
     ) AS similarity,
     count(*) AS shared_hyperedges
RETURN nbr.id AS neighbor_id,
       labels(nbr) AS neighbor_labels,
       similarity,
       shared_hyperedges
ORDER BY similarity DESC, shared_hyperedges DESC, neighbor_id
LIMIT $top_k
"""


KHOP_FRONTIER_EXPAND_QUERY = """
UNWIND $frontier AS item
MATCH (src:Node {id: item.node_id})<-[:HAS_PARTICIPANT]-(e:Hyperedge)-[:HAS_PARTICIPANT]->(dst:Node)
WHERE src <> dst
WITH item, dst, e, coalesce(e.rank, size([(e)-[:HAS_PARTICIPANT]->(:Node) | 1])) AS edge_rank, e.inv_rank AS inv_rank
WHERE edge_rank > 1 AND ($max_edge_rank IS NULL OR edge_rank <= $max_edge_rank)
WITH item,
     dst,
     e,
     edge_rank,
     (
        toFloat(item.score) *
        CASE
            WHEN $beta = 1.0 THEN coalesce(inv_rank, 1.0 / toFloat(edge_rank))
            ELSE 1.0 / pow(toFloat(edge_rank), $beta)
        END
     ) AS partial
WITH dst, sum(partial) AS hop_score
RETURN dst.id AS node_id, hop_score
ORDER BY hop_score DESC
LIMIT $hop_top_n
"""


def get_weighted_neighbors(
    driver: Driver,
    database: str,
    node_id: str,
    top_k: int = 10,
    beta: float = 1.0,
    max_edge_rank: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    一阶相似度查询：
    Similarity(v_i, v_j) = Σ_e (H_ie * H_je) / deg(e)^beta

    参数:
    - beta: 惩罚指数，=1 对应题目公式；>1 会更强惩罚大超边。
    - max_edge_rank: 可选硬阈值。设置后会忽略超大超边，加速大图场景。
    """
    with driver.session(database=database) as session:
        result = session.run(
            WEIGHTED_ONE_HOP_QUERY,
            {
                "node_id": node_id,
                "top_k": int(top_k),
                "beta": float(beta),
                "max_edge_rank": max_edge_rank,
            },
        )
        return [record.data() for record in result]


def get_weighted_neighbors_khop(
    driver: Driver,
    database: str,
    node_id: str,
    k_hop: int = 2,
    top_k: int = 20,
    beta: float = 1.0,
    decay: float = 0.85,
    max_edge_rank: Optional[int] = None,
    frontier_width: int = 200,
    hop_top_n: int = 2000,
) -> List[Dict[str, Any]]:
    """
    K-hop 相似度扩展（迭代式）：
    - 单跳沿用 1/deg(e)^beta 权重
    - 跳数衰减: contribution_h = hop_score * decay^(h-1)
    - 每轮保留 top frontier_width 前沿节点，控制组合爆炸
    """
    if k_hop <= 0:
        return []

    frontier: Dict[str, float] = {node_id: 1.0}
    accumulated: DefaultDict[str, float] = defaultdict(float)

    with driver.session(database=database) as session:
        for hop in range(1, k_hop + 1):
            frontier_rows = [{"node_id": key, "score": value} for key, value in frontier.items()]
            if not frontier_rows:
                break

            result = session.run(
                KHOP_FRONTIER_EXPAND_QUERY,
                {
                    "frontier": frontier_rows,
                    "beta": float(beta),
                    "max_edge_rank": max_edge_rank,
                    "hop_top_n": int(hop_top_n),
                },
            )

            next_frontier: DefaultDict[str, float] = defaultdict(float)
            for record in result:
                row = record.data()
                candidate_id = str(row.get("node_id", "")).strip()
                hop_score = float(row.get("hop_score", 0.0) or 0.0)
                if not candidate_id or candidate_id == node_id or hop_score <= 0.0:
                    continue

                accumulated[candidate_id] += hop_score * (float(decay) ** (hop - 1))
                next_frontier[candidate_id] += hop_score

            if not next_frontier:
                break

            sorted_frontier = sorted(next_frontier.items(), key=lambda item: item[1], reverse=True)[:frontier_width]
            frontier = dict(sorted_frontier)

        if not accumulated:
            return []

        ranked = sorted(accumulated.items(), key=lambda item: item[1], reverse=True)[:top_k]
        candidate_ids = [item[0] for item in ranked]
        labels_result = session.run(
            """
            UNWIND $candidate_ids AS node_id
            MATCH (n:Node {id: node_id})
            RETURN n.id AS node_id, labels(n) AS labels
            """,
            {"candidate_ids": candidate_ids},
        )
        labels_map = {record["node_id"]: record["labels"] for record in labels_result}

    return [
        {
            "neighbor_id": candidate_id,
            "neighbor_labels": labels_map.get(candidate_id, []),
            "similarity": score,
        }
        for candidate_id, score in ranked
    ]


def get_risk_pattern(
    driver: Driver,
    database: str,
    input_nodes: Sequence[str],
    limit: int = 5,
) -> List[Dict[str, Any]]:
    keywords = [normalize_text(keyword) for keyword in input_nodes if normalize_text(keyword)]
    if not keywords:
        return []

    query = """
    WITH $keywords AS keywords
    MATCH (n)
    WHERE (n:Concept OR n:Metric OR n:Mistake OR n:Artifact OR n:Project OR n:Task OR n:Method)
      AND any(kw IN keywords WHERE
          toLower(coalesce(n.name, '')) CONTAINS kw OR
          toLower(coalesce(n.description, '')) CONTAINS kw OR
          toLower(coalesce(n.tags_text, '')) CONTAINS kw)
    MATCH (edge:Risk_Pattern_Edge)-[:HAS_PARTICIPANT]->(n)
    MATCH (case:Case)-[:HAS_HYPEREDGE]->(edge)
    OPTIONAL MATCH (edge)-[part:HAS_PARTICIPANT]->(participant)
    OPTIONAL MATCH (case)-[:CONTAINS]->(strategy)
    WHERE strategy:Task OR strategy:Method OR strategy:Artifact
    RETURN edge.id AS edge_id,
           edge.name AS edge_name,
           edge.description AS edge_description,
           case.case_id AS case_id,
           case.title AS case_title,
           collect(DISTINCT n.name) AS matched_nodes,
           collect(DISTINCT {role: part.role, node: participant.name}) AS participants,
           collect(DISTINCT strategy.name)[0..5] AS fix_strategies
    LIMIT $limit
    """
    with driver.session(database=database) as session:
        result = session.run(query, {"keywords": keywords, "limit": limit})
        return [record.data() for record in result]


def print_verify_report(report: Dict[str, Dict[str, Any]]) -> None:
    print("\nRule verification report")
    print("-" * 72)
    for rule_id in RULE_IDS:
        payload = report.get(rule_id, {})
        status = "PASS" if payload.get("ok") else "FAIL"
        details = ", ".join(
            f"{key}={value}"
            for key, value in payload.items()
            if key not in {"ok", "description"}
        )
        print(f"{rule_id:>3} | {status:4} | {payload.get('description', '')}")
        if details:
            print(f"      {details}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build hypergraph structures and case associations in Neo4j.")
    parser.add_argument("--uri", default=DEFAULT_URI, help="Neo4j bolt URI")
    parser.add_argument("--user", default=DEFAULT_USER, help="Neo4j username")
    parser.add_argument("--password", default=DEFAULT_PASSWORD, help="Neo4j password")
    parser.add_argument("--database", default=DEFAULT_DATABASE, help="Neo4j database name")
    parser.add_argument(
        "--data-dir",
        default=str(DATA_DIR),
        help="Directory containing case JSON files (legacy D*.json or processed extraction JSON).",
    )
    parser.add_argument("--clear", action="store_true", help="Clear the database before import")
    parser.add_argument("--verify", action="store_true", help="Run H1-H15 structural verification after import")
    parser.add_argument(
        "--risk-keywords",
        nargs="*",
        default=[],
        help="Query related risk patterns by input keywords after import",
    )
    parser.add_argument("--risk-limit", type=int, default=5, help="Maximum number of risk patterns to return")
    parser.add_argument("--neighbor-node", default="", help="Run hypergraph similarity query for this node id")
    parser.add_argument("--neighbor-top-k", type=int, default=10, help="Top-K neighbors to return")
    parser.add_argument("--neighbor-k-hop", type=int, default=1, help="Hop count for similarity query")
    parser.add_argument("--neighbor-beta", type=float, default=1.0, help="Hyperedge-degree penalty exponent")
    parser.add_argument("--neighbor-decay", type=float, default=0.85, help="Hop decay factor for K-hop query")
    parser.add_argument(
        "--neighbor-max-edge-rank",
        type=int,
        default=None,
        help="Optional max hyperedge rank allowed in query",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise SystemExit(f"Data directory does not exist: {data_dir}")

    driver = build_driver(args.uri, args.user, args.password, args.database)
    try:
        if args.clear:
            print("Clearing database")
            clear_db(driver, args.database)

        print("Initializing constraints")
        initialize_constraints(driver, args.database)

        print("Initializing rule catalog")
        initialize_rule_nodes(driver, args.database)

        print(f"Importing cases from {data_dir}")
        summary = import_cases(driver, args.database, data_dir)
        print(
            "Import summary: "
            f"case_files={summary['case_files']}, "
            f"cases={summary['cases']}, "
            f"nodes={summary['nodes']}, "
            f"hyperedges={summary['hyperedges']}, "
            f"relations={summary['relations']}, "
            f"case_rules={summary['case_rules']}"
        )

        if args.verify:
            report = verify_h_rules(driver, args.database)
            print_verify_report(report)

        if args.risk_keywords:
            matches = get_risk_pattern(
                driver,
                args.database,
                args.risk_keywords,
                limit=args.risk_limit,
            )
            print("\nRisk pattern matches")
            print(json.dumps(matches, ensure_ascii=False, indent=2))

        if args.neighbor_node:
            if args.neighbor_k_hop <= 1:
                neighbors = get_weighted_neighbors(
                    driver,
                    args.database,
                    node_id=args.neighbor_node,
                    top_k=args.neighbor_top_k,
                    beta=args.neighbor_beta,
                    max_edge_rank=args.neighbor_max_edge_rank,
                )
            else:
                neighbors = get_weighted_neighbors_khop(
                    driver,
                    args.database,
                    node_id=args.neighbor_node,
                    k_hop=args.neighbor_k_hop,
                    top_k=args.neighbor_top_k,
                    beta=args.neighbor_beta,
                    decay=args.neighbor_decay,
                    max_edge_rank=args.neighbor_max_edge_rank,
                )
            print("\nHypergraph similarity neighbors")
            print(json.dumps(neighbors, ensure_ascii=False, indent=2))
    finally:
        close_managed_driver(args.uri, args.user, args.password)


if __name__ == "__main__":
    main()
