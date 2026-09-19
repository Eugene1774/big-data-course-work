from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np

from schema.case import KnowledgeGraphCase

DEGENERATE_AVERAGE_RANK_THRESHOLD = 2.0
HIGH_ORDER_MIN_RANK = 3
MIN_HIGH_ORDER_RATIO = 0.30


# 节点主题词典：用于将节点文本映射为可解释的主题标签。
TOPIC_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "financing": (
        "融资",
        "投资",
        "估值",
        "股权",
        "财务",
        "现金流",
        "funding",
        "investment",
        "valuation",
        "equity",
        "finance",
        "investor",
    ),
    "hardware": (
        "硬件",
        "硬件调试",
        "电路",
        "传感器",
        "焊接",
        "固件",
        "pcb",
        "firmware",
        "debug",
    ),
    "marketing": (
        "营销",
        "渠道",
        "获客",
        "转化",
        "广告",
        "media",
        "marketing",
        "acquisition",
        "channel",
    ),
    "product": (
        "产品",
        "需求",
        "用户",
        "痛点",
        "价值主张",
        "pmf",
        "product",
        "user",
        "persona",
        "value proposition",
    ),
}


# 超边意图规则：根据超边名称/描述推断其“应该包含/不应包含”的主题。
EDGE_INTENT_RULES = [
    {
        "name": "financing_edge",
        "patterns": ("融资", "funding", "investment", "equity", "估值"),
        "required_topics": {"financing"},
        "forbidden_topics": {"hardware"},
    },
    {
        "name": "hardware_edge",
        "patterns": ("硬件", "调试", "firmware", "pcb"),
        "required_topics": {"hardware"},
        "forbidden_topics": {"financing"},
    },
]


# edge_type 的结构语义约束（角色层面）。
EXPECTED_ROLES: Dict[str, Set[str]] = {
    "Value_Loop_Edge": {"market", "technology", "project"},
    "Risk_Pattern_Edge": {"project", "mistake", "outcome"},
}


@dataclass
class CaseReport:
    case_file: str
    case_id: str
    node_count: int
    hyperedge_count: int
    isolated_nodes: List[str]
    average_rank: float
    high_order_edge_ratio: float
    rank_distribution: Dict[str, int]
    degenerates_to_graph: bool
    incidence_ok: bool
    incidence_errors: List[str]
    semantic_warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_file": self.case_file,
            "case_id": self.case_id,
            "node_count": self.node_count,
            "hyperedge_count": self.hyperedge_count,
            "isolated_nodes": self.isolated_nodes,
            "average_rank": round(self.average_rank, 4),
            "high_order_edge_ratio": round(self.high_order_edge_ratio, 4),
            "rank_distribution": self.rank_distribution,
            "degenerates_to_graph": self.degenerates_to_graph,
            "incidence_ok": self.incidence_ok,
            "incidence_errors": self.incidence_errors,
            "semantic_warnings": self.semantic_warnings,
        }


def normalize_text(value: Any) -> str:
    text = str(value or "").lower()
    text = re.sub(r"[\u3000\s]+", " ", text)
    return text.strip()


def ensure_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def flatten_participants(participants: Dict[str, Any]) -> List[str]:
    """
    兼容两种参与者格式：
    1) {"role": "node_id"}
    2) {"role": ["node_id_a", "node_id_b"]}
    返回去重后的节点列表。
    """
    node_ids: List[str] = []
    for _, raw_value in participants.items():
        node_ids.extend(ensure_list(raw_value))
    return list(dict.fromkeys(node_ids))


def infer_topics(text: str) -> Set[str]:
    normalized = normalize_text(text)
    tags: Set[str] = set()
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(normalize_text(word) in normalized for word in keywords):
            tags.add(topic)
    return tags


def detect_edge_intents(edge_text: str) -> List[Dict[str, Any]]:
    normalized = normalize_text(edge_text)
    matched_rules: List[Dict[str, Any]] = []
    for rule in EDGE_INTENT_RULES:
        if any(normalize_text(pattern) in normalized for pattern in rule["patterns"]):
            matched_rules.append(rule)
    return matched_rules


def topic_coverage(required_topics: Set[str], actual_topics: Set[str]) -> float:
    """覆盖度评分：1.0 表示 required 全覆盖。"""
    if not required_topics:
        return 1.0
    return len(required_topics & actual_topics) / float(len(required_topics))


def build_incidence_matrix(
    node_ids: Sequence[str], edge_to_nodes: Dict[str, Sequence[str]]
) -> Tuple[List[str], np.ndarray]:
    edge_ids = list(edge_to_nodes.keys())
    node_index = {node_id: idx for idx, node_id in enumerate(node_ids)}
    edge_index = {edge_id: idx for idx, edge_id in enumerate(edge_ids)}

    matrix = np.zeros((len(node_ids), len(edge_ids)), dtype=np.int8)
    for edge_id, members in edge_to_nodes.items():
        col = edge_index[edge_id]
        for node_id in members:
            row = node_index.get(node_id)
            if row is None:
                continue
            matrix[row, col] = 1
    return edge_ids, matrix


def verify_incidence(
    node_ids: Sequence[str],
    edge_ids: Sequence[str],
    matrix: np.ndarray,
    edge_to_nodes: Dict[str, Sequence[str]],
) -> Tuple[bool, List[str]]:
    errors: List[str] = []

    if matrix.shape != (len(node_ids), len(edge_ids)):
        errors.append(
            f"incidence matrix shape mismatch: got {matrix.shape}, expected ({len(node_ids)}, {len(edge_ids)})"
        )

    if not np.isin(matrix, [0, 1]).all():
        errors.append("incidence matrix contains values other than 0/1")

    # 逐列校验：矩阵列表示的成员集合必须与 edge_to_nodes 一致。
    for col, edge_id in enumerate(edge_ids):
        expected_members = set(edge_to_nodes.get(edge_id, []))
        actual_members = {node_ids[row] for row in range(len(node_ids)) if matrix[row, col] == 1}
        if expected_members != actual_members:
            errors.append(
                f"edge {edge_id} mismatch: expected={sorted(expected_members)} actual={sorted(actual_members)}"
            )

    return len(errors) == 0, errors


def semantic_validation(
    hyperedges: Sequence[Dict[str, Any]],
    node_text_map: Dict[str, str],
) -> List[str]:
    """
    语义一致性验证分为两层：
    1) 角色结构一致性（edge_type -> expected roles）
    2) 文本主题一致性（required/forbidden topics）
    """
    warnings: List[str] = []

    for edge in hyperedges:
        edge_id = str(edge.get("id", "")).strip()
        edge_type = str(edge.get("edge_type", "")).strip()
        participants = edge.get("participants", {}) or {}
        role_set = set(participants.keys())

        expected_roles = EXPECTED_ROLES.get(edge_type)
        if expected_roles is not None and role_set != expected_roles:
            warnings.append(
                f"[{edge_id}] role mismatch: expected={sorted(expected_roles)} actual={sorted(role_set)}"
            )

        edge_text = " ".join(
            [
                str(edge.get("name", "")),
                str(edge.get("description", "")),
                edge_type,
            ]
        )
        intent_rules = detect_edge_intents(edge_text)
        if not intent_rules:
            continue

        member_ids = flatten_participants(participants)
        member_text = " ".join(node_text_map.get(node_id, "") for node_id in member_ids)
        member_topics = infer_topics(member_text)

        for rule in intent_rules:
            required_topics: Set[str] = set(rule["required_topics"])
            forbidden_topics: Set[str] = set(rule["forbidden_topics"])
            coverage = topic_coverage(required_topics, member_topics)
            missing_required = sorted(required_topics - member_topics)
            hit_forbidden = sorted(member_topics & forbidden_topics)

            if missing_required:
                warnings.append(
                    f"[{edge_id}] semantic weak-match ({rule['name']}): "
                    f"coverage={coverage:.2f}, missing required topics {missing_required}"
                )
            if hit_forbidden:
                warnings.append(
                    f"[{edge_id}] semantic conflict ({rule['name']}): "
                    f"coverage={coverage:.2f}, forbidden topics detected {hit_forbidden}"
                )

    return warnings


def analyze_case(
    case_path: Path,
    min_average_rank: float,
    min_high_order_ratio: float,
    high_order_min_rank: int,
) -> CaseReport:
    with case_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    raw = KnowledgeGraphCase.normalize_payload(raw, source_path=case_path)

    case_id = str(raw.get("case_id", case_path.stem))
    nodes = raw.get("nodes", []) or []
    hyperedges = raw.get("hyperedges", []) or []

    raw_node_ids = [str(node.get("id", "")).strip() for node in nodes if str(node.get("id", "")).strip()]
    node_id_counts = Counter(raw_node_ids)
    node_ids = list(dict.fromkeys(raw_node_ids))

    incidence_errors: List[str] = []
    duplicate_node_ids = sorted([node_id for node_id, count in node_id_counts.items() if count > 1])
    if duplicate_node_ids:
        incidence_errors.append(f"duplicate node ids found: {duplicate_node_ids}")

    node_text_map: Dict[str, str] = {}
    for node in nodes:
        node_id = str(node.get("id", "")).strip()
        if not node_id:
            continue
        tags = " ".join(ensure_list(node.get("tags", [])))
        node_text_map[node_id] = " ".join(
            [
                str(node.get("name", "")),
                str(node.get("description", "")),
                tags,
                str(node.get("node_type", "")),
            ]
        )

    edge_to_nodes: Dict[str, List[str]] = {}
    seen_edge_ids: Set[str] = set()

    for edge in hyperedges:
        edge_id = str(edge.get("id", "")).strip()
        if not edge_id:
            incidence_errors.append("found hyperedge with empty id")
            continue

        if edge_id in seen_edge_ids:
            incidence_errors.append(f"duplicate hyperedge id found: {edge_id}")
            continue
        seen_edge_ids.add(edge_id)

        participants = edge.get("participants", {}) or {}
        member_ids = flatten_participants(participants)
        if not member_ids:
            incidence_errors.append(f"edge {edge_id} has empty participant set")
        edge_to_nodes[edge_id] = member_ids

        # 成员存在性校验：避免出现未知节点引用。
        for node_id in member_ids:
            if node_id not in node_text_map:
                incidence_errors.append(f"edge {edge_id} references unknown node_id: {node_id}")

    edge_ids, incidence = build_incidence_matrix(node_ids, edge_to_nodes)
    incidence_ok, matrix_errors = verify_incidence(node_ids, edge_ids, incidence, edge_to_nodes)
    incidence_errors.extend(matrix_errors)

    if incidence.shape[1] == 0:
        node_degrees = np.zeros((len(node_ids),), dtype=np.int32)
        edge_ranks = np.zeros((0,), dtype=np.int32)
    else:
        node_degrees = incidence.sum(axis=1)
        edge_ranks = incidence.sum(axis=0)

    isolated_nodes = [node_ids[i] for i, degree in enumerate(node_degrees.tolist()) if int(degree) == 0]

    if edge_ranks.size == 0:
        average_rank = 0.0
        high_order_edge_ratio = 0.0
        rank_distribution: Dict[str, int] = {}
    else:
        average_rank = float(np.mean(edge_ranks))
        high_order_edge_ratio = float(np.mean(edge_ranks >= high_order_min_rank))
        rank_distribution = {
            str(rank): int((edge_ranks == rank).sum())
            for rank in sorted(set(int(item) for item in edge_ranks.tolist()))
        }

    # 退化判定：平均秩偏低，或高阶边占比不足，都会导致超图近似退化为普通图。
    degenerates_to_graph = (
        (not edge_to_nodes)
        or (average_rank <= min_average_rank)
        or (high_order_edge_ratio < min_high_order_ratio)
    )

    semantic_warnings = semantic_validation(hyperedges, node_text_map)

    return CaseReport(
        case_file=str(case_path),
        case_id=case_id,
        node_count=len(node_ids),
        hyperedge_count=len(edge_to_nodes),
        isolated_nodes=isolated_nodes,
        average_rank=average_rank,
        high_order_edge_ratio=high_order_edge_ratio,
        rank_distribution=rank_distribution,
        degenerates_to_graph=degenerates_to_graph,
        incidence_ok=incidence_ok and not incidence_errors,
        incidence_errors=incidence_errors,
        semantic_warnings=semantic_warnings,
    )


def iter_case_paths(explicit_cases: Iterable[str], data_dir: Path, pattern: str) -> List[Path]:
    case_paths = [Path(item) for item in explicit_cases]
    if case_paths:
        return case_paths
    paths: List[Path] = []
    for path in sorted(data_dir.glob(pattern)):
        if path.name.startswith("._"):
            continue
        if path.name.startswith("_failed_runs_"):
            continue
        paths.append(path)
    return paths


def print_report(reports: Sequence[CaseReport]) -> None:
    print("Hypergraph Skeleton Validation")
    print("=" * 72)
    for report in reports:
        print(f"\nCase: {report.case_id} ({report.case_file})")
        print(f"- node_count={report.node_count}, hyperedge_count={report.hyperedge_count}")
        print(f"- isolated_nodes={len(report.isolated_nodes)} -> {report.isolated_nodes}")
        print(
            f"- average_rank={report.average_rank:.4f}, "
            f"high_order_edge_ratio={report.high_order_edge_ratio:.4f}, "
            f"degenerates_to_graph={report.degenerates_to_graph}, "
            f"rank_distribution={report.rank_distribution}"
        )
        print(f"- incidence_ok={report.incidence_ok}")
        if report.incidence_errors:
            print("  incidence_errors:")
            for item in report.incidence_errors:
                print(f"  * {item}")
        if report.semantic_warnings:
            print("  semantic_warnings:")
            for item in report.semantic_warnings:
                print(f"  * {item}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate hypergraph skeleton and semantic consistency.")
    parser.add_argument("--case", action="append", default=[], help="Specific case json path (repeatable)")
    parser.add_argument(
        "--data-dir",
        default="processed_json",
        help="Directory containing case JSON files (legacy D*.json or processed extraction JSON).",
    )
    parser.add_argument(
        "--pattern",
        default="*.json",
        help="Glob pattern under data-dir when --case is absent",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON report")
    parser.add_argument(
        "--min-average-rank",
        type=float,
        default=2.0,
        help="Average rank threshold; lower means more graph-like structure",
    )
    parser.add_argument(
        "--min-high-order-ratio",
        type=float,
        default=0.30,
        help="Minimum ratio of high-order hyperedges required",
    )
    parser.add_argument(
        "--high-order-min-rank",
        type=int,
        default=3,
        help="Rank threshold to count an edge as high-order",
    )
    args = parser.parse_args()

    min_average_rank = args.min_average_rank if args.min_average_rank is not None else DEGENERATE_AVERAGE_RANK_THRESHOLD
    min_high_order_ratio = args.min_high_order_ratio if args.min_high_order_ratio is not None else MIN_HIGH_ORDER_RATIO
    high_order_min_rank = args.high_order_min_rank if args.high_order_min_rank is not None else HIGH_ORDER_MIN_RANK

    data_dir = Path(args.data_dir)
    case_paths = iter_case_paths(args.case, data_dir, args.pattern)
    if not case_paths:
        print("No case files found.")
        return 1

    reports = [
        analyze_case(
            path,
            min_average_rank=min_average_rank,
            min_high_order_ratio=min_high_order_ratio,
            high_order_min_rank=high_order_min_rank,
        )
        for path in case_paths
    ]

    if args.json:
        print(json.dumps([item.to_dict() for item in reports], ensure_ascii=False, indent=2))
    else:
        print_report(reports)

    # 任一 case 失败则返回 1，便于 CI 直接拦截。
    has_failure = any((not item.incidence_ok) or item.degenerates_to_graph for item in reports)
    return 1 if has_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
