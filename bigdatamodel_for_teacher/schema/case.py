from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional

from pydantic import BaseModel, Field

from schema.hyperedges import BaseHyperedge, parse_hyperedge
from schema.nodes import BaseNode, KnowledgeCard, Metric, Project, parse_node
from schema.relations import Relation


class KnowledgeGraphCase(BaseModel):
    """
    单个教学案例 / 创业项目的完整图谱数据。

    一个案例由三部分构成：
    1. nodes：节点集合，表达“有哪些对象”。
    2. relations：普通二元关系，表达“两个对象之间是什么关系”。
    3. hyperedges：超边，表达“多个对象一起组成什么模式”。
    """

    case_id: str = Field(..., description="案例唯一编号，例如 D001")
    title: str = Field(..., description="案例标题")
    description: str = Field(default="", description="案例简介")
    nodes: List[BaseNode] = Field(default_factory=list, description="案例中的全部节点")
    relations: List[Relation] = Field(default_factory=list, description="案例中的全部关系")
    hyperedges: List[BaseHyperedge] = Field(default_factory=list, description="案例中的全部超边")

    class Config:
        extra = "forbid"

    @staticmethod
    def _ensure_list(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [item.strip() for item in re.split(r"[,\n;；、|]+", value) if item.strip()]
        if isinstance(value, (list, tuple, set)):
            return [str(item).strip() for item in value if str(item).strip()]
        text = str(value).strip()
        return [text] if text else []

    @staticmethod
    def _dedupe_non_empty(items: Iterable[str]) -> List[str]:
        seen = set()
        result: List[str] = []
        for item in items:
            text = str(item).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            result.append(text)
        return result

    @classmethod
    def _build_case_id(cls, source_name: str, source_path: Optional[Path]) -> str:
        stem = Path(source_name or "case").stem
        hash_match = re.search(r"([0-9a-fA-F]{8})$", stem)
        if hash_match:
            token = hash_match.group(1).upper()
        else:
            token_input = f"{source_name}|{source_path or ''}"
            token = hashlib.sha1(token_input.encode("utf-8")).hexdigest()[:8].upper()
        return f"P_{token}"

    @classmethod
    def _convert_processed_payload(
        cls,
        raw: Dict[str, Any],
        source_path: Optional[Path],
    ) -> Dict[str, Any]:
        extraction = raw.get("extraction")
        if not isinstance(extraction, dict):
            raise ValueError("processed payload requires an object field named 'extraction'.")

        source_file = str(raw.get("source_file") or (source_path.name if source_path else "unknown.pdf")).strip()
        case_id = cls._build_case_id(source_file or "case", source_path)
        token = case_id.lower().replace("-", "_")

        source_stem = Path(source_file or "project").stem
        project_name = str(extraction.get("project_name") or source_stem or case_id).strip()
        if not project_name:
            project_name = case_id

        technical_barriers = cls._dedupe_non_empty(cls._ensure_list(extraction.get("technical_barriers")))
        innovation_points = cls._dedupe_non_empty(cls._ensure_list(extraction.get("innovation_points")))

        core_technology = str(extraction.get("core_technology") or "").strip()
        if not core_technology:
            core_technology = "Core technology pending clarification"

        application_scene = str(extraction.get("application_scene") or "").strip()
        if not application_scene:
            application_scene = "Application scenario pending clarification"

        if not technical_barriers:
            technical_barriers = ["Key technical risk pending clarification"]

        project_id = f"project_{token}"
        technology_id = f"concept_{token}_core_technology"
        market_id = f"concept_{token}_application_scene"
        evidence_id = f"evidence_{token}_source"

        nodes: List[Dict[str, Any]] = [
            {
                "id": project_id,
                "node_type": "Project",
                "name": project_name,
                "description": "Auto-generated from processed JSON extraction.",
                "stage": "idea",
                "team_name": "",
                "tags": ["auto_generated", "processed_json"],
                "metadata": {
                    "source_file": source_file,
                    "source_path": str(source_path) if source_path else "",
                    "value_propositions": innovation_points[:8],
                    "technical_route": core_technology,
                    "target_scenarios": [application_scene],
                    "risk_points": technical_barriers[:8],
                },
            },
            {
                "id": technology_id,
                "node_type": "Concept",
                "name": core_technology,
                "description": "Core technology extracted from project PDF.",
                "aliases": [],
                "tags": ["core_technology"],
                "metadata": {},
            },
            {
                "id": market_id,
                "node_type": "Concept",
                "name": application_scene,
                "description": "Application scene extracted from project PDF.",
                "aliases": [],
                "tags": ["application_scene"],
                "metadata": {},
            },
            {
                "id": evidence_id,
                "node_type": "Evidence",
                "name": "Structured extraction evidence",
                "description": "Generated by batch_processor from project PDF.",
                "source": source_file,
                "confidence": 0.7,
                "tags": ["llm_extraction"],
                "metadata": {},
            },
        ]

        innovation_ids: List[str] = []
        for index, item in enumerate(innovation_points[:12], start=1):
            node_id = f"concept_{token}_innovation_{index}"
            innovation_ids.append(node_id)
            nodes.append(
                {
                    "id": node_id,
                    "node_type": "Concept",
                    "name": item,
                    "description": "Innovation point extracted from project PDF.",
                    "aliases": [],
                    "tags": ["innovation_point"],
                    "metadata": {},
                }
            )

        mistake_ids: List[str] = []
        for index, item in enumerate(technical_barriers[:12], start=1):
            node_id = f"mistake_{token}_barrier_{index}"
            mistake_ids.append(node_id)
            nodes.append(
                {
                    "id": node_id,
                    "node_type": "Mistake",
                    "name": item,
                    "description": "Technical barrier extracted from project PDF.",
                    "symptom": item,
                    "tags": ["technical_barrier"],
                    "metadata": {},
                }
            )

        if not mistake_ids:
            fallback_mistake_id = f"mistake_{token}_barrier_1"
            mistake_ids.append(fallback_mistake_id)
            nodes.append(
                {
                    "id": fallback_mistake_id,
                    "node_type": "Mistake",
                    "name": "Key technical risk pending clarification",
                    "description": "Auto-generated fallback risk node.",
                    "symptom": "Risk details not provided in extraction payload.",
                    "tags": ["technical_barrier"],
                    "metadata": {},
                }
            )

        relations: List[Dict[str, Any]] = [
            {
                "id": f"relation_{token}_project_evidence",
                "source_id": project_id,
                "target_id": evidence_id,
                "relation_type": "EVIDENCED_BY",
                "description": "Project summary is supported by extracted evidence.",
                "weight": 0.7,
                "evidence_ids": [evidence_id],
                "metadata": {},
            },
            {
                "id": f"relation_{token}_tech_scene",
                "source_id": technology_id,
                "target_id": market_id,
                "relation_type": "PREREQ",
                "description": "Core technology is prerequisite for application scene realization.",
                "weight": 0.6,
                "evidence_ids": [evidence_id],
                "metadata": {},
            },
        ]

        for index, node_id in enumerate(innovation_ids[:6], start=1):
            relations.append(
                {
                    "id": f"relation_{token}_innovation_{index}",
                    "source_id": node_id,
                    "target_id": technology_id,
                    "relation_type": "PREREQ",
                    "description": "Innovation point contributes to core technology.",
                    "weight": 0.65,
                    "evidence_ids": [evidence_id],
                    "metadata": {},
                }
            )

        for index, node_id in enumerate(mistake_ids[:3], start=1):
            relations.append(
                {
                    "id": f"relation_{token}_risk_{index}",
                    "source_id": project_id,
                    "target_id": node_id,
                    "relation_type": "COMMON_MISTAKE",
                    "description": "Technical barrier may become project risk if unresolved.",
                    "weight": 0.7,
                    "evidence_ids": [evidence_id],
                    "metadata": {},
                }
            )

        fit_score = 0.8 if innovation_ids else 0.65
        risk_score = min(0.95, 0.45 + 0.05 * min(len(mistake_ids), 6))

        hyperedges = [
            {
                "id": f"hyperedge_{token}_value_loop",
                "edge_type": "Value_Loop_Edge",
                "name": "Auto value loop",
                "description": "Auto-generated value loop from processed JSON extraction.",
                "participants": {
                    "market": market_id,
                    "technology": [technology_id] + innovation_ids if innovation_ids else technology_id,
                    "project": project_id,
                },
                "fit_score": round(fit_score, 2),
                "metadata": {"source_file": source_file},
            },
            {
                "id": f"hyperedge_{token}_risk_pattern",
                "edge_type": "Risk_Pattern_Edge",
                "name": "Auto risk pattern",
                "description": "Auto-generated risk pattern from extracted technical barriers.",
                "participants": {
                    "project": project_id,
                    "mistake": mistake_ids,
                    "outcome": market_id,
                },
                "risk_score": round(risk_score, 2),
                "metadata": {"source_file": source_file},
            },
        ]

        description = (
            f"Auto-generated case from {source_file}. "
            f"Core technology: {core_technology}. "
            f"Application scene: {application_scene}."
        )

        return {
            "case_id": case_id,
            "title": project_name,
            "description": description,
            "nodes": nodes,
            "relations": relations,
            "hyperedges": hyperedges,
        }

    @classmethod
    def normalize_payload(
        cls,
        raw: Dict[str, Any],
        source_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        if not isinstance(raw, dict):
            raise TypeError("case payload must be a dictionary.")

        if isinstance(raw.get("extraction"), dict):
            return cls._convert_processed_payload(raw, source_path=source_path)

        nodes = raw.get("nodes")
        relations = raw.get("relations", [])
        hyperedges = raw.get("hyperedges", [])
        if isinstance(nodes, list) and isinstance(relations, list) and isinstance(hyperedges, list):
            normalized = dict(raw)
            case_id = str(normalized.get("case_id") or "").strip()
            if not case_id:
                source_name = str(normalized.get("source_file") or (source_path.stem if source_path else "case")).strip()
                normalized["case_id"] = cls._build_case_id(source_name or "case", source_path=source_path)
            title = str(normalized.get("title") or "").strip()
            if not title:
                normalized["title"] = str(normalized.get("case_id"))
            normalized["description"] = str(normalized.get("description") or "").strip()
            normalized["nodes"] = nodes
            normalized["relations"] = relations
            normalized["hyperedges"] = hyperedges
            return normalized

        raise ValueError(
            "Unsupported case payload format. Expected either legacy schema "
            "(case_id/title/nodes/relations/hyperedges) or processed extraction schema "
            "(source_file + extraction)."
        )

    @classmethod
    def _from_normalized_dict(cls, raw: Dict[str, Any]) -> "KnowledgeGraphCase":
        parsed_nodes = [parse_node(item) for item in raw.get("nodes", [])]
        parsed_relations = [Relation(**item) for item in raw.get("relations", [])]
        parsed_hyperedges = [parse_hyperedge(item) for item in raw.get("hyperedges", [])]

        return cls(
            case_id=raw["case_id"],
            title=raw["title"],
            description=raw.get("description", ""),
            nodes=parsed_nodes,
            relations=parsed_relations,
            hyperedges=parsed_hyperedges,
        )

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "KnowledgeGraphCase":
        """
        从原始字典构建案例对象。

        之所以不直接写成 KnowledgeGraphCase(**raw)，
        是因为 nodes / hyperedges 内部包含多种不同子类型，
        需要先用 parse_node / parse_hyperedge 做一次“类型分发”。
        """

        normalized = cls.normalize_payload(raw)
        return cls._from_normalized_dict(normalized)

    @classmethod
    def from_json_file(cls, file_path: Path) -> "KnowledgeGraphCase":
        """从 JSON 文件读取案例数据。"""

        path = Path(file_path)
        with path.open("r", encoding="utf-8") as file:
            raw = json.load(file)
        normalized = cls.normalize_payload(raw, source_path=path)
        return cls._from_normalized_dict(normalized)

    def get_node(self, node_id: str) -> Optional[BaseNode]:
        """按 id 查找单个节点。"""

        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def get_nodes_by_type(self, node_type: str) -> List[BaseNode]:
        """按 node_type 查找全部节点。"""

        return [node for node in self.nodes if node.node_type == node_type]

    def get_project(self) -> Optional[Project]:
        """返回案例中的第一个 Project 节点。"""

        for node in self.nodes:
            if isinstance(node, Project):
                return node
        return None

    def get_knowledge_cards(self) -> List[KnowledgeCard]:
        """返回案例中的全部 KnowledgeCard 节点。"""

        return [node for node in self.nodes if isinstance(node, KnowledgeCard)]

    def get_metric(self, metric_key: str) -> Optional[Metric]:
        """
        按指标键名查找 Metric 节点。

        这里同时兼容三种匹配方式：
        - metric_key 字段，例如 TAM
        - name 字段，例如 TAM
        - id 字段，例如 metric_tam
        这样后续编写规则时会更宽容、更好用。
        """

        normalized_key = metric_key.strip().upper()

        for node in self.nodes:
            if not isinstance(node, Metric):
                continue

            candidate_keys = {
                node.id.strip().upper(),
                node.name.strip().upper(),
            }

            if node.metric_key:
                candidate_keys.add(node.metric_key.strip().upper())

            if normalized_key in candidate_keys:
                return node

        return None

    def get_metric_value(self, metric_key: str) -> Optional[float]:
        """直接返回指标数值，找不到则返回 None。"""

        metric_node = self.get_metric(metric_key)
        if metric_node is None:
            return None
        return metric_node.value
