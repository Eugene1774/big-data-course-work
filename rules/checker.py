from __future__ import annotations

from collections import defaultdict, deque
from enum import Enum
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from pydantic import BaseModel, Field

from rules.rule_switches import is_rule_enabled
from schema.case import KnowledgeGraphCase
from schema.hyperedges import HyperedgeType
from schema.nodes import Artifact, BaseNode, Evidence, Metric, Method, Project, Task
from schema.relations import Relation, RelationType


KW = {
    "customer": {"customer", "persona", "segment", "user", "客户", "用户", "客群", "画像", "采购"},
    "value": {"value proposition", "solution", "benefit", "价值主张", "方案", "卖点", "解决方案"},
    "channel": {"channel", "distribution", "acquisition", "渠道", "获客", "投流", "展会", "直销", "推广"},
    "revenue": {"revenue", "pricing", "price", "subscription", "commission", "ads", "收入", "定价", "付费", "订阅", "佣金", "广告", "变现"},
    "payment": {"willingness to pay", "price sensitivity", "支付意愿", "价格敏感", "价格测试", "付费测试", "访谈", "问卷"},
    "demand": {"interview", "survey", "observation", "behavior", "访谈", "问卷", "观察", "原话", "日志"},
    "competitor": {"competitor", "benchmark", "comparison", "alternative", "substitute", "竞品", "竞争", "对比", "替代", "平替"},
    "innovation": {"innovation", "proprietary", "moat", "patent", "algorithm", "创新", "壁垒", "专利", "算法", "技术路线", "核心竞争力"},
    "growth": {"national", "nationwide", "scale", "franchise", "全国", "扩张", "复制", "连锁", "铺开"},
    "milestone": {"milestone", "timeline", "roadmap", "里程碑", "时间表", "路线图", "交付", "上线"},
    "deeptech": {"quantum", "chip", "robotics", "medical device", "大模型", "量子", "芯片", "机器人", "医疗器械", "工业设备"},
    "compliance_signal": {"ai", "privacy", "biometric", "education", "medical", "finance", "copyright", "children", "data", "ai伦理", "数据隐私", "隐私", "准入", "医疗", "金融", "未成年人", "版权", "教育"},
    "compliance_control": {"license", "permit", "regulatory", "consent", "ethics", "privacy policy", "资质", "许可", "审批", "合规", "授权", "同意书", "伦理"},
    "experiment": {"mvp", "pilot", "experiment", "ab test", "test", "验证", "试点", "实验", "最小可行产品", "灰度"},
    "control": {"control group", "ab test", "baseline", "对照组", "实验组", "基线"},
}


class RuleSeverity(str, Enum):
    INFO = "Info"
    WARNING = "Warning"
    HIGH = "High"
    ERROR = "High"


class RuleStatus(str, Enum):
    PASSED = "PASSED"
    WARNING = "WARNING"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class RuleDefinition(BaseModel):
    rule_id: str
    name: str
    description: str
    trigger_condition: str
    default_severity: RuleSeverity
    weight: int = 1
    implemented: bool = True

    class Config:
        extra = "forbid"


class RuleResult(BaseModel):
    rule_id: str
    name: str
    status: RuleStatus
    severity: RuleSeverity
    message: str
    is_triggered: bool
    evidence_nodes: List[str] = Field(default_factory=list)
    fix_suggestion: str = "补齐一条最关键的结构化证据。"
    details: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        extra = "forbid"


class DiagnosisReport(BaseModel):
    case_id: str
    title: str
    results: List[RuleResult] = Field(default_factory=list)

    class Config:
        extra = "forbid"

    @property
    def overall_status(self) -> RuleStatus:
        if any(item.status == RuleStatus.FAILED for item in self.results):
            return RuleStatus.FAILED
        if any(item.status == RuleStatus.WARNING for item in self.results):
            return RuleStatus.WARNING
        return RuleStatus.PASSED

    def count(self, status: RuleStatus) -> int:
        return sum(1 for item in self.results if item.status == status)


class GraphRuleChecker:
    """Heuristic H1-H15 checker compatible with the local case schema."""

    def __init__(self) -> None:
        self.rule_catalog = self._build_rule_catalog()

    def _build_rule_catalog(self) -> Dict[str, RuleDefinition]:
        rows = [
            ("H1", "客户-价值主张错位", "检查客户、价值、渠道、收入是否闭环。", "渠道无法触达客户或收入模式与价值主张不匹配。", RuleSeverity.HIGH, 2),
            ("H2", "渠道不可达", "检查渠道是否覆盖目标客群。", "渠道有效覆盖人群不包含目标客户。", RuleSeverity.HIGH, 2),
            ("H3", "定价无支付意愿证据", "检查定价是否有支付意愿证据。", "定价模型缺乏支付意愿或价格测试支撑。", RuleSeverity.WARNING, 1),
            ("H4", "TAM/SAM/SOM口径混乱", "检查 TAM >= SAM >= SOM。", "市场规模未满足 TAM >= SAM >= SOM。", RuleSeverity.HIGH, 2),
            ("H5", "需求证据不足", "检查需求是否由真实证据支撑。", "需求仅基于主观推测，缺乏访谈或行为证据。", RuleSeverity.WARNING, 1),
            ("H6", "竞品对比不可比", "检查竞品是否覆盖替代方案。", "竞品分析规避了核心替代品。", RuleSeverity.WARNING, 1),
            ("H7", "创新点不可验证", "检查创新是否有技术路线或量化对比。", "创新缺乏技术路径、专利状态或量化指标。", RuleSeverity.HIGH, 2),
            ("H8", "单位经济不成立", "检查 LTV >= 3 * CAC。", "LTV 未达到 CAC 的 3 倍。", RuleSeverity.WARNING, 1),
            ("H9", "增长逻辑跳跃", "检查是否缺乏 MVP 却直接扩张。", "尚未验证 MVP 就规划大规模扩张。", RuleSeverity.HIGH, 2),
            ("H10", "里程碑不可交付", "检查里程碑是否与资源匹配。", "里程碑与时间、团队能力或资源严重脱节。", RuleSeverity.HIGH, 3),
            ("H11", "合规/伦理缺口", "检查合规场景是否有说明。", "涉及 AI/隐私/准入却无合规方案。", RuleSeverity.HIGH, 2),
            ("H12", "技术路线与资源不匹配", "检查 TRL、技术路线与团队资源。", "TRL 过高且团队缺少专家或可行路径。", RuleSeverity.HIGH, 3),
            ("H13", "实验设计不合格", "检查实验是否形成验证闭环。", "实验缺少对照、核心指标或闭环。", RuleSeverity.WARNING, 1),
            ("H14", "路演叙事断裂", "检查问题-方案-盈利-增长是否连贯。", "问题、方案、盈利、增长之间存在断层。", RuleSeverity.WARNING, 1),
            ("H15", "评分项证据覆盖不足", "检查 Rubric 必需材料是否齐全。", "BP 或附件缺少 Rubric 关键证据。", RuleSeverity.WARNING, 1),
        ]
        return {
            rid: RuleDefinition(rule_id=rid, name=name, description=desc, trigger_condition=cond, default_severity=sev, weight=weight)
            for rid, name, desc, cond, sev, weight in rows
        }

    def check_case(self, case: KnowledgeGraphCase) -> DiagnosisReport:
        results: List[RuleResult] = []
        for rid, definition in self.rule_catalog.items():
            if not is_rule_enabled(rid):
                results.append(
                    RuleResult(
                        rule_id=definition.rule_id,
                        name=definition.name,
                        status=RuleStatus.SKIPPED,
                        severity=RuleSeverity.INFO,
                        message="Rule is disabled by admin switch.",
                        is_triggered=False,
                        evidence_nodes=[],
                        fix_suggestion="Rule disabled; no action required.",
                        details={"rule_enabled": False},
                    )
                )
                continue
            results.append(getattr(self, f"_check_{rid.lower()}")(case, definition))
        return DiagnosisReport(case_id=case.case_id, title=case.title, results=results)

    def verify_all(self, case: KnowledgeGraphCase) -> DiagnosisReport:
        """Backward-compatible alias for running the full H1-H15 verification set."""
        return self.check_case(case)

    def _norm(self, value: Any) -> str:
        return " ".join(str(value or "").strip().lower().split())

    def _contains(self, text: str, words: Iterable[str]) -> bool:
        text = self._norm(text)
        return any(self._norm(word) in text for word in words if self._norm(word))

    def _strings(self, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value] if value.strip() else []
        if isinstance(value, dict):
            result: List[str] = []
            for item in value.values():
                result.extend(self._strings(item))
            return result
        if isinstance(value, (list, tuple, set)):
            result: List[str] = []
            for item in value:
                result.extend(self._strings(item))
            return result
        return [str(value)]

    def _project(self, case: KnowledgeGraphCase) -> Optional[Project]:
        return case.get_project()

    def _meta(self, case: KnowledgeGraphCase, *keys: str) -> Any:
        project = self._project(case)
        meta = project.metadata if project else {}
        for key in keys:
            if key in meta:
                return meta[key]
        return None

    def _meta_list(self, case: KnowledgeGraphCase, *keys: str) -> List[Any]:
        value = self._meta(case, *keys)
        return value if isinstance(value, list) else ([] if value is None else [value])

    def _node_text(self, node: BaseNode) -> str:
        extra = []
        for attr in ("aliases", "metric_key", "source", "expected_output", "artifact_type", "stage"):
            extra.extend(self._strings(getattr(node, attr, None)))
        return self._norm(" ".join([node.name, node.description, " ".join(node.tags), " ".join(extra), " ".join(self._strings(node.metadata))]))

    def _relation_text(self, relation: Relation) -> str:
        return self._norm(" ".join([relation.relation_type.value, relation.description, " ".join(relation.evidence_ids), " ".join(self._strings(relation.metadata))]))

    def _case_text(self, case: KnowledgeGraphCase) -> str:
        project = self._project(case)
        parts = [case.title, case.description]
        if project:
            parts.extend([project.name, project.description, project.stage or "", " ".join(self._strings(project.metadata))])
        parts.extend(self._node_text(node) for node in case.nodes)
        parts.extend(self._relation_text(rel) for rel in case.relations)
        parts.extend(self._norm(" ".join([edge.name, edge.description, " ".join(edge.participants.keys()), " ".join(self._strings(edge.metadata))])) for edge in case.hyperedges)
        return self._norm(" ".join(parts))

    def _nodes(self, case: KnowledgeGraphCase, words: Iterable[str]) -> List[BaseNode]:
        return [node for node in case.nodes if self._contains(self._node_text(node), words)]

    def _relations(self, case: KnowledgeGraphCase, relation_type: RelationType) -> List[Relation]:
        return [relation for relation in case.relations if relation.relation_type == relation_type]

    def _metric_ids(self, case: KnowledgeGraphCase, *keys: str) -> List[str]:
        return [metric.id for key in keys if (metric := case.get_metric(key)) is not None]

    def _num_meta(self, case: KnowledgeGraphCase, *keys: str) -> Optional[float]:
        for key in keys:
            value = self._meta(case, key)
            if isinstance(value, (int, float)):
                return float(value)
            if value is not None and (match := re.search(r"(\\d+(?:\\.\\d+)?)", str(value))):
                return float(match.group(1))
        return None

    def _adjacency(self, case: KnowledgeGraphCase) -> Dict[str, Set[str]]:
        graph: Dict[str, Set[str]] = defaultdict(set)
        for rel in case.relations:
            graph[rel.source_id].add(rel.target_id)
            graph[rel.target_id].add(rel.source_id)
        for edge in case.hyperedges:
            ids = list(edge.node_ids)
            for source in ids:
                for target in ids:
                    if source != target:
                        graph[source].add(target)
        return graph

    def _component(self, graph: Dict[str, Set[str]], start: str) -> Set[str]:
        seen: Set[str] = set()
        queue: deque[str] = deque([start])
        while queue:
            node_id = queue.popleft()
            if node_id in seen:
                continue
            seen.add(node_id)
            queue.extend(graph.get(node_id, set()) - seen)
        return seen

    def _pass(self, d: RuleDefinition, msg: str, evidence: Sequence[str], details: Optional[Dict[str, Any]] = None) -> RuleResult:
        return RuleResult(rule_id=d.rule_id, name=d.name, status=RuleStatus.PASSED, severity=RuleSeverity.INFO, message=msg, is_triggered=False, evidence_nodes=list(dict.fromkeys(evidence)), fix_suggestion="保持当前结构，并继续补一条最强证据。", details=details or {})

    def _warn(self, d: RuleDefinition, msg: str, evidence: Sequence[str], fix: str, details: Optional[Dict[str, Any]] = None) -> RuleResult:
        return RuleResult(rule_id=d.rule_id, name=d.name, status=RuleStatus.WARNING, severity=RuleSeverity.WARNING, message=msg, is_triggered=True, evidence_nodes=list(dict.fromkeys(evidence)), fix_suggestion=fix, details=details or {})

    def _fail(self, d: RuleDefinition, msg: str, evidence: Sequence[str], fix: str, details: Optional[Dict[str, Any]] = None) -> RuleResult:
        return RuleResult(rule_id=d.rule_id, name=d.name, status=RuleStatus.FAILED, severity=d.default_severity, message=msg, is_triggered=True, evidence_nodes=list(dict.fromkeys(evidence)), fix_suggestion=fix, details=details or {})

    def _channel_findings(self, case: KnowledgeGraphCase) -> List[Dict[str, Any]]:
        channels = self._strings(self._meta(case, "primary_channels", "channels", "acquisition_channels", "channel_strategy"))
        segments = {self._norm(item) for item in self._strings(self._meta(case, "customer_profiles", "target_segments"))}
        scenarios = {self._norm(item) for item in self._strings(self._meta(case, "target_scenarios"))}
        findings: List[Dict[str, Any]] = []
        if not channels:
            return findings
        chosen = {self._norm(item) for item in channels}
        for card in [node for node in case.nodes if node.__class__.__name__ == "KnowledgeCard"]:
            for item in getattr(card, "industry_benchmarks", {}).get("channel_fit_data", []):
                channel = self._norm(item.get("channel"))
                if channel not in chosen:
                    continue
                card_segments = {self._norm(v) for v in self._strings(item.get("customer_segments"))}
                card_scenarios = {self._norm(v) for v in self._strings(item.get("applicable_scenarios"))}
                findings.append({
                    "card_id": card.id,
                    "channel": item.get("channel", ""),
                    "fit_score": float(item.get("fit_score", 0) or 0),
                    "segment_overlap": not segments or bool(card_segments & segments),
                    "scenario_overlap": not scenarios or bool(card_scenarios & scenarios),
                })
        return findings

    def _check_h1(self, case: KnowledgeGraphCase, d: RuleDefinition) -> RuleResult:
        project = self._project(case)
        customers = self._nodes(case, KW["customer"])
        values = self._nodes(case, KW["value"])
        channels = self._strings(self._meta(case, "primary_channels", "channels", "acquisition_channels"))
        revenues = self._strings(self._meta(case, "revenue_model", "revenue_streams", "pricing_model", "monetization"))
        evidence = ([project.id] if project else []) + [node.id for node in customers[:2] + values[:2]]
        if not customers and not self._meta_list(case, "customer_profiles", "customer_segments"):
            return self._warn(d, "数据不足：缺少明确的目标客户定义。", evidence, "重新定义一个最具体的目标客户画像。")
        if not values and not self._meta_list(case, "value_propositions"):
            return self._warn(d, "数据不足：缺少价值主张说明。", evidence, "补充一句只服务单一客群的价值主张。")
        if not channels:
            return self._warn(d, "数据不足：缺少主渠道策略。", evidence, "补充一个当前阶段的主获客渠道。")
        low_fit = [item for item in self._channel_findings(case) if item["fit_score"] < 0.35 or not item["segment_overlap"]]
        if low_fit:
            return self._fail(d, "渠道与目标客户或价值主张不匹配，商业闭环错位。", evidence + [item["card_id"] for item in low_fit], "重新选择一个能直接触达目标客户的主渠道。", {"mismatched_channels": low_fit})
        text = self._case_text(case)
        social = {self._norm(item) for item in ["社交媒体", "短视频投流", "linkedin", "tiktok", "微博", "小红书", "抖音"]}
        if self._contains(text, {"b2b", "高客单价", "工业设备", "企业服务", "采购"}) and {self._norm(item) for item in channels} & social:
            return self._fail(d, "高客单价或企业采购项目把弱成交渠道当作主渠道。", evidence, "将主渠道改成更贴近真实决策链路的成交渠道。")
        if revenues and self._contains(" ".join(revenues), {"广告", "流量变现", "ads", "advertising"}) and self._contains(text, {"企业服务", "工业设备", "长期售后", "高可靠"}):
            return self._fail(d, "收入模式与价值主张不匹配。", evidence, "把收入模式改成与核心交付价值一致的付费方式。", {"revenue_model": revenues})
        return self._pass(d, "未发现明显的客户-价值-渠道-收入错位。", evidence)

    def _check_h2(self, case: KnowledgeGraphCase, d: RuleDefinition) -> RuleResult:
        project = self._project(case)
        channels = self._strings(self._meta(case, "primary_channels", "channels", "acquisition_channels"))
        evidence = [project.id] if project else []
        if not channels:
            return self._warn(d, "数据不足：缺少渠道说明。", evidence, "补充一个当前最主要的获客渠道。")
        unreachable = [item for item in self._channel_findings(case) if item["fit_score"] < 0.2 or not item["segment_overlap"] or not item["scenario_overlap"]]
        if unreachable:
            return self._fail(d, "选定渠道无法有效覆盖目标客群或场景。", evidence + [item["card_id"] for item in unreachable], "提供目标用户在该渠道活跃的证据，或立即更换主渠道。", {"unreachable_channels": unreachable})
        text = self._case_text(case)
        combos = [({"小学生", "儿童", "未成年人"}, {"linkedin"}), ({"老年", "银发"}, {"discord"}), ({"工业设备", "采购", "企业服务"}, {"短视频投流", "社交媒体"})]
        chosen = {self._norm(item) for item in channels}
        for customer_words, bad_channels in combos:
            if self._contains(text, customer_words) and chosen & {self._norm(item) for item in bad_channels}:
                return self._fail(d, "目标客群与所选渠道的人群分布明显不匹配。", evidence, "改用目标客户真实活跃且可转化的渠道。")
        return self._pass(d, "未发现明确的渠道不可达问题。", evidence)

    def _check_h3(self, case: KnowledgeGraphCase, d: RuleDefinition) -> RuleResult:
        project = self._project(case)
        pricing = self._strings(self._meta(case, "pricing_model", "pricing", "price", "revenue_model", "revenue_streams"))
        evidence_nodes = [node for node in case.nodes if isinstance(node, Evidence)]
        evidence = ([project.id] if project else []) + [node.id for node in evidence_nodes[:3]]
        if not pricing and not self._contains(self._case_text(case), KW["revenue"]):
            return self._warn(d, "数据不足：尚未给出明确的定价或付费模型。", evidence, "补充一个可执行的首版定价模型。")
        pay_evidence = [node for node in evidence_nodes if self._contains(self._node_text(node), KW["payment"])]
        pay_rel = [rel for rel in case.relations if self._contains(self._relation_text(rel), KW["payment"])]
        if not pay_evidence and not pay_rel:
            return self._warn(d, "定价模型缺少支付意愿、价格敏感度或真实访谈证据。", evidence, "补充一份用户支付意愿或价格敏感度测试记录。", {"pricing_model": pricing})
        return self._pass(d, "定价模型已有支付意愿相关证据支撑。", evidence)

    def _check_h4(self, case: KnowledgeGraphCase, d: RuleDefinition) -> RuleResult:
        tam, sam, som = case.get_metric_value("TAM"), case.get_metric_value("SAM"), case.get_metric_value("SOM")
        evidence = self._metric_ids(case, "TAM", "SAM", "SOM")
        missing = [name for name, value in {"TAM": tam, "SAM": sam, "SOM": som}.items() if value is None]
        if missing:
            return self._warn(d, f"数据不足：缺少市场规模校验所需指标 {', '.join(missing)}。", evidence, "补齐 TAM、SAM、SOM 三个数值及各自口径说明。", {"missing_metrics": missing})
        if tam >= sam >= som:
            return self._pass(d, "市场规模口径满足 TAM >= SAM >= SOM。", evidence, {"TAM": tam, "SAM": sam, "SOM": som})
        return self._fail(d, "市场规模口径不一致，未满足 TAM >= SAM >= SOM。", evidence, "重新核算 TAM、SAM、SOM，确保三者是严格子集关系。", {"TAM": tam, "SAM": sam, "SOM": som})

    def _check_h5(self, case: KnowledgeGraphCase, d: RuleDefinition) -> RuleResult:
        project = self._project(case)
        evidences = [node for node in case.nodes if isinstance(node, Evidence)]
        interview_like = [node for node in evidences if self._contains(self._node_text(node), KW["demand"])]
        evidence = ([project.id] if project else []) + [node.id for node in interview_like[:3]]
        if not evidences and not self._relations(case, RelationType.EVIDENCED_BY):
            return self._warn(d, "核心需求描述缺乏访谈、问卷或行为观测证据。", evidence, "补充至少 5 份真实用户访谈或行为证据。")
        weak = [node for node in evidences if node.confidence is not None and float(node.confidence) < 0.5]
        if weak and not interview_like:
            return self._warn(d, "已有证据可信度偏低，仍不足以支撑核心需求判断。", [node.id for node in weak], "补充一份带原话摘录的高可信用户证据。")
        return self._pass(d, "需求描述已有基础用户证据支撑。", evidence)

    def _check_h6(self, case: KnowledgeGraphCase, d: RuleDefinition) -> RuleResult:
        project = self._project(case)
        text = self._case_text(case)
        nodes = self._nodes(case, KW["competitor"])
        evidence = ([project.id] if project else []) + [node.id for node in nodes[:4]]
        if bool(self._meta(case, "competitor_analysis_required")) and not nodes and not self._contains(text, KW["competitor"]):
            return self._warn(d, "项目要求做竞品分析，但当前没有任何结构化竞品材料。", evidence, "建立一个包含直接、间接和隐形替代品的竞品矩阵。", {"competitor_analysis_required": True})
        if self._contains(text, {"没有竞争对手", "no competitor", "first of its kind"}) and not nodes:
            return self._warn(d, "项目把“没有竞争对手”当作结论，竞品口径明显失真。", evidence, "补充一个真实的替代方案对比矩阵。")
        if (nodes or self._contains(text, KW["competitor"])) and not self._contains(text, {"direct", "indirect", "substitute", "直接", "间接", "替代", "隐形替代"}):
            return self._warn(d, "竞品分析存在口径偏差，缺少直接、间接或隐形替代方案。", evidence, "补齐一个包含直接、间接和隐形替代品的对比表。")
        return self._pass(d, "未发现明显的竞品可比性问题。", evidence)

    def _check_h7(self, case: KnowledgeGraphCase, d: RuleDefinition) -> RuleResult:
        project = self._project(case)
        text = self._case_text(case)
        methods = [node for node in case.nodes if isinstance(node, Method)]
        metrics = [node for node in case.nodes if isinstance(node, Metric)]
        evidence = ([project.id] if project else []) + [node.id for node in methods[:2] + metrics[:2]]
        if not self._contains(text, KW["innovation"]):
            return self._pass(d, "当前案例未出现需要验证的强创新宣称。", evidence)
        if not self._meta(case, "technical_route", "technology_path", "研发路线", "patent_status") and not methods and not metrics:
            return self._fail(d, "创新点描述缺乏技术路线、专利状态或量化对比指标支撑。", evidence, "补充一页技术路线图或量化对比指标。")
        return self._pass(d, "创新描述已有基本技术路径或量化指标支撑。", evidence)

    def _check_h8(self, case: KnowledgeGraphCase, d: RuleDefinition) -> RuleResult:
        cac, ltv = case.get_metric_value("CAC"), case.get_metric_value("LTV")
        evidence = self._metric_ids(case, "CAC", "LTV")
        missing = [name for name, value in {"CAC": cac, "LTV": ltv}.items() if value is None]
        if missing:
            return self._warn(d, f"数据不足：缺少单位经济校验所需指标 {', '.join(missing)}。", evidence, "补齐 LTV 和 CAC 两个数值及其计算口径。", {"missing_metrics": missing})
        if cac <= 0:
            return self._warn(d, "数据不足：CAC 小于等于 0，无法判断真实获客成本。", evidence, "重新估算 CAC，并纳入投放、人力和渠道成本。", {"CAC": cac, "LTV": ltv})
        threshold = 3 * cac
        if ltv >= threshold:
            return self._pass(d, "单位经济满足 LTV >= 3 * CAC。", evidence, {"CAC": cac, "LTV": ltv, "threshold": threshold})
        return self._warn(d, "单位经济不成立，当前 LTV 未达到 3 倍 CAC。", evidence, "先把 CAC 降下来或把 LTV 提上去，至少修正其中一个变量。", {"CAC": cac, "LTV": ltv, "threshold": threshold})

    def _check_h9(self, case: KnowledgeGraphCase, d: RuleDefinition) -> RuleResult:
        project = self._project(case)
        stage = self._norm(project.stage) if project and project.stage else ""
        text = self._case_text(case)
        artifacts = [node for node in case.nodes if isinstance(node, Artifact)]
        evidences = [node for node in case.nodes if isinstance(node, Evidence)]
        evidence = ([project.id] if project else []) + [node.id for node in artifacts[:2] + evidences[:2]]
        if not self._contains(text, KW["growth"]):
            return self._pass(d, "未检测到激进扩张表述。", evidence)
        if stage in {"idea", "mvp"} and (not artifacts or not evidences):
            return self._fail(d, "项目尚未完成 MVP 验证，却直接进入大规模扩张叙述。", evidence, "先定义一个单点 MVP 验证里程碑。", {"stage": stage})
        return self._pass(d, "增长规划与当前验证程度未发现明显跳跃。", evidence)

    def _check_h10(self, case: KnowledgeGraphCase, d: RuleDefinition) -> RuleResult:
        project = self._project(case)
        text = self._case_text(case)
        plans = self._strings(self._meta(case, "milestones", "timeline", "roadmap", "delivery_plan"))
        evidence = [project.id] if project else []
        if not plans and not self._contains(text, KW["milestone"]):
            return self._pass(d, "未检测到可判定的里程碑冲突。", evidence)
        methods = [node for node in case.nodes if isinstance(node, Method)]
        expert = self._contains(text, {"phd", "advisor", "专家", "技术合伙人", "资深工程师", "教授"})
        if (match := re.search(r"(\\d+)\\s*(天|周|个月|月)", " ".join(plans) + " " + text)) and int(match.group(1)) <= 2 and self._contains(text, KW["deeptech"]) and not expert:
            return self._fail(d, "里程碑周期与技术难度严重失配，当前计划不可交付。", evidence + [node.id for node in methods[:2]], "把里程碑拆成一个团队当前能力可交付的最小版本。", {"timeline_match": match.group(0)})
        if self._contains(text, KW["deeptech"]) and not expert and not methods:
            return self._warn(d, "技术交付计划偏激进，但团队资源与实现路径说明不足。", evidence, "补充一位关键技术负责人或外部资源来源。")
        return self._pass(d, "里程碑与团队资源未发现明显脱节。", evidence)

    def _check_h11(self, case: KnowledgeGraphCase, d: RuleDefinition) -> RuleResult:
        project = self._project(case)
        text = self._case_text(case)
        evidence = [project.id] if project else []
        if not self._contains(text, KW["compliance_signal"]):
            return self._pass(d, "当前案例未检测到强监管或高伦理风险场景。", evidence)
        controls = self._nodes(case, KW["compliance_control"])
        if not controls and not self._contains(text, KW["compliance_control"]):
            return self._fail(d, "项目涉及 AI、数据隐私或行业准入，但没有任何合规方案说明。", evidence, "补充一条数据合规或行业准入路径说明。")
        return self._pass(d, "已检测到基础合规或准入说明。", evidence + [node.id for node in controls[:2]])

    def _check_h12(self, case: KnowledgeGraphCase, d: RuleDefinition) -> RuleResult:
        project = self._project(case)
        text = self._case_text(case)
        trl = self._num_meta(case, "trl", "TRL", "technology_readiness_level")
        methods = [node for node in case.nodes if isinstance(node, Method)]
        expert = self._contains(text, {"phd", "advisor", "专家", "技术合伙人", "资深工程师", "教授"})
        evidence = ([project.id] if project else []) + [node.id for node in methods[:2]]
        if trl is not None and trl > 7 and not expert:
            return self._fail(d, "技术成熟度目标过高，但团队缺乏匹配的技术专家。", evidence, "引入一位能覆盖核心技术路线的外部技术负责人。", {"TRL": trl})
        if self._contains(text, KW["deeptech"]) and not expert and not self._meta(case, "technical_route", "technology_path", "研发路线") and not methods:
            return self._fail(d, "技术路线与现有资源不匹配，当前缺少可执行实现路径。", evidence, "把技术方案收敛到团队当前能完成的实现路径。")
        return self._pass(d, "技术路线与资源未发现明显失配。", evidence)

    def _check_h13(self, case: KnowledgeGraphCase, d: RuleDefinition) -> RuleResult:
        project = self._project(case)
        stage = self._norm(project.stage) if project and project.stage else ""
        text = self._case_text(case)
        tasks = [node for node in case.nodes if isinstance(node, Task)]
        artifacts = [node for node in case.nodes if isinstance(node, Artifact)]
        metrics = [node for node in case.nodes if isinstance(node, Metric)]
        evidences = [node for node in case.nodes if isinstance(node, Evidence)]
        evidence = ([project.id] if project else []) + [node.id for node in tasks[:1] + artifacts[:1] + metrics[:1] + evidences[:1]]
        if stage not in {"idea", "mvp"} and not self._contains(text, KW["experiment"]):
            return self._pass(d, "当前案例未进入需要严格实验设计的阶段。", evidence)
        closed_loop = int(bool(tasks or artifacts)) + int(bool(metrics)) + int(bool(evidences))
        if closed_loop < 3:
            return self._warn(d, "MVP 或实验设计缺少任务、指标或证据闭环。", evidence, "按“假设-实验-度量-学习”补齐一轮闭环实验。", {"closed_loop_parts": closed_loop})
        if self._contains(text, {"ab test", "实验", "对照"}) and not self._contains(text, KW["control"]):
            return self._warn(d, "实验提法已出现，但缺少对照组或基线说明。", evidence, "补充一个对照组或基线方案。")
        return self._pass(d, "实验设计已具备基础闭环。", evidence)

    def _check_h14(self, case: KnowledgeGraphCase, d: RuleDefinition) -> RuleResult:
        project = self._project(case)
        if project is None:
            return self._warn(d, "数据不足：缺少 Project 主节点。", [], "补充一个 Project 主节点。")
        problem = bool(self._nodes(case, KW["customer"]) or self._meta_list(case, "customer_profiles"))
        solution = bool(self._nodes(case, KW["value"]) or self._meta_list(case, "value_propositions"))
        monetization = bool(self._strings(self._meta(case, "revenue_model", "revenue_streams", "pricing_model", "pricing")) or case.get_metric_value("CAC") is not None or case.get_metric_value("LTV") is not None)
        growth = bool(self._strings(self._meta(case, "primary_channels", "channels", "acquisition_channels")) or any(edge.edge_type == HyperedgeType.VALUE_LOOP_EDGE.value for edge in case.hyperedges))
        missing = [name for name, present in {"问题/客户": problem, "方案": solution, "盈利": monetization, "增长": growth}.items() if not present]
        if missing:
            return self._warn(d, f"路演叙事存在断层，缺少: {', '.join(missing)}。", [project.id], "先补齐叙事链中最先缺失的一环。", {"missing_parts": missing})
        graph = self._adjacency(case)
        connected = self._component(graph, project.id)
        role_nodes = {
            "problem": {node.id for node in self._nodes(case, KW["customer"])},
            "solution": {node.id for node in self._nodes(case, KW["value"])},
            "growth": {node.id for node in self._nodes(case, KW["channel"])},
        }
        broken = [role for role, ids in role_nodes.items() if ids and not (connected & ids)]
        if broken:
            refs = [project.id] + [next(iter(role_nodes[role])) for role in broken if role_nodes[role]]
            return self._warn(d, "问题、方案与增长叙事没有落在同一主线上。", refs, "把断裂最明显的一环连接回 Project 主线。", {"disconnected_roles": broken})
        return self._pass(d, "问题、方案、盈利与增长叙事基本连贯。", [project.id])

    def _check_h15(self, case: KnowledgeGraphCase, d: RuleDefinition) -> RuleResult:
        project = self._project(case)
        artifacts = [node for node in case.nodes if isinstance(node, Artifact)]
        evidences = [node for node in case.nodes if isinstance(node, Evidence)]
        missing: List[str] = []
        if bool(self._meta(case, "competitor_analysis_required")) and not self._nodes(case, KW["competitor"]):
            missing.append("竞品对比表")
        if any(case.get_metric_value(key) is None for key in ("TAM", "SAM", "SOM")):
            missing.append("市场规模测算表")
        cac, ltv = case.get_metric_value("CAC"), case.get_metric_value("LTV")
        if cac is None or ltv is None or cac <= 0:
            missing.append("财务测算表")
        if not evidences:
            missing.append("用户证据包")
        stage = self._norm(project.stage) if project and project.stage else ""
        if stage in {"mvp", "growth"} and not artifacts:
            missing.append("原型或阶段性交付物")
        required = self._strings(self._meta(case, "rubric_required_evidence", "required_attachments"))
        uploaded = {self._norm(item) for item in self._strings(self._meta(case, "attachments", "uploaded_files"))}
        missing.extend(item for item in required if self._norm(item) not in uploaded)
        if missing:
            refs = ([project.id] if project else []) + [node.id for node in artifacts[:2] + evidences[:2]]
            items = list(dict.fromkeys(missing))
            return self._warn(d, "Rubric 要求的关键证据尚未覆盖完整。", refs, f"优先补上传 {items[0]}。", {"missing_items": items})
        return self._pass(d, "Rubric 关键证据覆盖完整。", [project.id] if project else [])
