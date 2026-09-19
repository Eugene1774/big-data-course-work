from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field


class HyperedgeType(str, Enum):
    """超边类型枚举。"""

    VALUE_LOOP_EDGE = "Value_Loop_Edge"
    RISK_PATTERN_EDGE = "Risk_Pattern_Edge"


ParticipantValue = Union[str, List[str]]


class BaseHyperedge(BaseModel):
    """
    超图中的基础超边模型。

    和普通关系不同，超边不是只连两个点，而是可以一次连接多个角色。
    这非常适合表达“市场 + 技术 + 项目”这种多元闭环结构。
    """

    id: str = Field(..., description="超边唯一标识")
    edge_type: str = Field(..., description="超边类型标识，用于 JSON 反序列化")
    name: str = Field(..., description="超边名称")
    description: str = Field(default="", description="超边说明")
    participants: Dict[str, ParticipantValue] = Field(..., description="角色名 -> 节点 id（支持单值或多值）")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="预留扩展字段")
    rule_triggered: str = Field(default="", description="触发规则，例如 H1_BusinessModelConsistency")
    logical_evaluation: str = Field(default="", description="多元闭环逻辑判断结果")
    impact: str = Field(default="", description="风险影响说明")

    class Config:
        extra = "forbid"

    @property
    def node_ids(self) -> List[str]:
        """返回当前超边关联的所有节点 id，便于后续遍历或校验。"""

        node_ids: List[str] = []
        for _, participant_id in self.iter_participant_pairs():
            if participant_id not in node_ids:
                node_ids.append(participant_id)
        return node_ids

    @staticmethod
    def _normalize_participant_values(value: ParticipantValue) -> List[str]:
        if isinstance(value, str):
            normalized = value.strip()
            return [normalized] if normalized else []
        normalized_items: List[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                normalized_items.append(text)
        return normalized_items

    def iter_participant_pairs(self) -> List[Tuple[str, str]]:
        """
        将 participants 统一展平为 (role, node_id) 对，便于导入和校验。
        """
        pairs: List[Tuple[str, str]] = []
        for role, raw_value in self.participants.items():
            for node_id in self._normalize_participant_values(raw_value):
                pairs.append((role, node_id))
        return pairs

    def validate_roles(self) -> None:
        """
        子类用于检查参与者角色是否完整。

        这里故意留成可重写方法，而不是放在父类中写死，
        是因为不同类型的超边，要求的角色名称完全不同。
        """

        return None


class ValueLoopEdge(BaseHyperedge):
    """
    价值闭环超边。

    该超边用于表达：
    - market：目标市场
    - technology：技术能力或实现手段
    - project：具体创业项目
    并通过 fit_score 表示三者之间的匹配程度。
    """

    edge_type: Literal["Value_Loop_Edge"] = Field(default="Value_Loop_Edge", description="固定为 Value_Loop_Edge")
    participants: Dict[str, ParticipantValue] = Field(
        ...,
        description="必须包含 market/technology/project 或 customer/value/channel 三个角色",
    )
    fit_score: float = Field(..., ge=0.0, le=1.0, description="价值闭环拟合度，范围 0 到 1")

    def validate_roles(self) -> None:
        allowed_role_sets = (
            {"market", "technology", "project"},
            {"customer", "value", "channel"},
            {"customer", "value_proposition", "channel"},
        )
        actual_roles = set(self.participants.keys())

        if actual_roles not in allowed_role_sets:
            required_union = {"market", "technology", "project", "customer", "value", "value_proposition", "channel"}
            missing_roles = sorted({"market", "technology", "project"} - actual_roles)
            extra_roles = sorted(actual_roles - required_union)
            raise ValueError(
                "ValueLoopEdge 的 participants 角色集合非法。"
                " 允许集合: {market, technology, project} / {customer, value, channel} / {customer, value_proposition, channel}。"
                f" 缺失角色: {missing_roles}；多余角色: {extra_roles}"
            )

        if not self.iter_participant_pairs():
            raise ValueError("ValueLoopEdge 的 participants 不能为空。")


class RiskPatternEdge(BaseHyperedge):
    """
    风险模式超边。

    该超边用于记录一个失败模式：
    - project：哪个项目或项目类型
    - mistake：出现了什么典型错误
    - outcome：最终带来了什么结果或后果
    可选的 risk_score 用于表示风险强度。
    """

    edge_type: Literal["Risk_Pattern_Edge"] = Field(default="Risk_Pattern_Edge", description="固定为 Risk_Pattern_Edge")
    participants: Dict[str, ParticipantValue] = Field(..., description="必须包含 project、mistake、outcome 三个角色")
    risk_score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="风险评分，范围 0 到 1")

    def validate_roles(self) -> None:
        allowed_role_sets = ({"project", "mistake", "outcome"},)
        actual_roles = set(self.participants.keys())

        if actual_roles not in allowed_role_sets:
            missing_roles = sorted({"project", "mistake", "outcome"} - actual_roles)
            extra_roles = sorted(actual_roles - {"project", "mistake", "outcome"})
            raise ValueError(
                "RiskPatternEdge 的 participants 必须且只能包含 project、mistake、outcome。"
                f" 缺失角色: {missing_roles}；多余角色: {extra_roles}"
            )

        if not self.iter_participant_pairs():
            raise ValueError("RiskPatternEdge 的 participants 不能为空。")


HyperedgeModel = Union[ValueLoopEdge, RiskPatternEdge]


HYPEREDGE_MODEL_MAP = {
    "Value_Loop_Edge": ValueLoopEdge,
    "Risk_Pattern_Edge": RiskPatternEdge,
}


def parse_hyperedge(raw: Dict[str, Any]) -> HyperedgeModel:
    """
    根据 edge_type 将原始 JSON 字典解析为具体的超边模型。

    同时会立即调用 validate_roles，确保参与角色完整无误。
    """

    edge_type = raw.get("edge_type")
    model_class = HYPEREDGE_MODEL_MAP.get(edge_type)
    if model_class is None:
        supported_types = ", ".join(sorted(HYPEREDGE_MODEL_MAP.keys()))
        raise ValueError(f"不支持的超边类型: {edge_type}。支持类型包括: {supported_types}")

    hyperedge = model_class(**raw)
    hyperedge.validate_roles()
    return hyperedge
