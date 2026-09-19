from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RelationType(str, Enum):
    """知识图谱中的关系类型枚举。"""

    PREREQ = "PREREQ"
    PRODUCES = "PRODUCES"
    MEASURED_BY = "MEASURED_BY"
    COMMON_MISTAKE = "COMMON_MISTAKE"
    EVIDENCED_BY = "EVIDENCED_BY"


class Relation(BaseModel):
    """
    图谱中的二元关系模型。

    说明：
    - source_id 和 target_id 分别指向两个节点的 id。
    - relation_type 约束关系的语义类型。
    - evidence_ids 可以挂接多个证据节点，用来追溯这条关系为什么成立。
    """

    id: str = Field(..., description="关系唯一标识")
    source_id: str = Field(..., description="起点节点 id")
    target_id: str = Field(..., description="终点节点 id")
    relation_type: RelationType = Field(..., description="关系类型")
    description: str = Field(default="", description="关系的自然语言说明")
    weight: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="关系权重，可用于后续推理或打分",
    )
    evidence_ids: List[str] = Field(default_factory=list, description="支撑该关系的证据节点 id 列表")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="预留扩展字段")

    class Config:
        extra = "forbid"
