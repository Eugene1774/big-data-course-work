from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class BaseNode(BaseModel):
    """
    所有知识图谱节点的共同父类。

    设计目的：
    1. 为所有节点提供统一的基础字段，便于后续做存储、检索、展示。
    2. 使用 node_type 保存节点真实类型，方便从 JSON 自动恢复为正确的 Python 类。
    3. metadata 作为扩展口，未来可以挂接课程编号、向量索引 id、标签体系等额外信息。
    """

    id: str = Field(..., description="节点唯一标识，例如 project_001 或 metric_tam")
    node_type: str = Field(..., description="节点类型标识，用于 JSON 反序列化时进行分发")
    name: str = Field(..., description="节点名称")
    description: str = Field(default="", description="节点说明")
    tags: List[str] = Field(default_factory=list, description="便于检索的标签列表")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="预留扩展字段")

    class Config:
        extra = "forbid"


class Concept(BaseNode):
    """概念节点，例如 PMF、TAM、护城河等创业知识点。"""

    node_type: Literal["Concept"] = Field(default="Concept", description="固定为 Concept")
    aliases: List[str] = Field(default_factory=list, description="概念别名，便于做同义词匹配")


class Method(BaseNode):
    """方法节点，例如精益画布、商业模式画布、AARRR 等方法论。"""

    node_type: Literal["Method"] = Field(default="Method", description="固定为 Method")
    steps: List[str] = Field(default_factory=list, description="方法的关键步骤")


class Task(BaseNode):
    """任务节点，例如用户访谈、竞品分析、MVP 测试等具体执行动作。"""

    node_type: Literal["Task"] = Field(default="Task", description="固定为 Task")
    expected_output: Optional[str] = Field(default=None, description="任务预期产出")


class Artifact(BaseNode):
    """产物节点，例如商业计划书、访谈报告、原型图等。"""

    node_type: Literal["Artifact"] = Field(default="Artifact", description="固定为 Artifact")
    artifact_type: Optional[str] = Field(default=None, description="产物类型，如文档、报告、原型")


class Metric(BaseNode):
    """
    指标节点，例如 CAC、LTV、TAM、SAM、SOM。

    这里特别加入 value 字段，原因是当前阶段我们使用 JSON 文件来模拟数据库，
    因此可以直接把测试案例中的指标值放在节点对象中，便于规则引擎读取。
    """

    node_type: Literal["Metric"] = Field(default="Metric", description="固定为 Metric")
    metric_key: Optional[str] = Field(default=None, description="指标键名，例如 TAM、CAC")
    value: Optional[float] = Field(default=None, description="指标数值")
    unit: Optional[str] = Field(default=None, description="指标单位，例如 CNY、percent、users")


class Mistake(BaseNode):
    """典型错误或逻辑谬误节点，例如伪需求、自嗨式创新、拍脑袋估算。"""

    node_type: Literal["Mistake"] = Field(default="Mistake", description="固定为 Mistake")
    symptom: Optional[str] = Field(default=None, description="该错误通常如何表现")


class Evidence(BaseNode):
    """证据节点，用于支撑声明、假设或结论。"""

    node_type: Literal["Evidence"] = Field(default="Evidence", description="固定为 Evidence")
    source: Optional[str] = Field(default=None, description="证据来源，例如访谈、问卷、财务系统")
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="证据可信度，范围 0 到 1",
    )


class KnowledgeCard(BaseNode):
    """
    文献知识卡片节点。

    该模型用于承接老师提供的文献摘要或行业研究结论，
    便于规则层、推理层和后续教学反馈层直接调用。

    典型用途：
    1. 存储某篇论文或行业报告的关键结论摘要。
    2. 标记它更适用于什么场景，例如 B2B SaaS、校园消费、内容社区等。
    3. 保存行业标准数据，例如推荐毛利率区间、CAC 回收周期、复购率基准线等。
    """

    node_type: Literal["KnowledgeCard"] = Field(default="KnowledgeCard", description="固定为 KnowledgeCard")
    card_id: Optional[str] = Field(default=None, description="知识卡片编号，例如 KC_001")
    type: Literal["政策标准", "行业基准", "失败案例", "渠道策略"] = Field(
        default="行业基准",
        description="知识卡片分类，用于教学智能体做不同风格的检索与提示",
    )
    labels: List[str] = Field(default_factory=list, description="知识卡片标签，例如 B2B、SaaS、增长、留存")
    applicable_scenarios: List[str] = Field(default_factory=list, description="适用场景列表")
    industry: Optional[str] = Field(
        default=None,
        description="识别出的所属行业，例如 AI、跨境电商、重工业、教育科技",
    )
    evidence_source: str = Field(
        ...,
        min_length=1,
        description="原文出处或页码定位信息，作为后续评价与追溯的证据链入口",
    )
    industry_benchmarks: Dict[str, Any] = Field(
        default_factory=dict,
        description="行业标准数据，例如 {'CAC': {'median': 120, 'unit': 'CNY/user'}}",
    )


class Project(BaseNode):
    """项目节点，代表学生创业项目主体。"""

    node_type: Literal["Project"] = Field(default="Project", description="固定为 Project")
    stage: Optional[str] = Field(default=None, description="项目阶段，例如 idea、mvp、growth")
    team_name: Optional[str] = Field(default=None, description="团队名称")


NodeModel = Union[
    Concept,
    Method,
    Task,
    Artifact,
    Metric,
    Mistake,
    Evidence,
    KnowledgeCard,
    Project,
]


NODE_MODEL_MAP = {
    "Concept": Concept,
    "Method": Method,
    "Task": Task,
    "Artifact": Artifact,
    "Metric": Metric,
    "Mistake": Mistake,
    "Evidence": Evidence,
    "KnowledgeCard": KnowledgeCard,
    "Project": Project,
}


def parse_node(raw: Dict[str, Any]) -> NodeModel:
    """
    根据 JSON 中的 node_type 字段，将原始字典转换为正确的节点模型。

    这是整个“JSON 模拟数据库”方案的关键函数：
    - JSON 文件天然只有字典，不带 Python 类型信息。
    - 通过 node_type，我们可以知道一条记录到底是 Concept、Metric 还是 KnowledgeCard。
    - 这样后续规则引擎就可以安心使用 isinstance(node, Metric) 之类的逻辑。
    """

    node_type = raw.get("node_type")
    model_class = NODE_MODEL_MAP.get(node_type)
    if model_class is None:
        supported_types = ", ".join(sorted(NODE_MODEL_MAP.keys()))
        raise ValueError(f"不支持的节点类型: {node_type}。支持类型包括: {supported_types}")
    return model_class(**raw)
