# BigDataModel AI编程助手提示词

## 项目背景

BigDataModel 是一个基于 `Streamlit + LangGraph + Neo4j` 的多角色创业教学智能体原型。

### 技术栈
- **前端框架**: Streamlit
- **多智能体编排**: LangGraph
- **图数据库**: Neo4j
- **大语言模型**: SiliconFlow API (Qwen/Qwen3-8B)
- **Python版本**: 3.12.x

### 核心功能模块
- `logic_adapter.py`: 核心逻辑适配器，包含图检索、超图推理、苏格拉底式对话
- `scoring_engine.py`: 能力画像与评分引擎，生成五力模型雷达图
- `core/rbac.py`: 基于角色的访问控制
- `core/state.py`: 多智能体状态管理
- `rules/checker.py`: H1-H15 规则检查器
- `kg_hypergraph.py`: 超图构建与验证
- `knowledge_base/cards/`: 知识卡片库 (111张)

### 角色系统
- **学生端 (student)**: 项目上传、规则诊断、Socratic对话、任务追踪
- **教师端 (teacher)**: 班级风险看板、个体诊断、干预计划
- **教务端 (admin)**: 全局监控骨架、规则开关示意

### 关键数据结构
- `AgentState`: 多智能体状态，包含 messages, user_role, active_agent, context_data 等
- `AbilityReport`: 五力模型能力画像，包含雷达图数据、评分表、审计追踪
- `KnowledgeGraphCase`: 项目案例结构，用于规则诊断

### 现有代码规范
1. 类型注解: 使用 Pydantic BaseModel 和 TypedDict
2. 日志规范: 使用 logging 模块，LOGGER = logging.getLogger(__name__)
3. 错误处理: 异常应记录到日志并返回降级结果
4. 配置读取: 优先读取环境变量，其次 Streamlit secrets
5. Neo4j连接: 使用 neo4j_resilience 模块的 ensure_external_driver_schema 和 get_managed_driver

---

## 改造需求

### 1. UI布局：从“模拟切换”转向“分级侧边栏”

**目标**: 解决目前使用 st.selectbox 切换角色导致的功能扩展受限问题

**具体改造**:

1. **侧边栏身份显示**
   - 在 st.sidebar 顶端动态展示当前登录用户的实名信息与角色标签
   - 格式: "学生 - 某某团队负责人" 或 "教师 - 张老师"
   - 使用 st.session_state 中的 user_id 和 role 字段

2. **权限与状态控制**
   - 直接利用 st.session_state["role"] 来控制不同门户页面的可见性
   - 确保 A 端用户无法越权访问 B 端视图
   - 复用 core/rbac.py 中的 enforce_rbac() 函数进行硬校验

3. **极简切换逻辑**
   - 在侧边栏底部设置"切换账号"按钮
   - 通过重置 Session 状态实现伪登录
   - 避开复杂的后端 Token 验证卡点

**参考实现位置**: main.py (当前使用 st.selectbox 切换角色)

---

### 2. 学生管理：解决"零数据"冷启动与画像生成

**目标**: 为学生端添加完善的多维能力画像体系

**具体改造**:

1. **多维能力画像 (Learning Profile)**

   ```python
   # 基础资料管理
   - 年级、专业及兴趣领域（如 AI、乡村振兴等）
   - 存储于 st.session_state["student_profile"]

   # 五力模型雷达图
   - 展示基于 H1-H15 规则评估生成的实时得分
   - 五力: 痛点发现、方案策划、商业建模、资源杠杆、逻辑表达
   - 使用 scoring_engine.py 中的 generate_ability_report() 生成数据
   - 使用 Plotly 渲染雷达图
   ```

2. **冷启动方案**

   ```python
   # 默认模版
   - 预设 default_profile.json，在无历史对话时调用
   - 模版路径: PROJECT_ROOT / "data" / "default_profile.json"

   # 初始归零
   - 若无日志得分，雷达图默认设为 0 或均值
   - 提示文案: "开始对话以解锁画像"
   ```

3. **项目与交互资产库**

   ```python
   # 项目归属
   - 以列表形式展示学生参与的项目及角色（组长/组员）
   - 从 Neo4j 中查询 student_id 关联的项目节点

   # 交互日志溯源
   - 提供对话摘要，方便学生回顾与 AI 教练的思维迭代过程
   - 使用 scoring_engine.py 中的 AuditTrail 结构
   ```

**参考实现位置**:
- scoring_engine.py (能力画像生成)
- core/state.py (AgentState 定义)

---

### 3. 教师端：极简"洞察与干预"中心

**目标**: 为教师端添加班级管理功能，实现"能看、能点、能改"

**具体改造**:

1. **班级全景看板**

   ```python
   # 通过 st.dataframe 展示花名册
   - 列: 学生姓名、平均分、H1-H15规则触发频次
   - 从 Neo4j 或 SQLite 数据库聚合查询

   # 高风险预警
   - 系统自动标红逻辑漏洞严重的项目
   - 如频繁触发"客户-价值主张错位"(H1)、"单位经济不成立"(H8)
   ```

2. **一键干预开关 (Intervention Hub)**

   ```python
   # 教师操作
   - 点击学生旁的按钮下发"教学策略"
   - 例如: 强制要求进行法律合规性压力测试

   # 实现方式
   - 仅修改后端的布尔值或策略标记
   - 使用 st.session_state 或 Neo4j 存储
   - 实时同步至学生端的 AI 话术逻辑中
   ```

**参考实现位置**:
- pages/teacher_portal.py (教师端页面)
- logic_adapter.py (AI话术逻辑)

---

### 4. 教务管理与技术"降级"路径

**目标**: 为教务端添加实用的管理功能，优先采用降级策略

**具体改造**:

1. **数据存储降级**

   ```python
   # 使用本地 JSON 存储用户信息与能力分数
   # 文件路径: PROJECT_ROOT / "data" / "users.json"
   # 格式:
   {
     "user_id": "S001",
     "name": "张三",
     "role": "student",
     "scores": [3, 4, 2, 5, 3],
     "projects": ["项目A", "项目B"]
   }

   # Neo4j 角色定位
   - 仅将其作为"证据库"存储画像属性得分
   - 用于项目教练动态调整"毒舌"程度
   - 不存储敏感信息
   ```

2. **解耦计算与展示**

   ```python
   # 让 validator.py 或 scoring_engine.py 专门负责算分
   # UI 只负责读取并渲染雷达图

   # 数据流
   - scoring_engine.calculate_scores() -> 算分
   - generate_ability_report() -> 生成画像
   - UI 组件 -> 渲染雷达图和数据表格
   ```

3. **教务宏观看板**

   ```python
   # 在后端支撑完成后实现
   - 全校项目的健康度分布
   - 业务漏洞 Top 3 统计
   - 用于指导教学改革

   # 可视化
   - 使用 Plotly 绑制健康度分布图
   - 使用 st.metric 显示关键指标
   ```

**参考实现位置**:
- scoring_engine.py (算分逻辑)
- core/db_init.py (数据库初始化)

---

## 技术约束

1. **必须复用现有模块**: 优先使用 scoring_engine.py, logic_adapter.py, core/rbac.py
2. **类型注解**: 所有新增函数必须包含完整的类型注解
3. **错误处理**: 异常应记录到日志并返回降级结果
4. **Session管理**: 使用 st.session_state 存储用户状态
5. **数据库兼容**: 支持 Neo4j 和 SQLite 两种存储后端
6. **响应式设计**: 页面必须适配不同的屏幕尺寸

## 代码风格要求

1. 使用中文注释
2. 函数文档使用 Pydantic BaseModel
3. 常量定义在模块顶部
4. 优先使用已有工具函数(如 _normalize_text, _clip_score)
5. 日志级别: INFO 用于正常流程, WARNING 用于异常情况, ERROR 用于错误

---

## 示例代码片段

### 侧边栏身份显示
```python
def render_sidebar_header():
    """在侧边栏顶端显示用户身份信息"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("👤 当前用户")
    user_name = st.session_state.get("user_name", "未登录")
    user_role = st.session_state.get("role", "student")
    role_labels = {"student": "学生", "teacher": "教师", "admin": "教务"}
    role_display = role_labels.get(user_role, "未知")
    st.sidebar.markdown(f"**{user_name}** - {role_display}")
    st.sidebar.markdown("---")
```

### 五力模型雷达图
```python
def render_radar_chart(ability_report: AbilityReport):
    """渲染五力模型雷达图"""
    import plotly.graph_objects as go

    radar_data = ability_report.get("radar_chart_data", {})
    fig = go.Figure(data=go.Scatterpolar(
        r=radar_data.get("r", []),
        theta=radar_data.get("theta", []),
        fill='toself'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
        showlegend=False
    )
    st.plotly_chart(fig)
```

### 冷启动默认画像
```python
DEFAULT_PROFILE = {
    "user_id": "",
    "name": "",
    "role": "student",
    "scores": [0, 0, 0, 0, 0],  # 五力模型初始值
    "projects": [],
    "diagnostic_history": []
}

def load_student_profile(user_id: str) -> dict:
    """加载学生画像，支持冷启动"""
    profile_path = PROJECT_ROOT / "data" / "profiles" / f"{user_id}.json"
    if profile_path.exists():
        return json.loads(profile_path.read_text())
    return DEFAULT_PROFILE.copy()
```

---

## 文件路径参考

- 主入口: `main.py`
- 学生端页面: `pages/student_portal.py`
- 教师端页面: `pages/teacher_portal.py`
- 教务端页面: `pages/admin_portal.py`
- 核心模块: `core/{rbac.py, state.py, graph.py, nodes.py, schemas.py}`
- 规则检查: `rules/checker.py`
- 评分引擎: `scoring_engine.py`
- 逻辑适配器: `logic_adapter.py`
- 知识卡片: `knowledge_base/cards/*.json`
- 数据目录: `data/`

---

## 注意事项

1. **不要修改已有函数签名**: 保持向后兼容
2. **优先使用现有工具函数**: 如 _normalize_text, _clip_score, calculate_scores
3. **Neo4j 连接使用现有工具**: 使用 neo4j_resilience 模块的 get_managed_driver
4. **Session 状态键名保持一致**: role, user_id, project_id, chat_history
5. **评分引擎已完整实现**: generate_ability_report() 和 calculate_scores() 可直接调用
