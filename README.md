# BigDataModel

基于 `Streamlit + LangGraph + Neo4j` 的多角色创业教学智能体原型。

## 1. 项目概览

当前包含 3 个前台角色与 2 条后端数据链路：

- 学生端：上传项目、规则诊断、Socratic 对话、任务追踪
- 教师端：班级风险看板、个体诊断、干预计划、教师对话
- 教务端：全局监控骨架、规则开关示意
- 文档处理链路：`knowledge_base/raw` PDF 批处理为 `processed_json` 结构化 JSON
- 图谱链路：知识卡 + 项目案例入库 Neo4j，构建超图并执行 H1-H15 结构校验

主入口：

- `main.py`：当前推荐入口（RBAC 多页面）
- `app.py`：旧入口，仅保留迁移提示

## 2. 环境要求

- Python `3.12.x`
- 建议使用独立虚拟环境
- 可选 Neo4j（未配置时可运行部分降级路径）

注意：

- `pytest` 未包含在 `requirements.txt` 中，运行单元测试前需单独安装
- 若环境混装过其他 `langchain` / `gradio` 依赖，建议新建干净虚拟环境

## 3. 快速开始

### 3.1 创建并激活虚拟环境

Windows PowerShell:

```powershell
cd C:\Users\low17\Desktop\bigdatamodel
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

如果 PowerShell 禁止脚本执行：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

### 3.2 安装依赖

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install pytest
```

国内网络可换清华源：

```powershell
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pytest
```

## 4. 环境变量配置

推荐在项目根目录创建 `.env`（也支持直接用系统环境变量）。

最小建议配置：

```dotenv
# LLM（前端对话 / 文本抽取）
SILICONFLOW_API_KEY=your_key
SILICONFLOW_BASE_URL=https://api.siliconflow.com/v1
SILICONFLOW_MODEL=Qwen/Qwen3-8B

# Neo4j（图检索 / 图谱构建）
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j
```

补充说明：

- `batch_processor.py` / `kg_pipeline.py` 会按优先级读取：
  - `LLM_API_KEY` -> `OPENAI_API_KEY` -> `SILICONFLOW_API_KEY`
  - `LLM_BASE_URL` -> `OPENAI_BASE_URL` -> `SILICONFLOW_BASE_URL`
  - `LLM_MODEL` -> `OPENAI_MODEL` -> `SILICONFLOW_MODEL`
- 云端 Neo4j 网络异常时，可额外配置 `NEO4J_RESOLVER_IP` / `NEO4J_RESOLVER_HOST` / `NEO4J_RESOLVER_PORT`

## 5. 运行前端

```powershell
streamlit run main.py
```

启动后可在侧边栏切换：

- 学生端
- 教师端
- 教务端

## 6. 文档批处理（PDF -> JSON）

将 `knowledge_base/raw` 中 PDF 抽取为 `processed_json/*.json`：

```powershell
python batch_processor.py --input-dir .\knowledge_base\raw --output-dir .\processed_json --mode first --max-workers 5
```

常用参数：

- `--mode first|chunk`：仅前 N 字符抽取 / 分块抽取全量
- `--max-chars`：`first` 模式字符上限（默认 3000）
- `--chunk-size`：`chunk` 模式分块大小（默认 4000）
- `--max-total-chars`：总字符上限（默认 40000，传 `-1` 表示不限）

失败任务会写入：`processed_json/_failed_runs_*.jsonl`。

## 7. Neo4j 图谱构建

### 7.1 初始化知识图（卡片 + 案例）

```powershell
python kg_init.py --uri $env:NEO4J_URI --user $env:NEO4J_USER --password $env:NEO4J_PASSWORD --database $env:NEO4J_DATABASE --cards-dir .\knowledge_base\cards --cases-dir .\processed_json
```

需要清库时追加 `--clear`。

### 7.2 增强管线导入（卡片实体对齐 + 关系入库）

```powershell
python kg_pipeline.py --uri $env:NEO4J_URI --user $env:NEO4J_USER --password $env:NEO4J_PASSWORD --database $env:NEO4J_DATABASE --cards-dir .\knowledge_base\cards
```

可选：`--limit N` 仅处理前 N 个卡片文件。

### 7.3 超图导入与 H 规则结构校验

```powershell
python kg_hypergraph.py --uri $env:NEO4J_URI --user $env:NEO4J_USER --password $env:NEO4J_PASSWORD --database $env:NEO4J_DATABASE --data-dir .\processed_json --verify
```

常用附加参数：

- `--risk-keywords ... --risk-limit 5`：导入后查询风险模式
- `--neighbor-node <node_id>`：执行超图相似邻居查询

### 7.4 Bash 一键构建

```bash
./run_kg_build.sh
```

可通过环境变量控制构建行为：

- `INSTALL_DEPS=1` 安装依赖
- `KG_CLEAR_PIPELINE=1` / `KG_CLEAR_HYPERGRAPH=1` 清库
- `KG_VERIFY_HYPERGRAPH=1` 导入后验证
- `KG_PIPELINE_LIMIT=<N>` 限制 pipeline 处理规模

### 7.5 回填超边（已有图数据补全）

```powershell
python backfill_hyperedges.py
```

该脚本读取 `.env` 中 Neo4j 连接配置并执行补全 Cypher。

## 8. 验证与测试

### 8.1 项目结构/智能体/规则自检

```powershell
python validator.py
```

### 8.2 超图骨架与语义一致性检查

```powershell
python validate_hypergraph_skeleton.py --data-dir .\processed_json
```

输出 JSON 报告可追加 `--json`。

### 8.3 单元测试

```powershell
python -m pytest
```

当前测试覆盖：

- `tests/test_rules.py`
- `tests/test_hypergraph_boundary.py`
- `tests/test_ability_report.py`
- `tests/test_student_agent_guardrails.py`

## 9. 关键目录与文件

- `main.py`：多角色导航入口
- `pages/student_portal.py`：学生端页面
- `pages/teacher_portal.py`：教师端页面
- `pages/admin_portal.py`：教务端页面
- `core/graph.py` / `core/router.py` / `core/nodes.py`：多智能体编排与路由
- `rules/checker.py`：H1-H15 规则检查
- `scoring_engine.py`：能力画像与报告
- `kg_init.py` / `kg_pipeline.py` / `kg_hypergraph.py`：图谱构建主链路
- `batch_processor.py`：PDF 批量结构化
- `validator.py` / `validate_hypergraph_skeleton.py`：离线校验工具

## 10. 常见问题

### 10.1 安装慢或依赖冲突

优先使用独立虚拟环境；国内网络可改镜像源。若最终出现 `Successfully installed ...`，通常本项目依赖已就绪。

### 10.2 教师端无数据

教师端依赖日志与学生交互产物。先在学生端完成上传与诊断，再切换教师端查看聚合看板。

### 10.3 未配置 Neo4j 能否运行

可以运行前端与部分逻辑，但图检索、超图推理、证据溯源能力会受限。
