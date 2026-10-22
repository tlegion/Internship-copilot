# agent-harness 数据库内容盘点报告

盘点日期：2026-10-22（Australia/Sydney）  
仓库：`/Users/tonyy/Internship/Project/agent-harness`  
分支：`stage3`  
数据库：`data/internship.db`，`PRAGMA user_version = 4`

本报告盘点数据库实际保存的内容，并对照仓库中的 Seed 源文件。数据库通过 SQLite `mode=ro` 查询。本轮不运行验收测试、真实 API、迁移、Seed 或索引同步；不修改业务代码或数据库。Evidence 仅提供内容摘要，不输出完整私人原文。

## 1. 数据总览

| 数据 | 数量 | 说明 |
|---|---:|---|
| Candidate | 1 | Evidence 中唯一的 `candidate_id` 为 `candidate_001`；没有独立 Candidate 表 |
| Evidence | 10 | 全部属于 `candidate_001`，与当前 Evidence Seed 逐字段一致 |
| Evidence Chunk | 40 | 每条 Evidence 对应 4 个 Chunk |
| Evidence Embedding | 40 | `text-embedding-3-small`，1536 维 |
| Evidence FTS 记录 | 40 | 保存 Chunk ID 和文本 |
| Job | 4 | `job_001` 至 `job_004` |
| Job Chunk / Embedding / FTS 记录 | 0 / 0 / 0 | 对应表存在，但目前没有职位索引数据 |

当前 Evidence Seed 完整；Jobs 则与当前仓库 Seed 不同：Seed 有 10 条，数据库有 4 条，缺少 6 条，现有 4 条也有字段差异。这里的“缺少”指相对 Seed 源文件，不代表这些数据一定应当被补入。

## 2. Candidate Evidence 明细

以下 10 条记录全部属于 `candidate_001`。日期保留数据库原始值；`mock_lab` 表示模拟实验材料，不应被当作真实生产或职业经历。

| Evidence ID | Source ID | Category | 日期 | Chunk 数 |
|---|---|---|---|---:|
| `ev_agent_001` | `agent_harness_project` | `ai_application_engineering` | `2026-09_mock_lab` | 4 |
| `ev_async_001` | `async_api_client` | `backend_engineering` | `2026-09_mock_lab` | 4 |
| `ev_backend_001` | `health_alerts_backend` | `backend_engineering` | `2026-02_to_2026-05` | 4 |
| `ev_data_001` | `data_processing_pipeline` | `data_engineering` | `2026-09_mock_lab` | 4 |
| `ev_database_001` | `database_service_project` | `database_engineering` | `2026-09_mock_lab` | 4 |
| `ev_devops_001` | `deployment_pipeline` | `devops` | `2026-09_mock_lab` | 4 |
| `ev_frontend_001` | `web_frontend_project` | `frontend_engineering` | `2026-09_mock_lab` | 4 |
| `ev_ml_001` | `synthetic_face_research` | `machine_learning` | `2026-06_to_present` | 4 |
| `ev_simulation_001` | `service_system_simulation` | `systems_modelling` | `2026-02_to_2026-05` | 4 |
| `ev_testing_001` | `backend_test_suite` | `software_testing` | `2026-09_mock_lab` | 4 |

| Evidence ID | 实际内容摘要 |
|---|---|
| `ev_agent_001` | 模拟 agent harness 对接 OpenAI Responses API，涉及函数调用、有界工具循环、写操作审批、SQLite 和向量检索。 |
| `ev_async_001` | 模拟 asyncio API 客户端，实现限并发、重试、超时、限流和部分失败恢复。 |
| `ev_backend_001` | 为团队健康警报应用实现后端 AI 子系统，使用 Django REST Framework。 |
| `ev_data_001` | 用模拟 CSV/JSON 事件构建 Python ETL，进行规范化、校验、去重、SQLite 加载和运行计数核对。 |
| `ev_database_001` | 模拟数据库服务使用 PostgreSQL 与 SQLite，涉及关系查询、索引优化和事务回滚测试。 |
| `ev_devops_001` | 用 Docker 打包演示 Python API，配置 GitHub Actions，并在模拟 AWS 环境排演部署、健康检查和回滚。该记录不是实际 AWS 生产部署记录。 |
| `ev_frontend_001` | 模拟 React/TypeScript 仪表板整合 REST API、筛选状态、表单校验和前后端字段。 |
| `ev_ml_001` | Honours thesis 研究 AI 生成人脸图像的检测与泛化能力。 |
| `ev_simulation_001` | Python 服务系统离散事件仿真，包含双服务器组、FIFO 队列、任务转向和随机工作负载，并比较服务器分配方案。 |
| `ev_testing_001` | 模拟后端 pytest 套件，覆盖校验、状态转换、外部 API mock、临时 SQLite 集成和回归缺陷。 |

按类别分布：`backend_engineering` 有 2 条，其余 8 类各 1 条。按日期分布：7 条为 `2026-09_mock_lab`，2 条为 `2026-02_to_2026-05`，1 条为 `2026-06_to_present`。摘要描述存储材料的内容，不代表独立核实了现实经历。

## 3. 数据库当前保存的全部 Jobs

`jobs` 表只有 `id`、`title`、`company`、`description`、`location` 五个字段，没有独立的技能要求、薪资、发布日期、申请链接或岗位类别字段。

| ID | 职位名（原值） | 公司（原值） | 地点（原值） | 内容概要 |
|---|---|---|---|---|
| `job_001` | Software Engineering Intern | Demo Company | Sydney | 使用 Python 构建后端 API 和内部工具。 |
| `job_002` | AI Engineering Intern | Demo AI Company | `NULL` | 构建由大型语言模型支持的应用功能。 |
| `job_003` | Backend Engineering Intern | Horizon Labs | Sydney | 开发 Python 后端 API、集成外部服务、编写 SQL、改进开发工具、调试 API 故障和处理结构化数据。 |
| `job_004` | Backend Engineering Intern | Horizon Labs | `NULL` | 与 `job_003` 的描述逐字相同。 |

### 数据库中的职位描述原文

**job_001**

> Build backend APIs and internal tools using Python.

**job_002**

> Build LLM-backed application features

**job_003 和 job_004**（两条记录保存完全相同的描述）

> The intern will work with the engineering team to build Python backend APIs, integrate external services, write SQL queries, and improve internal developer tools. The role involves debugging API failures and working with structured application data.

`job_002`、`job_004` 的地点是 SQL `NULL`，表示未填写，不能推断为远程。`job_003`、`job_004` 的职位名、公司和描述相同，但 ID 和地点不同；现有数据不足以判断是否为有意保留的两条记录。本轮未合并或删除。

## 4. Jobs Seed 全量盘点及数据库差异

源文件：`sql/seed_demo.sql`。以下是 Seed 内容，不能替代上节的数据库实际值。Seed 中的组织名均带有 Demo 标识。

| ID | Seed 职位名 | Seed 公司 | Seed 地点 | Seed 内容概要 | 数据库状态 |
|---|---|---|---|---|---|
| `job_001` | Backend Software Engineering Intern | Demo Harbour Software | Sydney | Python REST API、SQL、请求校验、集成测试和服务调试 | 已有；职位名、公司、描述不同，地点相同 |
| `job_002` | AI Engineering Intern | Demo Harbour AI | Sydney | Python LLM 应用、模型 API 与外部服务整合及响应评估 | 已有；公司、描述、地点不同，职位名相同 |
| `job_003` | Machine Learning Intern | Demo Vision Lab | Melbourne | PyTorch 视觉实验、图像分类评估及变换鲁棒性比较 | 已有；职位名、公司、描述、地点均不同 |
| `job_004` | Data Engineering Intern | Demo Data Works | Sydney | Python/SQL CSV/JSON ETL、校验、去重和报表查询 | 已有；职位名、公司、描述、地点均不同 |
| `job_005` | Platform / DevOps Intern | Demo Cloud Platform | Remote | Docker、GitHub Actions、AWS 部署环境、自动化构建测试和故障排查 | 未入库 |
| `job_006` | Full Stack Engineering Intern | Demo Web Studio | Sydney | React/TypeScript、REST/SQL 集成、状态管理及表单校验 | 未入库 |
| `job_007` | Software Test / QA Automation Intern | Demo Quality Systems | Sydney | pytest、mock 单测、集成/API 校验和回归测试 | 未入库 |
| `job_008` | Systems / Simulation Engineering Intern | Demo Service Analytics | Melbourne | Python 离散事件队列及随机系统仿真、资源分配和统计不确定性分析 | 未入库 |
| `job_009` | AI Application / RAG Intern | Demo Retrieval Systems | Sydney | Python RAG、Embeddings、向量检索、LLM 工具调用和 SQLite | 未入库 |
| `job_010` | Research Engineering Intern | Demo Robust Vision Research | Remote | PyTorch 视觉研究评估管线、鲁棒性实验和可复现实验报告 | 未入库 |

Seed 共 10 个 ID，数据库共 4 个 ID；相对 Seed 缺少 `job_005`、`job_006`、`job_007`、`job_008`、`job_009`、`job_010`，没有额外 ID。不能仅凭本次快照确定差异的历史原因。

## 5. 表结构与索引库存

| 表 | 业务字段 / 结构 | 行数 |
|---|---|---:|
| `jobs` | `id`（主键）、`title`、`company`、`description`、`location`（可空） | 4 |
| `evidence` | `evidence_id`（主键）、`candidate_id`、`source_id`、`category`、`date`（可空）、`text` | 10 |
| `chunks` | `chunk_id`（主键）、`evidence_id`、`chunk_index`、`text`、`content_hash` | 40 |
| `chunk_embeddings` | `content_hash` 与 `embedding_model` 组成主键；另有 `dimensions`、`embedding`（JSON 文本） | 40 |
| `chunk_fts` | FTS5：`chunk_id UNINDEXED`、`text` | 40 |
| `chunks_jobs` | `chunk_id`（主键）、`job_id`、`chunk_index`、`text`、`content_hash` | 0 |
| `chunk_embeddings_jobs` | `content_hash` 与 `embedding_model` 组成主键；另有 `dimensions`、`embedding` | 0 |
| `chunk_fts_jobs` | FTS5：`chunk_id UNINDEXED`、`text` | 0 |

Evidence 索引中有 40 条 `text-embedding-3-small` 向量，维度均为 1536。Job 业务记录存在，但其 Chunk、Embedding 和全文索引都没有记录。SQLite 另有 FTS5 自动维护的内部影子表，不计作额外 Evidence 或 Job；没有其他有数据的业务表。

## 6. Seed 入口及行为

| 入口 / 源文件 | 行为 |
|---|---|
| `scripts/init_db.py` | 完整初始化入口，要求数据库不存在；运行迁移、Job SQL Seed 和 Evidence Seed。不是已有数据库的重复填充入口。 |
| `sql/seed_demo.sql` | 10 条 Job Seed，使用 `INSERT OR IGNORE`；已有同 ID 记录保留原值，不更新，也不删除其他职位。 |
| `scripts/seed_rag.py` | 从 `rag_lab/data/evidence.json` 读取 10 条 Evidence，按 ID upsert；同 ID 字段会更新，不删除其他 Evidence。 |
| `rag_lab/data/embed_text.json` | 未被上述 Seed 入口引用，不属于当前初始化 Seed 数据源。 |
| 索引同步脚本 | 与 Seed 分开；Seed 本身不创建或同步 Chunk、Embedding 和 FTS 记录。 |

Evidence 源与数据库的 `evidence_id` 集合以及 `candidate_id`、`source_id`、`category`、`date`、`text` 字段全部一致：缺失 0、额外 0、字段差异 0。

## 7. 本轮操作记录与待确认数据事项

实际执行的是仓库内 Seed 脚本和源文件的只读检索、SQLite URI `mode=ro` 下的表结构/行数查询、全部 Job 字段读取、Evidence 元数据及内容摘要整理、源数据与数据库字段对照。没有运行单元测试、真实 API、迁移、Seed 或索引同步。唯一新增文件是本报告。

如后续要整理数据，需要先决定：是否让现有 Jobs 对齐这份 10 条 Demo Seed，还是保留目前的职位内容；`job_003` 与 `job_004` 是否分别保留；两条空地点是否补齐；是否建立 Job 检索索引。本轮仅记录这些现状，不自动作出数据变更。
