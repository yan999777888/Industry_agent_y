# Industry Agent

本项目用于实现赛题三“具有多模态能力的客服智能体设计”。当前已经完成知识库清洗、切分和索引，跑通了说明书问答、图片理解、多轮对话、复杂拆问和轻量客服策略的主链路，并补上了固定回归集与端到端质量观察脚本。

## 环境要求

- Python `3.10` 或更高版本
- Linux / macOS / WSL 环境均可
- 建议使用虚拟环境，避免污染系统 Python

## 环境构建

推荐使用下面这组 Bash 命令完成环境部署：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .
```

推荐在启动服务前显式配置 Ollama 文本模型和视觉模型：

```bash
export OLLAMA_BASE_URL=http://127.0.0.1:11434
export OLLAMA_MODEL=qwen3.5:2b
export OLLAMA_VISION_MODEL=llava-phi3
```

如果后续需要直接启动 API 服务，也可以继续执行：

```bash
uvicorn industry_agent.api.app:create_app --factory --host 0.0.0.0 --port 8000
```

## 项目文件树

下面给出当前项目的核心文件树。为方便阅读，`Knowledge_base/` 中的大量手册和图片文件仅保留结构示意，不逐一展开。

```text
Industry_Agent/
├── Knowledge_base/
│   ├── *.txt
│   └── 插图/
│       ├── *.jpg
│       └── *.png
├── data/
│   └── processed/
│       ├── .gitkeep
│       └── kb/
│           ├── build_summary.json
│           ├── chunks.jsonl
│           ├── images.jsonl
│           ├── index.sqlite
│           └── manuals.json
├── docs/
│   └── .gitkeep
├── scripts/
│   ├── build_kb.py
│   ├── evaluate_chat.py
│   ├── generate_submission.py
│   ├── observe_chat_quality.py
│   └── run_regression_suite.py
├── src/
│   └── industry_agent/
│       ├── __init__.py
│       ├── config.py
│       ├── agent/
│       │   ├── __init__.py
│       │   ├── context_manager.py
│       │   ├── customer_service_policy.py
│       │   ├── image_understanding.py
│       │   ├── question_router.py
│       │   ├── question_splitter.py
│       │   ├── response_formatter.py
│       │   ├── runtime_checks.py
│       │   ├── service.py
│       │   ├── session_store.py
│       │   └── test_api.py
│       ├── api/
│       │   ├── __init__.py
│       │   └── app.py
│       ├── kb/
│       │   ├── __init__.py
│       │   ├── build_index.py
│       │   ├── chunker.py
│       │   ├── index_store.py
│       │   ├── models.py
│       │   └── parser.py
│       ├── rag/
│       │   ├── __init__.py
│       │   └── retriever.py
│       └── utils/
│           └── __init__.py
├── tests/
│   ├── fixtures/
│   │   ├── quality_observation_cases.json
│   │   └── regression_cases.json
│   ├── test_agent_flow.py
│   ├── test_quality_observation.py
│   ├── test_regression_suite.py
│   ├── test_retriever.py
│   ├── test_runtime_checks.py
│   └── test_submission_generation.py
├── README.md
├── TASK.md
├── TODO.md
├── USE.md
├── pyproject.toml
├── requirements.txt
└── 赛题三附录.pdf
```

## 文件说明

- `Knowledge_base/`：赛题提供的原始知识库，包含说明书文本和配套插图。
- `data/processed/kb/`：知识库清洗、切分、图片关联和索引构建后的产物目录。
- `docs/`：后续技术文档、验证报告、流程图等材料的存放目录。
- `scripts/build_kb.py`：知识库构建入口脚本，负责调用 `src/industry_agent/kb/` 中的处理流程。
- `scripts/evaluate_chat.py`：对 `/chat` 做小样例端到端评测。
- `scripts/run_regression_suite.py`：执行固定回归验证集。
- `scripts/observe_chat_quality.py`：执行带分类标签的端到端质量观察，并输出问题分桶统计。
- `scripts/generate_submission.py`：读取测试问题并生成提交样例。
- `src/industry_agent/config.py`：项目路径和默认参数配置。
- `src/industry_agent/kb/parser.py`：手册解析、文本标准化、`<PIC>` 占位与图片 ID 对齐处理。
- `src/industry_agent/kb/chunker.py`：按照章节与长度约束切分知识块，生成可用于 RAG 的 chunk。
- `src/industry_agent/kb/models.py`：知识库处理过程中使用的数据模型。
- `src/industry_agent/kb/index_store.py`：将清洗结果写入 JSON、JSONL 和 SQLite 索引。
- `src/industry_agent/kb/build_index.py`：知识库清洗、切分、索引构建主流程。
- `src/industry_agent/rag/retriever.py`：当前的 SQLite 检索器，后续会扩展为混合检索。
- `src/industry_agent/agent/service.py`：客服智能体主编排层，负责检索、对话状态、图片理解结果注入和回答生成。
- `src/industry_agent/agent/question_router.py`：区分说明书问答、客服问题和寒暄输入。
- `src/industry_agent/agent/question_splitter.py`：复杂问题拆解模块。
- `src/industry_agent/agent/customer_service_policy.py`：轻量客服策略知识与场景化模板。
- `src/industry_agent/agent/response_formatter.py`：回答结构化与输出风格统一。
- `src/industry_agent/agent/runtime_checks.py`：服务启动前检查 Ollama、模型和索引状态。
- `src/industry_agent/agent/session_store.py`：结构化多轮会话状态存储。
- `src/industry_agent/agent/context_manager.py`：多轮上下文继承、产品补全和追问解析。
- `src/industry_agent/agent/image_understanding.py`：用户上传图片的 Base64 解析、元数据抽取和可选视觉描述。
- `src/industry_agent/api/app.py`：FastAPI 应用入口，目前已提供 `/health` 和 `/chat` 脚手架。
- `tests/fixtures/regression_cases.json`：固定回归集。
- `tests/fixtures/quality_observation_cases.json`：端到端质量观察样例。
- `tests/`：单元测试、回归验证和脚本逻辑测试目录。
- `TASK.md`：比赛任务说明。
- `TODO.md`：当前开发计划与阶段进度。
- `USE.md`：更详细的部署、调用和测试说明。
- `pyproject.toml`：项目打包配置。
- `requirements.txt`：当前项目需要安装的第三方依赖。

## 依赖说明

当前知识库构建脚本主要依赖 Python 标准库即可运行；`requirements.txt` 中列出的第三方库主要用于项目安装和 API 脚手架运行：

- `setuptools`：用于本地项目安装与打包。
- `wheel`：用于构建 Python wheel 包。
- `fastapi`：`/chat` RESTful API 服务框架。
- `uvicorn[standard]`：FastAPI 的 ASGI 运行服务。
- `httpx`：访问 Ollama 文本模型和可选视觉模型。
- `pillow`：解析上传图片的尺寸、格式等元数据。

## 构建知识库索引

```bash
python3 scripts/build_kb.py
```

默认输入为 `Knowledge_base/`，默认输出为 `data/processed/kb/`。

生成产物包括：

- `manuals.json`：手册解析与统计信息
- `chunks.jsonl`：可用于 RAG 的文本块，包含关联图片 ID
- `images.jsonl`：图片文件索引与引用情况
- `index.sqlite`：SQLite 检索索引，优先使用 FTS5
- `build_summary.json`：构建摘要、告警和统计指标

## 当前构建结果

当前已经完成首版知识库构建，主要结果如下：

- 已处理 `21` 份手册
- 已生成 `4132` 个知识块
- 已生成首版 `SQLite FTS5` 检索索引
- 目前有少数原始手册存在 `<PIC>` 数量与图片列表数量不一致的情况，这属于源数据问题，已在 `build_summary.json` 中记录告警

## 启动与验证

推荐按下面顺序启动和验证：

```bash
# 1. 构建知识库索引
python3 scripts/build_kb.py

# 2. 确认 Ollama 已准备好文本模型和视觉模型
ollama list

# 3. 启动 API
uvicorn industry_agent.api.app:create_app --factory --host 0.0.0.0 --port 8000
```

服务启动后，可执行：

```bash
# 健康检查
curl http://127.0.0.1:8000/health

# 小样例评测
python3 scripts/evaluate_chat.py

# 固定回归集
python3 scripts/run_regression_suite.py

# 端到端质量观察
python3 scripts/observe_chat_quality.py
```

其中：

- `scripts/run_regression_suite.py` 更适合做“是否回归”的硬校验。
- `scripts/observe_chat_quality.py` 更适合看当前系统在 `manual_rag`、`multimodal`、`customer_service`、`fallback` 等类别上的整体表现，并输出失败分桶。

## 接口概览

当前服务主要暴露两个 HTTP 接口：

| 接口 | 方法 | 作用 |
|------|------|------|
| `/health` | `GET` | 返回索引、Ollama 服务、文本模型和视觉模型的健康检查结果 |
| `/chat` | `POST` | 执行说明书问答、客服策略问答、多轮对话和图片辅助理解 |

`/chat` 请求体字段：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `question` | `string` | 是 | 用户问题，去掉空白后不能为空 |
| `images` | `list[string]` | 否 | Base64 图片列表 |
| `session_id` | `string` | 否 | 多轮对话会话 ID |

`/chat` 成功响应：

- HTTP `200`
- JSON 结构：`code`、`msg`、`data`
- `data` 中包含 `answer`、`session_id`、`image_ids`、`images`、`sources`、`references`、`confidence`、`timestamp`

## 错误码说明

当前常见错误码如下：

| HTTP 状态码 | 典型场景 | 返回示例 |
|------|------|------|
| `400` | 请求参数错误，例如 `question` 为空 | `{"detail": "question must not be empty"}` |
| `500` | 服务内部异常，例如对话编排或模型调用失败 | `{"detail": "chat failed: ... "}` |
| `503` | 依赖不可用，例如索引缺失或启动检查未通过 | `{"detail": "..."}`

说明：

- 错误响应当前沿用 FastAPI 默认的 `detail` 字段。
- `/docs` 中已同步标注 `/chat` 的输入结构和错误响应说明。

## 当前客服策略扩展

当前客服策略除了基础主题识别，还支持“主题内状态细分”。当前重点覆盖：

- 退款：`7天无理由`、`质量问题退款`、`已拆封/已使用`、`退款申请驳回`
- 物流：`物流停滞`、`已签收但未收到`、`改派/送错地址`
- 保修：`在保`、`过保`、`人为损坏`
- 发票：`抬头/税号/邮箱填错`、`已开具后申请重开`
- 安装：`预约安装`、`安装改约`、`师傅爽约`
- 补件：`补寄配件`、`补件申请驳回`

## 质量观察

除了固定回归集，当前还提供带标签的端到端质量观察样例：

- 样例文件：`tests/fixtures/quality_observation_cases.json`
- 运行命令：`python3 scripts/observe_chat_quality.py`
- 输出文件：`data/processed/quality_observation_report.json`

当前会按这些类别统计：

- `smalltalk`
- `manual_rag`
- `multimodal`
- `customer_service`
- `multiturn`
- `mixed`
- `fallback`
- `api_error`

当前会分桶观察这些常见问题：

- `answer_alignment`
- `source_routing`
- `image_binding`
- `low_confidence`
- `http_status`
- `error_detail`
