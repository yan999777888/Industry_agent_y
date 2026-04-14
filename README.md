# Industry Agent

本项目用于实现赛题三“具有多模态能力的客服智能体设计”。当前脚手架优先完成知识库清洗、切分和索引，后续将在此基础上补齐多模态理解、RAG 检索、回答生成、`/chat` API 和验证报告。

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
│   └── build_kb.py
├── src/
│   └── industry_agent/
│       ├── __init__.py
│       ├── config.py
│       ├── agent/
│       │   ├── __init__.py
│       │   ├── context_manager.py
│       │   ├── image_understanding.py
│       │   ├── question_splitter.py
│       │   ├── service.py
│       │   └── session_store.py
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
│   └── .gitkeep
├── README.md
├── TASK.md
├── TODO.md
├── pyproject.toml
├── requirements.txt
└── 赛题三附录.pdf
```

## 文件说明

- `Knowledge_base/`：赛题提供的原始知识库，包含说明书文本和配套插图。
- `data/processed/kb/`：知识库清洗、切分、图片关联和索引构建后的产物目录。
- `docs/`：后续技术文档、验证报告、流程图等材料的存放目录。
- `scripts/build_kb.py`：知识库构建入口脚本，负责调用 `src/industry_agent/kb/` 中的处理流程。
- `src/industry_agent/config.py`：项目路径和默认参数配置。
- `src/industry_agent/kb/parser.py`：手册解析、文本标准化、`<PIC>` 占位与图片 ID 对齐处理。
- `src/industry_agent/kb/chunker.py`：按照章节与长度约束切分知识块，生成可用于 RAG 的 chunk。
- `src/industry_agent/kb/models.py`：知识库处理过程中使用的数据模型。
- `src/industry_agent/kb/index_store.py`：将清洗结果写入 JSON、JSONL 和 SQLite 索引。
- `src/industry_agent/kb/build_index.py`：知识库清洗、切分、索引构建主流程。
- `src/industry_agent/rag/retriever.py`：当前的 SQLite 检索器，后续会扩展为混合检索。
- `src/industry_agent/agent/service.py`：客服智能体主编排层，负责检索、对话状态、图片理解结果注入和回答生成。
- `src/industry_agent/agent/question_splitter.py`：复杂问题拆解模块。
- `src/industry_agent/agent/session_store.py`：结构化多轮会话状态存储。
- `src/industry_agent/agent/context_manager.py`：多轮上下文继承、产品补全和追问解析。
- `src/industry_agent/agent/image_understanding.py`：用户上传图片的 Base64 解析、元数据抽取和可选视觉描述。
- `src/industry_agent/api/app.py`：FastAPI 应用入口，目前已提供 `/health` 和 `/chat` 脚手架。
- `tests/`：后续单元测试、检索测试和接口测试目录。
- `TASK.md`：比赛任务说明。
- `TODO.md`：当前开发计划与阶段进度。
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
