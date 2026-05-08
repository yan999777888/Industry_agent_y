# Industry Agent — 多模态客服智能体

基于 RAG（检索增强生成）的工业产品客服智能体系统。支持 21 类中文产品手册 + 英文汇总产品的说明书问答、多模态图片理解、多轮对话、客服策略路由和幻觉抑制。

本项目为赛题三"具有多模态能力的客服智能体设计"参赛作品。

## 技术架构

```
┌─────────────────────────────────────────────────────────┐
│                      用户请求                             │
│              (文字 + Base64 图片 + session_id)            │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│                  Agent Orchestrator                       │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────┐  │
│  │ Routing  │ │  Image   │ │Retrieval │ │ Evaluation │  │
│  │  Skill   │ │  Skill   │ │  Skill   │ │   Skill    │  │
│  └─────┬────┘ └─────┬────┘ └─────┬────┘ └────────────┘  │
│        │            │            │                        │
│   问题分类      图片分析      混合检索        回答自评        │
│  ┌─────▼────┐ ┌─────▼────┐ ┌─────▼────┐                 │
│  │闲聊/客服/ │ │Base64解析│ │SQLite+FAISS                │
│  │说明书问答 │ │视觉描述  │ │ RRF融合   │                 │
│  └──────────┘ └──────────┘ └─────┬────┘                 │
│                                  │                       │
│                           ┌──────▼──────┐                │
│                           │ LLM (MiMo)  │                │
│                           │ 云端 API 调用 │                │
│                           └──────┬──────┘                │
│                                  │                       │
│                           ┌──────▼──────┐                │
│                           │ 格式化 + 配图 │                │
│                           └─────────────┘                │
└──────────────────────────────────────────────────────────┘
```

## 系统要求

- **Python** >= 3.10
- **操作系统**：Linux / macOS / WSL
- **网络**：需访问小米 MiMo API（`api.xiaomimimo.com`）

## 快速部署

### 第一步：克隆项目并创建虚拟环境

```bash
cd /path/to/Industry_Agent

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
```

### 第二步：安装依赖

```bash
pip install -r requirements.txt
pip install -e .
```

核心依赖清单：

| 依赖 | 用途 |
|------|------|
| `openai` | MiMo API 调用（OpenAI-compatible） |
| `fastapi` + `uvicorn` | RESTful API 服务 |
| `sentence-transformers` | BAAI/bge-small-zh-v1.5 中文嵌入模型 |
| `faiss-cpu` | FAISS 本地向量索引 |
| `httpx` | HTTP 客户端 |
| `pillow` | 图片元数据解析 |
| `langchain` | RAG 管道组件（可选） |

### 第三步：配置 API 密钥

编辑 `src/industry_agent/config.py`，或通过环境变量设置：

```bash
# 小米 MiMo API
export LLM_API_KEY="your-api-key"
export LLM_BASE_URL="https://api.xiaomimimo.com/v1"
export LLM_MODEL="mimo-v2.5-pro"

# 嵌入模型（默认即可）
export EMBEDDING_MODEL="BAAI/bge-small-zh-v1.5"

# 检索模式：sqlite / vector / hybrid
export RETRIEVAL_MODE="hybrid"
```

当前 `config.py` 中已预配置 MiMo API，默认值：

```python
llm_api_key: str = os.getenv("LLM_API_KEY", "sk-xxx")
llm_base_url: str = os.getenv("LLM_BASE_URL", "https://api.xiaomimimo.com/v1")
llm_model: str = os.getenv("LLM_MODEL", "mimo-v2.5-pro")
```

### 第四步：构建知识库索引

```bash
# 4a. 构建 SQLite 关键词索引
PYTHONPATH=src python3 scripts/build_kb.py

# 4b. 构建 FAISS 向量索引（首次需下载嵌入模型，约需几分钟）
PYTHONPATH=src python3 -m industry_agent.rag.index_builder
```

构建产物位于 `data/processed/kb/`：

| 文件 | 说明 |
|------|------|
| `chunks.jsonl` | 3800 个知识块（含语义标签、清洗评分） |
| `index.sqlite` | SQLite FTS5 关键词索引 |
| `images.jsonl` | 2608 张配图元数据索引 |
| `vector.index` | FAISS 向量索引 |
| `vector.index.meta.jsonl` | 向量索引对应的 chunk 元数据 |
| `build_summary.json` | 构建统计与告警 |

### 第五步：验证 API 连通性

```bash
PYTHONPATH=src python3 -c "
from industry_agent.llm.client import LLMClient
client = LLMClient()
answer = client.chat([{'role': 'user', 'content': '你好'}])
print(answer)
"
```

预期输出：MiMo 模型的自我介绍回复。

### 第六步：启动 API 服务

```bash
PYTHONPATH=src python3 -m uvicorn industry_agent.api.app:create_app --factory --host 0.0.0.0 --port 8000
```

启动后访问：

- **API 文档**：`http://localhost:8000/docs`
- **健康检查**：`http://localhost:8000/health`

### 第七步：测试问答

```bash
# 健康检查
curl http://127.0.0.1:8000/health

# 说明书问答
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "电钻指示灯闪烁是什么意思？"}'
```

## 项目结构

```
Industry_Agent/
├── CLAUDE.md                         # Claude Code 项目指南
├── README.md                         # 本文档
├── TASK.md                           # 赛题说明
├── TODO.md                           # 开发计划
├── USE.md                            # 详细使用说明
├── pyproject.toml                    # 打包配置
├── requirements.txt                  # 依赖清单
├── 赛题三附录.pdf                      # 赛题附录
│
├── .claude/
│   ├── commands/                     # Claude 自定义命令
│   │   ├── rag-debug.md             # /rag-debug 检索调试
│   │   ├── kb-build.md              # /kb-build 知识库构建
│   │   ├── index-build.md           # /index-build 向量索引构建
│   │   ├── api-test.md              # /api-test 接口测试
│   │   ├── eval.md                  # /eval 质量评估
│   │   └── orchestrator-test.md     # /orchestrator-test 编排器测试
│   └── settings.local.json          # Claude 权限配置
│
├── Knowledge_base/                   # 原始知识库
│   ├── 21份中文产品手册 .txt
│   └── 插图/                        # 2608 张配图 (.jpg/.png)
│
├── data/processed/kb/                # 构建产物
│   ├── chunks.jsonl                  # 知识块 (3800条)
│   ├── index.sqlite                  # SQLite FTS5 索引
│   ├── vector.index                  # FAISS 向量索引
│   ├── images.jsonl                  # 图片索引
│   ├── manuals.json                  # 手册汇总
│   └── build_summary.json            # 构建统计
│
├── src/industry_agent/
│   ├── config.py                     # 全局配置
│   │
│   ├── llm/                          # LLM 云端调用
│   │   ├── __init__.py
│   │   └── client.py                 # MiMo API 客户端
│   │
│   ├── rag/                          # 检索模块
│   │   ├── retriever.py              # SQLite 关键词检索
│   │   ├── embedding.py              # BGE 嵌入模型
│   │   ├── vector_store.py           # FAISS 向量检索
│   │   ├── hybrid_retriever.py       # 混合检索 (RRF)
│   │   └── index_builder.py          # 向量索引构建
│   │
│   ├── agent/                        # 智能体核心
│   │   ├── orchestrator.py           # 编排器 (Skill 调度)
│   │   ├── service.py                # 编排器 (Ollama 版，兼容保留)
│   │   ├── skills/                   # 技能目录
│   │   │   ├── retrieval_skill.py    # 检索技能
│   │   │   ├── image_skill.py        # 图像理解技能
│   │   │   ├── routing_skill.py      # 路由技能
│   │   │   └── evaluation_skill.py   # 评估技能
│   │   ├── question_router.py        # 问题路由分类
│   │   ├── question_splitter.py      # 复杂问题拆分
│   │   ├── context_manager.py        # 多轮对话上下文
│   │   ├── session_store.py          # 会话状态存储
│   │   ├── image_understanding.py    # 图片理解
│   │   ├── customer_service_policy.py# 客服策略模板
│   │   ├── response_formatter.py     # 回答格式化
│   │   └── runtime_checks.py         # 启动健康检查
│   │
│   ├── api/                          # API 层
│   │   └── app.py                    # FastAPI /chat /health
│   │
│   ├── kb/                           # 知识库构建
│   │   ├── build_index.py            # 构建入口
│   │   ├── parser.py                 # 手册解析
│   │   ├── chunker.py                # 文本切分
│   │   ├── index_store.py            # 索引写入
│   │   └── models.py                 # 数据模型
│   │
│   └── utils/
│
├── scripts/                          # 工具脚本
│   ├── build_kb.py                   # 构建知识库
│   ├── evaluate_chat.py              # 批量评测
│   ├── generate_submission.py        # 生成提交文件
│   ├── observe_chat_quality.py       # 质量观察
│   ├── run_regression_suite.py       # 回归测试
│   └── reprocess_submission.py       # 提交后处理
│
├── submission/                       # 提交文件
│   ├── question_public.csv           # 测试问题
│   └── submission_generated.csv      # 生成的答案
│
└── tests/                            # 测试
    ├── fixtures/
    │   ├── regression_cases.json     # 回归测试集
    │   └── quality_observation_cases.json
    └── test_*.py
```

## API 接口

### POST /chat

客服问答接口。

**请求：**

```json
{
  "question": "电钻指示灯闪烁是什么意思？",
  "images": ["base64_string_1", "base64_string_2"],
  "session_id": "user_123"
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `question` | string | 是 | 用户问题 |
| `images` | list[string] | 否 | Base64 图片列表 |
| `session_id` | string | 否 | 多轮对话会话 ID |

**响应：**

```json
{
  "code": 0,
  "msg": "success",
  "data": {
    "answer": "当正确连接电源并插入电池后，红色指示灯持续闪烁表示充电已开始。<PIC>...",
    "session_id": "user_123",
    "image_ids": ["drill_04", "drill_05"],
    "images": [
      {"image_id": "drill_04", "file_name": "drill_04.jpg", "path": "...", "exists": true}
    ],
    "sources": ["电钻"],
    "references": [
      {"chunk_id": "...", "title": "...", "text_snippet": "...", "product_name": "电钻", "score": "31.2"}
    ],
    "confidence": 0.8,
    "timestamp": 1776137645
  }
}
```

### GET /health

健康检查接口，返回知识库索引和模型服务状态。

### 错误码

| HTTP 状态码 | 场景 |
|------------|------|
| 400 | `question` 为空 |
| 500 | 对话编排或模型调用失败 |
| 503 | 知识库索引不存在或依赖不可用 |

## 使用示例

### 说明书问答

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "VR头显佩戴时有什么安全注意事项？"}'
```

### 多轮对话

```bash
# 第一轮
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "健身追踪器怎么充电？"}'

# 第二轮（使用返回的 session_id）
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "充满电需要多久？", "session_id": "s_xxx"}'
```

### 图片辅助问答

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "这个指示灯是什么意思？",
    "images": ["iVBORw0KGgo..."]
  }'
```

### 代码直接调用（不经过 HTTP）

```python
from industry_agent.agent.orchestrator import AgentOrchestrator

agent = AgentOrchestrator()
resp = agent.run(
    question="洗碗机安装有什么要求？",
    session_id="test_001",
)
print(resp.answer)
print(resp.image_ids)
print(resp.confidence)
```

## 检索系统

系统采用三路混合检索：

```
用户查询
  │
  ├─→ SQLite LIKE    ──→ 关键词模糊匹配候选
  ├─→ SQLite FTS5    ──→ BM25 全文检索候选
  └─→ FAISS 向量检索 ──→ BGE 语义相似度候选
         │
         ▼
    RRF 融合排序 (Reciprocal Rank Fusion, k=60)
         │
         ▼
    证据过滤 (产品对齐 + 语义类型匹配 + 分数阈值)
         │
         ▼
    Top-K chunk → 送入 LLM 生成回答
```

### 检索模式配置

```bash
export RETRIEVAL_MODE="sqlite"   # 仅关键词检索（轻量，无需安装 faiss）
export RETRIEVAL_MODE="vector"   # 仅向量检索（语义匹配）
export RETRIEVAL_MODE="hybrid"   # 混合检索（默认，效果最好）
```

### Chunk 语义标签

知识块带有以下语义类型，用于意图对齐排序：

| 类型 | 含义 | 匹配场景 |
|------|------|---------|
| `procedure` | 操作步骤 | "怎么安装"、"如何充电" |
| `safety_warning` | 安全警告 | "安全注意事项"、"危险" |
| `troubleshooting` | 故障排除 | "故障"、"不工作"、"闪烁" |
| `parts_list` | 部件清单 | "包含什么"、"配件" |
| `specification` | 规格参数 | "尺寸"、"重量"、"默认密码" |

## 测试与评估

```bash
# 构建知识库
PYTHONPATH=src python3 scripts/build_kb.py

# 批量评测
PYTHONPATH=src python3 scripts/evaluate_chat.py

# 回归测试
PYTHONPATH=src python3 scripts/run_regression_suite.py

# 质量观察
PYTHONPATH=src python3 scripts/observe_chat_quality.py

# 生成提交文件
PYTHONPATH=src python3 scripts/generate_submission.py
PYTHONPATH=src python3 scripts/generate_submission.py --limit 5   # 小批量试跑

# 单元测试
python3 -m unittest discover -s tests -p 'test_*.py'
```

## Claude Code 自定义命令

项目为 Claude Code 配置了 6 个专用命令：

| 命令 | 用途 |
|------|------|
| `/rag-debug <问题>` | 调试检索质量，查看各路检索结果和排序 |
| `/kb-build` | 构建/重建知识库索引（SQLite + FAISS） |
| `/index-build` | 构建 FAISS 向量索引 |
| `/api-test <问题>` | 测试 /chat 接口问答效果 |
| `/eval <数据>` | 按竞赛标准评估回答质量（1-5 分） |
| `/orchestrator-test` | 测试编排器完整流程 |

## 产品覆盖范围

### 中文产品（21 类）

VR头显、冰箱、吹风机、电钻、儿童电动摩托车、发电机、功能键盘、健身单车、健身追踪器、烤箱、可编程温控器、空调、空气净化器、蓝牙激光鼠标、摩托艇、人体工学椅、水泵、洗碗机、相机、蒸汽清洁机

### 英文汇总产品

boat, airfryer, vacuum, lawn mower, snowmobile, motherboard, microwave, pressure cooker, earphone, ereader, fax, grill, toothbrush, coffee machine, landline, camera, television, washing machine

## 赛题评分标准

| 分数 | 标准 |
|------|------|
| 1 分 | 回答未回应问题，结构混乱或缺失，图片无关或无帮助 |
| 2 分 | 回答部分回应问题，但不完整；结构较弱，图文结合较差 |
| 3 分 | 回答回应了问题，但缺乏深度；结构清晰但可优化 |
| 4 分 | 回答清晰、较为全面；结构逻辑清晰，图片有助于理解 |
| 5 分 | 回答详细、有深度；结构严谨连贯，图片与文本完美互补 |
