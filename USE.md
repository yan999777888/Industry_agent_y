# USE

本文档记录当前项目的实际使用方法。后续每次功能更新后，都会同步补充这里的操作说明。

---

## 1. 环境准备

### 1.1 Python 环境

```bash
cd /mnt/i/Industry_agent/Industry_Agent   # 项目根目录
python3 -m venv .venv
source .venv/bin/activate
```

### 1.2 安装依赖

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .
```

### 1.3 Ollama（本地 LLM）

需要预先安装 [Ollama](https://ollama.com/) 并拉取模型：

```bash
# 安装 Ollama（Linux）
curl -fsSL https://ollama.com/install.sh | sh

# 拉取当前使用的模型
ollama pull qwen3.5:2b

# 确认 Ollama 服务已启动（默认端口 11434）
curl http://localhost:11434/api/tags
```

> **注意**：当前使用 `qwen3.5:2b` 进行测试。如需更好的效果，可拉取更大的模型（如 `qwen3:8b`），并修改 `src/industry_agent/agent/service.py` 中的 `OLLAMA_MODEL` 配置。

---

## 2. 构建知识库索引

当原始知识库更新，或需要重新生成检索索引时，执行：

```bash
python3 scripts/build_kb.py
```

构建完成后，索引产物会写到 `data/processed/kb/` 目录：

| 文件 | 说明 |
|------|------|
| `index.sqlite` | SQLite 检索索引（主查询库） |
| `chunks.jsonl` | 所有知识块明细 |
| `images.jsonl` | 图片元数据与关联关系 |
| `manuals.json` | 手册级别汇总信息 |
| `build_summary.json` | 构建摘要（含 warning 信息） |

---

## 3. 启动 API 服务

```bash
uvicorn industry_agent.api.app:create_app --factory --host 0.0.0.0 --port 8000
```

### 健康检查

```bash
curl http://127.0.0.1:8000/health
# 预期返回：{"status":"ok"}
```

---

## 4. 调用 /chat 接口

### 请求格式

```
POST /chat
Content-Type: application/json
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `question` | string | ✅ | 用户的客服问题 |
| `images` | list[string] | ❌ | Base64 编码的图片列表（预留） |
| `session_id` | string | ❌ | 会话 ID，用于多轮对话上下文 |

### 返回格式

```json
{
  "code": 0,
  "msg": "success",
  "data": {
    "answer": "根据参考资料，电钻电池的安装步骤如下……",
    "session_id": "s_a1b2c3d4",
    "image_ids": ["drill0_15", "Manual11_8"],
    "sources": ["电钻"],
    "references": [
      {
        "chunk_id": "chunk_22326ebfac83",
        "title": "钻孔（图7）",
        "text_snippet": "# 钻孔（图7） 图7 注意：钻薄材料时……"
      }
    ],
    "timestamp": 1776137645
  }
}
```

### 示例 1：电钻使用

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "电钻怎么钻孔？"}'
```

### 示例 2：VR 头显安全

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "VR头显佩戴时有什么安全注意事项？"}'
```

### 示例 3：多轮对话

```bash
# 第一轮：获取 session_id
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "电钻的电池怎么安装？"}'

# 第二轮：传入上一轮返回的 session_id
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "充电时有什么注意事项？", "session_id": "s_xxx"}'
```

---

## 5. 运行测试脚本

```bash
# 端到端冒烟测试（需要先启动 API 服务和 Ollama）
python3 src/industry_agent/agent/test_api.py
```

---

## 6. 本地直接调用（不经过 HTTP）

```python
from industry_agent.rag.retriever import SQLiteRetriever
from industry_agent.agent.service import AgentService, ChatRequest

agent = AgentService()
resp = agent.chat(ChatRequest(question="洗碗机安装有什么要求？"))
print(resp.answer)
print(resp.image_ids)
print(resp.sources)
```

---

## 7. 当前技术架构

```
用户问题
  │
  ▼
关键词提取（中文 bigram + ASCII 合并）
  │
  ▼
SQLite LIKE 评分检索（产品名×5 + 标题×2 + 正文×1）
  │
  ▼
Context 组装（top-5 chunks，截断 4000 字）
  │
  ▼
Ollama qwen3.5:2b（think=false，原生 /api/chat）
  │
  ▼
结构化返回（answer + image_ids + sources + references）
```

### 关键文件

| 文件 | 职责 |
|------|------|
| `src/industry_agent/api/app.py` | FastAPI 路由与请求/响应模型 |
| `src/industry_agent/agent/service.py` | Agent 编排：检索 → 组装 → LLM |
| `src/industry_agent/rag/retriever.py` | 关键词提取 + SQLite 评分检索 |
| `src/industry_agent/kb/build_index.py` | 知识库构建主流程 |
| `src/industry_agent/kb/parser.py` | 手册解析与文本规范化 |
| `src/industry_agent/kb/chunker.py` | 文本切块策略 |
| `src/industry_agent/kb/index_store.py` | SQLite 索引写入 |
| `src/industry_agent/kb/models.py` | 数据模型定义 |
| `src/industry_agent/config.py` | 项目路径与全局配置 |
