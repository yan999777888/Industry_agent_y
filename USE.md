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

当前版本也支持通过环境变量切换 Ollama 地址和模型：

```bash
export OLLAMA_BASE_URL=http://localhost:11434
export OLLAMA_MODEL=qwen3.5:2b
```

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
    "images": [
      {
        "image_id": "drill0_15",
        "file_name": "drill0_15.png",
        "path": "Knowledge_base/插图/drill0_15.png",
        "exists": true
      }
    ],
    "sources": ["电钻"],
    "references": [
      {
        "chunk_id": "chunk_22326ebfac83",
        "title": "钻孔（图7）",
        "text_snippet": "# 钻孔（图7） 图7 注意：钻薄材料时……",
        "product_name": "电钻",
        "score": "31.2"
      }
    ],
    "confidence": 0.86,
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

当前版本的多轮对话管理已经支持：

- 保存结构化会话状态，而不只是简单历史消息
- 自动继承上一轮识别出的产品名和型号
- 识别“这个”“它”“刚才那个”“还有呢”这类追问表达
- 在检索前自动补全上下文，例如把“这个还有其他尺寸吗？”解析成带产品名的检索查询

例如：

```bash
# 第一轮
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "我想更换健身追踪器的表带"}'

# 第二轮：即使不再重复写产品名，也会自动继承上一轮上下文
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "这个还有其他尺寸吗？", "session_id": "s_xxx"}'
```

如果你想切换到新的产品主题，建议直接使用新的 `session_id`，避免沿用上一轮的产品上下文。

### 示例 4：复杂问题拆解

当前系统已经支持把一次提问中的多个问题拆开后逐项回答，例如：

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "\"请问你们家的商品支持7天无理由退换货吗？\",\n\"需要自己承担运费吗？\""
  }'
```

返回答案会尽量按：

```text
问题1：...
问题2：...
```

的形式组织，内部会先做子问题拆解，再分别检索和生成回答。

---

## 5. 运行测试脚本

```bash
# 端到端冒烟测试（需要先启动 API 服务和 Ollama）
python3 src/industry_agent/agent/test_api.py
```

### 5.1 批量评测 /chat

启动 API 服务后，可以用内置小样例集批量请求 `/chat`：

```bash
python3 scripts/evaluate_chat.py
```

默认结果会保存到：

```text
data/processed/eval_chat_results.jsonl
```

也可以指定服务地址：

```bash
python3 scripts/evaluate_chat.py --base-url http://127.0.0.1:8000
```

如果你有自定义问题列表，可以创建一个每行一个问题的文本文件：

```bash
python3 scripts/evaluate_chat.py --questions docs/my_questions.txt
```

### 5.2 生成平台提交文件

启动 API 服务后，可以读取 `submission/question_public.csv` 并生成提交文件：

```bash
python3 scripts/generate_submission.py
```

默认输入：

```text
submission/question_public.csv
```

默认输出：

```text
submission/submission_generated.csv
```

同时会生成调试日志：

```text
submission/submission_generated_debug.jsonl
```

如果想先小批量试跑，例如只生成前 5 条：

```bash
python3 scripts/generate_submission.py --limit 5
```

如果 API 服务不是默认地址，可以指定：

```bash
python3 scripts/generate_submission.py --base-url http://127.0.0.1:8000
```

生成后的 `submission/submission_generated.csv` 格式为：

```csv
id,ret
1,回答内容
2,回答内容
```

该格式与 `submission/submission_example.csv` 保持一致，可用于提交到测试平台。

---

## 6. 本地直接调用（不经过 HTTP）

```python
from industry_agent.rag.retriever import SQLiteRetriever
from industry_agent.agent.service import AgentService, ChatRequest

agent = AgentService()
resp = agent.chat(ChatRequest(question="洗碗机安装有什么要求？"))
print(resp.answer)
print(resp.image_ids)
print(resp.images)
print(resp.sources)
print(resp.confidence)
```

---

## 7. 当前技术架构

```
用户问题
  │
  ▼
查询解析（产品别名 + 型号识别 + 中文关键词）
  │
  ▼
SQLite LIKE 候选召回 + Python 侧重排
  │
  ▼
证据筛选（低置信拒答 + 同产品过滤 + top chunks）
  │
  ▼
Ollama qwen3.5:2b（think=false，原生 /api/chat）
  │
  ▼
结构化返回（answer + image_ids/images + sources + references + confidence）
```

当前多轮链路补充为：

```
session_id
  │
  ▼
结构化会话状态（current_product / current_models / dialog_summary / history）
  │
  ▼
追问解析（产品继承 + 代词消解 + 多问题拆解）
  │
  ▼
检索与回答生成
```

### 关键文件

| 文件 | 职责 |
|------|------|
| `src/industry_agent/api/app.py` | FastAPI 路由与请求/响应模型 |
| `src/industry_agent/agent/service.py` | Agent 编排：检索 → 组装 → LLM |
| `src/industry_agent/agent/question_splitter.py` | 复杂问题拆解模块 |
| `src/industry_agent/agent/session_store.py` | 结构化会话状态存储 |
| `src/industry_agent/agent/context_manager.py` | 多轮上下文继承与追问解析 |
| `src/industry_agent/rag/retriever.py` | 关键词提取 + SQLite 评分检索 |
| `src/industry_agent/kb/build_index.py` | 知识库构建主流程 |
| `src/industry_agent/kb/parser.py` | 手册解析与文本规范化 |
| `src/industry_agent/kb/chunker.py` | 文本切块策略 |
| `src/industry_agent/kb/index_store.py` | SQLite 索引写入 |
| `src/industry_agent/kb/models.py` | 数据模型定义 |
| `src/industry_agent/config.py` | 项目路径与全局配置 |
