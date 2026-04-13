# USE

本文档记录当前项目的实际使用方法。后续每次功能更新后，都会同步补充这里的操作说明。

## 1. 启动环境

```bash
cd /home/lancegan/Datas/Codes/Python/Industry_Agent
source .venv/bin/activate
```

如果尚未安装依赖：

```bash
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .
```

## 2. 构建知识库索引

当原始知识库更新，或需要重新生成检索索引时，执行：

```bash
python3 scripts/build_kb.py
```

构建完成后，索引产物会写到：

- `data/processed/kb/build_summary.json`
- `data/processed/kb/chunks.jsonl`
- `data/processed/kb/images.jsonl`
- `data/processed/kb/index.sqlite`
- `data/processed/kb/manuals.json`

## 3. 运行自动化测试

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```

## 4. 启动 API 服务

```bash
uvicorn industry_agent.api.app:create_app --factory --host 0.0.0.0 --port 8000
```

### 健康检查

```bash
curl http://127.0.0.1:8000/health
```

预期返回：

```json
{"status":"ok"}
```

## 5. 调用 /chat 接口

### 示例 1：健身追踪器表带问题

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "我想更换健身追踪器的表带，有其他尺寸可选吗？",
    "top_k": 5
  }'
```

### 示例 2：电钻指示灯问题

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "我的DCB107或DCB112型号电钻指示灯闪烁时，这些闪烁标识代表什么含义？",
    "top_k": 5
  }'
```

### 返回字段说明

- `answer`：系统生成的最终回答
- `image_ids`：建议返回给前端展示的相关图片 ID
- `sources`：本次回答引用的知识块来源
- `confidence`：当前回答的置信度估计
- `debug`：调试信息，包括查询解析结果和最终选中的知识块

## 6. 本地直接调用智能体

如果不通过 HTTP 接口，也可以直接在 Python 里调用：

```bash
python3 - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path('src').resolve()))

from industry_agent.agent.service import ChatRequest, CustomerServiceAgent

agent = CustomerServiceAgent()
response = agent.chat(ChatRequest(question="洗碗机安装有什么要求？"))
print(response.to_record())
PY
```
