# CLAUDE.md — Industry Agent 项目指南

## 项目概述

**Industry Agent** 是一个多模态客服智能体系统，用于工业产品说明书的 RAG 问答。核心能力：
- 多模态理解：处理用户文字 + 图片输入
- RAG 检索增强：从 21 份产品说明书中精准检索并生成回答
- 多轮对话：会话上下文管理和产品切换
- 路由分流：自动区分说明书问答 vs 客服策略 vs 闲聊

## 技术栈

- Python >=3.10
- FastAPI (API 服务)
- SQLite FTS5 (关键词检索) + FAISS (向量检索，新增)
- Sentence-Transformers / BAAI/bge-small-zh-v1.5 (嵌入模型)
- OpenAI-compatible API (LLM：DeepSeek/Kimi 等)

## 目录结构

```
Industry_Agent/
├── src/industry_agent/
│   ├── config.py                  # 全局配置（路径、模型、检索模式）
│   ├── llm/                       # 云端 LLM 客户端
│   │   └── client.py              # OpenAI-compatible API 封装
│   ├── rag/                       # 检索模块
│   │   ├── retriever.py           # SQLite FTS5 关键词检索 (现有)
│   │   ├── embedding.py           # BGE 嵌入模型管理 (新增)
│   │   ├── vector_store.py        # FAISS 向量检索 (新增)
│   │   ├── hybrid_retriever.py    # 混合检索 RRF (新增)
│   │   └── index_builder.py       # 向量索引构建 (新增)
│   ├── agent/                     # 智能体模块
│   │   ├── service.py             # 主编排逻辑 (现有，Ollama 版)
│   │   ├── orchestrator.py        # 新编排器，skill 调度 (新增)
│   │   ├── skills/                # 技能目录 (新增)
│   │   │   ├── retrieval_skill.py # 检索技能
│   │   │   ├── image_skill.py     # 图像理解技能
│   │   │   ├── routing_skill.py   # 路由技能
│   │   │   └── evaluation_skill.py# 评估技能
│   │   ├── question_router.py     # 问题路由分类
│   │   ├── question_splitter.py   # 复杂问题拆分
│   │   ├── context_manager.py     # 多轮对话上下文
│   │   ├── session_store.py       # 会话状态存储
│   │   ├── image_understanding.py # 图片理解（Base64 + Vision）
│   │   ├── customer_service_policy.py # 客服策略模板
│   │   └── response_formatter.py  # 回答格式化
│   ├── api/                       # API 层
│   │   └── app.py                 # FastAPI /chat /health 端点
│   ├── kb/                        # 知识库构建
│   │   ├── build_index.py         # KB 构建入口
│   │   ├── parser.py              # 手册解析、OCR 修复
│   │   ├── chunker.py             # 文本切分
│   │   ├── index_store.py         # JSONL/SQLite 索引写入
│   │   └── models.py              # 数据模型
│   └── utils/
├── data/processed/kb/             # 已构建的知识库数据
│   ├── chunks.jsonl               # 3800 个知识块（含清洗、语义标签）
│   ├── images.jsonl               # 图片索引
│   ├── index.sqlite               # SQLite FTS5 索引
│   └── vector.index               # FAISS 向量索引 (构建后生成)
├── Knowledge_base/                # 原始数据
│   ├── *.txt                      # 21 份中文产品手册
│   └── 插图/                      # 2608 张配图
├── scripts/                       # 工具脚本
├── tests/                         # 测试
└── requirements.txt
```

## 关键工作流

### 构建向量索引
```bash
cd /Users/mac/Documents/AGENT_RAG/Industry_Agent
PYTHONPATH=src python3 -m industry_agent.rag.index_builder
```

### 运行 API 服务
```bash
PYTHONPATH=src python3 -m uvicorn industry_agent.api.app:create_app --reload
```

### 使用新编排器
```python
from industry_agent.agent.orchestrator import AgentOrchestrator
agent = AgentOrchestrator()
resp = agent.run(question="电钻指示灯闪烁", session_id="test")
```

### 环境变量
```bash
LLM_API_KEY="sk-xxx"           # API 密钥
LLM_BASE_URL="https://api.deepseek.com"  # API 地址
LLM_MODEL="deepseek-chat"      # 模型名
EMBEDDING_MODEL="BAAI/bge-small-zh-v1.5" # 嵌入模型
RETRIEVAL_MODE="hybrid"        # sqlite/vector/hybrid
```

## 核心设计模式

### 检索流程
1. 查询分析 → 提取产品名、型号、关键词 (`analyze_query`)
2. 多路召回 → SQLite LIKE + FTS5 + FAISS 向量 (`HybridRetriever`)
3. RRF 融合 → Reciprocal Rank Fusion 合并排序
4. 证据过滤 → 产品对齐 + 语义类型匹配 + 分数阈值

### Skill 架构
每个 Skill 继承 `BaseSkill`，实现 `execute(**kwargs) -> SkillResult`：
- `retrieval` — 封装 HybridRetriever，统一检索接口
- `image` — 封装 ImageUnderstander，分析上传图片
- `routing` — 封装 QuestionRouter，问题分类分流
- `evaluation` — 回答质量自评（忠实度、相关性、完整性）

### 回答评分标准（竞赛要求）
- 1分：未回应，结构混乱
- 2分：部分回应，不完整
- 3分：回应但缺乏深度
- 4分：清晰全面，图文结合
- 5分：详细有深度，图文互补

## 产品覆盖范围

21 类中文产品：VR头显、冰箱、吹风机、电钻、儿童电动摩托车、发电机、功能键盘、健身单车、健身追踪器、烤箱、可编程温控器、空调、空气净化器、蓝牙激光鼠标、摩托艇、人体工学椅、水泵、洗碗机、相机、蒸汽清洁机

英文汇总产品：boat, airfryer, vacuum, lawn mower, snowmobile, motherboard, microwave, pressure cooker, earphone, ereader, fax, grill, toothbrush, coffee machine, landline, camera, television, washing machine

## Chunk 元数据语义类型

`procedure`(操作步骤), `safety_warning`(安全警告), `troubleshooting`(故障排除), `parts_list`(部件清单), `specification`(规格参数), `toc`(目录), `general`(通用)

## 开发注意事项

- 现有 `service.py` 使用 Ollama 本地调用，新的 `llm/client.py` 使用云端 API，两者并存
- 新模块的 numpy/faiss/sentence-transformers 导入都是 lazy 的，未安装时不影响其他功能
- SQLite 检索已有完善的中文分词和语义排序逻辑，不要随意修改 `retriever.py`
- 图片处理流程：Base64 解码 → Vision 模型描述 → 提取视觉特征词 → 辅助检索
