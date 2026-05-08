# 开发日志

## 2026-05-08

### 完成内容

#### 1. 架构升级：从 Ollama 本地 LLM 迁移到云端 MiMo API

- **原因**：本地 Ollama 模型（qwen3.5:2b）效果有限，切换为小米 MiMo 云端大模型
- **改动**：新增 `src/industry_agent/llm/client.py`，封装 OpenAI-compatible API 调用
- **配置**：`config.py` 新增 `llm_api_key`、`llm_base_url`（`api.xiaomimimo.com`）、`llm_model`（`mimo-v2.5-pro`）
- **适配**：MiMo API 使用 `max_completion_tokens`（非 `max_tokens`）、`top_p=0.95`、`frequency_penalty=0`

#### 2. 新增 RAG 检索模块

- `src/industry_agent/rag/embedding.py` — BAAI/bge-small-zh-v1.5 嵌入模型管理，lazy 加载
- `src/industry_agent/rag/vector_store.py` — FAISS 本地向量检索，支持 save/load/build/search
- `src/industry_agent/rag/hybrid_retriever.py` — 混合检索（SQLite + FAISS + RRF 融合，k=60）
- `src/industry_agent/rag/index_builder.py` — 从 chunks.jsonl 构建 FAISS 向量索引，支持 CLI 运行

#### 3. 新增 Agent Skill 框架

- `src/industry_agent/agent/skills/__init__.py` — 技能注册表 `SKILL_REGISTRY`
- `src/industry_agent/agent/skills/retrieval_skill.py` — 检索技能，封装 sqlite/vector/hybrid 三种模式
- `src/industry_agent/agent/skills/image_skill.py` — 图像理解技能，对接云端 vision API
- `src/industry_agent/agent/skills/routing_skill.py` — 路由技能，封装 question_router + 闲聊匹配
- `src/industry_agent/agent/skills/evaluation_skill.py` — 评估技能，支持启发式 + LLM-as-Judge 两种评估

#### 4. 新增 Agent Orchestrator 编排器

- `src/industry_agent/agent/orchestrator.py` — 基于 Skill 调度的模块化编排器
- 流程：路由 → 图像理解 → 检索 → LLM 生成 → 格式化 → 评估
- 兼容现有 `/chat` API 接口，可直接替换 `AgentService`

#### 5. 配置与依赖更新

- `config.py` 新增：embedding_model、vector_index_path、retrieval_mode、llm_api_key/base_url/model
- `requirements.txt` 新增：openai、sentence-transformers、faiss-cpu、langchain 系列

#### 6. Claude Code 配置

- 创建 `CLAUDE.md` — 项目完整技术文档
- 创建 6 个自定义命令：`/rag-debug`、`/kb-build`、`/index-build`、`/api-test`、`/eval`、`/orchestrator-test`
- 配置 `.claude/settings.local.json` 权限白名单
- 创建 memory 文件，记录项目概况

#### 7. README 重写

- 补充完整部署流程（7 步）
- 更新项目结构（新增模块）
- 新增架构图、检索流程图
- 补充 MiMo API 配置说明
- 新增 Claude Code 命令说明

### 验证结果

- [x] MiMo API 连通性测试通过（`mimo-v2.5-pro` 正常响应）
- [x] 闲聊路由正常（"你好" → smalltalk）
- [x] 说明书问答正常（"电钻指示灯闪烁" → 置信度 0.8，6 张配图）
- [x] 客服路由正常（"退货需要准备什么" → customer_service）
- [x] 所有新模块导入正常（无 numpy/faiss 环境下也可导入，lazy 加载）

### 后续计划

- [ ] 安装 sentence-transformers + faiss-cpu，构建 FAISS 向量索引
- [ ] 测试 hybrid 模式下的混合检索效果
- [ ] 用回归测试集对比 sqlite vs hybrid 检索质量
- [ ] 评估 skill 自评功能的实际效果
- [ ] 优化 MiMo 的 system prompt，提升回答质量
