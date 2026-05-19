# 项目模块总表

## 项目概况

当前项目已经完成从原始知识库处理、RAG 检索、多模态输入理解、多轮对话管理到 API 服务封装、测试验证和提交脚本生成的完整闭环。整体上，这不是一个单点问答 Demo，而是一套围绕比赛赛题构建的多模态客服智能体系统。

系统当前默认主链路由 `service` 后端驱动，提供标准 `/chat` 接口；离线侧已经完成知识库构建并落地产物；在线侧已经具备说明书问答、客服问答、多轮追问、图片辅助检索和回答后处理能力；验证侧已经具备回归测试、质量观察和提交质量分析工具。

---

## 模块总表

| 模块名称 | 核心职责 | 主要实现文件 | 当前实现细节 | 当前状态 | 可继续优化点 |
| --- | --- | --- | --- | --- | --- |
| 配置管理模块 | 统一管理路径、模型后端、检索模式和运行参数 | `src/industry_agent/config.py` | 通过环境变量统一配置知识库路径、LLM 后端、嵌入模型、检索模式、DashScope/OpenAI-compatible/Ollama 参数 | 已完成 | 可以补更多运行配置校验和配置模板说明 |
| API 服务模块 | 对外提供标准 RESTful 接口 | `src/industry_agent/api/app.py` | 已实现 `/health` 和 `/chat`；启动时自动进行健康检查并创建智能体实例 | 已完成 | 可继续补充 Bearer Token 校验、请求日志、流式响应能力 |
| 启动检查模块 | 检查依赖、索引和模型可用性 | `src/industry_agent/agent/runtime_checks.py` | 检查 `index.sqlite`、`images.jsonl`、模型配置、API Key、DashScope 向量表等 | 已完成 | 可增加更细粒度的性能检查、网络连通性诊断 |
| 知识库解析模块 | 读取原始说明书、修复文本噪声、绑定图片 | `src/industry_agent/kb/parser.py` | 支持 JSON、JSONL、literal、tail recovery；修复 OCR 粘连；处理 `<PIC>` 与图片 ID 对齐 | 已完成 | 仍可继续加强英文 OCR 清洗、异常格式恢复能力 |
| 知识切块模块 | 将说明书切成适合 RAG 的知识块 | `src/industry_agent/kb/chunker.py` | 按章节、语义类型和长度约束切块；识别 `procedure`、`troubleshooting`、`safety_warning`、`specification` 等类型；对步骤类内容保留 overlap | 已完成 | 可继续优化 chunk 粒度，进一步降低跨段信息割裂 |
| 知识库构建总控模块 | 串联清洗、切块、索引构建与统计 | `src/industry_agent/kb/build_index.py`、`scripts/build_kb.py` | 生成 `manuals.json`、`chunks.jsonl`、`images.jsonl`、`index.sqlite`、`build_summary.json`；统计知识质量 | 已完成 | 可补充更丰富的构建指标导出和构建过程缓存 |
| 索引存储模块 | 将知识块和图片索引写入持久化文件 | `src/industry_agent/kb/index_store.py` | 输出 JSON/JSONL/SQLite 三类索引，支撑后续检索与统计 | 已完成 | 可增加索引版本管理与增量更新能力 |
| Query 分析模块 | 抽取产品名、型号、关键词、短语和同义词 | `src/industry_agent/rag/retriever.py` | 包含较重的 query analysis 逻辑，支持中英文关键词、同义词扩展、产品领域提示 | 已完成 | 仍可继续优化复杂问题下的关键词精炼和英文域判定 |
| 词法检索模块 | 基于 SQLite 执行主检索并对候选重打分 | `src/industry_agent/rag/retriever.py` | 结合关键词、标题命中、产品名、语义类型、英文域信息对 chunk 进行重打分 | 已完成 | 可继续减少跨产品误召回，优化长 query 下的候选排序 |
| 向量检索模块 | 提供稠密向量检索通道 | `src/industry_agent/rag/vector_store.py` | 支持无依赖 hashing embedding、本地 `sentence-transformers`、DashScope embedding 三种模式 | 已完成 | 应进一步收敛主用向量通道，减少不同环境下效果波动 |
| 检索工厂模块 | 根据运行模式装配检索器 | `src/industry_agent/rag/factory.py` | 支持 `sqlite`、`vector`、`hybrid`、DashScope 模式；可叠加 BM25、cross-encoder、reranker | 已完成 | 可继续简化配置分支，统一最优默认策略 |
| 混合检索模块 | 融合词法、向量与重排序能力 | `src/industry_agent/rag/hybrid_retriever.py`、`src/industry_agent/rag/factory.py` | 默认更偏 hybrid；可引入 cross-encoder 或 DashScope reranker 做候选重排 | 基本完成 | 可继续验证不同融合策略对榜单得分的实际收益 |
| Query Expansion 模块 | 生成扩展 query 提升召回 | `src/industry_agent/rag/query_expansion.py` | 在 `service.py` 中作为可选增强路径，对原始 query 做扩展检索 | 已接入 | 需进一步验证扩展 query 是否引入更多噪声 |
| 问题路由模块 | 区分说明书问题与客服问题 | `src/industry_agent/agent/question_router.py` | 依据客服词、说明书词、产品名、型号、how-to 意图做启发式路由 | 已完成 | 路由边界题仍可继续优化，尤其是混合型问题 |
| 复杂问题拆分模块 | 把多问输入拆成多个子问题 | `src/industry_agent/agent/question_splitter.py` | 识别多问句、多子句和引用式提问，输出 `SubQuestion` 序列 | 已完成 | 合并回答时仍可进一步增强自然度与压缩冗余 |
| 多轮上下文管理模块 | 管理追问、上下文继承与话题切换 | `src/industry_agent/agent/context_manager.py` | 基于结构化状态做产品补全、型号继承、上下文重置和话题切换检测 | 已完成 | 可继续提升跨轮混合主题场景下的鲁棒性 |
| 会话状态存储模块 | 存储当前产品、型号、历史问题和摘要 | `src/industry_agent/agent/session_store.py` | 使用内存型状态存储结构，带 TTL、历史裁剪和多字段状态维护 | 已完成 | 可增加持久化会话和多实例共享能力 |
| 图片理解模块 | 解析用户图片并转成检索辅助信号 | `src/industry_agent/agent/image_understanding.py` | 解析 Base64、读取元数据、可选调用视觉模型生成描述，并抽取部件词、状态词、故障词 | 已完成 | 视觉理解质量和图像关键词抽取仍可继续优化 |
| 客服策略模块 | 提供客服场景骨架知识 | `src/industry_agent/agent/customer_service_policy.py` | 基于 topic/scenario 定义退款、物流、发票、售后、投诉等规则化知识 | 已完成 | 仍偏骨架化，需继续扩充真实客服知识颗粒度 |
| 客服知识库模块 | 提供结构化客服条目检索 | `src/industry_agent/agent/customer_service_kb.py`、`src/industry_agent/agent/customer_service_kb_data.json` | 先检索数据文件条目，再融合策略骨架；支持场景化命中和上下文 topic 继承 | 已完成 | 可继续沉淀成更完整的独立知识库，提高覆盖率 |
| Prompt 模块 | 为说明书问答和客服问答构造约束提示词 | `src/industry_agent/agent/prompts.py` | 已对语言一致性、拒答模板、手册口吻、幻觉抑制做显式约束 | 已完成 | 可继续针对高频错误样例做 Prompt 微调 |
| 回答格式化模块 | 清洗 LLM 输出，统一风格 | `src/industry_agent/agent/response_formatter.py` | 去掉 Markdown、结构噪声、模板痕迹，使结果更接近客服风格 | 已完成 | 仍可让说明书回答更贴近“短抽取 + 精准图”的比赛风格 |
| LLM 接入模块 | 统一调用 Ollama 与 OpenAI-compatible 模型 | `src/industry_agent/llm/client.py` | 支持文本对话和图像对话调用，屏蔽不同后端差异 | 已完成 | 可继续增加重试、限流、调用耗时统计和 fallback 策略 |
| 主服务编排模块 | 串联检索、图片理解、路由、生成、后处理 | `src/industry_agent/agent/service.py` | 负责闲聊识别、上下文恢复、子问题拆分、路由、检索候选合并、证据过滤、Prompt 构造、LLM 调用、抽取式 fallback、图片 grounding 和结果合并 | 核心完成 | 仍是最主要优化点，尤其是路由精度、图片 grounding、回答风格和多问覆盖 |
| 可选编排器模块 | 提供 skill 化的可替代编排实现 | `src/industry_agent/agent/orchestrator.py`、`src/industry_agent/agent/skills/` | 将路由、检索、图片理解、自评包装为 skill；保留另一种模块化编排方案 | 已完成但非主链路 | 可继续评估是否替代主链路，或仅保留作实验框架 |
| 检索技能模块 | skill 化封装检索调用 | `src/industry_agent/agent/skills/retrieval_skill.py` | 对检索器调用做统一包装，便于 orchestrator 使用 | 已完成 | 可补更详细的调试信息和错误恢复 |
| 路由技能模块 | skill 化封装问题路由 | `src/industry_agent/agent/skills/routing_skill.py` | 支持小聊识别与客服上下文继承路由 | 已完成 | 可与主服务路由逻辑进一步统一 |
| 评估技能模块 | 对回答做轻量自评 | `src/industry_agent/agent/skills/evaluation_skill.py` | 从 faithfulness、relevancy、completeness 三维做启发式评分 | 已完成 | 目前是轻量启发式，后续可换成更稳定的离线评估器 |
| 提交生成模块 | 批量调用 `/chat` 生成比赛提交 | `scripts/generate_submission.py` | 支持问题读取、回答清洗、`<PIC>` 标记插入、调试日志保存 | 已完成 | 清洗逻辑仍偏重写式，需谨慎避免二次损伤答案 |
| 提交重处理模块 | 对已有提交进行二次修正 | `scripts/reprocess_submission.py` | 面向 debug 日志重构提交文本，尝试提升提交风格与格式 | 已完成 | 风险较高，建议继续向“轻清洗”而不是“重改写”方向收敛 |
| 回归测试模块 | 验证核心能力是否退化 | `scripts/run_regression_suite.py`、`tests/fixtures/regression_cases.json` | 固定样例覆盖说明书问答、多轮对话、英文题、复杂多问等 | 已完成 | 仍可补更多多模态和客服多意图专项用例 |
| 质量观察模块 | 分类观察 `/chat` 的实际表现 | `scripts/observe_chat_quality.py`、`tests/fixtures/quality_observation_cases.json` | 按类别统计漏答、图文错配、来源异常、语言不一致等问题 | 已完成 | 可扩充为更系统的指标报表 |
| 提交质量分析模块 | 分析提交文件的整体风险 | `scripts/analyze_submission_quality.py` | 统计 fallback、答案过长、问题回显、多问题疑似漏答等风险特征 | 已完成 | 可继续增加图片相关性和语义正确性分析能力 |
| 测试模块 | 对知识库、检索、智能体、接口和脚本做验证 | `tests/` | 已覆盖知识清洗、chunk 质量、检索打分、agent 流程、API 集成、提交分析等 | 已完成 | 可继续提升端到端覆盖率和回归样例数量 |

---

## 当前完成度判断

从工程角度看，项目当前已经完成以下关键工作：

1. 已完成原始说明书与插图知识库的清洗、切分、索引构建。
2. 已完成说明书问答、客服问答、多轮对话和图片辅助检索主链路。
3. 已完成 `/chat` API 封装与启动健康检查。
4. 已完成回归测试、质量观察、提交生成和提交质量分析工具。
5. 已形成“知识构建 -> 在线问答 -> 测试验证 -> 提交分析 -> 继续优化”的基本工程闭环。

综合判断，目前项目处于“主系统已成型、质量仍在持续打磨”的阶段。

---

## 优先优化项

结合当前代码实现和项目现状，后续最值得优先推进的优化方向如下：

1. **继续优化路由精度**：尤其是说明书问题与客服问题同时含有重叠关键词的边界场景。
2. **继续增强图片 grounding**：从“相关图片返回”进一步提升到“答案所引用知识点对应图片返回”。
3. **继续扩充客服知识库**：让客服回答从规则骨架进一步走向高覆盖、细粒度的结构化知识。
4. **继续降低英文汇总手册干扰**：优化英文多产品混合场景下的域识别与召回排序。
5. **继续压缩模板化回答痕迹**：让客服答案更像真实客服表达，减少保守措辞与泛化解释。
6. **继续强化多问场景覆盖**：确保复杂多意图输入既答全，也答得自然。
7. **继续沉淀验证结果表格与案例库**：方便比赛技术文档、验证报告与答辩材料复用。

