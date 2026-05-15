# 后续改进计划

> 本次提交生成完成后，根据实际评分和答案质量重新评估优先级。

## 本次改动回顾

| 改动 | 文件 | 预期效果 |
|------|------|----------|
| prompt 重写（详细全面、事实优先） | `prompts.py` | 答案更深、更准、不遗漏子问题 |
| 长度限制回滚（800/1500） | `service.py` | 不截断详细回答 |
| cross-encoder 默认开启 | `factory.py` | 检索结果更相关 |
| query expansion 默认开启 | `service.py` | 多角度检索，召回更全 |
| 编码修复 | `service.py` | 消除乱码 |
| 语言检测 | `service.py` | 英文问题英文回答 |
| bge-m3 1024维嵌入 | `vector_store.py` | 跨语言语义理解增强 |

---

## P0 — 影响面大、成本低

### 1. CS policy 数据去套话

**问题**：`customer_service_policy.py` 的 `_TOPIC_RULES` 每条规则都充满"通常""建议""以平台规则为准"，LLM 读到的是套话骨架，自然产出套话答案。

**方案**：
- 逐条审查 topic overview/materials/timeline/fees 字段
- 去掉冗余的"通常""一般""可能""建议"限定词
- 保留必要的"以平台规则为准"但每 topic 最多 1 次
- 将"建议准备订单号..."改为"需要准备订单号..."
- 将"建议先..."改为"操作步骤：..."

**涉及文件**：`src/industry_agent/agent/customer_service_policy.py`

**风险评估**：低——只是把模糊表述变确定，不改变实际策略内容

### 2. 验证向量索引是否使用 bge-m3

**问题**：不确定 `index.sqlite` 的 `chunk_vectors` 表是否真的用 bge-m3 重建了，可能还是旧的 hashing-ngram。

**方案**：
- 查询 `vector_metadata` 表确认 `embedding_model` 和 `dimensions`
- 如果不是 bge-m3 / 1024，运行 `scripts/build_kb.py` 重建
- 重建耗时约 15-45 分钟（4132 chunks × 1024 dim）

**涉及文件**：`data/processed/kb/index.sqlite`、`scripts/build_kb.py`

**风险评估**：低——重建索引不影响服务，完成后再重启即可

---

## P1 — 特定场景修复

### 3. "找不到信息"答案优化

**问题**：英文场景下 LLM 找不到信息时会输出冗长的道歉+无关信息，如：
> "Based on the available information, there is no description of a 'battery conversion feature' in the provided documentation. Before sailing, the documented battery-related procedures are: 1. ... 2. ..."

**方案**：
- 在 `_build_extractive_manual_answer` 的选句逻辑中，增加对"refusal preamble"的检测和裁剪
- 或在 `_final_answer_cleanup` 中检测以 "Based on the available" / "I'm sorry" / "I don't have" 开头的句子，截断后从第一个实质句子开始
- 更好的方案：在 prompt 里明确"如果确实没有相关信息，直接说'手册未提及X。以下是相关的最接近信息：...'，不要展开解释为什么没有"

**涉及文件**：`service.py`、`prompts.py`

### 4. 图片绑定：从"检索驱动"到"答案驱动"

**问题**（已知问题 #7）：当前图片绑定逻辑是"检索到什么 chunk 就绑定什么图片"，而非"答案提到什么才绑定什么图片"。

**方案**：
- 在 `_select_grounded_manual_image_ids` 中增加答案文本分析
- 提取答案中的关键名词/动词，与候选图片的 chunk 文本进行匹配
- 只绑定与答案内容直接相关的图片

**涉及文件**：`service.py`

---

## P2 — 锦上添花

### 5. CS 知识库独立化

**问题**（已知问题 #8）：`customer_service_kb_data.json` 是从 CS 规则推导的，不是独立的真实客服知识数据源。

**方案**：
- 收集或构造独立的客服知识条目
- 或增强现有条目，使其与 policy rules 有差异化信息
- 目标是让 policy 提供策略骨架，kb 提供补充细节，两者互补而非重复

**涉及文件**：`customer_service_kb_data.json`、`customer_service_kb.py`

### 6. 回归测试集扩展

**问题**（已知问题 #13）：缺少覆盖多意图 + 多模态 + 混合路由的综合回归测试集。

**方案**：
- 从 400 题中筛选出多意图（含子问题）题目
- 加入多模态（带图片）测试用例
- 加入混合路由（同时涉及产品手册和客服策略）题目
- 建立自动化回归流程

**涉及文件**：`tests/fixtures/`、`scripts/run_regression_suite.py`

### 7. Response formatter 自然化

**问题**（已知问题 #12）：当前 formatter 偏模板化，不够自然。

**方案**：
- 进一步简化 `format_manual_answer` 和 `format_customer_service_answer`
- 去掉过度的结构化清理（如强行去"结论："标题）
- 让 LLM 的自然输出直接通过，只做最小必要清理

**涉及文件**：`response_formatter.py`

---

## 执行顺序建议

```
本次提交生成 → 查看实际评分/问题
    ↓
P0.2 验证向量索引 → 确保基础设施正确
    ↓
P0.1 CS policy 去套话 → 消除答案模板感
    ↓
P1.3 "找不到信息"优化 → 减少无效回答
    ↓
P1.4 图片绑定优化 → 提升多模态分数
    ↓
P2.7 Formatter 自然化 → 润色输出
    ↓
P2.5 / P2.6 → 长期建设
```
