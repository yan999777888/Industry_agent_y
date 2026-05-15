# 后续改进计划

> 基于 0.405 分提交数据的实际分析。分析日期：2026-05-14。

## 本次 0.405 提交数据分析

| 指标 | 数值 | 评估 |
|------|------|------|
| 总题目数 | 400 | |
| 平均答案长度 | 676 字符 | 中文 478，英文 902 |
| 含 `<PIC>` 的答案 | 162/400 (40.5%) | **238 题无图** |
| 总 `<PIC>` 标记数 | 325 | 平均 0.8/题 |
| 模型空回复 | 4 题 (IDs 127, 146, 168, 414) | 已修复代码但未重启生效 |
| 英文答案平均长度 | 902 字符 | 比中文长 89%，含大量铺垫 |
| 英文拒绝式回答 | 9 题 | "Based on the available information..." |
| 英文题中文回答 | 1 题 (ID 418) | 语言检测失效 |
| 重度"通常"（≥3次）| 16 题 | CS policy 套话 |
| 重度"建议"（≥3次）| 18 题 | CS policy 套话 |
| 重度"可能"（≥4次）| 11 题 | 模糊表述 |
| 平均耗时 | 49.1 秒/题 | 偏慢 |
| 拒绝/致歉语言 | 11 题 | 已大幅减少 |

---

## P0 — 阻塞性问题（影响面大、修复成本低）

### 1. ~~sentence-transformers 未安装 → 向量搜索完全失效~~ ✅ 已修复

**问题**：服务器环境未安装 `sentence-transformers`，导致：
- bge-m3 语义向量检索返回 **0 条结果**（模型加载抛异常，被 `except Exception: return []` 静默吞掉）
- cross-encoder 重排序导入失败（`factory.py:14` try/except 设 `None`）
- 整个 retrieval 退化为纯关键词匹配（FTS5 + LIKE），丢失跨语言语义理解能力

**验证**：
```
Vector search results: 0  ← 完全失效
Model creation error: sentence-transformers is not installed
```

**修复**：`pip install sentence-transformers`，然后重启服务。

**预期提升**：这是 0.405 分数低的最大单一原因。修复后语义检索恢复，中英文跨语言理解增强，检索命中率应有明显提升。

### 2. 模型空回复 4 题未拦截

**问题**：IDs 127, 168, 414 答案仍为"模型未返回有效回答"（含图片后缀）。代码修复已完成但**服务器未用新代码重启**。

**修复**：重启服务使 `_should_use_extractive_manual_answer` 中的空回复检测生效。

---

## P1 — 高影响修复

### 3. 图片绑定覆盖率：238/400 (59.5%) 无图

**数据**：
- 仅 162/400 答案含 `<PIC>` 标记
- 总标记数 325，平均 0.8/题
- 4 道模型空回复题错误地携带了图片

**根因**：
- 当前逻辑是"检索到 chunk 就绑定图片"，而非"答案提到才绑定"
- 大量 chunk 不含图片标记 → 检索命中无图 chunk → 答案无图
- 166 道英文题（IDs 241-436）覆盖的产品手册可能图片标记稀疏

**方案**：
- 在 `_select_grounded_manual_image_ids` 中分析**答案文本内容**
- 提取答案中提到的操作/部件/步骤，与候选图片的 chunk 文本匹配
- 不只依赖检索结果中的图片，增加"答案驱动"的图片搜索通路
- 对英文手册区域检查图片标记是否完整

**涉及文件**：`service.py`

### 4. 英文拒绝式回答（9 题）

**问题**：9 道英文题答案以 "Based on the available information, there is no..." 开头，冗长地解释为什么找不到信息，然后列出不相关内容。

**示例（ID 243）**：
> "Based on the available information, there is no specific procedure described as a 'battery conversion feature.' However, the standard procedure to prepare the battery for sailing involves..."

**方案**：
- Prompt 增加明确指令："If the reference does not contain the exact information requested, state in ONE sentence what is missing, then provide the closest relevant information. Do NOT explain why information is missing."
- 在 `_final_answer_cleanup` 中检测并截断 "Based on the available" / "I'm sorry" 类开场白
- 更好的方案：检索阶段如果召回不足，直接走 extractive fallback 而非让 LLM 编造铺垫

**涉及文件**：`prompts.py`、`service.py`

### 5. 英文题中文回答（1 题：ID 418）

**问题**：ID 418 的英文问题被用中文回答。

**根因**：语言检测 `_is_english_text` 可能对短查询或含产品名（如 "V-Belt Holder"）的英文题判断失效。

**方案**：检查 ID 418 的原始问题文本，调试语言检测逻辑。

---

## P2 — 质量提升

### 6. CS policy 去套话

**数据**：16 题"通常"≥3、18 题"建议"≥3、11 题"可能"≥4。

**方案**（同前）：
- 逐条审查 `_TOPIC_RULES` 中的 topic overview/materials/timeline/fees
- 去掉冗余限定词，将模糊表述变明确指令
- "建议准备订单号..." → "需要准备订单号..."
- "通常可以享受..." → "享受..."

**涉及文件**：`customer_service_policy.py`

### 7. 短答案问题（4 题 <100 字符）

**数据**：
- IDs 127, 168, 414：模型空回复（P0.2 已修复）
- ID 217：安装烤箱门步骤，46 字符，内容正确但极其简短

**方案**：对 extractive fallback 路径增加"至少抽取 3 句相关原文"的要求。

### 8. 提升答案详细度

**数据**：中文答案平均仅 478 字符（含空格和换行），远低于英文的 902。

**分析**：
- 中文题目多为客服类（短平快回答）
- 也可能是中文检索召回不足导致 LLM 可用的上下文少

**方案**：在 prompt 中强化"尽可能提供所有相关细节"——当前 prompt 已有，但 CS prompt 可能需要加强。

---

## 执行顺序

```
P0.1 安装 sentence-transformers → 恢复语义检索
    ↓
P0.2 重启服务（使所有代码修复生效）
    ↓
重新生成 submission → 获取新分数基线
    ↓
P1.3 图片绑定优化 → 提升多模态分
    ↓
P1.4 英文拒绝式回答 → 提升英文题质量
    ↓
P1.5 语言检测修复 → 消除乱码
    ↓
P2.6 CS policy 去套话 → 消除模板感
    ↓
P2.7 / P2.8 → 润色
```

---

## 本次改动回顾

| 改动 | 文件 | 状态 |
|------|------|------|
| prompt 重写（详细全面、事实优先） | `prompts.py` | 已部署 |
| 长度限制放宽（800/1500） | `service.py` | 已部署 |
| cross-encoder 默认开启 | `factory.py` | 代码已改，**sentence-transformers 未安装** |
| query expansion 默认开启 | `service.py` | 代码已改，待验证效果 |
| 编码修复 | `service.py` | 已部署（0 乱码残留） |
| 语言检测 | `service.py` | 已部署（1/187 英文题漏网） |
| bge-m3 1024维嵌入 | `vector_store.py` | 索引已构建，**查询时未生效** |
| 模型空回复修复 | `service.py` | 代码已改，**服务器未重启** |
