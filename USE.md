# USE

本文档只保留当前项目最常用、最必要的使用方式。

## 1. 当前默认配置

当前代码默认与 `Industry_agent_y` 对齐：

- Agent 后端：`service`
- LLM 后端：`openai_compatible`
- 文本模型：`mimo-v2.5-pro`
- 视觉模型：`mimo-v2.5-pro`
- 嵌入模型：`BAAI/bge-small-zh-v1.5`
- 检索模式：`hybrid`

如果你不改环境变量，系统就按这套默认值运行。

## 2. 环境准备

### 2.1 创建虚拟环境

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2.2 安装依赖

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .
```

## 3. 配置模型

### 3.1 默认方式：云端 API

启动前建议显式设置：

```bash
export INDUSTRY_AGENT_AGENT_BACKEND=service
export INDUSTRY_AGENT_LLM_BACKEND=openai_compatible

export LLM_API_KEY=your-api-key
export LLM_BASE_URL=https://api.xiaomimimo.com/v1
export LLM_MODEL=mimo-v2.5-pro
export LLM_VISION_MODEL=mimo-v2.5-pro

export EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
export RETRIEVAL_MODE=hybrid
```

### 3.2 可选方式：本地 Ollama

如果你想改成本地模型：

```bash
export INDUSTRY_AGENT_AGENT_BACKEND=service
export INDUSTRY_AGENT_LLM_BACKEND=ollama

export OLLAMA_BASE_URL=http://127.0.0.1:11434
export OLLAMA_MODEL=qwen3.5:2b
export OLLAMA_VISION_MODEL=llava-phi3

export EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
export RETRIEVAL_MODE=hybrid
```

如果没装 Ollama，就不要设置成 `ollama`。

## 4. 构建知识库索引

首次运行，或修改了知识库 / 嵌入模型 / 检索配置后，需要重建索引：

```bash
python3 scripts/build_kb.py
```

成功后会生成：

- `data/processed/kb/index.sqlite`
- `data/processed/kb/chunks.jsonl`
- `data/processed/kb/images.jsonl`
- `data/processed/kb/manuals.json`
- `data/processed/kb/english_summary_segments.json`
- `data/processed/kb/build_summary.json`

当前的分块策略是：

- 先按 `#` 章节切分
- 再按内容类型组织 chunk 单元
- `procedure / troubleshooting` 优先按步骤成块
- `specification / parts_list / safety_warning` 优先按行和键值对成块
- chunk 之间会保留轻量 overlap
- 图片标记会尽量和前后说明文字绑定到同一块
- 如果某本手册的 `<PIC>` 数量与真实图片列表严重失配，系统会自动停用顺序硬绑定，避免把错误图片批量挂到错误文本上
- 会过滤只有章节标题、没有正文内容的低价值 chunk
- 会额外修正常见英文 OCR 连字、缺空格和布局码噪声
- 英文汇总手册会补充 `domain_segment_index` / `domain_segment_label`，便于排查跨产品混杂
- 英文汇总手册会先按 section 做产品域分段，再进入 chunk 流程

构建完成后，建议优先检查 `data/processed/kb/build_summary.json` 里的：

- `chunk_quality.chunk_type_counts`
- `chunk_quality.semantic_type_counts`
- `chunk_quality.with_image_ratio`
- `chunk_quality.low_clean_score_ratio`
- `chunk_quality.oversized_chunk_ratio`
- `chunk_quality.heading_only_ratio`
- `chunk_quality.long_title_ratio`
- `manual_quality.top_chunk_manuals`
- `manual_quality.parse_mode_counts`
- `manual_quality.attachment_outliers`
- `english_summary_segments`
- `english_summary_segment_quality`

如果你想抽样检查真实 chunk 内容，可以执行：

```bash
python3 scripts/sample_kb_chunks.py --limit 6
```

例如只看步骤型 chunk：

```bash
python3 scripts/sample_kb_chunks.py --chunk-type procedure --limit 6
```

例如只看英文汇总手册里的相机类 chunk：

```bash
python3 scripts/sample_kb_chunks.py --manual-id 汇总英文手册 --domain-label camera --limit 6
```

只看带图 chunk：

```bash
python3 scripts/sample_kb_chunks.py --has-image yes --limit 6
```

## 5. 启动服务

```bash
uvicorn industry_agent.api.app:create_app --factory --host 0.0.0.0 --port 8000
```

健康检查：

```bash
curl http://127.0.0.1:8000/health
```

如果启动失败，先检查：

- `LLM_API_KEY` / `LLM_BASE_URL` 是否配置正确
- `data/processed/kb/index.sqlite` 是否已经生成
- 如果使用 Ollama，`OLLAMA_MODEL` 和 `OLLAMA_VISION_MODEL` 是否已拉取

## 6. 调用接口

### 6.1 最小请求

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "洗碗机安装有什么要求？"
  }'
```

### 6.2 多轮对话

第一轮：

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "电钻的电池怎么充电？"
  }'
```

返回里会有 `session_id`。第二轮带上它：

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "充电时有什么注意事项？",
    "session_id": "上一轮返回的session_id"
  }'
```

### 6.3 图片问答

接口支持：

- `question`
- `images`
- `session_id`

其中 `images` 是 Base64 字符串数组。

### 6.4 平台客服类问题

下面这类问题会优先走“客服策略”分支，而不是说明书检索：

- 退款 / 退货 / 换货 / 退款到账
- 发票 / 发票类型 / 抬头 / 税号 / 重开发票
- 尺寸更换 / 换大一号 / 换小一号 / 尺寸差价 / 补差价
- 物流 / 催发货 / 签收异常 / 改地址
- 售后 / 保修 / 人为损坏 / 维修费用
- 少件 / 漏发 / 补寄 / 包装破损
- 投诉 / 虚假宣传 / 试用 / 优惠券 / 以旧换新

这些客服问题里，像“更大尺寸 / 补差价 / 乡镇配送 / 返修很久 / 重复扣款 / 已发货改地址”也会被细化到更具体的场景模板。

这类问题现在会优先返回：

- 更贴近场景的客服处理建议，而不是只给规则骨架
- 多意图问题会尽量并行覆盖，例如“发票类型 + 多久收到”

同时，当前路由已经做了一层纠偏：

- 如果问题里有明确产品/型号，并且核心是在问“怎么安装 / 怎么充电 / 指示灯含义 / 默认密码 / 安全注意事项”这类操作说明，系统会优先保留手册检索路线
- 只有明确命中订单、退款、发票、物流、地址、扣款这类平台客服信号时，才会强制走客服分支

- `结论`
- `处理步骤`
- `时效/费用`
- `补充说明`

当前实现不是纯模板直出，而是：

- 先用规则命中客服场景，生成“客服策略骨架”
- 再从独立的客服知识数据文件 `src/industry_agent/agent/customer_service_kb_data.json` 里检索最接近当前场景的客服知识条目
- 如果数据文件里暂时没有覆盖到该场景，再退回到 policy projection 生成的补充知识条目
- 最后调用大模型基于“策略骨架 + 客服知识参考”生成更自然的最终客服回答
- 如果模型输出异常或跑偏，再自动回退到规则骨架答案

这条客服知识检索链是内置的，不需要重新执行 `scripts/build_kb.py`。如果你只修改了 `customer_service_kb_data.json`，重启 API 后即可生效。
当前独立客服数据里已经补了几类高频场景，例如：

- 退款到账
- 发票重开 / 更正
- 已签收但未收到
- 催发货 / 发货延迟
- 价保条件
- 上门安装收费
- 质量问题退换
- 退款驳回
- 重复扣款
- 补寄配件被驳回
如果一个客服问题里同时包含两个明显不同的客服意图，例如“退款多久到账，以及发票抬头写错了还能重开吗”，系统现在会并行覆盖多个客服主题，而不是只保留第一个主题。

### 6.5 复杂问题与说明书回答约束

当前还有两个默认行为：

- 如果一个问题里同时包含多个独立子问，例如 `支持退款吗，电钻怎么充电？`，系统会先拆成多个子问题，再分别路由到客服策略或说明书检索。
- 多个子问题在最终回答里会被合并成一条自然回复，默认不再输出 `问题1 / 问题2` 这类标签。
- 对说明书类的 `步骤 / 安装 / 设置 / 充电 / 指示灯含义 / 故障含义` 问题，系统会优先输出基于检索证据整理的回答，减少模型自由发挥。
- 说明书类答案现在默认更短，更接近“直接抽取关键证据 + 必要步骤”的风格，不再强制套固定三段式模板。
- 说明书类返回图片时，会先筛掉和当前答案关联度不高的图；默认优先返回和答案真正相关的 1 到 3 张图，而不是把检索命中的所有图全部带回。
- 对客服类问题，系统会尽量直接回答当前子场景，不再默认加上“先看订单状态/先看签收时间”这类泛化前缀；像“建议先准备订单号/截图/联系客服”这类统一尾句，也只会在确实问到材料、联系渠道或需要兜底时才保留。
- 说明书检索现在更偏向结构化内容块：步骤型问题优先命中步骤块，故障/指示灯问题优先命中状态说明块，避免被“概览/简介”类泛化段落带偏。

## 7. 常用脚本

批量评测：

```bash
python3 scripts/evaluate_chat.py
```

生成提交文件：

```bash
python3 scripts/generate_submission.py
```

如果已经有 `submission/submission_generated_debug.jsonl`，想只重做“提交答案清洗”而不重新调用接口：

```bash
python3 scripts/generate_submission.py --from-debug \
  --debug-output submission/submission_generated_debug.jsonl \
  --output submission/submission_generated.csv
```

当前提交导出会自动清理这些内容：

- `问题1 / 问题2 / 问题3` 这类内部拆问标记
- HTML / Markdown / 内部标题噪声
- `与上一问处理思路一致`、`模型未返回有效回答` 等占位语
- 重复句、明显问题回显、格式化残留

同时，提交清洗现在默认遵循一条更保守的原则：

- 如果主链路已经给出了比较直接、比较干净的最终答案，提交清洗会尽量只做轻量格式收口，不再强行重写
- 只有在答案明显带有兜底话术、内部标签、英文内部标题、问题回显或泛化客服模板时，才会进入更强的重写/压缩分支

同时会保留比赛需要的图文提交格式：

- 说明书类答案会输出 `<PIC>` 标记
- 有关联图片时会在答案后拼接图片列表，格式为 `"答案<PIC>";["image_1", "image_2"]`
- 客服类答案如果没有配图，不会强行追加图片后缀

注意：

- `--from-debug` 只会重做提交清洗，不会重新跑检索、路由和回答生成。
- 如果你刚修改了主链路逻辑（例如复杂拆问、客服路由、抽取式兜底），要先重启 API，再重新执行 `python3 scripts/generate_submission.py` 才会体现在新提交文件里。
- 如果你刚修改了英文问答主链路，尤其是英文抽取式兜底或 `service.py` 中的参考片段长度，`--from-debug` 不会补回这些新能力；要重新跑 `python3 scripts/generate_submission.py`，让 API 重新生成一份新的 `submission_generated_debug.jsonl`。

质量分析：

```bash
python3 scripts/analyze_submission_quality.py
```

回归测试：

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```

固定回归：

```bash
python3 scripts/run_regression_suite.py
```

这组固定回归现在不只检查答案文本，还会检查一部分 `retrieval_debug` 关键信息，例如：

- 是否走到了预期的 `manual_rag / customer_service / smalltalk`
- 多轮追问是否命中了上下文继承
- 客服问题命中的知识来源是 `data_file` 还是 `policy_projection`
- 图片问答是否真的返回了图片、且数量没有失控

端到端质量观察：

```bash
python3 scripts/observe_chat_quality.py
```

这组观察样例会按 `smalltalk / manual_rag / multimodal / customer_service / multiturn / mixed / fallback` 分类输出，并额外记录：

- `answer_alignment`
- `source_routing`
- `image_binding`
- `debug_alignment`
- `low_confidence`

## 8. 常见问题

### 8.1 `sentence-transformers is not installed`

说明你现在使用的是神经嵌入模型 `BAAI/bge-small-zh-v1.5`，但环境里没装依赖。

安装方式：

```bash
pip install -r requirements.txt
```

然后重建索引：

```bash
python3 scripts/build_kb.py
```

如果你暂时不想用神经嵌入，可以切回轻量方案：

```bash
export EMBEDDING_MODEL=hashing-ngram-v1
python3 scripts/build_kb.py
```

### 8.2 `/chat` 返回 500 或 503

优先检查：

1. 是否已经执行 `python3 scripts/build_kb.py`
2. 云端 API 的 `LLM_API_KEY` 是否有效
3. 如果使用 Ollama，本地模型是否已拉取

### 8.3 修改了嵌入模型或检索方式后效果不对

改了下面任一项后，都建议重建索引：

- `EMBEDDING_MODEL`
- `RETRIEVAL_MODE`
- `Knowledge_base/` 中的原始知识库文件

命令：

```bash
python3 scripts/build_kb.py
```

## 9. 一套最短可执行流程

如果你要从零开始跑通项目，按这个顺序：

1. 创建虚拟环境并安装依赖
2. 配置云端 API 环境变量
3. 执行 `python3 scripts/build_kb.py`
4. 启动 `uvicorn industry_agent.api.app:create_app --factory --host 0.0.0.0 --port 8000`
5. 用 `curl http://127.0.0.1:8000/health` 检查服务
6. 用 `/chat` 发送问题验证主链路
