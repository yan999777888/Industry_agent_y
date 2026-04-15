# USE

本文档记录当前项目的实际使用方法。后续每次功能更新后，都会同步补充这里的操作说明。

---

## 1. 环境准备

### 1.1 Python 环境

```bash
cd ./Industry_Agent   # 如果当前位于项目上级目录
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
ollama pull llava-phi3

# 确认 Ollama 服务已启动（默认端口 11434）
curl http://localhost:11434/api/tags
```

> **注意**：当前使用 `qwen3.5:2b` 进行测试。如需更好的效果，可拉取更大的模型（如 `qwen3:8b`），并修改 `src/industry_agent/agent/service.py` 中的 `OLLAMA_MODEL` 配置。

当前版本也支持通过环境变量切换 Ollama 地址和模型：

```bash
export OLLAMA_BASE_URL=http://localhost:11434
export OLLAMA_MODEL=qwen3.5:2b
export OLLAMA_VISION_MODEL=llava-phi3
```

说明：

- 不设置 `OLLAMA_VISION_MODEL` 时，系统仍然可以接收 `images` 字段，但只会解析图片格式、尺寸、大小等元数据，不会做视觉内容描述。
- 设置 `OLLAMA_VISION_MODEL` 后，系统会尝试调用该视觉模型，生成简短图片描述，并把描述注入检索链路。
- 当前代码中的默认组合为：
  - 文本模型：`qwen3.5:2b`
  - 视觉模型：`llava-phi3`

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
# 预期返回：包含 status 和组件检查结果
```

当前 `/health` 会在启动后返回运行时健康检查结果，主要包括：

- `index.sqlite` 是否存在
- `images.jsonl` 是否存在
- `Ollama` 服务是否可访问
- 文本模型是否已拉取
- 视觉模型是否已拉取

如果缺少必需组件，服务会在启动阶段直接报错，而不是等到第一次 `/chat` 调用才失败。

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

### 错误码说明

当前 `/chat` 的成功响应固定为：

- HTTP `200`
- 业务字段：`code=0`、`msg=success`

当前常见错误响应如下：

| HTTP 状态码 | 场景 | 返回形式 |
|------|------|------|
| `400` | 请求参数错误，例如 `question` 为空 | `{"detail": "question must not be empty"}` |
| `500` | 服务内部异常，例如对话编排或模型调用失败 | `{"detail": "chat failed: ... "}` |
| `503` | 依赖不可用，例如索引缺失、知识库未构建 | `{"detail": "..."}`

说明：

- 当前错误响应仍沿用 FastAPI 默认 `detail` 字段。
- `/docs` 中已经同步标注了这些错误码及说明。
- `/health` 主要用于启动后检查依赖状态；若关键依赖缺失，服务通常会在启动阶段直接失败，而不是等到 `/chat` 才报错。

### 错误请求示例

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "   "}'
```

预期返回：

```json
{
  "detail": "question must not be empty"
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

### 示例 5：上传图片辅助理解

当前 `/chat` 已支持接收 Base64 图片。推荐两种方式：

1. 直接传纯 Base64 字符串
2. 传 `data:image/...;base64,...` 形式的 Data URL

例如：

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "这个指示灯是什么意思？",
    "images": [
      "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/w8AAgMBgN8L1n4AAAAASUVORK5CYII="
    ]
  }'
```

图片理解链路的当前行为：

- 会先解析上传图片是否为有效 Base64
- 会提取格式、尺寸、文件大小等元数据
- 如果配置了 `OLLAMA_VISION_MODEL`，会额外生成图片内容摘要
- 图片摘要会参与当前轮检索和回答生成
- 相关调试信息当前保存在服务层 `retrieval_debug` 中，后续如果需要，也可以继续透出到 HTTP 响应

适合当前版本的图片类型：

- 设备局部照片
- 指示灯、按钮、接口、屏幕界面
- 安装位置、损坏现象、零部件近景

当前仍不建议完全依赖图片输入来替代文字问题，最好同时提供：

- 产品名
- 型号
- 故障现象或想咨询的具体问题

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

当前脚本在写入 `submission_generated.csv` 之前，还会自动做一层“提交专用清洗”，主要包括：

- 去除 `结论 / 操作说明 / 注意事项 / 相关图片` 这类内部结构化标签
- 去除 `Manual16_51`、`drill0_17` 这类图片 ID 直接写入答案正文
- 去除部分“参考资料仅包含...”这类内部说明口吻
- 将纯拒答答案改写成更自然的客服式提示语
- 如果一个多问题答案里只有部分子问题拒答，会尽量保留已经答出的有效内容
- 如果清洗后只剩下题目原文回显，会回落到更稳妥的 fallback，而不是把问题原样提交
- 对中文客服策略答案做句级去重和长度压缩，减少多个子问题合并后的模板重复

这层清洗只影响提交文件生成，不影响 `/chat` 接口本身的原始返回。

生成后的 `submission/submission_generated.csv` 格式为：

```csv
id,ret
1,回答内容
2,回答内容
```

该格式与 `submission/submission_example.csv` 保持一致，可用于提交到测试平台。

### 5.3 检索专项回归

Phase A 检索质量专项优化当前已经覆盖以下能力：

- 长问题切词与短语抽取
- 领域同义词扩展，例如 `腕带 -> 表带`、`红灯 -> 指示灯`
- 标题意图加权，例如“安装 / 充电 / 表带尺寸 / 默认密码”类标题优先
- 扩展词降噪，降低仅靠泛化词命中的“设置类”章节被错误顶上来
- 检索后二次证据重排，综合考虑 query 命中、图片线索命中、配图存在性和候选多路命中情况
- 英文问题去噪，过滤 `how / what / the / if` 这类低价值英文停用词
- 英文短语抽取，例如 `approval label`、`battery compartment`、`over temperature`
- 对 `汇总英文` 这类英文汇总知识源增加更强的英文语义对齐排序，降低被通用英文目录误召回
- 对 `汇总英文` 增加英文子领域提示词，区分 boat / camera / airfryer / eReader 等内部混合手册内容
- 候选召回改为“按短语和关键词分批拉取再合并”，避免常见词如 `battery` 抢占候选上限，导致后续 `sailing / eReader` 等关键信号进不了重排
- 增加少量英文意图别名，例如 `record voice -> voice recording`、`battery conversion -> battery switches`，用于提升说明书常见表达差异下的召回稳定性

如需只验证检索层，不依赖 Ollama 和 API，可以直接运行：

```bash
python3 -m unittest discover -s tests -p 'test_retriever.py'
```

当前这组测试主要覆盖：

- 型号、产品名、关键短语抽取
- 同义词扩展
- 长问题关键词提取去噪
- 标题意图加权是否生效
- “扩展词误抬分”是否被抑制
- 多路候选融合是否优先保留同时被文本检索和多模态检索命中的 chunk

### 5.4 多模态融合验证

Phase B 第一批多模态融合优化当前已经接入主链路，核心变化是：

- 图片理解除了返回 `retrieval_hint` 字符串外，还会输出结构化 `retrieval_terms`
- `/chat` 在说明书问答场景下会同时跑：
  - 文本 query 检索
  - 文本 query + 图片检索词 的融合检索
- 两路候选会做去重合并，再进入证据筛选
- 证据筛选阶段会额外参考图片相关词命中情况，优先保留更贴近图片内容的 chunk

如果想重点验证多模态融合，可以运行：

```bash
python3 -m unittest discover -s tests -p 'test_agent_flow.py'
```

当前已覆盖的多模态测试包括：

- 图片检索词抽取是否保留高价值视觉词
- Agent 是否把结构化图片检索词真正传入主链路
- 多路候选融合是否会抬高同时被文本路和图片路召回的证据
- 图片相关证据是否会在二次筛选中优先保留

Phase B 第二批多模态融合优化新增了两点：

- 图片检索词现在会继续拆成 `component_terms / status_terms / issue_terms / other_terms`
- 证据排序会额外计算“图文一致性”，优先保留同时命中部件词和状态词的 chunk

对应调试信息可以在 `retrieval_debug.image_features` 和 `retrieval_debug.image_understanding.visual_features` 中看到。

---

### 5.5 回答格式验证

Phase C 第一批回答自然度与模板优化当前已经完成这些改进：

- 会先解析模型原始输出中的 `结论 / 操作 / 注意 / 相关图片` 等段落
- 会自动归一化成统一模板，避免标签写法不一致
- 对没有结构的普通文本回答，会自动拆成更自然的 `结论 / 操作/说明 / 注意事项`
- 会尽量避免把同一句话机械地重复到三个小节里

如果想重点验证格式层，可以运行：

```bash
python3 -m unittest discover -s tests -p 'test_agent_flow.py'
```

当前已覆盖的格式化测试包括：

- 纯文本回答是否能拆成不同功能的小节
- `操作:`、`注意:` 这类标签是否会被归一化
- 相关图片小节是否稳定补齐

Phase C 第二批新增了两点：

- 多问题回答不再依赖额外 LLM 重写，而是按子问题结果做确定性合并
- 合并后的答案会保留每个子问题自己的结构化内容，减少二次改写带来的格式漂移

近期又补充了英文说明书题的保守抽取回答优化：

- 当英文问题已经检索到高相关证据，但本地小模型仍返回拒答时，系统会从证据中抽取短句生成保守答案
- 抽取时会做标题/正文去重，减少 `Approval label... Approval label...` 这类重复拼接
- 对 `how / operate / use` 类问题，会优先选择 `press / select / go to / turn` 等操作句，减少把备注句当成结论

---

### 5.6 多轮鲁棒性验证

Phase D 第一批多轮鲁棒性优化当前已经完成这些改进：

- 支持通过 `清空上下文 / 重新开始` 一类输入重置当前 `session_id` 的上下文
- 如果用户说了“换个产品”“另一个产品”但没有给出新产品名，会先要求补充，而不是误继承上一轮产品
- 如果用户明确给出新的产品名，系统会识别为话题切换并更新当前会话产品

如果想重点验证多轮鲁棒性，可以运行：

```bash
python3 -m unittest discover -s tests -p 'test_agent_flow.py'
```

当前已覆盖的多轮鲁棒性测试包括：

- 会话重置后是否停止继承上一轮产品
- 未明确的新话题切换是否会进入澄清
- 明确产品切换后是否会更新当前会话产品

Phase D 第二批新增了客服类多轮承接：

- 如果上一轮是退款、物流、售后等客服策略问题，下一轮短追问会继承上一轮客服话题
- 例如第一轮问“我想退款，退款多久能到账？”，第二轮问“那需要准备什么材料？”，系统会继续走客服策略分支
- 会话状态会记录 `current_route` 和 `current_service_topics`，避免把客服追问误送进说明书检索

---

### 5.7 客服策略知识验证

Phase E 第一批客服策略知识扩展当前新增覆盖：

- 修改收货地址
- 预约安装 / 上门安装
- 价保 / 降价补差
- 支付失败 / 重复扣款
- 催发货 / 延迟发货
- 保修期 / 质保期
- 配件、附件、包装盒和补寄配件

如果想重点验证客服策略，可以运行：

```bash
python3 -m unittest discover -s tests -p 'test_agent_flow.py'
```

也可以启动 API 后用固定回归集验证：

```bash
python3 scripts/run_regression_suite.py
```

Phase E 第二批进一步增强了“主题内追问”能力：

- 会识别客服问题是在追问“材料/凭证”
- 会识别客服问题是在追问“多久/时效”
- 会识别客服问题是在追问“收费/费用”
- 会识别客服问题是在追问“条件/是否支持”
- 会识别客服问题是在追问“流程/怎么申请”
- 会识别客服问题是在追问“联系谁/找谁处理”

这意味着像下面这些问法，会得到更细粒度的话术，而不只是泛泛地返回一个客服大类说明：

- `修改收货地址需要准备什么材料？`
- `催发货一般多久能处理？`
- `上门安装需要收费吗？`
- `刚买完就降价了，可以申请价保吗？`
- `刚买完就降价了，申请价保需要满足什么条件？`
- `支付失败但是扣款了，应该怎么申请核查？`
- `发票抬头填错了，这种情况应该联系谁处理？`

本轮又继续补齐了“同一主题下不同业务状态”的场景模板，重点包括：

- 退款原因细分：`7天无理由`、`质量问题退款`、`已拆封/已使用`
- 物流状态细分：`物流停滞`、`已签收但未收到`、`改派/送错地址`
- 保修审核细分：`在保`、`过保`、`人为损坏`
- 退款状态细分：`退款/退货申请被驳回`
- 发票状态细分：`发票已开具但信息填写错误，需重开或更正`
- 安装状态细分：`安装改约`、`师傅爽约/未按约上门`
- 补件状态细分：`补寄配件申请被驳回`

例如下面这些问法，会比之前更接近真实客服的处理路径：

- `这个商品有质量问题，我想退款，需要我承担运费吗？`
- `物流显示已签收但我没收到，这种情况应该怎么处理？`
- `设备进水了，还能走保修吗？`
- `我的退款申请被驳回了，我现在该怎么办？`
- `发票抬头填错了，而且已经开出来了，还能重开吗？`
- `上门安装已经约好了，但是师傅没来，可以改约吗？`
- `我申请补寄配件被驳回了，还能重新提交吗？`

---

### 5.8 回归验证集

Phase F 已经把固定回归验证集继续扩展到了这些新增场景：

- 客服多轮追问
- 修改地址
- 价保
- 上门安装收费
- 支付异常
- 催发货时效
- 价保条件追问
- 支付异常流程追问
- 发票联系对象追问
- 退款申请驳回
- 发票重开
- 安装改约
- 补寄配件驳回
- 会话重置

固定回归集位置：

```text
tests/fixtures/regression_cases.json
```

当前如果服务已启动，建议在本地执行：

```bash
python3 scripts/run_regression_suite.py
```

如果只想检查脚本本身和样例结构是否正常，可以执行：

```bash
python3 -m unittest discover -s tests -p 'test_regression_suite.py'
```

### 5.9 端到端质量观察

除了固定回归集之外，当前项目还补了一套“观察型”样例，用于查看不同类别请求的整体表现和失败类型分桶。

样例文件：

```text
tests/fixtures/quality_observation_cases.json
```

启动 API 后执行：

```bash
python3 scripts/observe_chat_quality.py
```

默认会输出到：

```text
data/processed/quality_observation_report.json
```

这个脚本更适合做端到端质量观察，而不是硬性回归拦截。它会按类别统计：

- `smalltalk`
- `manual_rag`
- `multimodal`
- `customer_service`
- `multiturn`
- `mixed`
- `fallback`
- `api_error`

同时还会把问题大致分桶到这些失败类型：

- `answer_alignment`
- `source_routing`
- `image_binding`
- `low_confidence`
- `http_status`
- `error_detail`

如果只想验证观察脚本本身，可以执行：

```bash
python3 -m unittest discover -s tests -p 'test_quality_observation.py'
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
SQLite LIKE 候选召回 + FTS/BM25 风格候选召回 + Python 混合重排
（含短语抽取 / 同义词扩展 / 标题意图加权 / 扩展词降噪）
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
会话控制（上下文重置 / 话题切换澄清 / 明确产品切换）
  │
  ▼
检索与回答生成
```

当前多模态补充链路为：

```
用户上传 Base64 图片
  │
  ▼
图片解析（格式 / 尺寸 / 大小）
  │
  ▼
可选视觉描述（OLLAMA_VISION_MODEL）
  │
  ▼
图片线索去噪，提取结构化 retrieval_terms
  │
  ▼
文本检索 + 多模态融合检索 双路召回
  │
  ▼
候选合并重排 + 图片感知证据筛选
  │
  ▼
图文一致性加权（部件词 / 状态词 / 故障词）
  │
  ▼
注入回答上下文
```

当前还额外支持一层轻量分流：

```
日常寒暄 / 致谢 / 告别
  │
  ▼
直接返回通用客服引导语
  │
  ▼
不进入知识库检索，不返回配图
```

例如：

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "你好"}'
```

这类输入当前会直接返回简短引导语，而不会误命中说明书内容。

### 示例 6：通用客服问题路由

当前系统已经支持把“说明书问答”和“通用客服/售后问题”分开处理。

这意味着像下面这类问题：

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "我想退款，退款多久能到账？"}'
```

不会再强行去说明书里检索“退款”相关段落，而是优先走轻量客服策略分支，返回更接近真实客服话术的保守答复，例如：

- 提示需要订单号、支付方式、购买渠道
- 说明到账时间通常取决于支付渠道和订单状态
- 引导用户继续补充关键信息

当前通用客服分支优先覆盖：

- 退款 / 退货 / 换货
- 发票 / 发票抬头
- 物流 / 运费 / 补发 / 签收
- 投诉 / 假货 / 虚假宣传 / 赔偿
- 售后 / 维修 / 保修
- 破损 / 瑕疵 / 少件 / 保质期异常

如果一个请求里同时包含多个子问题，系统会按子问题分别路由；说明书类问题继续走 RAG，售后类问题走客服策略分支。

### 示例 7：固定回归验证集

为了避免后续优化依赖公开题数据，项目现在内置了一套固定回归验证集：

```bash
python3 scripts/run_regression_suite.py
```

默认会读取：

```text
tests/fixtures/regression_cases.json
```

当前覆盖的通用场景包括：

- 说明书问答
- 寒暄分流
- 通用客服问题路由
- 混合多问题输入
- 多轮追问
- 图片辅助问答
- 复杂拆问
- 拒答场景

后续如果我们新增功能，也应该优先把验证样例补到这套固定回归集中。

### 关键文件

| 文件 | 职责 |
|------|------|
| `src/industry_agent/api/app.py` | FastAPI 路由与请求/响应模型 |
| `src/industry_agent/agent/service.py` | Agent 编排：检索 → 组装 → LLM |
| `src/industry_agent/agent/question_splitter.py` | 复杂问题拆解模块 |
| `src/industry_agent/agent/session_store.py` | 结构化会话状态存储 |
| `src/industry_agent/agent/context_manager.py` | 多轮上下文继承与追问解析 |
| `src/industry_agent/agent/image_understanding.py` | 上传图片解析与可选视觉描述 |
| `src/industry_agent/agent/question_router.py` | 问题路由，区分说明书问答与通用客服问题 |
| `src/industry_agent/agent/customer_service_policy.py` | 轻量客服策略知识 |
| `src/industry_agent/agent/response_formatter.py` | 回答后处理与格式稳定 |
| `src/industry_agent/agent/runtime_checks.py` | 启动健康检查 |
| `src/industry_agent/rag/retriever.py` | 关键词提取 + SQLite 评分检索 |
| `src/industry_agent/kb/build_index.py` | 知识库构建主流程 |
| `src/industry_agent/kb/parser.py` | 手册解析与文本规范化 |
| `src/industry_agent/kb/chunker.py` | 文本切块策略 |
| `src/industry_agent/kb/index_store.py` | SQLite 索引写入 |
| `src/industry_agent/kb/models.py` | 数据模型定义 |
| `src/industry_agent/config.py` | 项目路径与全局配置 |
