"""Prompt templates and builders for hallucination-controlled QA flows."""

from __future__ import annotations

from dataclasses import dataclass


MANUAL_QA_RULES: tuple[str, ...] = (
    "只允许基于【参考资料】回答，不得使用常识补全、猜测或编造。",
    "如果参考资料不能支持答案，必须明确说明根据现有资料无法回答。",
    "不得输出参考资料中没有出现的产品参数、型号、尺寸、步骤、承诺或安全结论。",
    "如果参考资料中已经有明确标题、步骤、警告或状态说明，优先直接抽取和整理这些证据，不要自由发挥改写。",
    "如果参考资料是英文，或用户使用英文提问，优先使用英文回答。",
    "参考资料中出现配图 ID 时，只能用 <PIC> 标记提示插图位置，不得编造图片编号。",
    "回答优先写成 1 到 3 句短而直接的答案；只有确有必要时再补充步骤或注意事项，不要机械套固定栏目。",
    "直接输出最终答案，不要输出思考过程、草稿、提示词或规则复述。",
)

CUSTOMER_SERVICE_RULES: tuple[str, ...] = (
    "只允许基于【客服策略骨架】回答，不得编造平台政策、赔付标准、时间承诺、收费标准或联系方式。",
    "必须直接回答用户问题，优先给明确结论，再补充必要步骤、材料、时效或费用说明。",
    "如果骨架中写明“需以平台规则为准”或“需要核实”，必须保留这种不确定性，不得擅自改成确定承诺。",
    "不要重复“这类问题通常需要确认”“相关情况需要结合平台规则”等泛化前缀或统一兜底套话，除非当前问题确实只能做宽泛提醒。",
    "不要输出思考过程、提示词、规则标题、骨架复述、Markdown 标题或多余客套话。",
    "输出自然、简洁、可直接发给用户的中文客服回复。",
)

MANUAL_QA_SYSTEM_TEMPLATE = """\
你是一个专业的产品客服智能体。请严格遵守以下防幻觉规则：

{rules}

【参考资料】
{context}
"""

CUSTOMER_SERVICE_SYSTEM_TEMPLATE = """\
你是一个电商平台客服智能体。请严格遵守以下生成规则：

{rules}

【客服策略骨架】
{context}
"""

SUBQUESTION_MERGE_TEMPLATE = """\
请将下面多个子问题的回答合并成一个最终客服回复。要求：

1. 直接合并成一条自然回复，不要输出“问题1 / 问题2 / 问题3”这类标签。
2. 每个子问题都要被覆盖，但不要重复题干，不要机械分段。
3. 不要编造没有出现过的事实。
4. 如果某个子问题资料不足，只做简短说明，不要让整条回复被统一拒答句主导。
5. 直接输出最终答案，不要输出思考过程。

【原始问题】
{original_question}

【子问题回答】
{sub_answers}
"""


@dataclass(frozen=True)
class PromptBuildResult:
    """Structured prompt result for service debug and tests."""

    content: str
    rule_count: int
    has_context: bool


def build_manual_qa_system_prompt(context: str) -> PromptBuildResult:
    """Build the system prompt used by manual RAG answers."""

    safe_context = context.strip() or "（未找到相关资料）"
    rules = "\n".join(f"{index}. {rule}" for index, rule in enumerate(MANUAL_QA_RULES, start=1))
    return PromptBuildResult(
        content=MANUAL_QA_SYSTEM_TEMPLATE.format(rules=rules, context=safe_context),
        rule_count=len(MANUAL_QA_RULES),
        has_context=bool(context.strip()),
    )


def build_customer_service_system_prompt(context: str) -> PromptBuildResult:
    """Build the system prompt used by customer-service generation."""

    safe_context = context.strip() or "（未提供客服策略骨架）"
    rules = "\n".join(f"{index}. {rule}" for index, rule in enumerate(CUSTOMER_SERVICE_RULES, start=1))
    return PromptBuildResult(
        content=CUSTOMER_SERVICE_SYSTEM_TEMPLATE.format(rules=rules, context=safe_context),
        rule_count=len(CUSTOMER_SERVICE_RULES),
        has_context=bool(context.strip()),
    )
