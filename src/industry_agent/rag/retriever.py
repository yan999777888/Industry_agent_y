"""SQLite-backed retriever with Chinese query analysis and reranking.

SQLite FTS5's default tokenizer cannot segment Chinese text, so the first
iteration uses lightweight keyword extraction plus Python-side scoring.  The
retriever intentionally returns ordinary dictionaries to keep compatibility
with the current AgentService.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from industry_agent.config import settings
from industry_agent.rag.vector_store import SQLiteVectorSearcher, VectorSearcher, describe_vector_retrieval

# ---------------------------------------------------------------------------
# Query analysis resources
# ---------------------------------------------------------------------------

_STOPWORDS: set[str] = {
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一",
    "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着",
    "没有", "看", "好", "自己", "这", "他", "她", "它", "吗", "什么",
    "怎么", "怎样", "如何", "请问", "能", "可以", "吧", "呢", "啊",
    "那", "这个", "那个", "哪", "哪个", "多少", "为什么", "谁",
    "请", "帮", "告诉", "一下", "关于", "需要", "是否", "哪些",
}
_EN_STOPWORDS: set[str] = {
    "a", "an", "and", "are", "as", "at", "be", "before", "can", "could",
    "do", "does", "for", "from", "how", "i", "if", "in", "into", "is",
    "it", "me", "my", "of", "on", "or", "should", "the", "this", "to",
    "use", "using", "want", "what", "when", "where", "while", "with",
    "after", "about", "please", "tell", "need", "first", "time",
    "you", "your", "that", "they", "them", "their", "its", "has", "have",
    "had", "been", "was", "were", "will", "would", "may", "might", "shall",
    "not", "but", "also", "some", "any", "all", "each", "every", "both",
    "there", "here", "then", "than", "more", "most", "very", "much",
    "just", "only", "even", "still", "well", "too", "so", "such",
    "which", "who", "whom", "whose", "other",
}

_DOMAIN_PHRASES: tuple[str, ...] = (
    # 通用
    "指示灯", "闪烁", "标识", "充电", "充电器", "电池组", "表带", "尺寸",
    "更换", "安装", "维修", "故障", "清洁", "连接", "设置", "显示",
    "程序", "控制台", "佩戴", "模式", "温度", "延迟", "开机", "关机",
    "按键", "默认密码", "安全注意事项", "注意事项", "售后", "保修",
    "自动运行", "自动模式", "制冷", "制热", "除湿", "送风", "风速",
    "遥控器", "滤网", "传感器", "节能", "静音", "定时", "预约",
    # 空调
    "空调", "单冷型", "冷暖型", "极速制冷", "极速制热", "自清洁",
    "等离子净化", "节能制冷", "睡眠模式", "自动转换", "导风板",
    "摆风", "显示屏亮度", "无遥控器操作", "空调自动重启",
    # 空气净化器
    "空气净化器", "涡轮模式", "常规模式", "灰尘传感器", "预过滤网",
    "滤网更换", "空气质量", "VOC", "风速档位",
    # 冰箱
    "冰箱", "冰柜", "除臭装置", "内部灯", "温控器", "冷冻室",
    "冷藏室", "制冰", "温度调节",
    # 洗碗机
    "洗碗机", "洗涤剂", "碗篮", "喷淋臂", "排水管", "进水管",
    "预洗", "烘干", "洗涤程序", "漂洗", "高温消毒",
    # 烤箱
    "烤箱", "烤架", "烤盘", "接油盘", "烘焙", "烧烤",
    "催化侧面板", "自动清洁", "热风循环", "上下加热",
    # 电钻
    "电钻", "无键夹头", "批头", "腰带挂钩", "电池组充电",
    "充电器指示灯", "扭矩", "转速", "钻头",
    # 发电机
    "发电机", "燃油开关", "油箱通气", "经济控制开关", "消音器",
    "机油", "空气滤清器", "火花塞", "直流保护器", "暖机",
    "怠速", "额定转速", "排空机油",
    # 水泵
    "水泵", "注水", "排水", "密封", "水封",
    # 吹风机
    "吹风机", "阻风门", "化油器", "启动绳", "泵油膜片",
    "停机开关", "半油门", "全油门", "预热",
    # 健身追踪器
    "健身追踪器", "心率", "EDA扫描", "GPS", "运动追踪",
    "睡眠追踪", "非接触式支付", "表盘", "通知", "蓝牙配对",
    "充电线", "磁吸", "佩戴位置",
    # 健身单车
    "健身单车", "阻力等级", "心率控制", "体能测试", "用户档案",
    "预设程序", "座椅调节", "踏板", "快速启动",
    # 摩托艇
    "摩托艇", "拖曳速度", "滑航速度", "半滑航", "转向",
    "熄火绳", "喷水推进", "舵", "登艇", "靠泊",
    "倾斜调节", "稳速器", "燃油表", "发动机过热",
    # 相机
    "相机", "快门", "光圈", "ISO", "白平衡", "对焦",
    "AF模式", "连拍", "闪光灯", "取景器", "肩带", "镜头",
    "相纸", "打印模式", "自动打印", "手动打印",
    # VR头显
    "VR头显", "头带", "控制器", "定位", "刷新率", "瞳距",
    # 人体工学椅
    "人体工学椅", "扶手", "头枕", "腰枕", "后仰", "气杆",
    "底盘", "脚轮", "升降", "椅背", "按摩",
    # 洗碗机
    "洗碗机", "洗涤", "碗篮调节", "排水连接", "软水盐",
    "洗涤剂盒",
    # 温控器
    "温控器", "供暖", "制冷温度", "程序日程", "风扇模式",
    "智能响应", "热泵", "接线端子",
    # 蒸汽清洁机
    "蒸汽清洁机", "蒸汽拖把", "水箱", "蒸馏水", "清洁头",
    "硬质地面", "消毒",
    # 蓝牙鼠标
    "蓝牙鼠标", "蓝牙驱动", "WIDCOMM", "配对按钮", "USB接收器",
    # 功能键盘
    "功能键盘", "RGB灯光", "宏命令", "按键重映射", "轴体",
    "拔轴器", "板载配置", "CAM软件",
    # 冰箱
    "冰箱", "冷藏", "冷冻", "温控", "除霜",
    # 微波炉
    "微波炉", "烧烤", "解冻", "预设菜单", "童锁",
    # 打印机/传真
    "打印机", "传真", "墨盒", "扫描", "复印",
    # 耳机
    "耳机", "降噪", "通透模式", "主动降噪", "ANC",
    "低延迟模式", "语音助手",
    # 电钻
    "电钻", "冲击钻", "螺丝刀", "钻孔",
    # 儿童摩托车
    "儿童电动摩托车", "挡泥板", "前轴", "把手管",
    # 草坪机
    "草坪机", "割草", "刀片", "油门", "座椅",
    # 喷射器
    "喷射器", "喷嘴", "进气口", "叶轮",
)
_DOMAIN_SYNONYMS: dict[str, tuple[str, ...]] = {
    "红灯": ("指示灯", "闪烁"),
    "蓝灯": ("指示灯",),
    "绿灯": ("指示灯",),
    "灯闪": ("指示灯", "闪烁"),
    "腕带": ("表带", "健身追踪器"),
    "带子": ("表带",),
    "大小": ("尺寸",),
    "配对": ("连接",),
    "连不上": ("连接", "故障"),
    "连不上网": ("连接", "故障"),
    "充满电": ("充电", "电池组"),
    "没电": ("电池", "充电"),
    "重置": ("设置",),
    "密码": ("默认密码",),
    "pin": ("密码", "设备锁"),
    "pin码": ("密码", "设备锁"),
    "pin code": ("密码", "设备锁"),
    "死机": ("故障",),
    "卡住": ("故障",),
    "发热": ("温度",),
    "过热": ("延迟", "温度"),
    "拆卸": ("更换", "安装"),
    # 空调相关
    "开不了机": ("开机", "故障"),
    "不制冷": ("制冷", "故障"),
    "不制热": ("制热", "故障"),
    "风不大": ("风速",),
    "风太小": ("风速",),
    "有异味": ("清洁", "滤网"),
    "漏水": ("故障", "排水"),
    "噪音大": ("故障", "静音"),
    "不显示": ("显示", "故障"),
    "遥控失灵": ("遥控器", "故障"),
    "自动模式": ("自动运行", "模式"),
    "智能模式": ("自动运行", "模式"),
    # 净化器相关
    "净化效果差": ("滤网", "清洁"),
    "指示灯红": ("指示灯", "滤网更换"),
    "空气质量差": ("空气净化器", "自动运行"),
    # 冰箱相关
    "不制冷冰箱": ("冰箱", "制冷", "故障"),
    "结冰": ("冰箱", "冷冻", "故障"),
    "有霜": ("冰箱", "除霜"),
    # 洗碗机相关
    "洗不干净": ("洗碗机", "洗涤", "故障"),
    "不排水": ("洗碗机", "排水", "故障"),
    "不烘干": ("洗碗机", "烘干", "故障"),
    # 发电机相关
    "启动不了": ("启动", "故障"),
    "不发电": ("发电机", "故障"),
    "油箱": ("燃油开关", "发电机"),
    # 摩托艇相关
    "转向不灵": ("转向", "摩托艇", "故障"),
    "发动机熄火": ("发动机", "故障"),
    "无法启动": ("启动", "故障"),
    # 相机相关
    "拍不了": ("快门", "相机", "故障"),
    "对不上焦": ("对焦", "相机", "故障"),
    "不开机": ("开机", "故障"),
    "屏幕黑": ("显示", "故障"),
    # 健身追踪器相关
    "心率不准": ("心率", "健身追踪器"),
    "GPS定位慢": ("GPS", "健身追踪器"),
    "不显示通知": ("通知", "蓝牙配对"),
    # 售后相关
    "退货": ("售后", "退货"),
    "换货": ("售后", "换货"),
    "退款": ("售后", "退款"),
    "保修期": ("保修", "售后"),
    "维修费": ("维修", "售后"),
    "发票": ("售后", "发票"),
    "物流": ("物流", "售后"),
    "快递": ("物流", "快递"),
    "包装破损": ("包装", "破损", "售后"),
}
_LONG_TOKEN_SPLIT_RE = re.compile(r"[的了和及与并或后前时再先把将并且然后如果则呢吗啊呀啦如何为何能否是否可以应该需要]")
_QUERY_PHRASE_RE = re.compile(r"[\u4e00-\u9fff]{3,}")
_ASCII_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9._-]*")
_TITLE_INTENT_BOOSTS: tuple[tuple[tuple[str, ...], tuple[str, ...], float], ...] = (
    (("安全注意事项", "注意事项", "佩戴"), ("安全", "注意", "警告"), 5.0),
    (("安装", "更换", "拆卸"), ("安装", "组装", "更换"), 4.5),
    (("充电", "电池组", "充满电"), ("充电", "电池", "电池组"), 4.5),
    (("尺寸", "表带", "腕带"), ("尺寸", "表带"), 4.0),
    (("默认密码", "密码", "重置", "pin"), ("密码", "默认密码", "pin", "设备锁", "重置"), 4.0),
    (("连接", "配对", "连不上"), ("连接", "接口", "配对"), 4.0),
)
_ENGLISH_DOMAIN_HINTS: dict[str, tuple[str, ...]] = {
    "boat": (
        "boat", "sailing", "onboard", "on board", "anchor", "jet wash", "steering",
        "starboard", "port side", "bow", "stern", "hull", "engine compartment",
        "emission control", "storage compartment", "wet storage", "watercraft",
    ),
    "camera": (
        "camera", "lens", "shutter", "viewfinder", "battery grip", "autofocus",
        "af mode", "aperture", "iso", "flash photography", "image playback",
        "dc coupler", "eos",
    ),
    "airfryer": (
        "air fryer", "airfryer", "nutriu", "preset", "keep warm", "basket",
        "hot air", "remote cooking", "wifi", "voice control", "food table",
    ),
    "ereader": (
        "e reader", "ereader", "e-book reader", "ebook", "voice recording",
        "photo viewer", "photo mode", "browser history", "record", "main menu",
    ),
    "vacuum": (
        "vacuum", "vacuum cleaner", "home base", "full bin", "side brush",
        "caster wheel", "dual-mode virtual wall", "dust bin", "roomba",
    ),
    "motherboard": (
        "motherboard", "tpm connector", "pci express", "cpu", "system memory",
        "raid", "rear panel connectors", "onboard led", "bios", "sata", "usb 3.1",
        "intel lan", "serial port", "apm configuration", "erp ready", "pxe option",
    ),
    "pressure_cooker": (
        "pressure cooker", "quick release", "float valve", "steam release",
        "anti-block shield", "condensation collector", "sealing ring",
    ),
    "microwave": (
        "microwave", "over-the-range", "auto defrost", "grease filter",
        "charcoal filter", "oven light", "light timer",
    ),
    "snowmobile": (
        "snowmobile", "throttle cable", "v-belt", "spark plug", "brake lever", "ski",
        "vk540", "suspension", "spring preload", "fresh snow", "fuel tank", "rider",
    ),
    "landline": (
        "landline", "base station", "handset", "answering machine", "phonebook",
    ),
    "lawn_mower": (
        "lawn mower", "mower deck", "blade-control switch", "height-of-cut",
        "grass deflector", "spark-plug", "parking brake", "cutting blade", "pto",
    ),
    "coffee_machine": (
        "coffee", "espresso", "lungo", "capsule", "descaling", "water tank",
        "coffee preparation", "drip tray", "milk frother",
    ),
    "fax": (
        "fax", "telephone line cord", "phone line", "mfc-", "ink cartridge",
        "document feeder", "scanner glass", "telephone wall jack",
    ),
    "toothbrush": (
        "toothbrush", "brush head", "brushing", "pressure sensor", "brush pacer",
        "senseiq", "gum", "bristles", "toothpaste",
    ),
    "grill": (
        "grill", "grilling", "burner", "cooking surface", "grease tray",
        "spider alert", "propane", "bristle brush",
    ),
    "earphone": (
        "earphone", "earphones", "earbud", "earbuds", "charging case",
        "bluetooth", "noise canceling", "pairing",
    ),
    "television": (
        "television", "tv", "hdmi", "remote control", "channel", "picture mode",
        "screen", "antenna", "audio output",
    ),
    "washing_machine": (
        "washer", "washing machine", "washtub", "wash timer", "spin timer",
        "drain filter", "overflow filter", "drain hose", "rinse", "cycle selector",
    ),
}
_SEMANTIC_INTENT_HINTS: dict[str, tuple[str, ...]] = {
    "procedure": (
        "怎么", "如何", "步骤", "方法", "安装", "更换", "拆卸", "清洁", "设置",
        "连接", "操作", "使用", "充电", "佩戴", "调节", "how", "procedure",
        "install", "replace", "remove", "clean", "set", "connect", "use", "charge",
    ),
    "safety_warning": (
        "安全", "注意", "警告", "危险", "风险", "禁忌", "注意事项", "safety",
        "warning", "caution", "danger", "risk", "avoid",
    ),
    "troubleshooting": (
        "故障", "错误", "报错", "无法", "不能", "不工作", "闪烁", "指示灯",
        "蜂鸣", "失败", "troubleshoot", "problem", "error", "fault", "fails",
        "not working", "flashing", "blinking", "indicator",
    ),
    "parts_list": (
        "部件", "零件", "配件", "组成", "包含", "包装", "清单", "parts",
        "accessories", "components", "included", "overview",
    ),
    "specification": (
        "规格", "参数", "尺寸", "重量", "容量", "默认", "密码", "型号", "温度",
        "specification", "dimensions", "weight", "capacity", "default", "model",
    ),
}
_GENERIC_SECTION_HINTS: tuple[str, ...] = (
    "概览", "总览", "简介", "说明", "保养说明", "产品介绍",
    "overview", "general", "introduction", "camera care", "cleaning safety",
)
_PROCEDURE_CUES: tuple[str, ...] = (
    "步骤", "先", "再", "然后", "最后", "取下", "装入", "插入", "连接",
    "按下", "点击", "选择", "确认", "检查", "press", "select", "install",
    "remove", "connect", "turn", "open", "close",
)
_TROUBLESHOOTING_CUES: tuple[str, ...] = (
    "表示", "代表", "含义", "闪烁", "指示灯", "报警", "错误", "故障", "状态",
    "means", "indicates", "warning", "flash", "flashing", "blink", "blinking",
    "status", "error", "fault",
)
_SPECIFICATION_CUES: tuple[str, ...] = (
    "规格", "参数", "尺寸", "重量", "容量", "默认", "密码", "型号", "温度",
    "specification", "dimensions", "weight", "capacity", "default", "model",
)
_ENGLISH_QUERY_ALIASES: dict[str, tuple[str, ...]] = {
    "battery conversion": ("battery switches", "battery switch assembly", "emerg parallel"),
    "battery switching": ("battery switches", "battery switch assembly"),
    "record voice": ("voice recording", "record mode"),
    "voice record": ("voice recording", "record mode"),
    "photo viewer": ("photo mode", "photo rotation", "previous or next photo"),
}

_PRODUCT_ALIASES: dict[str, str] = {
    "vr头显": "VR头显",
    "头显": "VR头显",
    "ps vr": "VR头显",
    "人体工学椅": "人体工学椅",
    "椅子": "人体工学椅",
    "办公椅": "人体工学椅",
    "健身单车": "健身单车",
    "单车": "健身单车",
    "动感单车": "健身单车",
    "健身追踪器": "健身追踪器",
    "追踪器": "健身追踪器",
    "手表": "健身追踪器",
    "腕表": "健身追踪器",
    "腕带": "健身追踪器",
    "表带": "健身追踪器",
    "儿童电动摩托车": "儿童电动摩托车",
    "电动摩托车": "儿童电动摩托车",
    "冰箱": "冰箱",
    "功能键盘": "功能键盘",
    "键盘": "功能键盘",
    "发电机": "发电机",
    "可编程温控器": "可编程温控器",
    "温控器": "可编程温控器",
    "吹风机": "吹风机",
    "摩托艇": "摩托艇",
    "水泵": "水泵",
    "洗碗机": "洗碗机",
    "烤箱": "烤箱",
    "电钻": "电钻",
    "冲击钻": "电钻",
    "起子": "电钻",
    "电动工具": "电钻",
    "相机": "相机",
    "空气净化器": "空气净化器",
    "净化器": "空气净化器",
    "空调": "空调",
    "蒸汽清洁机": "蒸汽清洁机",
    "清洁机": "蒸汽清洁机",
    "蓝牙激光鼠标": "蓝牙激光鼠标",
    "鼠标": "蓝牙激光鼠标",
    # English product aliases → 汇总英文
    "jetski": "汇总英文",
    "jet ski": "汇总英文",
    "watercraft": "汇总英文",
    "boat": "汇总英文",
    "airfryer": "汇总英文",
    "air fryer": "汇总英文",
    "vacuum": "汇总英文",
    "vacuum cleaner": "汇总英文",
    "lawn mower": "汇总英文",
    "snowmobile": "汇总英文",
    "motherboard": "汇总英文",
    "microwave": "汇总英文",
    "pressure cooker": "汇总英文",
    "earphone": "汇总英文",
    "earphones": "汇总英文",
    "ereader": "汇总英文",
    "e-reader": "汇总英文",
    "e reader": "汇总英文",
    "fax": "汇总英文",
    "grill": "汇总英文",
    "toothbrush": "汇总英文",
    "coffee machine": "汇总英文",
    "landline": "汇总英文",
    "camera": "汇总英文",
    "washer": "汇总英文",
    "washing machine": "汇总英文",
}

_TOKEN_RE = re.compile(
    r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]+"
    r"|[A-Za-z][A-Za-z0-9._-]*"
    r"|[0-9]+(?:\.[0-9]+)*",
)
_MODEL_RE = re.compile(r"[A-Za-z]{2,}\d+[A-Za-z0-9._-]*")
_FTS_UNSAFE_RE = re.compile(r'["\'():*]+')


@dataclass(frozen=True)
class QueryAnalysis:
    raw_query: str
    keywords: list[str]
    products: list[str]
    models: list[str]
    phrases: list[str]
    expanded_keywords: list[str]


def analyze_query(query: str) -> QueryAnalysis:
    """Analyze product scope, model numbers and useful search keywords."""

    normalized = _normalize(query)
    products = _unique(
        product
        for alias, product in _PRODUCT_ALIASES.items()
        if alias and alias in normalized
    )
    models = _unique(match.group(0).upper() for match in _MODEL_RE.finditer(query))
    phrases = extract_query_phrases(query)
    keywords = extract_keywords(query)
    for phrase in _DOMAIN_PHRASES:
        if phrase in query:
            keywords.append(phrase)
    expanded_keywords = expand_keywords(query, keywords)
    keywords.extend(expanded_keywords)
    keywords.extend(products)
    keywords.extend(models)
    return QueryAnalysis(
        raw_query=query,
        keywords=_unique(keywords),
        products=products,
        models=models,
        phrases=_unique(phrases),
        expanded_keywords=_unique(expanded_keywords),
    )


def extract_keywords(query: str, *, min_len: int = 2) -> list[str]:
    """Extract Chinese and ASCII keywords from a user query."""

    normalized_query = _normalize_query_text(query)
    raw_tokens = _TOKEN_RE.findall(normalized_query)
    keywords: list[str] = []

    def add(term: str) -> None:
        term = term.strip()
        if term and term not in _STOPWORDS and len(term) >= min_len:
            keywords.append(term)

    merged_tokens = _merge_ascii_cjk_tokens(raw_tokens)
    for token in merged_tokens:
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9._-]*|[0-9]+(?:\.[0-9]+)*", token):
            if re.fullmatch(r"[A-Za-z][A-Za-z0-9._-]*", token) and token.lower() in _EN_STOPWORDS:
                continue
            add(token.upper())
            continue

        if len(token) <= 6:
            add(token)
            for size in (3, 2):
                for index in range(len(token) - size + 1):
                    add(token[index : index + size])
        else:
            for term in _extract_long_token_terms(token):
                add(term)

    return _unique(keywords)


def extract_query_phrases(query: str) -> list[str]:
    phrases: list[str] = []
    for match in _QUERY_PHRASE_RE.finditer(query):
        phrase = match.group(0).strip()
        if len(phrase) >= 4:
            phrases.append(phrase[:12])
    phrases.extend(_extract_ascii_query_phrases(query))
    normalized_query = re.sub(r"\s+", " ", query.lower())
    for alias, values in _ENGLISH_QUERY_ALIASES.items():
        if alias in normalized_query:
            phrases.extend(values)
    for term in _DOMAIN_PHRASES:
        if term in query:
            phrases.append(term)
    return _unique(phrases)


def _extract_ascii_query_phrases(query: str) -> list[str]:
    words = [
        word.lower()
        for word in _ASCII_WORD_RE.findall(query)
        if len(word) >= 3 and word.lower() not in _EN_STOPWORDS
    ]
    phrases: list[str] = []
    for size in (3, 2):
        for index in range(0, max(0, len(words) - size + 1)):
            phrase = " ".join(words[index : index + size])
            if len(phrase) >= 8:
                phrases.append(phrase)
    return phrases[:8]


def expand_keywords(query: str, keywords: list[str]) -> list[str]:
    expanded: list[str] = []
    normalized = _normalize(query)
    for key, values in _DOMAIN_SYNONYMS.items():
        if _normalize(key) in normalized:
            expanded.extend(values)
    for keyword in keywords:
        for key, values in _DOMAIN_SYNONYMS.items():
            if _normalize(key) == _normalize(keyword):
                expanded.extend(values)
    return _unique(expanded)


def _merge_ascii_cjk_tokens(tokens: list[str]) -> list[str]:
    merged_tokens: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if (
            re.fullmatch(r"[A-Za-z][A-Za-z0-9._-]*", token)
            and index + 1 < len(tokens)
            and re.fullmatch(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]+", tokens[index + 1])
            and len(token) + len(tokens[index + 1]) <= 8
        ):
            merged_tokens.extend([token + tokens[index + 1], token, tokens[index + 1]])
            index += 2
            continue
        merged_tokens.append(token)
        index += 1
    return merged_tokens


def _prioritize_search_terms(
    *,
    phrases: list[str],
    keywords: list[str],
    products: list[str],
) -> list[str]:
    prioritized = _unique([*products, *phrases, *keywords])
    return sorted(
        prioritized,
        key=lambda term: (
            term not in products,
            term not in phrases,
            -len(str(term).split()),
            -len(str(term)),
        ),
    )


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


class SQLiteRetriever:
    """Keyword-based retriever backed by the SQLite knowledge index."""

    def __init__(
        self,
        db_path: Path = settings.processed_dir / "index.sqlite",
        *,
        vector_searcher: VectorSearcher | None = None,
    ) -> None:
        self.db_path = db_path
        self.vector_searcher = vector_searcher or SQLiteVectorSearcher(db_path)

    def retrieval_status(self) -> dict[str, Any]:
        """Describe the active retrieval stack for docs/debug output."""

        return {
            "strategy": "hybrid_lexical_with_optional_vector",
            "channels": ["like", "fts5_bm25", "vector"],
            "lexical_channels": ["like", "fts5_bm25"],
            "vector": describe_vector_retrieval(db_path=self.db_path),
        }

    def search(self, query: str, *, limit: int = 5) -> list[dict[str, Any]]:
        """Return reranked chunks for the query."""

        if not self.db_path.exists():
            raise FileNotFoundError(f"index not found: {self.db_path}")

        analysis = analyze_query(query)
        keywords = analysis.keywords or [query.strip()]
        fetch_limit = max(limit * 20, 180)

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            like_candidates = self._candidate_search(
                conn,
                keywords=keywords,
                phrases=analysis.phrases,
                products=analysis.products,
                limit=fetch_limit,
            )
            fts_candidates = self._fts_candidate_search(
                conn,
                keywords=keywords,
                phrases=analysis.phrases,
                products=analysis.products,
                limit=fetch_limit,
            )
        finally:
            conn.close()

        vector_candidates = self._vector_candidate_search(query=query, limit=fetch_limit)
        candidate_rows = self._merge_candidate_rows(like_candidates, fts_candidates, vector_candidates)
        scored = [
            self._score_row(dict(row), analysis)
            for row in candidate_rows
        ]
        scored = [row for row in scored if row["_score"] > 0]
        scored.sort(
            key=lambda item: (
                item["_score"],
                item["_product_match"],
                item["_title_hits"],
                len(_parse_json_list(item.get("image_ids"))),
            ),
            reverse=True,
        )
        return scored[:limit]

    # ------------------------------------------------------------------

    def _candidate_search(
        self,
        conn: sqlite3.Connection,
        *,
        keywords: list[str],
        phrases: list[str],
        products: list[str],
        limit: int,
    ) -> list[sqlite3.Row]:
        """Fetch broad candidates, then Python does the precise scoring."""

        where_parts: list[str] = []
        params: list[str] = []
        prioritized_terms = _prioritize_search_terms(
            phrases=phrases,
            keywords=keywords,
            products=products,
        )

        if not prioritized_terms:
            return []

        product_clause = ""
        product_params: list[str] = []
        if products:
            placeholders = ", ".join("?" for _ in products)
            product_clause = f"product_name IN ({placeholders}) AND "
            product_params.extend(products)

        merged: dict[str, sqlite3.Row] = {}
        per_term_limit = max(16, min(80, limit // max(1, min(len(prioritized_terms), 6))))
        single_term_sql = f"""
            SELECT *
            FROM chunks
            WHERE {product_clause}(text LIKE ? OR title LIKE ? OR product_name LIKE ?)
            ORDER BY
              (CASE WHEN title LIKE ? THEN 3 ELSE 0 END) +
              (CASE WHEN product_name LIKE ? THEN 2 ELSE 0 END) +
              (CASE WHEN text LIKE ? THEN 1 ELSE 0 END) DESC,
              LENGTH(title) ASC,
              LENGTH(text) ASC
            LIMIT ?
        """
        for term in prioritized_terms[:14]:
            like = f"%{term}%"
            rows = conn.execute(
                single_term_sql,
                [
                    *product_params,
                    like,
                    like,
                    like,
                    like,
                    like,
                    like,
                    per_term_limit,
                ],
            ).fetchall()
            for row in rows:
                merged[str(row["chunk_id"])] = row

        if len(merged) >= max(limit, 40):
            return list(merged.values())

        for term in prioritized_terms[:10]:
            like = f"%{term}%"
            where_parts.append("(text LIKE ? OR title LIKE ? OR product_name LIKE ?)")
            params.extend([like, like, like])

        if not where_parts:
            return list(merged.values())

        fallback_sql = f"""
            SELECT *
            FROM chunks
            WHERE {product_clause}({' OR '.join(where_parts)})
            ORDER BY LENGTH(title) ASC, LENGTH(text) ASC
            LIMIT ?
        """
        for row in conn.execute(fallback_sql, [*product_params, *params, limit]).fetchall():
            merged[str(row["chunk_id"])] = row
        return list(merged.values())

    def _fts_candidate_search(
        self,
        conn: sqlite3.Connection,
        *,
        keywords: list[str],
        phrases: list[str],
        products: list[str],
        limit: int,
    ) -> list[sqlite3.Row]:
        """Fetch FTS5 candidates when the virtual table is available."""

        usable_terms = [
            _sanitize_fts_term(term)
            for term in _unique([*products, *phrases, *keywords])
            if _sanitize_fts_term(term)
        ]
        if not usable_terms:
            return []

        match_terms = usable_terms[:10]
        match_query = " OR ".join(f'"{term}"' for term in match_terms)
        try:
            rows = conn.execute(
                """
                SELECT
                  chunks.*,
                  bm25(chunks_fts) AS fts_rank,
                  1 AS fts_hit
                FROM chunks_fts
                JOIN chunks ON chunks.chunk_id = chunks_fts.chunk_id
                WHERE chunks_fts MATCH ?
                LIMIT ?
                """,
                (match_query, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return []

        if not products:
            return rows
        return [
            row
            for row in rows
            if str(row["product_name"]) in products
        ]

    def _merge_candidate_rows(
        self,
        like_rows: list[sqlite3.Row],
        fts_rows: list[sqlite3.Row],
        vector_rows: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        for row in like_rows:
            record = dict(row)
            record.setdefault("fts_rank", None)
            record.setdefault("fts_hit", 0)
            record["_retrieval_channels"] = ["like"]
            merged[str(record["chunk_id"])] = record

        for row in fts_rows:
            record = dict(row)
            chunk_id = str(record["chunk_id"])
            existing = merged.get(chunk_id)
            if existing is None:
                record["_retrieval_channels"] = ["fts"]
                merged[chunk_id] = record
                continue
            existing["fts_hit"] = max(int(existing.get("fts_hit", 0)), int(record.get("fts_hit", 0)))
            if record.get("fts_rank") is not None:
                existing["fts_rank"] = record.get("fts_rank")
            _append_channel(existing, "fts")
        for record in vector_rows or []:
            chunk_id = str(record.get("chunk_id", ""))
            if not chunk_id:
                continue
            existing = merged.get(chunk_id)
            if existing is None:
                record.setdefault("fts_rank", None)
                record.setdefault("fts_hit", 0)
                record["_retrieval_channels"] = ["vector"]
                merged[chunk_id] = record
                continue
            if record.get("_vector_score") is not None:
                existing["_vector_score"] = record.get("_vector_score")
            _append_channel(existing, "vector")
        return list(merged.values())

    def _vector_candidate_search(self, *, query: str, limit: int) -> list[dict[str, Any]]:
        """Fetch optional vector candidates if a vector searcher is configured."""

        try:
            rows = self.vector_searcher.search(query, limit=limit)
        except Exception:
            return []
        return [dict(row) for row in rows if row.get("chunk_id")]

    def _score_row(self, row: dict[str, Any], analysis: QueryAnalysis) -> dict[str, Any]:
        title = str(row.get("title", ""))
        text = str(row.get("text", ""))
        product = str(row.get("product_name", ""))
        title_norm = _normalize(title)
        text_norm = _normalize(text)
        product_norm = _normalize(product)
        exact_keywords = {
            keyword
            for keyword in analysis.keywords
            if keyword not in analysis.expanded_keywords
        }

        score = 0.0
        title_hits = 0
        text_hits = 0
        product_match = 0
        matched_keywords: list[str] = []
        matched_distinct_terms: set[str] = set()

        if analysis.products:
            if product in analysis.products:
                product_match = 1
                score += 20.0
                matched_distinct_terms.add(product)
            else:
                score -= 4.0
        elif product == "汇总英文":
            score -= 3.0

        for model in analysis.models:
            model_norm = _normalize(model)
            if model_norm in title_norm:
                score += 12.0
                title_hits += 1
                matched_keywords.append(model)
                matched_distinct_terms.add(model)
            elif model_norm in text_norm:
                score += 7.0
                text_hits += 1
                matched_keywords.append(model)
                matched_distinct_terms.add(model)

        for keyword in analysis.keywords:
            if keyword in analysis.products:
                continue
            kw = _normalize(keyword)
            if not kw:
                continue
            is_expanded_only = keyword in analysis.expanded_keywords and keyword not in exact_keywords
            product_boost = 0.6 if is_expanded_only else 1.0
            title_boost = 2.0 if is_expanded_only else 3.5
            text_boost = 0.8 if is_expanded_only else 1.2
            prefix_boost = 0.5 if is_expanded_only else 2.0
            domain_title_boost = 2.0 if is_expanded_only else 4.0
            domain_prefix_boost = 3.0 if is_expanded_only else 8.0
            domain_text_boost = 0.8 if is_expanded_only else 1.5
            if kw in product_norm:
                score += product_boost
                product_match = max(product_match, 1)
                matched_keywords.append(keyword)
                matched_distinct_terms.add(keyword)
            if kw in title_norm:
                score += title_boost
                title_hits += 1
                matched_keywords.append(keyword)
                matched_distinct_terms.add(keyword)
                if title_norm.startswith(kw):
                    score += prefix_boost
                    if keyword in _DOMAIN_PHRASES:
                        score += domain_prefix_boost
                if keyword in _DOMAIN_PHRASES:
                    score += domain_title_boost
            elif kw in text_norm:
                score += text_boost
                text_hits += 1
                matched_keywords.append(keyword)
                matched_distinct_terms.add(keyword)
                if keyword in _DOMAIN_PHRASES:
                    score += domain_text_boost

        for phrase in analysis.phrases:
            phrase_norm = _normalize(phrase)
            if len(phrase_norm) < 4:
                continue
            if phrase_norm in title_norm:
                score += 10.0
                if " " in phrase:
                    score += 2.0
                matched_distinct_terms.add(phrase)
            elif phrase_norm in text_norm:
                score += 2.0
                matched_distinct_terms.add(phrase)

        for query_terms, title_terms, boost in _TITLE_INTENT_BOOSTS:
            if any(term in analysis.keywords or term in analysis.expanded_keywords for term in query_terms):
                if any(_normalize(term) in title_norm for term in title_terms):
                    score += boost

        image_ids = _parse_json_list(row.get("image_ids"))
        if image_ids and any(term in analysis.keywords for term in ("指示灯", "表带", "尺寸", "安装", "更换")):
            score += 1.2

        metadata = _parse_json_object(row.get("metadata"))
        clean_score = metadata.get("clean_score")
        if isinstance(clean_score, (int, float)):
            score += (float(clean_score) - 0.5) * 2.0
        if metadata.get("is_toc"):
            score -= 3.0
        if metadata.get("has_ocr_noise"):
            score -= 1.5
        intents = _detect_query_semantic_intents(analysis)
        score += _semantic_alignment_score(metadata=metadata, analysis=analysis, intents=intents)
        score += _exact_phrase_alignment_boost(title_norm=title_norm, text_norm=text_norm, analysis=analysis)
        score += _query_structure_alignment_score(
            title_norm=title_norm,
            text_norm=text_norm,
            metadata=metadata,
            intents=intents,
        )

        if int(row.get("fts_hit", 0)):
            score += 5.0
            rank_bonus = _fts_rank_bonus(row.get("fts_rank"))
            score += rank_bonus
        vector_score = row.get("_vector_score")
        if isinstance(vector_score, (int, float)):
            score += min(max(float(vector_score), 0.0), 1.0) * 8.0

        if product == "汇总英文":
            score += _english_manual_alignment_score(title_norm=title_norm, text_norm=text_norm, analysis=analysis)
            score += _english_title_focus_score(title_norm=title_norm, text_norm=text_norm, analysis=analysis)
            query_groups = _detect_english_domain_groups(" ".join([*analysis.keywords, *analysis.phrases]))
            domain_label = str(metadata.get("domain_label") or "")
            if domain_label and query_groups:
                if domain_label in query_groups:
                    score += 12.0
                else:
                    score -= 4.0

        if title_hits >= 2:
            score += 3.0
        if title_hits + text_hits >= 4:
            score += 2.0

        exact_intent_terms = _unique(
            [
                *analysis.models,
                *analysis.phrases[:4],
                *[
                    keyword
                    for keyword in analysis.keywords
                    if keyword not in analysis.products and keyword not in analysis.expanded_keywords
                ],
            ]
        )
        if exact_intent_terms:
            has_exact_intent_match = any(term in matched_distinct_terms for term in exact_intent_terms)
            has_expansion_only_match = any(term in matched_distinct_terms for term in analysis.expanded_keywords)
            if has_expansion_only_match and not has_exact_intent_match:
                score -= 4.0

        signal_terms = _unique([*analysis.products, *analysis.models, *analysis.expanded_keywords, *analysis.phrases[:4]])
        if signal_terms:
            coverage = len(matched_distinct_terms) / max(1, min(len(signal_terms), 6))
            score += min(coverage * 4.0, 4.0)

        row["_score"] = round(score, 3)
        row["_matched_keywords"] = _unique(matched_keywords)
        row["_query_products"] = analysis.products
        row["_query_models"] = analysis.models
        row["_title_hits"] = title_hits
        row["_text_hits"] = text_hits
        row["_product_match"] = product_match
        row["_fts_rank"] = row.get("fts_rank")
        row["_fts_hit"] = int(row.get("fts_hit", 0))
        row["_vector_score"] = row.get("_vector_score")
        row["_retrieval_channels"] = _unique(row.get("_retrieval_channels", []))
        row["_retrieval_strategy"] = "like+fts+vector_optional"
        return row


def _english_manual_alignment_score(*, title_norm: str, text_norm: str, analysis: QueryAnalysis) -> float:
    english_keywords = _unique(
        keyword
        for keyword in analysis.keywords
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9._-]*", keyword)
    )
    english_phrases = _unique(
        phrase
        for phrase in analysis.phrases
        if _ASCII_WORD_RE.search(phrase)
    )
    if not english_keywords and not english_phrases:
        return 0.0

    keyword_hits = 0
    title_keyword_hits = 0
    long_keyword_hits = 0
    for keyword in english_keywords:
        token = _normalize(keyword)
        if not token:
            continue
        if token in title_norm:
            title_keyword_hits += 1
            keyword_hits += 1
            if len(token) >= 6:
                long_keyword_hits += 1
        elif token in text_norm:
            keyword_hits += 1
            if len(token) >= 6:
                long_keyword_hits += 1

    phrase_hits = 0
    title_phrase_hits = 0
    for phrase in english_phrases:
        token = _normalize(phrase)
        if not token:
            continue
        if token in title_norm:
            title_phrase_hits += 1
            phrase_hits += 1
        elif token in text_norm:
            phrase_hits += 1

    score = 0.0
    score += min(keyword_hits * 1.6, 9.0)
    score += min(title_keyword_hits * 1.8, 6.0)
    score += min(phrase_hits * 4.0, 12.0)
    score += min(title_phrase_hits * 3.0, 9.0)

    if keyword_hits <= 1 and phrase_hits == 0:
        if long_keyword_hits >= 1:
            score += 4.0
        else:
            score -= 12.0
    elif keyword_hits >= 3 or phrase_hits >= 1:
        score += 4.0

    query_groups = _detect_english_domain_groups(
        " ".join([*analysis.keywords, *analysis.phrases])
    )
    if query_groups:
        row_groups = _detect_english_domain_groups(f"{title_norm} {text_norm}")
        overlap = query_groups & row_groups
        if overlap:
            score += 10.0 * len(overlap)
        elif row_groups:
            score -= 5.0
        else:
            score -= 2.0
    return score


def _english_title_focus_score(*, title_norm: str, text_norm: str, analysis: QueryAnalysis) -> float:
    english_exact_keywords = _unique(
        keyword
        for keyword in analysis.keywords
        if keyword not in analysis.expanded_keywords and re.fullmatch(r"[A-Za-z][A-Za-z0-9._-]*", keyword)
    )
    if not english_exact_keywords:
        return 0.0

    title_hits = 0
    text_only_hits = 0
    for keyword in english_exact_keywords:
        token = _normalize(keyword)
        if not token:
            continue
        if token in title_norm:
            title_hits += 1
        elif token in text_norm:
            text_only_hits += 1

    score = min(title_hits * 2.2, 6.0)
    if title_hits == 0 and text_only_hits >= 2:
        score -= 3.0
    elif title_hits == 0 and text_only_hits == 1:
        score -= 1.2
    return score


def _append_channel(row: dict[str, Any], channel: str) -> None:
    channels = list(row.get("_retrieval_channels", []))
    if channel not in channels:
        channels.append(channel)
    row["_retrieval_channels"] = channels


def _detect_english_domain_groups(text: str) -> set[str]:
    normalized = re.sub(r"\s+", " ", text.lower())
    groups: set[str] = set()
    for group_name, hints in _ENGLISH_DOMAIN_HINTS.items():
        for hint in hints:
            if _english_hint_occurs(normalized, hint):
                groups.add(group_name)
                break
    return groups


def _english_hint_occurs(normalized_text: str, hint: str) -> bool:
    normalized_hint = re.sub(r"\s+", " ", hint.lower()).strip()
    if not normalized_hint:
        return False
    if " " in normalized_hint or "-" in normalized_hint:
        return normalized_hint in normalized_text
    return bool(re.search(rf"\b{re.escape(normalized_hint)}\b", normalized_text))


def _exact_phrase_alignment_boost(*, title_norm: str, text_norm: str, analysis: QueryAnalysis) -> float:
    score = 0.0
    for phrase in analysis.phrases[:6]:
        token = _normalize(phrase)
        if len(token) < 4:
            continue
        if title_norm.startswith(token):
            score += 4.5
        elif token in title_norm:
            score += 2.5
        elif token in text_norm:
            score += 0.8
    return score


def _query_structure_alignment_score(
    *,
    title_norm: str,
    text_norm: str,
    metadata: dict[str, Any],
    intents: set[str],
) -> float:
    if not intents:
        return 0.0

    combined = f"{title_norm} {text_norm}"
    score = 0.0
    generic_title = any(_normalize(hint) in title_norm for hint in _GENERIC_SECTION_HINTS)
    procedure_cue_hits = sum(1 for cue in _PROCEDURE_CUES if _normalize(cue) in combined)
    troubleshooting_cue_hits = sum(1 for cue in _TROUBLESHOOTING_CUES if _normalize(cue) in combined)
    specification_cue_hits = sum(1 for cue in _SPECIFICATION_CUES if _normalize(cue) in combined)

    if "procedure" in intents:
        score += min(procedure_cue_hits * 1.1, 4.5)
        if metadata.get("is_procedure"):
            score += 1.5
        if generic_title and procedure_cue_hits == 0:
            score -= 3.0

    if "troubleshooting" in intents:
        score += min(troubleshooting_cue_hits * 1.2, 4.8)
        if generic_title and troubleshooting_cue_hits == 0:
            score -= 3.2

    if "specification" in intents:
        score += min(specification_cue_hits * 1.0, 3.8)
        if generic_title and specification_cue_hits == 0:
            score -= 2.2

    if "parts_list" in intents and metadata.get("semantic_type") == "parts_list":
        score += 1.8
    return score


def _semantic_alignment_score(*, metadata: dict[str, Any], analysis: QueryAnalysis, intents: set[str] | None = None) -> float:
    semantic_type = str(metadata.get("semantic_type") or "general")
    if not semantic_type or semantic_type == "general":
        return 0.0

    intents = intents or _detect_query_semantic_intents(analysis)
    if not intents:
        return -1.5 if metadata.get("is_warning_only") else 0.0

    score = 0.0
    if "procedure" in intents:
        if semantic_type == "procedure":
            score += 5.5
        elif semantic_type == "safety_warning":
            score -= 3.5
    if "safety_warning" in intents:
        if semantic_type == "safety_warning":
            score += 5.0
        elif semantic_type == "procedure":
            score += 0.8
    if "troubleshooting" in intents:
        if semantic_type == "troubleshooting":
            score += 6.0
        elif semantic_type == "procedure":
            score += 1.2
        elif semantic_type == "safety_warning":
            score -= 2.0
    if "parts_list" in intents:
        if semantic_type == "parts_list":
            score += 5.0
        elif semantic_type == "specification":
            score += 1.5
    if "specification" in intents:
        if semantic_type == "specification":
            score += 5.0
        elif semantic_type == "parts_list":
            score += 1.0

    if metadata.get("is_warning_only") and "safety_warning" not in intents:
        score -= 1.5
    return score


def _detect_query_semantic_intents(analysis: QueryAnalysis) -> set[str]:
    query_text = " ".join(
        [
            analysis.raw_query.lower(),
            *[keyword.lower() for keyword in analysis.keywords],
            *[phrase.lower() for phrase in analysis.phrases],
        ]
    )
    intents: set[str] = set()
    for intent, hints in _SEMANTIC_INTENT_HINTS.items():
        if any(hint.lower() in query_text for hint in hints):
            intents.add(intent)
    return intents


def _parse_json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if not isinstance(value, str):
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return [str(v) for v in parsed] if isinstance(parsed, list) else []


def _parse_json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", text.lower())


def _normalize_query_text(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"[，,。；;：:！!？?\(\)\[\]\"“”‘’]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def _extract_long_token_terms(token: str) -> list[str]:
    terms: list[str] = []
    for phrase in _DOMAIN_PHRASES:
        if phrase in token:
            terms.append(phrase)
    for key, values in _DOMAIN_SYNONYMS.items():
        if key in token:
            terms.append(key)
            terms.extend(values)

    parts = [part.strip() for part in _LONG_TOKEN_SPLIT_RE.split(token) if 2 <= len(part.strip()) <= 8]
    terms.extend(parts[:6])

    # Always extract 2-3 char n-grams for Chinese text matching
    for size in (3, 2):
        step = max(1, size // 2)
        for index in range(0, max(0, len(token) - size + 1), step):
            ngram = token[index : index + size]
            if ngram and len(ngram) >= 2:
                terms.append(ngram)
    return _unique(terms)[:20]


def _sanitize_fts_term(term: str) -> str:
    cleaned = _FTS_UNSAFE_RE.sub(" ", str(term)).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned if len(cleaned) >= 2 else ""


def _fts_rank_bonus(value: Any) -> float:
    try:
        rank = float(value)
    except (TypeError, ValueError):
        return 0.0
    rank = abs(rank)
    return max(0.0, 4.0 - min(4.0, math.log1p(rank + 1e-6)))


def _unique(values: Any) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value)
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result
