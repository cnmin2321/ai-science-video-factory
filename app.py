r"""
AI 科普视频工厂 · GitHub / Streamlit 单文件版
================================================

这是从本地 D:\AI-Video-Factory 项目整理出的自包含 Streamlit 应用，目标是：
1. 可以直接作为 GitHub 仓库根目录的 app.py 提交；
2. 保留“主题 -> 结构化科普脚本 -> 分镜 -> 画面 -> 字幕/视频”的核心逻辑；
3. 保留对本地 Ollama 的真实 HTTP 调用（/api/chat、/api/tags）；
4. 在没有 Ollama、没有 Piper、没有 FFmpeg 的云端环境里，仍能生成脚本、
   分镜预览和可下载的 SRT/JSON，不会直接报错退出。

部署：
    pip install -r requirements.txt
    streamlit run app.py

Ollama 配置（本地或远程）：
    环境变量：
        OLLAMA_BASE_URL=http://127.0.0.1:11434
        OLLAMA_MODEL=qwenlocal
    或 Streamlit secrets：
        OLLAMA_BASE_URL = "http://127.0.0.1:11434"
        OLLAMA_MODEL = "qwenlocal"

说明：
- GitHub/Streamlit Cloud 上的 127.0.0.1 指向容器自身，不是你的电脑；
  云端要调用 Ollama，需要把 OLLAMA_BASE_URL 指向可访问的远程地址/隧道。
- 没有 Ollama 时，默认主题会使用内置的已核查离线脚本；内置《为什么 AI 会产生幻觉？》和《科普汽车发展史》两套事实库。
- 没有 Piper 时生成静音视频；没有 FFmpeg 时只提供分镜 PNG/SRT/JSON 下载。
"""

from __future__ import annotations

import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import wave
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# ---------------------------------------------------------------------------
# 基础配置
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AI 科普视频工厂",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _secret(key: str, default: str = "") -> str:
    try:
        value = st.secrets.get(key, None)
        if value:
            return str(value)
    except Exception:
        pass
    return os.getenv(key, default)


OLLAMA_BASE_URL = _secret("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = _secret("OLLAMA_MODEL", "qwenlocal")
OLLAMA_TIMEOUT = int(_secret("OLLAMA_TIMEOUT", "600"))
APP_ROOT = Path(__file__).resolve().parent
APP_DIR = Path(tempfile.gettempdir()) / "ai_science_factory_web"
APP_DIR.mkdir(parents=True, exist_ok=True)

W, H = 1080, 1920
FPS = 30
BG_TOP = (3, 7, 16)
BG_BOTTOM = (7, 14, 30)
WHITE = (241, 248, 255)
MUTED = (151, 168, 190)
DIM = (100, 119, 143)
CYAN = (34, 211, 238)
BLUE = (59, 130, 246)
PURPLE = (167, 139, 250)
GREEN = (52, 211, 153)
AMBER = (251, 191, 36)
RED = (248, 113, 113)

ALLOWED_DIAGRAMS = {
    "hook", "definition", "next_token", "plausibility", "data_noise",
    "reward", "snowball", "rag", "uncertainty", "mitigation", "summary",
    "timeline", "comparison",
}


# ---------------------------------------------------------------------------
# 内置事实库（与本地项目 config/topics/ai_hallucination.json 同源，做了精简）
# ---------------------------------------------------------------------------

PROFILE: dict[str, Any] = {
    "topic": "为什么 AI 会产生幻觉？",
    "aliases": ["AI 幻觉", "大模型幻觉", "AI为什么会胡说八道", "为什么AI会一本正经地胡说八道"],
    "knowledge_goal": "解释大语言模型幻觉的定义、生成机制、数据与偏好因素、RAG 的局限、检测与缓解方法。",
    "sources": [
        {"id": "S1", "title": "Survey of Hallucination in Natural Language Generation", "year": 2023, "url": "https://arxiv.org/abs/2202.03629"},
        {"id": "S2", "title": "A Survey on Hallucination in Large Language Models", "year": 2023, "url": "https://arxiv.org/abs/2311.05232"},
        {"id": "S3", "title": "Calibrated Language Models Must Hallucinate", "year": 2024, "url": "https://arxiv.org/abs/2311.14648"},
        {"id": "S4", "title": "Detecting hallucinations using semantic entropy", "year": 2024, "url": "https://www.nature.com/articles/s41586-024-07421-0"},
        {"id": "S5", "title": "Towards Understanding Sycophancy in Language Models", "year": 2023, "url": "https://arxiv.org/abs/2310.13548"},
        {"id": "S6", "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", "year": 2020, "url": "https://arxiv.org/abs/2005.11401"},
        {"id": "S7", "title": "Lost in the Middle: How Language Models Use Long Contexts", "year": 2023, "url": "https://arxiv.org/abs/2307.03172"},
        {"id": "S8", "title": "Enabling Large Language Models to Generate Text with Citations", "year": 2023, "url": "https://arxiv.org/abs/2305.14627"},
        {"id": "S9", "title": "Teaching Models to Express Their Uncertainty in Words", "year": 2022, "url": "https://arxiv.org/abs/2205.14334"},
        {"id": "S10", "title": "GPT-4 Technical Report", "year": 2023, "url": "https://arxiv.org/abs/2303.08774"},
    ],
    "claims": [
        {"id": "C1", "text": "大语言模型幻觉指生成看似流畅、但缺乏事实依据，或与可靠事实/输入不一致的内容。", "source_ids": ["S1", "S2"], "risk": "high", "status": "supported"},
        {"id": "C2", "text": "主流大模型以自回归方式逐词元预测下一个词元，训练目标是建模语言分布，而不是查询事实数据库。", "source_ids": ["S1"], "risk": "high", "status": "supported"},
        {"id": "C3", "text": "模型输出的是概率上更可能的续写；语言连贯只说明统计上常见，并不保证事实为真。", "source_ids": ["S3", "S1"], "risk": "high", "status": "supported"},
        {"id": "C4", "text": "训练语料来自大规模网页、书籍等，可能包含错误、过时、矛盾或覆盖不足的信息。", "source_ids": ["S2"], "risk": "medium", "status": "supported"},
        {"id": "C5", "text": "偏好优化可能奖励自信、顺耳的回答，从而放大猜测；偏好优化不等于事实优化。", "source_ids": ["S5"], "risk": "high", "status": "supported"},
        {"id": "C6", "text": "自回归生成逐词展开，早期错误会进入后续上下文，模型可能继续生成自洽但错误的解释。", "source_ids": ["S1", "S2"], "risk": "medium", "status": "supported"},
        {"id": "C7", "text": "RAG 通过引入外部文档降低幻觉，但检索错误、文档冲突或忽略证据时仍会产生幻觉。", "source_ids": ["S6", "S7"], "risk": "high", "status": "supported"},
        {"id": "C8", "text": "语义熵等方法通过多次采样并比较答案语义是否一致来估计不确定性。", "source_ids": ["S4"], "risk": "high", "status": "supported"},
        {"id": "C9", "text": "降低幻觉的工程手段包括引用来源、外部核验、允许模型表达不确定/拒答，以及人工审核。", "source_ids": ["S8", "S9"], "risk": "medium", "status": "supported"},
        {"id": "C10", "text": "幻觉不是模型具有欺骗意图的证据，而是当前训练与生成机制的局限。", "source_ids": ["S1", "S2"], "risk": "medium", "status": "supported"},
        {"id": "C11", "text": "先进商用模型也不能保证零幻觉；高风险场景仍需外部验证与人工把关。", "source_ids": ["S10"], "risk": "high", "status": "supported"},
    ],
}

FALLBACK_SCRIPT: dict[str, Any] = {
    "title": "为什么 AI 会产生幻觉？",
    "subtitle": "大模型“一本正经胡说八道”的科学解释",
    "hook": "AI 说得越自信，我们越容易忘记：它可能只是在生成一个“最像答案”的答案。",
    "knowledge_goal": PROFILE["knowledge_goal"],
    "target_duration": 120,
    "scenes": [
        {"id": 1, "heading": "它不是在“撒谎”", "narration": "如果 AI 用非常肯定的语气告诉你一个错误事实，你会不会上当？这种看似流畅、实则失实的回答，就是大模型幻觉。", "on_screen": ["流畅 ≠ 正确", "“一本正经地胡说八道”"], "key_takeaway": "幻觉：说得像真的，但不一定有事实依据。", "diagram": "hook", "keywords": ["幻觉", "流畅", "失实"], "claim_ids": ["C1"]},
        {"id": 2, "heading": "什么是 AI 幻觉", "narration": "幻觉不一定是语法错误。它可能条理清晰、语气自信，甚至编出不存在的论文。判断标准只有一条：有没有可靠证据。", "on_screen": ["不是错别字", "不是语气问题", "看证据"], "key_takeaway": "判断幻觉，看证据，不看语气。", "diagram": "definition", "keywords": ["证据", "自信", "论文"], "claim_ids": ["C1", "C10"]},
        {"id": 3, "heading": "一个词元一个词元地猜", "narration": "大模型不是在数据库里查答案，而是根据上文，一个词元一个词元地预测下一个词最可能是什么。", "on_screen": ["猫坐在 ___ 上", "垫子 72%", "沙发 18%", "月亮 0.3%"], "key_takeaway": "它生成的是“最可能的续写”，不是“已核验的事实”。", "diagram": "next_token", "keywords": ["词元", "预测", "概率"], "claim_ids": ["C2"]},
        {"id": 4, "heading": "连贯 ≠ 真实", "narration": "它优化的是“像真的”文本。语言连贯只代表概率高，不代表事实为真。", "on_screen": ["高概率", "高流畅", "≠ 高可信"], "key_takeaway": "像真的，不等于就是真的。", "diagram": "plausibility", "keywords": ["连贯", "概率", "真实"], "claim_ids": ["C3"]},
        {"id": 5, "heading": "学到的世界有噪声", "narration": "它的知识来自海量网页、书籍和代码。语料里有错误、过时信息、矛盾说法，也有覆盖不到的长尾问题。", "on_screen": ["网页", "书籍", "代码", "噪声与空白"], "key_takeaway": "语料有噪声，模型也会把噪声学进去。", "diagram": "data_noise", "keywords": ["语料", "噪声", "长尾"], "claim_ids": ["C4"]},
        {"id": 6, "heading": "讨好人类，也可能跑偏", "narration": "为了让回答更讨喜，模型还会做偏好优化。可自信、顺耳、像专家，往往更容易得高分，猜测也可能被奖励。", "on_screen": ["人类偏好", "自信 + 顺耳", "≠ 事实正确"], "key_takeaway": "偏好优化优化的是“讨喜”，不是“求真”。", "diagram": "reward", "keywords": ["偏好优化", "自信", "奖励"], "claim_ids": ["C5"]},
        {"id": 7, "heading": "错误会滚雪球", "narration": "生成是逐词进行的。开头一旦选错，错误就会进入下文，模型再围绕它继续编，越编越自洽。", "on_screen": ["错误前提", "→ 自洽解释", "→ 更像真的"], "key_takeaway": "第一个错误，会被后面的文字不断“圆回来”。", "diagram": "snowball", "keywords": ["错误累积", "自洽", "上下文"], "claim_ids": ["C6"]},
        {"id": 8, "heading": "RAG 不是万能药", "narration": "检索增强生成会把外部资料放进上下文。它能降低幻觉，但检索错了、资料矛盾，或者模型没用好证据，错误仍会发生。", "on_screen": ["问题 → 检索 → 文档 → 回答", "降低 ≠ 消除"], "key_takeaway": "RAG 能补证据，但不能替你做最终判断。", "diagram": "rag", "keywords": ["RAG", "检索", "证据"], "claim_ids": ["C7"]},
        {"id": 9, "heading": "如何发现它在编", "narration": "怎么发现它在编？让模型对同一问题多次作答，再比较答案在语义上是否一致。越分散，通常越不确定。", "on_screen": ["同一问题 × N 次", "语义聚类", "不确定性 ↑"], "key_takeaway": "答案越“飘”，越值得警惕。", "diagram": "uncertainty", "keywords": ["语义熵", "采样", "不确定"], "claim_ids": ["C8"]},
        {"id": 10, "heading": "把事实关进流程", "narration": "更可靠的做法，是要求引用来源，对关键数字做外部核验，允许模型说“不知道”，并保留人工审核。", "on_screen": ["引用来源", "外部核验", "允许拒答", "人工审核"], "key_takeaway": "流程上的证据，比模型语气更可靠。", "diagram": "mitigation", "keywords": ["引用", "核验", "人工审核"], "claim_ids": ["C9", "C11"]},
        {"id": 11, "heading": "记住这三句话", "narration": "记住三句话：流畅不等于真实；自信不等于可靠；AI 可以帮你查资料，但最后一道关，必须由证据和人把守。", "on_screen": ["流畅 ≠ 真实", "自信 ≠ 可靠", "证据 + 人 把关"], "key_takeaway": "AI 可以加速理解，但不能替代证据。", "diagram": "summary", "keywords": ["真实", "可靠", "证据"], "claim_ids": ["C1", "C10"]},
    ],
    "summary": ["流畅不等于真实", "自信不等于可靠", "证据和人工审核是最后一道关"],
    "platform": {"description": "AI 为什么会“一本正经地胡说八道”？用 2 分钟看懂大模型幻觉的机制、局限与应对方法。", "hashtags": ["#AI科普", "#人工智能", "#大模型", "#AI幻觉"]},
    "meta": {"source": "builtin_fallback", "model": None, "reason": "Ollama unavailable or validation failed"},
}


# ---------------------------------------------------------------------------
# 第二个内置事实库：汽车发展史（可继续扩展更多 topic profile）
# ---------------------------------------------------------------------------

AUTO_PROFILE: dict[str, Any] = {
    "topic": "科普汽车发展史",
    "aliases": ["汽车发展史", "汽车史", "汽车的历史", "汽车是怎么发明的"],
    "series": "MOBILITY SCIENCE LAB",
    "visual_subtitle": "汽车发展史 · 科学解释",
    "knowledge_goal": "按时间线解释汽车从 1886 年诞生、福特流水线、安全与排放法规、混合动力到电动化与智能驾驶的关键变化。",
    "sources": [
        {"id": "S1", "title": "History of the automobile", "year": 2024, "url": "https://en.wikipedia.org/wiki/History_of_the_automobile"},
        {"id": "S2", "title": "Benz Patent-Motorwagen", "year": 2024, "url": "https://en.wikipedia.org/wiki/Benz_Patent-Motorwagen"},
        {"id": "S3", "title": "Ford Model T", "year": 2024, "url": "https://en.wikipedia.org/wiki/Ford_Model_T"},
        {"id": "S4", "title": "Assembly line", "year": 2024, "url": "https://en.wikipedia.org/wiki/Assembly_line"},
        {"id": "S5", "title": "Seat belt", "year": 2024, "url": "https://en.wikipedia.org/wiki/Seat_belt"},
        {"id": "S6", "title": "Catalytic converter", "year": 2024, "url": "https://en.wikipedia.org/wiki/Catalytic_converter"},
        {"id": "S7", "title": "Toyota Prius", "year": 2024, "url": "https://en.wikipedia.org/wiki/Toyota_Prius"},
        {"id": "S8", "title": "Tesla Model S", "year": 2024, "url": "https://en.wikipedia.org/wiki/Tesla_Model_S"},
        {"id": "S9", "title": "Global EV Outlook 2024", "year": 2024, "url": "https://www.iea.org/reports/global-ev-outlook-2024"},
        {"id": "S10", "title": "Advanced driver-assistance system", "year": 2024, "url": "https://en.wikipedia.org/wiki/Advanced_driver-assistance_system"},
    ],
    "claims": [
        {"id": "C1", "text": "1886 年卡尔·本茨的 Benz Patent-Motorwagen 被广泛认为是第一辆实用的内燃机汽车，采用三轮、后置单缸发动机。", "source_ids": ["S1", "S2"], "risk": "high", "status": "supported"},
        {"id": "C2", "text": "1908 年福特 Model T 让汽车从昂贵手工产品走向大众；1913 年移动流水线进一步降低了装配时间和成本。", "source_ids": ["S3", "S4"], "risk": "high", "status": "supported"},
        {"id": "C3", "text": "20 世纪上半叶，封闭车身、电启动、道路网络和加油站逐步普及，汽车从实验机器变成日常交通工具。", "source_ids": ["S1"], "risk": "medium", "status": "supported"},
        {"id": "C4", "text": "二战后高速公路和郊区扩张推动了汽车普及，也带来拥堵、事故和城市形态变化。", "source_ids": ["S1"], "risk": "medium", "status": "supported"},
        {"id": "C5", "text": "1959 年沃尔沃把三点式安全带推向市场，随后安全带法规和碰撞测试成为汽车安全的重要 baseline。", "source_ids": ["S5"], "risk": "medium", "status": "supported"},
        {"id": "C6", "text": "1970 年代石油危机和排放法规推动燃油喷射、轻量化、催化转化器和无铅汽油普及。", "source_ids": ["S6"], "risk": "medium", "status": "supported"},
        {"id": "C7", "text": "1980-90 年代电子控制单元、ABS、安全气囊和车身碰撞结构逐步成为主流配置。", "source_ids": ["S1"], "risk": "medium", "status": "supported"},
        {"id": "C8", "text": "1997 年丰田 Prius 成为首款大规模量产的混合动力车，用电机回收制动能量并降低油耗。", "source_ids": ["S7"], "risk": "medium", "status": "supported"},
        {"id": "C9", "text": "2000 年代后锂离子电池成本下降，特斯拉 Roadster（2008）和 Model S（2012）推动电动车从 niche 走向主流。", "source_ids": ["S8", "S9"], "risk": "medium", "status": "supported"},
        {"id": "C10", "text": "2020 年代电动车销量快速增长，但充电网络、电池成本、电网和软件安全仍是普及的关键约束；ADAS 仍以辅助驾驶为主。", "source_ids": ["S9", "S10"], "risk": "high", "status": "supported"},
    ],
}

AUTO_FALLBACK_SCRIPT: dict[str, Any] = {
    "title": "汽车发展史：从三轮车到软件定义汽车",
    "subtitle": "140 年，汽车如何变成今天的样子",
    "hook": "1886 年，一辆只有三个轮子、不到一马力的车，改变了人类的出行方式。",
    "knowledge_goal": AUTO_PROFILE["knowledge_goal"],
    "target_duration": 120,
    "scenes": [
        {"id": 1, "heading": "三个轮子的起点", "narration": "1886 年，卡尔·本茨造出 Benz Patent-Motorwagen。它只有三个轮子，后置单缸发动机，速度不快，却被视为第一辆实用的内燃机汽车。", "on_screen": ["1886", "Benz Patent-Motorwagen", "三轮 · 单缸"], "key_takeaway": "汽车不是突然出现的，它从一辆实验三轮车开始。", "diagram": "timeline", "keywords": ["1886", "本茨", "内燃机"], "claim_ids": ["C1"]},
        {"id": 2, "heading": "福特让汽车变便宜", "narration": "1908 年，福特 Model T 把汽车从手工奢侈品变成大众产品。1913 年移动流水线投入使用，装配时间和成本大幅下降。", "on_screen": ["1908 Model T", "1913 流水线", "从手工到量产"], "key_takeaway": "流水线不是小改进，它改变了汽车的价格和普及速度。", "diagram": "timeline", "keywords": ["福特", "Model T", "流水线"], "claim_ids": ["C2"]},
        {"id": 3, "heading": "从机器变成日常工具", "narration": "封闭车身、电启动、道路和加油站逐渐普及。汽车不再只是机械爱好者的玩具，而成为普通人通勤、购物和旅行的工具。", "on_screen": ["封闭车身", "电启动", "道路 + 加油站"], "key_takeaway": "汽车普及，靠的不只是发动机，还有整套基础设施。", "diagram": "definition", "keywords": ["电启动", "道路", "通勤"], "claim_ids": ["C3"]},
        {"id": 4, "heading": "高速公路与郊区", "narration": "二战后，高速公路和郊区扩张把汽车推成生活中心。它带来自由和效率，也带来拥堵、事故和城市形态的改变。", "on_screen": ["高速公路", "郊区化", "自由与代价"], "key_takeaway": "汽车重塑了城市，也制造了新的城市问题。", "diagram": "definition", "keywords": ["高速公路", "郊区", "拥堵"], "claim_ids": ["C4"]},
        {"id": 5, "heading": "安全带成为底线", "narration": "1959 年，沃尔沃把三点式安全带推向市场。此后安全带法规、碰撞测试和安全车身，让“安全”从卖点变成汽车的基本底线。", "on_screen": ["1959 三点式安全带", "碰撞测试", "安全法规"], "key_takeaway": "安全不是配置表上的一行字，而是几十年事故换来的标准。", "diagram": "timeline", "keywords": ["安全带", "沃尔沃", "碰撞测试"], "claim_ids": ["C5"]},
        {"id": 6, "heading": "石油危机逼出效率", "narration": "1970 年代石油危机和排放法规，逼着汽车提高效率、减少污染。燃油喷射、轻量化、催化转化器和无铅汽油逐步普及。", "on_screen": ["石油危机", "排放法规", "催化转化器"], "key_takeaway": "油价和法规，往往比工程师的偏好更能改变汽车。", "diagram": "definition", "keywords": ["石油危机", "排放", "催化转化器"], "claim_ids": ["C6"]},
        {"id": 7, "heading": "电子系统接管汽车", "narration": "1980 到 90 年代，电子控制单元、ABS、安全气囊和碰撞结构进入主流。汽车开始从纯机械产品，变成机电一体的复杂系统。", "on_screen": ["ECU", "ABS", "安全气囊"], "key_takeaway": "汽车的竞争力，开始从发动机扩展到电子和软件。", "diagram": "definition", "keywords": ["ECU", "ABS", "气囊"], "claim_ids": ["C7"]},
        {"id": 8, "heading": "混动的折中方案", "narration": "1997 年，丰田 Prius 成为首款大规模量产的混合动力车。它用电机回收制动能量，在城市工况下显著降低油耗。", "on_screen": ["1997 Prius", "电机 + 发动机", "能量回收"], "key_takeaway": "混动不是过渡的妥协，而是把浪费的能量捡回来。", "diagram": "timeline", "keywords": ["Prius", "混动", "能量回收"], "claim_ids": ["C8"]},
        {"id": 9, "heading": "锂电池把电动车推上台", "narration": "2000 年代后，锂离子电池成本持续下降。特斯拉 Roadster 和 Model S 把电动车从边缘产品推向主流视野，续航和性能不再是笑话。", "on_screen": ["锂离子电池", "Roadster 2008", "Model S 2012"], "key_takeaway": "电动车翻身，关键不只是电机，而是电池成本曲线。", "diagram": "timeline", "keywords": ["锂电池", "特斯拉", "续航"], "claim_ids": ["C9"]},
        {"id": 10, "heading": "软件定义汽车，但别神化", "narration": "2020 年代，电动车销量快速增长，充电网络、电池成本、电网和软件安全成为新约束。ADAS 仍以辅助驾驶为主，离真正的无人驾驶还有距离。", "on_screen": ["电动化", "充电网络", "ADAS 辅助驾驶"], "key_takeaway": "汽车在变聪明，但“辅助”和“自动”之间还有很长的路。", "diagram": "comparison", "keywords": ["电动车", "充电", "ADAS"], "claim_ids": ["C10"]},
    ],
    "summary": ["汽车从机械产品变成机电+软件系统", "每次普及都伴随基础设施和法规变化", "电动化与智能化仍有现实约束"],
    "platform": {"description": "用 2 分钟看懂汽车 140 年的关键转折：流水线、安全带、排放法规、混动、锂电池和 ADAS。", "hashtags": ["#汽车发展史", "#汽车科普", "#电动车", "#知识科普"]},
    "meta": {"source": "builtin_fallback", "model": None, "reason": "汽车发展史内置已核查脚本"},
}

TOPIC_PROFILES = [PROFILE, AUTO_PROFILE]
ACTIVE_PROFILE: dict[str, Any] = PROFILE


def resolve_profile(topic: str) -> dict[str, Any] | None:
    key = normalize(topic)
    for p in TOPIC_PROFILES:
        if key in {normalize(x) for x in [p["topic"]] + p.get("aliases", [])}:
            return p
    return None


def fallback_script(profile: dict[str, Any]) -> dict[str, Any]:
    if profile.get("topic") == AUTO_PROFILE["topic"]:
        return json.loads(json.dumps(AUTO_FALLBACK_SCRIPT, ensure_ascii=False))
    return json.loads(json.dumps(FALLBACK_SCRIPT, ensure_ascii=False))


# ---------------------------------------------------------------------------
# 任意主题：通用解释模板（无事实库时使用，明确标记未核查）
# ---------------------------------------------------------------------------

GENERIC_SYSTEM_PROMPT = """你是一位中文科普视频编剧。用户没有提供事实清单，你要写的是“通用解释框架”，不是具体事实报道。
不要编造具体年份、数字、人名、论文名、公司名或“研究表明”。可以解释概念、机制、常见误解、影响和判断方法。
每个分镜 narration 35-60 个汉字；claim_ids 留空数组；diagram 只能从 hook, definition, timeline, comparison, summary 中选。
总旁白 400-520 个汉字。只输出合法 JSON，不要 Markdown，不要解释。
"""


def make_generic_profile(topic: str, facts_text: str = "") -> dict[str, Any]:
    facts = [x.strip() for x in (facts_text or "").splitlines() if len(x.strip()) >= 6][:8]
    claims = [
        {"id": f"U{i + 1}", "text": fact, "source_ids": ["USER"], "risk": "unknown", "status": "unverified"}
        for i, fact in enumerate(facts)
    ]
    sources = []
    if facts:
        sources.append({"id": "USER", "title": "用户提供的材料（未独立核验）", "year": datetime.now().year, "url": ""})
    return {
        "topic": topic.strip(),
        "aliases": [],
        "series": "OPEN SCIENCE LAB",
        "visual_subtitle": f"{topic.strip()} · 通用解释",
        "knowledge_goal": f"用通用框架解释“{topic.strip()}”：定义、背景、机制、误解、影响和判断方法。",
        "sources": sources,
        "claims": claims,
        "unverified": True,
        "facts": facts,
    }


def generic_script(topic: str, target_duration: int, scene_count: int, profile: dict[str, Any]) -> dict[str, Any]:
    topic = topic.strip()
    facts = profile.get("facts", [])
    rows = [
        ("先问三个问题", f"“{topic}”听起来很大，其实可以先拆成三个问题：它是什么，为什么现在被讨论，它和普通人有什么关系。", ["是什么", "为什么", "与我何关"], "先拆问题，再下判断，比急着站队更有用。"),
        ("它到底是什么", f"理解“{topic}”，先看边界：它包含哪些部分，又不包含哪些部分。定义清楚，后面的判断才不会被概念带偏。", ["定义", "边界", "不包含什么"], "先定义，再讨论，能减少一半的无效争论。"),
        ("为什么会发生", f"任何现象都有条件。“{topic}”背后通常有技术、需求、成本或规则的变化，几股力量一起推着它走到今天。", ["技术", "需求", "成本 / 规则"], "别找单一原因，先看几股力量如何叠加。"),
        ("关键机制", f"把“{topic}”拆开，会看到输入、过程、结果三个环节。真正决定走向的，往往不是最显眼的那一步，而是中间的约束。", ["输入", "过程", "结果 / 约束"], "看懂约束，比记住结论更有用。"),
        ("一个常见误解", f"关于“{topic}”，最常见的误解是把它当成单一原因的结果。现实通常更复杂：短期看情绪，长期看结构和成本。", ["单一原因", "短期情绪", "长期结构"], "越像“一句话解释”，越要警惕它省略了什么。"),
        ("对普通人的影响", f"对普通人来说，“{topic}”的影响不在口号里，而在价格、时间、选择和风险的变化里。先看谁受益，谁承担成本。", ["价格", "时间", "选择 / 风险"], "判断影响，先问收益和成本落在谁身上。"),
        ("争议与边界", f"“{topic}”不是没有争议。有人说它被高估，有人说它被低估；判断时要看证据、样本和利益相关方，而不是只看立场。", ["高估？", "低估？", "证据 / 样本"], "有争议不可怕，可怕的是只看立场不看证据。"),
        ("怎么判断信息", f"面对“{topic}”的信息，先问三个问题：来源是谁，数据怎么来的，结论有没有被夸大。能回答这三点，才值得继续相信。", ["来源是谁", "数据怎么来", "结论是否夸大"], "信息来源、数据和结论，是三道最基本的过滤器。"),
        ("最后一句", f"“{topic}”不会只有一个答案。把它放回具体场景，分清事实、解释和判断，才能做出自己的结论。", ["回到场景", "区分事实/解释", "再做判断"], "通用框架给你地图，具体结论仍要靠证据。"),
    ]
    scenes = []
    for i, (heading, narration, on_screen, takeaway) in enumerate(rows[:max(9, min(10, scene_count))], 1):
        claim_ids = [f"U{i}"] if i <= len(facts) else []
        scenes.append({
            "id": i,
            "heading": heading,
            "narration": narration,
            "on_screen": on_screen,
            "key_takeaway": takeaway,
            "diagram": ["definition", "timeline", "comparison", "summary"][i % 4],
            "keywords": [topic, heading],
            "claim_ids": claim_ids,
        })
    return {
        "title": f"{topic}：一次讲清楚",
        "subtitle": "从定义、机制到影响，拆开看",
        "hook": f"如果你只记住一个结论，可能会错过“{topic}”真正重要的部分。",
        "knowledge_goal": profile["knowledge_goal"],
        "target_duration": target_duration,
        "scenes": scenes,
        "summary": ["先定义，再讨论", "看机制和约束，不只看结论", "分清事实、解释和判断"],
        "platform": {"description": f"用通用框架拆解“{topic}”：定义、背景、机制、误解、影响和判断方法。", "hashtags": ["#知识科普", f"#{topic[:12]}"]},
        "meta": {"source": "generic_template", "model": None, "verified": False, "reason": "通用模板草稿，未使用事实库"},
    }


def generic_user_prompt(topic: str, target_duration: int, scene_count: int) -> str:
    return f"""主题：{topic}
目标时长：{target_duration} 秒，竖屏 9:16，普通话科普。
分镜数量：9-10 个。
这是通用解释草稿，没有事实清单：
- 不要编造具体年份、数字、人名、公司名、论文名；
- 可以写定义、背景、机制、常见误解、对普通人的影响、争议边界、判断方法；
- 每个分镜 narration 35-60 个汉字，claim_ids 留空数组；
- diagram 从 hook, definition, timeline, comparison, summary 中选。
输出 JSON，结构：
{{"title":"标题","subtitle":"副标题","hook":"开场钩子","knowledge_goal":"知识目标","target_duration":{target_duration},"scenes":[{{"id":1,"heading":"短标题","narration":"35-60字","on_screen":["要点1","要点2"],"key_takeaway":"一句话","diagram":"definition","keywords":["词"],"claim_ids":[]}}],"summary":["总结1","总结2","总结3"],"platform":{{"description":"简介","hashtags":["#知识科普"]}}}}
"""


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------

def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def normalize(text: str) -> str:
    return re.sub(r"[\s，。？！?？!、：:；“”\"'《》<>]", "", text or "").lower()


def chinese_chars(text: str) -> int:
    return len(re.findall(r"[\u4e00-\u9fff]", text or ""))


def wrap(draw: ImageDraw.ImageDraw, text: str, f: ImageFont.FreeTypeFont, max_w: int) -> list[str]:
    lines, cur = [], ""
    for ch in text.replace("\r", ""):
        if ch == "\n":
            lines.append(cur)
            cur = ""
            continue
        trial = cur + ch
        if draw.textlength(trial, font=f) <= max_w or not cur:
            cur = trial
        else:
            if ch in "，。！？；：、,.!?;:）】》":
                cur += ch
                lines.append(cur)
                cur = ""
            else:
                lines.append(cur)
                cur = ch
    if cur:
        lines.append(cur)
    return lines


def fit_font(draw: ImageDraw.ImageDraw, text: str, max_w: int, start: int, minimum: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    size = start
    while size > minimum:
        f = font(size, bold)
        if max(draw.textlength(line, font=f) for line in text.split("\n")) <= max_w:
            return f
        size -= 2
    return font(minimum, bold)


_FONT_CACHE: dict[tuple[bool, int], ImageFont.FreeTypeFont] = {}


def font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    key = (bold, size)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    candidates = [
        r"C:\Windows\Fonts\msyhbd.ttc" if bold else r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc" if bold else "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/System/Library/Fonts/PingFang.ttc",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                loaded = ImageFont.truetype(path, size=size, index=0)
                _FONT_CACHE[key] = loaded
                return loaded
            except Exception:
                continue
    loaded = ImageFont.load_default()
    _FONT_CACHE[key] = loaded
    return loaded


# ---------------------------------------------------------------------------
# Ollama 客户端
# ---------------------------------------------------------------------------

class OllamaError(RuntimeError):
    pass


def ollama_get(path: str, timeout: int = 8) -> dict[str, Any]:
    req = urllib.request.Request(OLLAMA_BASE_URL + path, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        raise OllamaError(str(exc)) from exc


def ollama_status() -> dict[str, Any]:
    try:
        tags = ollama_get("/api/tags")
        names = [m.get("name", "") for m in tags.get("models", [])]
        present = any(n == OLLAMA_MODEL or n.split(":")[0] == OLLAMA_MODEL.split(":")[0] for n in names)
        return {"available": True, "models": names, "model_present": present, "error": ""}
    except Exception as exc:
        return {"available": False, "models": [], "model_present": False, "error": str(exc)}


def ollama_chat(messages: list[dict[str, str]], json_mode: bool = True, timeout: int | None = None) -> str:
    payload: dict[str, Any] = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "format": "json" if json_mode else None,
        "options": {"temperature": 0.4, "num_ctx": 8192, "num_predict": 2500},
    }
    if payload["format"] is None:
        payload.pop("format")
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_BASE_URL + "/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout or OLLAMA_TIMEOUT) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise OllamaError(f"Ollama HTTP {exc.code}: {detail[:500]}") from exc
    except Exception as exc:
        raise OllamaError(f"无法调用 Ollama：{exc}") from exc
    return ((body.get("message") or {}).get("content") or "").strip()


def strip_fence(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if start >= 0 and end > start else text


# ---------------------------------------------------------------------------
# 脚本校验
# ---------------------------------------------------------------------------

class ScriptError(ValueError):
    pass


def validate_script(data: dict[str, Any], profile: dict[str, Any] | None = None, allow_unverified: bool = False) -> dict[str, Any]:
    profile = profile or ACTIVE_PROFILE
    issues: list[str] = []
    if not isinstance(data, dict):
        raise ScriptError("顶层必须是 JSON 对象")
    if not str(data.get("title") or "").strip():
        issues.append("缺少 title")
    scenes = data.get("scenes")
    if not isinstance(scenes, list) or not (9 <= len(scenes) <= 14):
        issues.append("scenes 必须是 9-14 个分镜")
        scenes = scenes if isinstance(scenes, list) else []
    known = {c["id"] for c in profile["claims"]}
    total = 0
    for i, s in enumerate(scenes, 1):
        if not isinstance(s, dict):
            issues.append(f"第 {i} 个分镜不是对象")
            continue
        s.setdefault("id", i)
        s["narration"] = re.sub(r"\s+", "", str(s.get("narration") or ""))
        s["heading"] = str(s.get("heading") or "").strip()
        s["key_takeaway"] = str(s.get("key_takeaway") or "").strip()
        s["diagram"] = str(s.get("diagram") or "definition").strip()
        if len(s["narration"]) < 20:
            issues.append(f"第 {i} 个分镜旁白过短（{len(s['narration'])} 字）")
        if len(s["narration"]) > 120:
            issues.append(f"第 {i} 个分镜旁白过长（{len(s['narration'])} 字）")
        if s["diagram"] not in ALLOWED_DIAGRAMS:
            issues.append(f"第 {i} 个分镜 diagram 不支持：{s['diagram']}")
        if not isinstance(s.get("on_screen"), list) or not s["on_screen"]:
            issues.append(f"第 {i} 个分镜缺 on_screen")
        if not isinstance(s.get("claim_ids"), list):
            if allow_unverified:
                s["claim_ids"] = []
            else:
                issues.append(f"第 {i} 个分镜缺 claim_ids")
        elif not s["claim_ids"] and not allow_unverified:
            issues.append(f"第 {i} 个分镜缺 claim_ids")
        if not allow_unverified:
            unknown = [c for c in s.get("claim_ids", []) if c not in known]
            if unknown:
                issues.append(f"第 {i} 个分镜引用未知论断：{unknown}")
        total += chinese_chars(s["narration"])
    if total < 360:
        issues.append(f"总旁白过短（{total} 个汉字），120 秒视频建议 420-560 字")
    if total > 650:
        issues.append(f"总旁白过长（{total} 个汉字）")
    if issues:
        raise ScriptError("；".join(issues))
    existing_meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    if existing_meta.get("verified") is False:
        existing_meta.update({"generated_at": now_iso(), "chars": total})
        data["meta"] = existing_meta
    else:
        data["meta"] = {"source": "ollama", "model": OLLAMA_MODEL, "generated_at": now_iso(), "chars": total}
    return data


# ---------------------------------------------------------------------------
# 提示词与生成
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """你是一位严谨的中文科普视频编剧。你只能使用用户提供的事实清单，不得新增数字、论文名或因果断言。
每个分镜必须绑定 claim_ids，只能从清单中选择。输出必须是合法 JSON，不要 Markdown，不要解释。
总旁白 430-540 个汉字；每个分镜 narration 必须 35-60 个汉字，不能只写一句 10-19 字的短语。
diagram 只能填一个值，可选：hook, definition, next_token, plausibility, data_noise, reward, snowball, rag, uncertainty, mitigation, summary。
"""


def user_prompt(topic: str, target_duration: int, scene_count: int, profile: dict[str, Any] | None = None) -> str:
    profile = profile or ACTIVE_PROFILE
    facts = [
        {"claim_id": c["id"], "statement": c["text"], "risk": c["risk"]}
        for c in profile["claims"]
    ]
    schema = {
        "title": "标题，20 字以内",
        "subtitle": "副标题，25 字以内",
        "hook": "一句话开场钩子",
        "knowledge_goal": "本片知识目标",
        "target_duration": target_duration,
        "scenes": [{
            "id": 1,
            "heading": "屏幕短标题，12 字以内",
            "narration": "本段旁白，35-60 个汉字",
            "on_screen": ["屏幕要点 1", "屏幕要点 2"],
            "key_takeaway": "一句关键结论，25 字以内",
            "diagram": "hook",
            "keywords": ["高亮词"],
            "claim_ids": ["C1"],
        }],
        "summary": ["总结 1", "总结 2", "总结 3"],
        "platform": {"description": "平台简介", "hashtags": ["#AI科普"]},
    }
    return f"""主题：{topic}
目标时长：{target_duration} 秒，竖屏 9:16，普通话科普。
分镜数量：{scene_count} 个左右。
知识目标：{profile.get('knowledge_goal', '')}

【已核查事实清单】
{json.dumps(facts, ensure_ascii=False, indent=2)}

【写作要求】
- 每个分镜 narration 35-60 个汉字，把解释、例子、过渡都写进 narration。
- on_screen 每项不超过 16 字；key_takeaway 不超过 25 字。
- 不要编造百分比、年份、人名、论文名。
- 只输出下面结构的 JSON：

{json.dumps(schema, ensure_ascii=False, indent=2)}
"""


def generate_script(topic: str, target_duration: int, scene_count: int, prefer_llm: bool = True, profile: dict[str, Any] | None = None) -> tuple[dict[str, Any], str]:
    """返回 (script, mode)。mode 为 ollama / fallback。"""
    profile = profile or ACTIVE_PROFILE
    if profile.get("unverified"):
        stt = ollama_status()
        if prefer_llm and stt["available"] and stt["model_present"]:
            try:
                raw = ollama_chat([
                    {"role": "system", "content": GENERIC_SYSTEM_PROMPT},
                    {"role": "user", "content": generic_user_prompt(topic, target_duration, scene_count)},
                ])
                data = json.loads(strip_fence(raw))
                data["meta"] = {"source": "ollama_generic", "model": OLLAMA_MODEL, "verified": False, "reason": "通用解释草稿"}
                return validate_script(data, profile, allow_unverified=True), "ollama_generic"
            except (json.JSONDecodeError, ScriptError, OllamaError) as exc:
                st.warning(f"Ollama 通用草稿未通过校验，改用模板：{str(exc)[:240]}")
        return generic_script(topic, target_duration, scene_count, profile), "generic_template"
    if not prefer_llm:
        data = fallback_script(profile)
        data["meta"]["reason"] = "用户选择离线脚本"
        return data, "fallback"
    stt = ollama_status()
    if not stt["available"] or not stt["model_present"]:
        data = fallback_script(profile)
        data["meta"]["reason"] = "Ollama 不可用或模型未安装"
        return data, "fallback"
    raw = ""
    last_error = ""
    for attempt in range(1, 3):
        try:
            raw = ollama_chat([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt(topic, target_duration, scene_count, profile)},
            ])
            data = json.loads(strip_fence(raw))
            return validate_script(data, profile), "ollama"
        except (json.JSONDecodeError, ScriptError, OllamaError) as exc:
            last_error = str(exc)
            st.warning(f"Ollama 第 {attempt} 次输出未通过校验：{last_error[:300]}")
    data = fallback_script(profile)
    data["meta"]["reason"] = f"Ollama 输出未通过校验：{last_error[:200]}"
    return data, "fallback"


# ---------------------------------------------------------------------------
# 画面渲染
# ---------------------------------------------------------------------------

def _gradient() -> Image.Image:
    top = np.array(BG_TOP, dtype=np.float32)
    bottom = np.array(BG_BOTTOM, dtype=np.float32)
    rows = np.linspace(0, 1, H, dtype=np.float32)[:, None]
    arr = top[None, :] * (1 - rows) + bottom[None, :] * rows
    arr = np.repeat(arr[:, None, :], W, axis=1).astype(np.uint8)
    return Image.fromarray(arr, "RGB").convert("RGBA")


def _background(seed: int) -> Image.Image:
    img = _gradient()
    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    gd.ellipse((690, -210, 1320, 420), fill=(34, 211, 238, 42))
    gd.ellipse((-260, 1220, 420, 1900), fill=(124, 58, 237, 44))
    glow = glow.filter(ImageFilter.GaussianBlur(150))
    img = Image.alpha_composite(img, glow)
    d = ImageDraw.Draw(img)
    for x in range(0, W, 54):
        d.line((x, 0, x, H), fill=(90, 140, 180, 12), width=1)
    for y in range(0, H, 54):
        d.line((0, y, W, y), fill=(90, 140, 180, 12), width=1)
    rng = np.random.default_rng(seed)
    for _ in range(80):
        x, y = int(rng.integers(30, W - 30)), int(rng.integers(90, H - 90))
        r = int(rng.choice([1, 1, 1, 2, 2, 3]))
        color = [CYAN, BLUE, PURPLE, (200, 230, 255)][int(rng.integers(0, 4))]
        d.ellipse((x - r, y - r, x + r, y + r), fill=color)
    return img


def _panel(img: Image.Image, box: tuple[int, int, int, int]) -> None:
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.rounded_rectangle(box, radius=34, fill=(10, 19, 36, 232), outline=(56, 189, 248, 62), width=2)
    img.alpha_composite(layer)


def _chips(d: ImageDraw.ImageDraw, x: int, y: int, w: int, items: list[str]) -> int:
    f = font(27, True)
    gap = 12
    cw = (w - gap) // 2
    for i, item in enumerate(items[:4]):
        cx = x + (i % 2) * (cw + gap)
        cy = y + (i // 2) * 60
        d.rounded_rectangle((cx, cy, cx + cw, cy + 48), radius=24, fill=(20, 36, 62), outline=(56, 189, 248, 70), width=1)
        d.ellipse((cx + 16, cy + 19, cx + 26, cy + 29), fill=CYAN if i % 2 == 0 else PURPLE)
        d.text((cx + 38, cy + 24), item, font=f, fill=WHITE, anchor="lm")
    return ((min(len(items), 4) + 1) // 2) * 60


def _diagram(d: ImageDraw.ImageDraw, box: tuple[int, int, int, int], scene: dict[str, Any]) -> None:
    x, y, w, h = box
    kind = scene.get("diagram", "definition")
    if kind == "next_token":
        d.text((x + 20, y + 8), "下一个词元，最可能是什么？", font=font(26, True), fill=MUTED)
        d.rounded_rectangle((x + 22, y + 50, x + w - 22, y + 142), radius=18, fill=(7, 15, 30), outline=CYAN, width=2)
        d.text((x + w // 2, y + 96), "猫坐在  ?  上", font=font(48, True), fill=WHITE, anchor="mm")
        for i, (label, pct, color) in enumerate([("垫子", 72, CYAN), ("沙发", 18, BLUE), ("月亮", 0.3, PURPLE)]):
            yy = y + 180 + i * 80
            d.text((x + 30, yy + 16), label, font=font(28, True), fill=WHITE, anchor="lm")
            bx0, bx1 = x + 140, x + w - 110
            d.rounded_rectangle((bx0, yy, bx1, yy + 38), radius=9, fill=(24, 39, 60))
            d.rounded_rectangle((bx0, yy, bx0 + max(8, int((bx1 - bx0) * pct / 100)), yy + 38), radius=9, fill=color)
            d.text((x + w - 28, yy + 19), f"{pct:g}%", font=font(25, True), fill=color, anchor="rm")
        d.text((x + 22, y + h - 50), "概率高，只代表“更像常见续写”，不代表事实为真。", font=font(24, True), fill=AMBER)
    elif kind == "rag":
        stages = ["问题", "检索器", "文档库", "模型", "回答"]
        bw = 126
        gap = (w - 36 - len(stages) * bw) / (len(stages) - 1)
        cy = y + 90
        for i, label in enumerate(stages):
            bx = int(x + 18 + i * (bw + gap))
            d.rounded_rectangle((bx, cy - 42, bx + bw, cy + 42), radius=14, fill=(15, 29, 52), outline=(56, 189, 248, 110), width=2)
            d.text((bx + bw // 2, cy), label, font=font(23, True), fill=WHITE, anchor="mm")
            if i < len(stages) - 1:
                d.line((bx + bw + 4, cy, bx + bw + int(gap) - 4, cy), fill=MUTED, width=3)
        d.text((x + w // 2, y + 210), "检索错 / 文档矛盾 / 忽略证据  →  仍可能幻觉", font=font(26, True), fill=RED, anchor="mm")
        d.text((x + w // 2, y + 265), "RAG 降低风险，不提供“零幻觉”保证。", font=font(23, False), fill=MUTED, anchor="mm")
    elif kind == "summary":
        items = scene.get("on_screen", [])[:3]
        if not items:
            items = ["先定义", "看机制", "再判断"]
        for i, title in enumerate(items):
            yy = y + 10 + i * 120
            color = [CYAN, AMBER, PURPLE][i % 3]
            d.rounded_rectangle((x + 16, yy, x + w - 16, yy + 102), radius=18, fill=(15, 29, 52), outline=color, width=2)
            d.text((x + 38, yy + 50), f"{i + 1:02d}", font=font(34, True), fill=color, anchor="lm")
            d.text((x + 100, yy + 50), title, font=fit_font(d, title, w - 140, 28, 18, True), fill=WHITE, anchor="lm")
    elif kind == "hook":
        d.rounded_rectangle((x + 20, y + 10, x + w - 20, y + 230), radius=24, fill=(13, 24, 45), outline=(56, 189, 248, 80), width=2)
        d.text((x + 40, y + 28), "AI 助手 · 对话演示", font=font(22, True), fill=MUTED)
        d.rounded_rectangle((x + w - 330, y + 72, x + w - 40, y + 126), radius=18, fill=(30, 58, 95))
        d.text((x + w - 60, y + 99), "长城在月球上肉眼可见吗？", font=font(22, False), fill=WHITE, anchor="rm")
        d.rounded_rectangle((x + 36, y + 144, x + 390, y + 206), radius=18, fill=(18, 38, 64), outline=CYAN, width=1)
        d.text((x + 56, y + 175), "当然可以，这是常识。", font=font(25, True), fill=WHITE, anchor="lm")
        d.rounded_rectangle((x + 404, y + 156, x + 504, y + 196), radius=8, outline=RED, width=2)
        d.text((x + 454, y + 176), "错误示例", font=font(18, True), fill=RED, anchor="mm")
        d.text((x + 36, y + 270), "流畅和自信，可能是最危险的“正确感”。", font=font(26, True), fill=AMBER)
    elif kind == "timeline":
        items = scene.get("on_screen", [])[:4]
        if not items:
            items = ["起点", "普及", "安全", "电动化"]
        cw = (w - 44) // len(items)
        for i, item in enumerate(items):
            cx = x + 14 + i * cw
            color = [CYAN, BLUE, PURPLE, GREEN][i % 4]
            d.rounded_rectangle((cx + 6, y + 40, cx + cw - 6, y + 190), radius=14, fill=(15, 29, 52), outline=color, width=2)
            d.text((cx + cw // 2, y + 78), f"{i + 1:02d}", font=font(24, True), fill=color, anchor="mm")
            fit = fit_font(d, item, cw - 22, 24, 18, True)
            d.text((cx + cw // 2, y + 138), item, font=fit, fill=WHITE, anchor="mm")
            if i < len(items) - 1:
                d.line((cx + cw - 4, y + 115, cx + cw + 4, y + 115), fill=MUTED, width=3)
        d.text((x + w // 2, y + 245), scene.get("key_takeaway", ""), font=fit_font(d, scene.get("key_takeaway", ""), w - 40, 26, 20, True), fill=AMBER, anchor="mm")
    elif kind == "comparison":
        items = scene.get("on_screen", [])[:2] or ["过去", "现在"]
        for i, item in enumerate(items):
            cx = x + 16 + i * (w // 2)
            color = CYAN if i == 0 else PURPLE
            d.rounded_rectangle((cx, y + 30, cx + w // 2 - 26, y + 220), radius=16, fill=(15, 29, 52), outline=color, width=2)
            d.text((cx + (w // 2 - 26) // 2, y + 90), item, font=fit_font(d, item, w // 2 - 60, 30, 20, True), fill=WHITE, anchor="mm")
            d.text((cx + (w // 2 - 26) // 2, y + 155), "vs" if i == 0 else "", font=font(22, True), fill=MUTED, anchor="mm")
        d.text((x + w // 2, y + 270), scene.get("key_takeaway", ""), font=fit_font(d, scene.get("key_takeaway", ""), w - 40, 26, 20, True), fill=AMBER, anchor="mm")
    else:
        # 通用定义卡 / 因果卡
        cards = scene.get("on_screen", [])[:3]
        cw = (w - 64) // max(1, len(cards))
        for i, item in enumerate(cards):
            cx = x + 16 + i * (cw + 16)
            d.rounded_rectangle((cx, y + 20, cx + cw, y + 210), radius=18, fill=(15, 29, 52), outline=(56, 189, 248, 70), width=2)
            d.text((cx + cw // 2, y + 105), item, font=fit_font(d, item, cw - 20, 24, 18, True), fill=WHITE, anchor="mm")
        d.text((x + w // 2, y + 300), scene.get("key_takeaway", "判断标准：有没有可靠证据"), font=fit_font(d, scene.get("key_takeaway", "判断标准：有没有可靠证据"), w - 40, 32, 20, True), fill=CYAN, anchor="mm")


def render_scene(scene: dict[str, Any], index: int, total: int, out: Path, profile: dict[str, Any] | None = None) -> Path:
    profile = profile or ACTIVE_PROFILE
    img = _background(1000 + index)
    d = ImageDraw.Draw(img)
    d.text((64, 96), profile.get("series", "AI  SCIENCE  LAB"), font=font(22, True), fill=CYAN)
    d.text((W - 64, 96), f"EP 01   /   {index:02d} OF {total:02d}", font=font(19, True), fill=MUTED, anchor="ra")
    d.rounded_rectangle((64, 138, 270, 174), radius=18, fill=(34, 211, 238, 22), outline=(34, 211, 238, 70), width=1)
    d.text((82, 156), "本地生成 · 事实核查", font=font(17, True), fill=(190, 220, 235), anchor="lm")

    title = scene.get("heading", "")
    tf = fit_font(d, title, 850, 76, 50, True)
    d.text((66, 222), title, font=tf, fill=WHITE)
    d.rounded_rectangle((66, 330, 150, 340), radius=5, fill=CYAN)
    d.text((166, 320), profile.get("visual_subtitle", "大模型幻觉 · 科学解释"), font=font(22, False), fill=MUTED)

    panel = (60, 386, W - 60, 1150)
    _panel(img, panel)
    d = ImageDraw.Draw(img)
    chips_h = _chips(d, panel[0] + 26, panel[1] + 24, panel[2] - panel[0] - 52, scene.get("on_screen", []))
    dbox = (panel[0] + 16, panel[1] + 42 + chips_h, panel[2] - panel[0] - 32, 1150 - (panel[1] + 42 + chips_h) - 20)
    _diagram(d, dbox, scene)

    d.rounded_rectangle((60, 1186, W - 60, 1338), radius=24, fill=(11, 32, 48), outline=(34, 211, 238, 90), width=2)
    d.rounded_rectangle((60, 1186, 74, 1338), radius=7, fill=CYAN)
    d.text((96, 1206), "KEY  TAKEAWAY", font=font(17, True), fill=CYAN)
    kf = fit_font(d, scene.get("key_takeaway", ""), 800, 33, 24, True)
    d.text((96, 1244), scene.get("key_takeaway", ""), font=kf, fill=WHITE)

    d.line((64, 1792, W - 64, 1792), fill=(56, 189, 248, 40), width=1)
    d.text((64, 1812), "AI 科普视频工厂  ·  单文件 GitHub 版", font=font(19, False), fill=DIM)
    d.text((W - 64, 1812), "证据  ·  解释  ·  可视化", font=font(19, False), fill=DIM, anchor="ra")
    for i in range(total):
        d.ellipse((64 + i * 17, 1852, 70 + i * 17, 1858), fill=CYAN if i == index - 1 else (56, 189, 248, 60))
    out.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(out, quality=94, optimize=True)
    return out


# ---------------------------------------------------------------------------
# 音频 / 字幕 / 视频（可选）
# ---------------------------------------------------------------------------

def local_piper_paths() -> tuple[Path | None, Path | None]:
    env_model = os.getenv("PIPER_MODEL", "").strip()
    candidates = [
        APP_ROOT / "models/piper/zh_CN-huayan-medium.onnx",
        Path(env_model) if env_model else None,
        Path("/app/models/piper/zh_CN-huayan-medium.onnx"),
    ]
    for model in candidates:
        if model and model.name and model.exists():
            cfg = model.with_suffix(model.suffix + ".json")
            if cfg.exists():
                return model, cfg
    return None, None


def piper_available() -> bool:
    try:
        import piper  # noqa: F401
    except Exception:
        return False
    model, cfg = local_piper_paths()
    return bool(model and cfg)


def synthesize_voice(scenes: list[dict[str, Any]], out_dir: Path) -> tuple[Path | None, list[float]]:
    model, cfg = local_piper_paths()
    if not model or not cfg:
        return None, []
    try:
        import piper  # noqa: F401
    except Exception:
        return None, []
    seg_dir = out_dir / "voice"
    seg_dir.mkdir(parents=True, exist_ok=True)
    paths, durations = [], []
    for s in scenes:
        txt = seg_dir / f"scene_{s['id']:02d}.txt"
        wav = seg_dir / f"scene_{s['id']:02d}.wav"
        txt.write_text(s["narration"], encoding="utf-8")
        try:
            subprocess.run(
                [sys.executable, "-m", "piper", "-m", str(model), "-c", str(cfg), "-i", str(txt), "-f", str(wav),
                 "--length-scale", "1.05", "--sentence-silence", "0.08"],
                check=True, capture_output=True, timeout=300,
            )
        except Exception:
            return None, []
        if not wav.exists():
            return None, []
        with wave.open(str(wav), "rb") as fh:
            dur = fh.getnframes() / fh.getframerate()
        paths.append(wav)
        durations.append(dur)
    # 拼接为一条旁白（统一到 22050Hz/单声道）
    if not paths:
        return None, []
    with wave.open(str(paths[0]), "rb") as fh:
        sr, ch, sw = fh.getframerate(), fh.getnchannels(), fh.getsampwidth()
    total = sum(durations) + 0.8 * (len(paths) - 1)
    arr = np.zeros(int(total * sr), dtype=np.float32)
    pos = 0.0
    for path, dur in zip(paths, durations):
        with wave.open(str(path), "rb") as fh:
            raw = fh.readframes(fh.getnframes())
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if ch > 1:
            data = data.reshape(-1, ch).mean(axis=1)
        begin = int(pos * sr)
        arr[begin:begin + len(data)] += data
        pos += dur + 0.8
    out = out_dir / "narration.wav"
    with wave.open(str(out), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sr)
        fh.writeframes((np.clip(arr, -1, 1) * 32767).astype(np.int16).tobytes())
    return out, durations


def split_cues(text: str, max_chars: int = 15) -> list[str]:
    pieces = re.findall(r"[^，。！？；：,.!?;:]+[，。！？；：,.!?;:]?", text)
    chunks, cur = [], ""
    for p in pieces:
        p = p.strip()
        if not p:
            continue
        while len(p) > max_chars:
            chunks.append(p[:max_chars])
            p = p[max_chars:]
        if len(cur) + len(p) <= max_chars:
            cur += p
        else:
            if cur:
                chunks.append(cur)
            cur = p
    if cur:
        chunks.append(cur)
    return chunks


def srt_time(sec: float) -> str:
    ms = int(round(max(0, sec) * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def make_srt(scenes: list[dict[str, Any]], durations: list[float], out: Path) -> Path:
    lines, idx, cursor = [], 1, 0.0
    for scene, dur in zip(scenes, durations):
        cues = split_cues(scene["narration"])
        total_chars = max(1, sum(len(c) for c in cues))
        for cue in cues:
            cd = dur * len(cue) / total_chars
            lines += [str(idx), f"{srt_time(cursor)} --> {srt_time(cursor + cd)}", cue, ""]
            idx += 1
            cursor += cd
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def find_ffmpeg() -> str | None:
    for key in ("FFMPEG_BINARY", "IMAGEIO_FFMPEG_EXE"):
        p = os.getenv(key, "")
        if p and Path(p).exists():
            return p
    # 本地完整版自带的 FFmpeg
    for candidate in sorted((APP_ROOT / "tools").glob("**/ffmpeg.exe")):
        if candidate.stat().st_size > 10_000_000:
            return str(candidate)
    p = shutil.which("ffmpeg")
    if p:
        return p
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def build_video(images: list[Path], durations: list[float], audio: Path | None, srt: Path | None, out_dir: Path) -> Path | None:
    ff = find_ffmpeg()
    if not ff:
        return None
    out = out_dir / "preview.mp4"
    concat = out_dir / "concat.txt"
    with open(concat, "w", encoding="utf-8") as fh:
        for img, dur in zip(images, durations):
            fh.write(f"file '{img.as_posix()}'\nduration {dur:.3f}\n")
        fh.write(f"file '{images[-1].as_posix()}'\n")
    vf = "fps=30,scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2,format=yuv420p"
    base = [ff, "-y", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(concat)]
    if audio and audio.exists():
        base += ["-i", str(audio)]
    cmd = base + ["-vf", vf, "-r", "30", "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p"]
    if audio and audio.exists():
        cmd += ["-c:a", "aac", "-b:a", "160k"]
    else:
        cmd += ["-an"]
    cmd += ["-t", f"{sum(durations):.3f}", str(out)]
    # 尝试烧录 SRT；如果当前 FFmpeg 没有 libass/字体，则回退到无字幕版本。
    if srt and srt.exists():
        font_name = "Microsoft YaHei" if os.name == "nt" else "Noto Sans CJK SC"
        sub_vf = vf + f",subtitles={srt.name}:force_style='FontName={font_name},FontSize=8,PrimaryColour=&H00FFFFFF,OutlineColour=&H00101010,Outline=1,MarginV=18'"
        sub_cmd = list(cmd)
        sub_cmd[sub_cmd.index("-vf") + 1] = sub_vf
        try:
            subprocess.run(sub_cmd, cwd=str(out_dir), check=True, capture_output=True, timeout=900)
            return out
        except Exception:
            pass
    try:
        subprocess.run(cmd, cwd=str(out_dir), check=True, capture_output=True, timeout=900)
        return out
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 一键生成辅助
# ---------------------------------------------------------------------------

def estimate_durations(scenes: list[dict[str, Any]]) -> list[float]:
    return [max(6.0, len(s.get("narration", "")) / 4.2 + 1.0) for s in scenes]


def build_storyboard(script: dict[str, Any], profile: dict[str, Any] | None = None) -> tuple[list[str], list[float]]:
    scenes = script.get("scenes", [])
    out_dir = APP_DIR / "scenes"
    images: list[str] = []
    for i, scene in enumerate(scenes, 1):
        images.append(str(render_scene(scene, i, len(scenes), out_dir / f"scene_{i:02d}.png", profile)))
    return images, estimate_durations(scenes)


def build_export_bundle(script: dict[str, Any], images: list[str], durations: list[float]) -> dict[str, Any]:
    scenes = script.get("scenes", [])
    out_dir = APP_DIR / "export"
    out_dir.mkdir(parents=True, exist_ok=True)
    audio, audio_durations = synthesize_voice(scenes, out_dir)
    if audio and len(audio_durations) == len(scenes):
        durations = [d + 0.8 for d in audio_durations]
        durations[-1] += 0.6
        # 配音已经生成后不再拉伸语音；把差额留到最后一张总结卡的停留时间。
        target = float(script.get("target_duration") or 120)
        total = sum(durations)
        if total < target:
            durations[-1] += target - total
    else:
        target = float(script.get("target_duration") or 120)
        total = sum(durations)
        if total > 0 and abs(total - target) > 8:
            durations = [d * target / total for d in durations]
    srt = make_srt(scenes, durations, out_dir / "subtitles.srt")
    video = build_video([Path(p) for p in images], durations, audio, srt, out_dir) if images else None
    return {
        "srt": str(srt),
        "audio": str(audio) if audio else None,
        "video": str(video) if video else None,
        "durations": durations,
    }


def factcheck_payload(profile: dict[str, Any] | None = None) -> dict[str, Any]:
    profile = profile or ACTIVE_PROFILE
    return {
        "topic": profile["topic"],
        "checked_at": now_iso(),
        "claims": [
            {
                "id": c["id"],
                "text": c["text"],
                "risk": c["risk"],
                "source_ids": c["source_ids"],
                "verification_status": "supported" if c.get("status") == "supported" else "needs_review",
            }
            for c in profile["claims"]
        ],
        "sources": profile["sources"],
        "blocked": False,
        "requires_human_review": True,
        "notes": (
            ["这是通用解释模板草稿，没有内置事实库；发布前必须补充来源并逐条核对。"]
            if profile.get("unverified")
            else [
                "单文件版内置的是已整理的来源映射；自动状态不等于人工读过原文。",
                "发布前必须人工核对关键事实、字幕、配音和画面。",
            ]
        ),
    }


def sources_markdown(profile: dict[str, Any] | None = None) -> str:
    profile = profile or ACTIVE_PROFILE
    lines = [
        f"# 参考资料与事实核查：{profile['topic']}",
        "",
        f"- 生成时间：{now_iso()}",
        "- 说明：以下为单文件版内置的已整理来源映射，**不替代人工阅读原文**。",
        "",
        "## 来源清单",
        "",
        "| ID | 来源 | 年份 | 链接 |",
        "|---|---|---|---|",
    ]
    for s in profile["sources"]:
        link = f"[{s['url']}]({s['url']})" if s.get("url") else "（用户提供，无链接）"
        lines.append(f"| {s['id']} | {s['title']} | {s['year']} | {link} |")
    if not profile["sources"]:
        lines += ["", "> 这是通用解释模板草稿，没有内置来源。发布前请补充权威来源并逐条核对。", ""]
    lines += ["", "## 逐条论断", ""]
    for c in profile["claims"]:
        refs = "、".join(c["source_ids"])
        lines += [f"### {c['id']}（{c['risk']}）", "", c["text"], "", f"来源：{refs}", ""]
    lines += [
        "## 人工审核记录",
        "",
        "- [ ] 已阅读关键来源原文或摘要",
        "- [ ] 已确认脚本没有断章取义、过时结论或夸大表述",
        "- [ ] 已确认画面与旁白一致",
        "- [ ] 审核人：__________ 日期：__________",
        "",
    ]
    return "\n".join(lines)


def export_manifest(script: dict[str, Any], bundle: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "project": "github_single_file",
        "title": script.get("title", ""),
        "generated_at": now_iso(),
        "script_source": script.get("meta", {}).get("source", "builtin"),
        "model": OLLAMA_MODEL,
        "files": {
            "script.json": True,
            "fact_check.json": True,
            "sources.md": True,
            "subtitles.srt": bool(bundle and bundle.get("srt")),
            "preview.mp4": bool(bundle and bundle.get("video")),
        },
        "review": {"fact_check": "pending", "publish": "pending"},
        "notes": ["程序生成不等于可发布；发布前必须人工审核。"],
    }


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

CSS = """
<style>
:root { --cyan:#22D3EE; --purple:#A78BFA; }
.stApp { background: radial-gradient(1200px 600px at 85% -10%, rgba(34,211,238,.10), transparent 60%),
                    radial-gradient(900px 500px at 5% 100%, rgba(124,58,237,.12), transparent 60%), #050B18; }
[data-testid="stSidebar"] { background:#070F1E; border-right:1px solid rgba(56,189,248,.12); }
#MainMenu, footer { visibility:hidden; }
.block-container { padding-top:1.4rem; max-width:1360px; }
.hero { border:1px solid rgba(56,189,248,.18); background:linear-gradient(120deg, rgba(11,20,38,.95), rgba(7,15,30,.75));
        border-radius:22px; padding:26px 30px; margin-bottom:18px; position:relative; overflow:hidden; }
.hero:after { content:""; position:absolute; right:-80px; top:-120px; width:360px; height:360px;
        background:radial-gradient(circle, rgba(34,211,238,.22), transparent 65%); }
.badge { color:var(--cyan); font-size:12px; letter-spacing:.18em; font-weight:700; }
.hero h1 { font-size:34px; margin:8px 0 6px; }
.hero p { color:#9FB3C8; margin:0; }
.card { border:1px solid rgba(56,189,248,.14); background:rgba(11,20,38,.72); border-radius:16px; padding:16px 18px; }
.pill { display:inline-block; padding:4px 10px; border-radius:999px; font-size:12px; font-weight:600;
        border:1px solid rgba(34,211,238,.35); color:#A5F3FC; background:rgba(34,211,238,.08); margin-right:6px; }
.pill.green { border-color:rgba(52,211,153,.4); color:#6EE7B7; background:rgba(52,211,153,.08); }
.pill.amber { border-color:rgba(251,191,36,.35); color:#FDE68A; background:rgba(251,191,36,.08); }
.small { color:#7E93AC; font-size:12px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str, badge: str = "GITHUB / STREAMLIT EDITION") -> None:
    st.markdown(
        f'<div class="hero"><div class="badge">{badge}</div><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def init_state() -> None:
    if "script" not in st.session_state:
        st.session_state.script = json.loads(json.dumps(FALLBACK_SCRIPT, ensure_ascii=False))
    if "mode" not in st.session_state:
        st.session_state.mode = "fallback"
    if "images" not in st.session_state:
        st.session_state.images = []
    if "audio" not in st.session_state:
        st.session_state.audio = None
    if "video" not in st.session_state:
        st.session_state.video = None
    if "durations" not in st.session_state:
        st.session_state.durations = []
    if "bundle" not in st.session_state:
        st.session_state.bundle = None
    if "fact_reviewed" not in st.session_state:
        st.session_state.fact_reviewed = False


init_state()

if st.session_state.get("profile"):
    ACTIVE_PROFILE = st.session_state.profile
elif st.session_state.get("profile_topic"):
    _saved_profile = resolve_profile(st.session_state.profile_topic)
    if _saved_profile:
        ACTIVE_PROFILE = _saved_profile

with st.sidebar:
    st.markdown(
        '<div style="font-size:22px;font-weight:800;">🎬 AI 科普视频工厂</div>'
        '<div class="small" style="margin:4px 0 16px;">单文件 · GitHub 可部署 · 人工审核</div>',
        unsafe_allow_html=True,
    )
    page = st.radio("导航", ["制作", "事实核查", "分镜与脚本", "导出", "部署诊断"], label_visibility="collapsed")
    st.divider()
    st.markdown('<div class="small">Ollama</div>', unsafe_allow_html=True)
    st.code(f"{OLLAMA_BASE_URL}\n{OLLAMA_MODEL}", language="text")
    status = ollama_status()
    if status["available"] and status["model_present"]:
        st.markdown('<span class="pill green">Ollama 已连接</span>', unsafe_allow_html=True)
    elif status["available"]:
        st.markdown('<span class="pill amber">模型未安装</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="pill amber">Ollama 离线 · 用内置脚本</span>', unsafe_allow_html=True)
    st.caption("GitHub/Cloud 上的 127.0.0.1 不是你的电脑；远程 Ollama 请设置 OLLAMA_BASE_URL。")
    st.divider()
    st.caption("不自动发布 · 不抓取商业音乐 · 事实需人工复核")


if page == "制作":
    hero("把一个问题，做成一条可核查的科普短片", "Ollama 写脚本 · 程序化分镜 · 字幕/视频可选 · 没有本地模型也能预览")
    c1, c2 = st.columns([1.15, 0.85], gap="large")
    with c1:
        with st.form("generate_form"):
            topic = st.text_input("视频主题", value=ACTIVE_PROFILE["topic"])
            st.caption("内置主题：为什么 AI 会产生幻觉？ / 科普汽车发展史。其他主题可走下方通用模式。")
            facts_input = st.text_area(
                "补充事实 / 来源（可选，每行一条）",
                height=80,
                placeholder="例如：某个年份、某个定义、某个官方数据来源；不填则只生成通用解释框架。",
            )
            allow_generic = st.checkbox("任意主题：允许生成未核查草稿", value=True)
            a, b = st.columns(2)
            with a:
                duration = st.select_slider("目标时长（秒）", options=[60, 90, 120, 150, 180], value=120)
                scene_count = st.slider("分镜数量", 9, 14, 11)
            with b:
                prefer_llm = st.checkbox("优先调用 Ollama", value=True)
                auto_build = st.checkbox("生成脚本后自动做分镜/字幕/视频", value=True)
            submitted = st.form_submit_button("一键生成脚本与视频素材", type="primary", use_container_width=True)
        if submitted:
            profile = resolve_profile(topic)
            if profile is None:
                if not allow_generic:
                    st.error("这个主题没有内置事实库。勾选“任意主题：允许生成未核查草稿”，或先在代码里补一份 profile。")
                else:
                    profile = make_generic_profile(topic, facts_input)
            if profile is not None:
                ACTIVE_PROFILE = profile
                st.session_state.profile = profile
                st.session_state.profile_topic = profile["topic"]
                with st.spinner("正在生成脚本（Ollama 可能需要几分钟）..."):
                    script, mode = generate_script(topic, duration, scene_count, prefer_llm, profile)
                st.session_state.script = script
                st.session_state.mode = mode
                st.session_state.images = []
                st.session_state.audio = None
                st.session_state.video = None
                st.session_state.bundle = None
                st.session_state.fact_reviewed = False
                mode_label = {
                    "ollama": "Ollama",
                    "fallback": "内置已核查脚本",
                    "generic_template": "通用解释模板（未核查）",
                    "ollama_generic": "Ollama 通用草稿（未核查）",
                }.get(mode, mode)
                st.success("脚本已生成：" + mode_label)
                if profile.get("unverified"):
                    st.warning("这是未核查草稿：没有内置事实库，发布前必须补充权威来源并逐条核对。")
                if mode == "fallback":
                    st.info("Ollama 不可用或输出未通过校验，已回退到内置脚本；这不会阻止你继续生成分镜。")
                if auto_build:
                    with st.spinner("正在渲染分镜..."):
                        images, durations = build_storyboard(script, profile)
                    st.session_state.images = images
                    st.session_state.durations = durations
                    with st.spinner("正在生成字幕、配音并尝试合成视频..."):
                        bundle = build_export_bundle(script, images, durations)
                    st.session_state.bundle = bundle
                    st.session_state.srt = bundle.get("srt")
                    st.session_state.audio = bundle.get("audio")
                    st.session_state.video = bundle.get("video")
                    if bundle.get("video"):
                        st.success("一键生成完成：分镜、SRT、配音和 preview.mp4 都已完成。")
                    else:
                        st.info("分镜和 SRT 已完成；当前环境没有 FFmpeg 或视频合成失败，可稍后在导出页重试。")
    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 这一版保留的核心逻辑")
        st.markdown(
            "- 主题 → 结构化 JSON 脚本\n"
            "- 内置主题绑定 `claim_ids`；通用模式明确标记未核查\n"
            "- 真实调用 `/api/chat`、`/api/tags`\n"
            "- 9–11 张 1080×1920 程序化信息图\n"
            "- SRT 字幕 + 可选 MP4 合成\n"
            "- 失败自动回退，不中断页面"
        )
        st.markdown("</div>", unsafe_allow_html=True)
        st.info("GitHub 上没有本地 Piper/字体/FFmpeg 时，仍可生成脚本、PNG、SRT；视频合成会自动跳过。")

elif page == "事实核查":
    hero("事实核查", "先看证据，再看画面；自动状态不等于人工读过原文", "FACT CHECK")
    fact = factcheck_payload()
    st.markdown(f"### {fact['topic']}")
    st.caption(f"生成时间：{fact['checked_at']} · 登记来源 {len(fact['sources'])} 条 · 论断 {len(fact['claims'])} 条")
    if not fact["claims"]:
        st.warning("通用解释模板草稿：没有内置事实库。请补充权威来源，逐条核对后再发布。")
    cols = st.columns(3)
    cols[0].metric("高风险论断", sum(1 for c in fact["claims"] if c["risk"] == "high"))
    cols[1].metric("已登记来源", len(fact["sources"]))
    cols[2].metric("人工审核", "已确认" if st.session_state.fact_reviewed else "待确认")
    rows = [
        {
            "ID": c["id"],
            "风险": c["risk"],
            "状态": "✅ 有来源" if c["verification_status"] == "supported" else "⚠️ 待复核",
            "来源": "、".join(c["source_ids"]),
            "论断": c["text"],
        }
        for c in fact["claims"]
    ]
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
    st.markdown("#### 来源清单")
    if fact["sources"]:
        for s in fact["sources"]:
            link = f"[{s['title']}]({s['url']})" if s.get("url") else s["title"]
            st.markdown(f"- **{s['id']}** {link} · {s['year']}")
    else:
        st.info("没有登记来源。通用模式只能生成解释框架，不能当作事实核查通过。")
    st.download_button("下载 fact_check.json", json.dumps(fact, ensure_ascii=False, indent=2).encode("utf-8"), file_name="fact_check.json", mime="application/json")
    st.download_button("下载 sources.md", sources_markdown().encode("utf-8"), file_name="sources.md", mime="text/markdown")
    st.markdown("#### 人工确认")
    checked = st.checkbox("我已阅读关键来源，并确认脚本没有断章取义、过时结论或夸大表述", value=st.session_state.fact_reviewed)
    st.session_state.fact_reviewed = checked
    if checked:
        st.success("已记录人工事实审核确认。发布前仍需完整观看视频。")
    else:
        st.warning("未确认前不要发布。自动核查只验证来源映射，不替代阅读原文。")

elif page == "分镜与脚本":
    hero("分镜与脚本", "逐段检查旁白、屏幕要点和事实绑定；可改文案后重新生成画面", "REVIEW")
    script = st.session_state.script
    st.markdown(f"### {script.get('title', '')}")
    st.caption(script.get("subtitle", ""))
    scenes = script.get("scenes", [])
    for i, scene in enumerate(scenes, 1):
        with st.expander(f"{i:02d} · {scene.get('heading', '')}", expanded=(i == 1)):
            scene["heading"] = st.text_input("屏幕标题", value=scene.get("heading", ""), key=f"h{i}")
            scene["narration"] = st.text_area("旁白", value=scene.get("narration", ""), key=f"n{i}", height=100)
            scene["key_takeaway"] = st.text_input("关键结论", value=scene.get("key_takeaway", ""), key=f"k{i}")
            st.caption("绑定论断：" + "、".join(scene.get("claim_ids", [])) + " · 图示：" + scene.get("diagram", ""))
    st.session_state.script = script
    if st.button("重新生成分镜画面", type="primary", use_container_width=True):
        with st.spinner("正在渲染 1080×1920 分镜..."):
            images, durations = build_storyboard(script, ACTIVE_PROFILE)
            st.session_state.images = images
            st.session_state.durations = durations
        st.success(f"已生成 {len(images)} 张分镜")
    if st.session_state.images:
        cols = st.columns(3)
        for i, path in enumerate(st.session_state.images):
            with cols[i % 3]:
                st.image(path, caption=f"分镜 {i + 1:02d}", use_container_width=True)

elif page == "导出":
    hero("导出", "脚本、分镜、SRT 与可选 MP4；发布前必须人工核对", "EXPORT")
    script = st.session_state.script
    scenes = script.get("scenes", [])
    a, b, c = st.columns(3)
    a.download_button("下载 script.json", json.dumps(script, ensure_ascii=False, indent=2).encode("utf-8"), file_name="script.json", mime="application/json", use_container_width=True)
    if st.session_state.images:
        b.download_button("下载第一张分镜 PNG", Path(st.session_state.images[0]).read_bytes(), file_name="scene_01.png", mime="image/png", use_container_width=True)
    if st.button("生成 SRT / 尝试合成 MP4", type="primary", use_container_width=True):
        with st.spinner("正在生成字幕、配音并尝试合成视频..."):
            images = st.session_state.images
            durations = st.session_state.durations or estimate_durations(scenes)
            bundle = build_export_bundle(script, images, durations)
            st.session_state.bundle = bundle
            st.session_state.srt = bundle.get("srt")
            st.session_state.audio = bundle.get("audio")
            st.session_state.video = bundle.get("video")
            if not images:
                st.warning("还没有分镜 PNG：请先到「分镜与脚本」生成画面，或回「制作」一键生成。")
        st.success("导出处理完成")
        if not st.session_state.audio:
            st.info("未检测到本地 Piper 模型：已生成 SRT；MP4 为静音或未合成。")
    fact = factcheck_payload()
    manifest = export_manifest(script, st.session_state.get("bundle"))
    st.markdown("#### 可下载文件")
    d1, d2, d3 = st.columns(3)
    d1.download_button("fact_check.json", json.dumps(fact, ensure_ascii=False, indent=2).encode("utf-8"), file_name="fact_check.json", mime="application/json", use_container_width=True)
    d2.download_button("sources.md", sources_markdown().encode("utf-8"), file_name="sources.md", mime="text/markdown", use_container_width=True)
    d3.download_button("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"), file_name="manifest.json", mime="application/json", use_container_width=True)
    if st.session_state.get("srt"):
        st.download_button("下载 subtitles.srt", Path(st.session_state.srt).read_bytes(), file_name="subtitles.srt", mime="text/plain", use_container_width=True)
    if st.session_state.video:
        st.video(st.session_state.video)
        st.download_button("下载 preview.mp4", Path(st.session_state.video).read_bytes(), file_name="preview.mp4", mime="video/mp4", use_container_width=True)
    if not st.session_state.fact_reviewed:
        st.warning("事实核查尚未人工确认。请在「事实核查」页确认后再发布。")
    st.warning("导出前请人工核对事实、字幕、配音和画面。程序通过不等于可以发布。")

elif page == "部署诊断":
    hero("部署诊断", "检查 Ollama、FFmpeg、Piper、字体和 GitHub 配置", "DIAGNOSTICS")
    st.markdown("### 运行环境")
    stt = ollama_status()
    ff = find_ffmpeg()
    piper_ok = piper_available()
    rows = [
        {"项目": "Ollama 地址", "状态": OLLAMA_BASE_URL, "说明": "可用" if stt["available"] else "不可达"},
        {"项目": "Ollama 模型", "状态": OLLAMA_MODEL, "说明": "已安装" if stt["model_present"] else "未安装"},
        {"项目": "FFmpeg", "状态": ff or "未找到", "说明": "可合成 MP4" if ff else "只能导出 PNG/SRT"},
        {"项目": "Piper 本地配音", "状态": "可用" if piper_ok else "未安装", "说明": "本地模型存在" if piper_ok else "MP4 可能静音"},
        {"项目": "中文字体", "状态": "已检测" if "Font" in type(font(20)).__name__ else "默认字体", "说明": "云端请安装 fonts-noto-cjk"},
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)
    st.markdown("### GitHub / Streamlit Cloud")
    st.code(
        "提交文件：app.py, requirements.txt, packages.txt, .streamlit/config.toml\n"
        "requirements.txt 由 requirements-github.txt 重命名\n"
        "Secrets:\nOLLAMA_BASE_URL = \"https://your-ollama-host\"\nOLLAMA_MODEL = \"qwenlocal\"",
        language="text",
    )
    st.info("Streamlit Cloud 的 127.0.0.1 不是你的电脑；没有远程 Ollama 时会使用内置已核查脚本。")
    st.warning("单文件版不包含本地 Piper 模型和完整 ASS 烧录；需要完整质量请用本地 ui/app.py。")
