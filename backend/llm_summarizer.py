from __future__ import annotations

import asyncio
import inspect
import os
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from contextlib import suppress
from typing import Any, Sequence
from urllib.parse import urlparse

from .transcript import (
    TranscriptSegment,
    chunk_segments,
    format_timestamp,
    merge_segment_lines,
    segments_to_prompt,
)


PROVIDER_DEFAULTS = {
    # 与 frontend/script.js 的 PROVIDER_CONFIG.defaultModel 保持一致
    "deepseek": ("https://api.deepseek.com", "deepseek-flash"),
    "openai": ("https://api.openai.com/v1", "gpt-5.6-terra"),
    "openai_gpt4": ("https://api.openai.com/v1", "gpt-4o"),
    "openai_gpt35": ("https://api.openai.com/v1", "gpt-4o-mini"),
    "qwen": (
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "qwen3.7-plus",
    ),
    "glm": ("https://open.bigmodel.cn/api/paas/v4", "glm-4.7-flash"),
    "moonshot": ("https://api.moonshot.cn/v1", "kimi-k3"),
}


def default_base_url(model_type: str) -> str:
    """内置 Provider 的官方接口地址；custom 或未知 Provider 返回空串。"""
    entry = PROVIDER_DEFAULTS.get((model_type or "").strip().lower())
    return entry[0] if entry else ""


def normalize_endpoint_host(base_url: str, model_type: str = "") -> str:
    """把接口地址归一成可比对的身份（host[:port] + 去掉尾斜杠的路径）。

    本机保存的 API Key 以此为键，语义是"这把 Key 只可能被发给它当初那个端点"：
    因此只取 host+path，忽略大小写、尾斜杠、默认端口和 query/fragment
    （query 里可能带凭据，绝不能成为身份的一部分）。地址完全无法解析时返回空串，
    调用方据此拒绝复用任何已保存的 Key。
    """
    value = (base_url or "").strip() or default_base_url(model_type)
    if not value:
        return ""
    if "://" not in value:
        value = f"https://{value}"
    try:
        parts = urlparse(value)
        host = (parts.hostname or "").lower()
        port = parts.port
    except ValueError:
        return ""
    if not host:
        return ""
    if port and port not in (80, 443):
        host = f"{host}:{port}"
    path = (parts.path or "").rstrip("/").lower()
    return f"{host}{path}"

SUMMARY_CHUNK_CHARACTERS = 9_000
MERGE_INPUT_CHARACTERS = 14_000
LLM_TIMEOUT_SECONDS = 300
LLM_MAX_RETRIES = 0
DEEPSEEK_HIGH_TOKEN_BUDGET = 12_000
# max 档的输出额度按正文需求的倍数抬高：思考链与正文共用一个池子，额度只是油箱
# 容量而不是油门——正常调用想停就停、用不满，抬高不影响成本；但太低会让长思考
# 吃满后整条流被截断、正文为 0（实测 max 档思考 2.2 万字撞 1.38 万 token 上限，
# 正文一个字没拿到，靠关思考重试才救回）。×5 给思考留出收尾余地，正常产出规模
# 仍由各阶段的 max_tokens 把住。
MAX_EFFORT_BUDGET_MULTIPLIER = 5
# 观测到某条通道确实在返回思考链之后，给它后续每次请求的输出额度补一份思考预留。
# 靠实测而不是猜档位：ollama / vLLM / 各种免费兼容网关上我们对思考没有任何把手
# （``_apply_reasoning`` 对 custom 什么都不下发），链长只能由模型决定，而思考链计入
# 输出 token——额度不够时挤掉的是正文（用户实测：一次压缩请求思考 4505 字 / 正文 0）。
# 同一进程内按 (provider, model, base_url) 记住，第一次请求之后就会自动放宽。
# DeepSeek 兼容通道不走这里，它已有 MAX_EFFORT_BUDGET_MULTIPLIER 那套膨胀机制。
THINKING_RESERVE_TOKENS = 2_000
_THINKING_SEEN_CHANNELS: set[tuple[str, str, str]] = set()
# auto 档在 DeepSeek 兼容通道的默认推理档位：开思考（high）。
# 2026-10-03 同一份 25 分钟转录（13952 字 / 167 段）三档对照：off 36 秒出 30 个小节、
# 平均每节 183 字、正文里一个时间戳都没有（带时间轴的目录，不像笔记）；high 3 分 29 秒
# 出 11 个话题小节、平均每节 384 字、正文另带 22 个时间点，且最长一次思考 18955 字仍在
# 额度内没被截断；max 5 分 25 秒，两个块思考吃满 16960 额度后退化成 off，成稿因此整段
# 少了 09:29→12:04 的内容。默认从 off 换成 high 就是为了最后这两条。
# 代价要写清：中短视频默认慢一个数量级（36s → 3.5min），想快请显式选「极速」。
# 这条默认同样作用于"模型名里带 deepseek 的第三方免费网关"（_uses_deepseek_compatibility
# 按名字判），那类通道有长思考流被 peer closed 掐断的旧记录——现在断流与正文为 0 都不再
# 判死任务，但会更慢。长视频逐段直写那一档仍走 SECTION_EFFORT=off：每段只几千字，思考
# 帮不上忙（实测 high 档每段一万多字思考只换一千字正文，28 分钟跑了 44 分钟）。
STYLE_DEFAULT_EFFORT = "high"
# 逐段直写（长视频）在 auto 下的档位：直接关思考。这一步是“照着几千字原文整理成稿”
# 的局部任务，思考链帮不上忙——实测 high 档每段 1 万多字思考只换 1 千字正文，
# 28 分钟的视频总共跑了 44 分钟；关掉后输出额度全部留给正文。
SECTION_EFFORT = "off"
# 「挑重点、不保证逐字覆盖」的风格：与 concise 同属一类——不做长视频逐段直写，也不做尾部
# 覆盖核对（覆盖度是"是否写到结尾"的一致性检查，对清单式产出不适用）。
CONCISE_LIKE_STYLES = frozenset({"concise", "keypoints"})
# 笔记风格与内容场景白名单。风格管"怎么写"，场景管"写什么结构"，两者正交拼装；
# general 拼空块，行为与历史版本完全一致（老任务缺 content_type 时缺省 general）。
SUMMARY_STYLES = frozenset({"detailed", "faithful", "concise", "keypoints"})
CONTENT_TYPES = frozenset({"general", "tutorial", "knowledge", "learning"})
# 笔记末尾时间戳落后转写结尾超过该秒数视为丢尾。
TAIL_GAP_SECONDS = 60.0
# 补尾时附带给模型参考的已有笔记结尾长度。
NOTE_TAIL_CHARACTERS = 600
# 补写结尾的三道护栏。成稿提示词明确允许“不必每段都加时间戳”，所以“笔记末尾时间戳
# 落后于转写结尾”既可能是真的丢尾，也可能只是省略了时间戳。不加护栏时，后者会把
# 整段剩余转录塞进一次 max 思考档的补写请求（几十分钟无任何进度反馈，表现为“卡住不动”，
# 而 detailed 只记告警不发请求，所以只有“详细复原”会卡）。
# 缺口超过该秒数视为“省略时间戳”而不是丢尾，只告警不补写。
TAIL_PATCH_MAX_GAP_SECONDS = 600.0
# 补写材料字符上限，超过时只保留最靠近结尾的部分。
TAIL_PATCH_MAX_INPUT_CHARACTERS = 4_000
# 用户手动补写（自己按的按钮）时的材料上限：比自动补写宽。自动补写没有征求同意，
# 缺口一大会退化成“把剩余转录重写一遍”，几十分钟没有进度；手动这次是用户看着缺口、
# 代价和超时上限做的选择，请求规模仍有上限，就没必要只补最后一段给他留个洞。
TAIL_PATCH_MAX_INPUT_CHARACTERS_MANUAL = 12_000
# 补写请求的整体耗时上限：超时则保留原笔记，绝不让任务停在补写阶段不放。
TAIL_PATCH_TIMEOUT_SECONDS = 240.0
# 流式读取心跳间隔：把"模型仍在输出"反映到任务进度上，避免长时间同一句话看起来像死锁。
# 与转写心跳同节奏（30s）：20s 时日志滚动太快，反而看不清进度。
LLM_HEARTBEAT_SECONDS = 30.0
# 转录净字数超过该值时改走“逐段直写成稿”管线（见 ``_write_long_transcript_notes``）。
# 根因：``_reduce_chunks`` 把成稿可见材料压到恒 ≤ ``MERGE_INPUT_CHARACTERS`` 字、成稿输出
# 恒 ≤ 4_600 token，两个上限都与视频时长无关。3 小时课（6.2 万字）因此丢约 2 小时内容，
# 而 37 分钟这一档逐段压缩后本来就 < 14000 字、归并未触发、额度也够用，实测是正常的，
# 所以阈值以下一律保持原管线，不拿已验证良好的行为去换未验证的路径。
# 24_000 净字数 ≈ 70~80 分钟口播，是策略选择而非推导值：中长度视频若反馈变差就下调。
LONG_TRANSCRIPT_CHARACTERS = 24_000
# 每段直写的转录净字数预算（时间戳开销不计入，见 chunk_segments 的同名参数）。
NOTE_SECTION_CHARACTERS = 6_500
# 提示词内把碎 ASR 行合并到的目标字数：whisper 常切成平均 7.5 字一段，逐行加时间戳会让
# 时间戳吃掉 81% 的输入预算，模型读到的也不像话。
SECTION_LINE_MERGE_CHARACTERS = 48
# 同时直写的段数。这是本仓第一处并行模型调用：task_slots 只管任务级，不与之冲突。
SECTION_WRITE_CONCURRENCY = 3
# 每段输出额度 = 净字数 / 字每 token × 保留率，夹在上下限之间。
# 1.6 字/token 与 ``_tail_patch_material`` 里“+33 字行开销”同一套估算口径；
# 0.8 是口语转写成书面笔记的经验保留率（去掉语气词与重复后仍略短于原文）。
SECTION_TOKENS_PER_CHARACTER = 1.6
SECTION_TOKEN_KEEP_RATIO = 0.8
SECTION_MAX_TOKENS_MIN = 1_200
SECTION_MAX_TOKENS_MAX = 5_000
# 分片压缩与分层归并这两次「整理」请求的输出额度：按材料净字数估正文需要多少，
# 再加一份思考链预留。旧值是恒定 1200 / 1400，同时犯了两错——
# ① 一个 9000 预算的块里真语音从 1700 字到 9000 字都有（whisper 碎段把预算吃在时间戳
#    开销上），恒定额度对大块不够、对小块过宽；
# ② 1200 连思考链都装不下：ollama 上 gemma 实测一次压缩请求思考 4505 字 / 正文 0 /
#    finish_reason=length，整条任务死在 10%（该通道不响应关思考，见 _apply_reasoning
#    对 custom 的处理——我们没有任何把手）。
# 额度只是油箱容量不是油门（见 MAX_EFFORT_BUDGET_MULTIPLIER 那条注释），用不满不加成本。
NOTE_BUDGET_KEEP_RATIO = 0.45
NOTE_BUDGET_THINKING_RESERVE_TOKENS = 1_500
NOTE_BUDGET_MAX_TOKENS_MIN = 1_600
NOTE_BUDGET_MAX_TOKENS_MAX = 5_000
# 每段最多尝试次数（首次 + 重试一次）与段内续写次数。
SECTION_ATTEMPTS = 2
SECTION_CONTINUATION_MAX = 3
# 撞到限流时，重试前先等多久。实测并发 3 会让免费兼容网关（如 ModelScope 推理服务）瞬间
# 把整批请求全打成 429，而紧接着的第二次重试等于继续砸请求 ⇒ 11 段全部落到「内嵌原文」
# 兜底，笔记虽没丢内容但完全没法看。等待期间仍占着并发额度，顺带起到节流作用。
SECTION_RATE_LIMIT_BACKOFF_SECONDS = 20.0
_RATE_LIMIT_RE = re.compile(
    r"429|rate.?limit|too many requests|insufficient_quota|out of quota", re.IGNORECASE
)
# 兜底小节的标题字样（测试与核对脚本按它区分「整理过的正文」和「内嵌的原文」）。
RAW_FALLBACK_HEADING = "### 未能整理的原文"
# 各段标题写完之后，概览文本的一次小请求额度。
OVERVIEW_MAX_TOKENS = 600
# 兜底时附在提示词里的“上一段结尾原话”长度，只用于让模型看懂指代。
SECTION_CONTEXT_CHARACTERS = 160
# 背景资料（视频简介）进提示词的预算：帮助模型认出这是哪门课/谁的视频、
# 核对术语与专有名词写法；只用于理解与校对，不允许当成视频内容写进笔记。
# 评论区热评多为玩梗，误导风险大于信息量，刻意不采集。
BRIEF_DESCRIPTION_CHARACTERS = 400
# 匹配笔记中的 [MM:SS]、[MM:SS-MM:SS]、[HH:MM:SS] 等时间戳（起点与区间终点都计入）。
# 分钟位允许 1~3 位：长视频模型常把 1 小时 45 分写成 [105:30]，识别不到会被误判成丢尾。
_NOTE_TIMESTAMP_RE = re.compile(
    r"\[(\d{1,3}:\d{2}(?::\d{2})?)(?:\s*[-–—~]\s*(\d{1,3}:\d{2}(?::\d{2})?))?\]"
)
# 标题结尾的 [MM:SS] / [起点-终点]（用户要求小标题带时间更好读，但目录行前已印过区间，
# 所以取标题时剥掉；只认时钟样式，避免误删标题里正常的方括号内容）
_HEADING_TIMESTAMP_RE = re.compile(
    r"\s*\[\d{1,3}:\d{2}(?::\d{2})?(?:\s*[-–—~]\s*\d{1,3}:\d{2}(?::\d{2})?)?\]$"
)
# 模型（尤其开思考那几档）爱把标题区间写成 [00:00 – 00:33]，而提示词与代码生成的目录用的
# 是半角 -。同一次运行内部一致、跨次不一致，收尾统一掉；只认时间:时间两侧的空格长破折号，
# 不动正文里正常的破折号。
_TIMESTAMP_DASH_RE = re.compile(
    r"(\d{1,3}:\d{2}(?::\d{2})?)\s*[–—―−]\s*(\d{1,3}:\d{2}(?::\d{2})?)"
)
# 模型常把公式写成 \(...\) / \[...\]，而多数 Markdown 渲染器只认 $...$ / $$...$$。
# 成稿收尾时统一归一；代码块内是字面文本，跳过不改。
_MATH_DISPLAY_RE = re.compile(r"\\\[(.+?)\\\]", re.S)
_MATH_INLINE_RE = re.compile(r"\\\((.+?)\\\)", re.S)
# 思考类参数兼容差异：官方 DeepSeek 认私有 thinking，很多第三方 OpenAI 兼容网关
# 只认标准 reasoning_effort（v1.2.3 用户反馈 400 unsupported_parameter "thinking"）。
# 被拒过的参数按 (provider, model, base_url) 记住，同进程后续任务不再重复踩。
REASONING_PARAM_NAMES = ("thinking", "reasoning_effort", "enable_thinking", "reasoning")
# 输出额度参数也可能被第三方网关拒绝（值超过它的上限）
TOKEN_BUDGET_PARAM_NAMES = ("max_tokens", "max_completion_tokens")
_DEGRADABLE_PARAM_NAMES = REASONING_PARAM_NAMES + TOKEN_BUDGET_PARAM_NAMES
_UNSUPPORTED_PARAM_RE = re.compile(
    r"""["'`](thinking|reasoning_effort|enable_thinking|reasoning|max_tokens|max_completion_tokens)["'`]
    [^"',()]{0,40}?\s+(?:is|are|was|were|seems\s+to\s+be)\s+not\s+supported""",
    re.IGNORECASE | re.VERBOSE,
)
# 另一种形状：网关不否认参数名，只否认我们给的取值（ModelScope 对
# `reasoning_effort: none` 回 "'reasoning_effort' must be one of: 'low', ..."）。
# 对我们的目的来说这等同于「这个参数在这条通道上不能用」，同样该走降级而不是判失败。
_REJECTED_VALUE_RE = re.compile(
    r"""["'`](thinking|reasoning_effort|enable_thinking|reasoning|max_tokens|max_completion_tokens)["'`]
    [^"']{0,40}?\s+(?:must\s+be\s+one\s+of|must\s+be\b|is\s+invalid|invalid\s+value)""",
    re.IGNORECASE | re.VERBOSE,
)
_REJECTED_REASONING_PARAMS: dict[tuple[str, str, str], set[str]] = {}
# 参数被拒后的最多降级次数（thinking → reasoning_effort → 不注入）
MAX_PARAM_RETRIES = 2

def _looks_rate_limited(exc: BaseException) -> bool:
    """这次失败是否像限流/配额（决定逐段直写重试前要不要先等一会）。"""
    return bool(_RATE_LIMIT_RE.search(f"{type(exc).__name__}: {exc}"))


def _looks_disconnected(exc: BaseException) -> bool:
    """这次失败是否像流被网络/网关中途掐断（决定要不要自动重投一次）。

    实测昇腾免费通道会在十几分钟的长思考流上直接 peer closed；openai SDK 把
    这类错误统一包成 APIConnectionError，不认时再按消息特征兜底。
    """
    try:
        from openai import APIConnectionError
    except ImportError:
        return False
    if isinstance(exc, APIConnectionError):
        return True
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(
        marker in text
        for marker in (
            "peer closed",
            "incomplete chunked read",
            "connection reset",
            "connection aborted",
            "remotedisconnected",
        )
    )


def utf8_safe(text: str) -> str:
    """把模型输出里编不进 UTF-8 的字符换成占位符。

    输出被 max_tokens 截断时，emoji 常常只剩下半个代理对（U+D800–U+DFFF）。这种
    字符拼进 str 不报错，一到 JSON 序列化才抛 UnicodeEncodeError，而且炸点在端点
    return 之后：/api/llm-test 会崩成裸 500（前端只剩"测试请求失败（HTTP 500）"，
    看着完全像上游挂了），笔记正文则会把 notes.md 与 task.json 的写入一起带崩。
    """
    return text.encode("utf-8", "replace").decode("utf-8", "replace")


def _rejected_params_from_env() -> set[str]:
    """逃生口：`AINOTE_REJECTED_LLM_PARAMS=thinking,reasoning_effort`。

    跟 pi/pivi 的 `compat.supportsReasoningEffort` 同一思路：通道能力靠声明而不是
    猜名字。自动降级能覆盖绝大多数情况，但“每阶段都先吃一次 400 再重试”仍然很浪费
    时，遇到这种通道让用户一行环境变量声明掉即可（多个用逗号分隔）。
    """
    raw = (os.getenv("AINOTE_REJECTED_LLM_PARAMS") or "").lower()
    return {
        name.strip()
        for name in raw.replace(" ", ",").split(",")
        if name.strip() in REASONING_PARAM_NAMES
    }

_PROXY_ENV_NAMES = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY")


def mask_proxy_target(value: str) -> str:
    """把代理地址里的账号段打码（http://user:pass@host:port → http://***@host:port）。

    诊断信息里允许出现"代理指向哪里"，但绝不允许把代理凭据带出去。
    """
    text = str(value or "").strip()
    if "@" not in text:
        return text[:120]
    scheme, sep, rest = text.partition("://")
    head = f"{scheme}://" if sep else ""
    host = rest.rsplit("@", 1)[-1]
    return f"{head}***@{host}"[:120]


def env_proxy_summary() -> str | None:
    """环境变量里配置的出站代理（打码后）；没有则 None。

    httpx（openai SDK 底层）只认这一份代理配置；Windows 注册表里的系统代理
    影响的是浏览器与 urllib。分开看才说得清"请求到底走的哪条路"。
    """
    for name in _PROXY_ENV_NAMES:
        value = os.environ.get(name)
        if value:
            return f"{name}={mask_proxy_target(value)}"
    return None


def _is_connect_failure(exc: Exception | None) -> bool:
    """只有"连不上"这一类失败才值得绕代理重试；401/404 与代理无关。"""
    if exc is None:
        return False
    try:
        from openai import APIConnectionError
    except ImportError:
        return False
    return isinstance(exc, APIConnectionError)  # APITimeoutError 是它的子类


ProgressCallback = Callable[[int, str], Awaitable[None] | None]


@dataclass
class TailCoverage:
    """笔记覆盖度核对结果：缺口多大、能不能补、补写要花多少预算。

    只陈述事实与预算，不做决定——自动补写与否由 ``_ensure_tail_coverage`` 按风格判断；
    用户手动补写则由接口把这份报告原样交给前端，让用户看着代价自己选。
    """

    note_end: float | None
    coverage_end: float
    gap_seconds: float
    missing_segments: int
    missing_characters: int
    material_segments: int
    material_characters: int
    patch_max_tokens: int
    patch_available: bool
    patch_blocked_reason: str | None

    @property
    def covers_full_gap(self) -> bool:
        """补写材料是否覆盖整个缺口；否则只补最靠近结尾的一段。"""
        return self.missing_segments > 0 and self.material_segments >= self.missing_segments

    def to_dict(self) -> dict[str, Any]:
        """给前端看的视图：时间戳转成 MM:SS，字段名与页面文案对齐。"""
        return {
            "note_end": (
                format_timestamp(self.note_end) if self.note_end is not None else None
            ),
            "coverage_end": format_timestamp(self.coverage_end),
            "gap_seconds": round(self.gap_seconds, 1),
            "missing_segments": self.missing_segments,
            "missing_characters": self.missing_characters,
            "material_characters": self.material_characters,
            "covers_full_gap": self.covers_full_gap,
            "patch_available": self.patch_available,
            "patch_blocked_reason": self.patch_blocked_reason,
            "patch_max_tokens": self.patch_max_tokens,
            "patch_timeout_seconds": int(TAIL_PATCH_TIMEOUT_SECONDS),
        }


@dataclass
class TailPatchOutcome:
    """一次按需补写的结果：成功给正文，失败给能直接展示的原因。

    ``text`` 是补写后的整篇正文（生成流程用），``body`` 是这次补写的片段
    （补写已有笔记时用——归档那份的图片引用与任务目录不同，得各自拼接）。
    """

    text: str | None = None
    body: str | None = None
    missing_start: str = ""
    error_kind: str | None = None  # timeout / error / empty / unavailable
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.text is not None


class LLMSummarizer:
    def __init__(
        self,
        model_type: str,
        api_key: str,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("An API key is required")
        if model_type == "custom":
            if not base_url or not model:
                raise ValueError("Custom providers require base_url and model")
        else:
            if model_type not in PROVIDER_DEFAULTS:
                raise ValueError(f"Unsupported model provider: {model_type}")
            default_base_url, default_model = PROVIDER_DEFAULTS[model_type]
            base_url = base_url or default_base_url
            model = model or default_model

        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "openai is not installed. Run pip install -r backend/requirements.txt"
            ) from exc

        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=LLM_TIMEOUT_SECONDS,
            max_retries=LLM_MAX_RETRIES,
        )
        self._api_key = api_key  # 直连重试要重建客户端；任何日志/响应出口统一脱敏
        self.model_type = model_type
        self.base_url = base_url or ""
        self.model = model
        self.warnings: list[str] = []
        # 最近一次覆盖度核对结果（见 ``_ensure_tail_coverage``）：任务结果要靠它把
        # “笔记没写到结尾”推给前端，让用户自己决定补不补。
        self.coverage: TailCoverage | None = None
        # rejected_params（该配置已被拒的思考参数）由同名属性从全局缓存按需读取，
        # 同进程内后续任务不会再重复踩同一个 400

    async def test_connection(self, timeout_seconds: float = 20.0) -> tuple[bool, str, float]:
        """轻量连通性测试：发一个极小请求，返回 (是否成功, 可读消息, 耗时秒)。

        带环境代理的失败会用"不走代理"的直连再试一次：1.4.2 群里那批"只有重启
        电脑才好"的故障，最像代理/加速器把出站请求带进僵死状态——直连通了就把
        结论说给用户（关代理即可），别让人重启电脑碰运气。
        """
        start = time.monotonic()
        ok, message, latency, failure = await self._ping(self.client, timeout_seconds)
        proxy_in_env = env_proxy_summary()
        if ok or proxy_in_env is None or not _is_connect_failure(failure):
            return ok, utf8_safe(message), latency
        bare, bare_http = self._direct_client(timeout_seconds)
        if bare is None:
            return False, utf8_safe(message), latency
        try:
            direct_ok, direct_message, direct_latency, _ = await self._ping(bare, timeout_seconds)
        finally:
            with suppress(Exception):
                await bare.close()
            with suppress(Exception):
                await bare_http.aclose()
        if direct_ok:
            return True, (
                f"{direct_message}（不走系统代理直连成功；走 {proxy_in_env} 的请求失败：{message}）"
                "——本机代理/加速器大概率僵死了，关掉它即可，不用重启电脑"
            ), direct_latency
        return False, (
            f"{message}；绕开代理直连也一样失败：{direct_message}"
            "——到该地址的整条网络都不通，先关代理/加速器试一次，不行再重启电脑"
        ), direct_latency

    async def _ping(self, client: Any, timeout_seconds: float) -> tuple[bool, str, float, Exception | None]:
        start = time.monotonic()
        try:
            response = await client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=8,
                timeout=timeout_seconds,
            )
            latency = time.monotonic() - start
            reply = utf8_safe(response.choices[0].message.content or "").strip()
            if reply:
                return True, f"连接成功（{latency * 1000:.0f} ms）模型响应：{reply[:40]}", latency, None
            return True, f"连接成功（{latency * 1000:.0f} ms）", latency, None
        except Exception as exc:
            latency = time.monotonic() - start
            return False, self._describe_llm_error(exc, latency), latency, exc

    def _direct_client(self, timeout_seconds: float) -> tuple[Any | None, Any | None]:
        """绕开环境代理的临时客户端；依赖缺失时返回 (None, None)，退回原失败文案。"""
        try:
            import httpx
            from openai import AsyncOpenAI
        except ImportError:
            return None, None
        http = httpx.AsyncClient(trust_env=False, timeout=timeout_seconds)
        client = AsyncOpenAI(
            api_key=self._api_key,
            base_url=self.base_url or None,
            http_client=http,
            max_retries=0,
        )
        return client, http

    @staticmethod
    def _describe_llm_error(exc: Exception, latency: float) -> str:
        """把连接异常翻译成用户可读的诊断信息（区分 key 错误 / URL 错误 / 其他）。"""
        try:
            from openai import (
                APIConnectionError,
                APITimeoutError,
                AuthenticationError,
                BadRequestError,
                NotFoundError,
                PermissionDeniedError,
                RateLimitError,
            )
        except ImportError:
            message = (
                f"连接失败（{latency * 1000:.0f} ms）：{type(exc).__name__}: {exc}"
            )
            return message

        elapsed_ms = f"{latency * 1000:.0f} ms"
        if isinstance(exc, AuthenticationError):
            return (
                f"连接失败（HTTP 401，{elapsed_ms}）：API Key 无效或已过期，请检查密钥"
            )
        if isinstance(exc, PermissionDeniedError):
            return (
                f"连接失败（HTTP 403，{elapsed_ms}）：API Key 无权限访问该模型，"
                f"请检查密钥权限或模型名称"
            )
        if isinstance(exc, NotFoundError):
            return (
                f"连接失败（HTTP 404，{elapsed_ms}）：接口/模型不存在——大概率是 "
                f"Base URL 或模型名填错了"
            )
        if isinstance(exc, BadRequestError):
            return (
                f"连接失败（HTTP 400，{elapsed_ms}）：请求被拒绝——通常是模型名不识别"
                f"或参数不受支持（{exc}）"
            )
        if isinstance(exc, RateLimitError):
            return (
                f"连接失败（HTTP 429，{elapsed_ms}）：请求频率超限或余额不足，请稍后重试"
            )
        if isinstance(exc, APITimeoutError):
            return (
                f"连接失败（{elapsed_ms}）：请求超时——服务无响应，或网络不通"
            )
        if isinstance(exc, APIConnectionError):
            hint = "Base URL 填错或网络无法访问该地址"
            if "resolve" in str(exc).lower() or "dns" in str(exc).lower():
                hint = "域名解析失败——Base URL 填错或网络不通"
            return (
                f"连接失败（{elapsed_ms}）：{hint}（{type(exc).__name__}: {exc}）"
            )
        detail = getattr(exc, "status_code", None)
        if detail:
            return f"连接失败（HTTP {detail}，{elapsed_ms}）：{exc}"
        return f"连接失败（{elapsed_ms}）：{type(exc).__name__}: {exc}"

    async def sprout_report(self, markdown: str) -> str:
        """「发芽」：把笔记里的种子想法撞到别的领域，长出一张跨界新知卡片。

        规则照搬「得到大脑会员式思考技能包」的发芽模式（ClawHub 的
        ``dedao-thinking-skills/sprout.md``）：先提炼**一条**种子想法，再去别的
        领域找**真有结构相似**的对应物，最后落成「种子 / 跨界碰撞 2–4 条 / 启发」
        三块的新知卡片。凑不出跨界就老实说，绝不硬扯、绝不编造。

        与原版唯一的差异是输入：原版吃"一条想法"，这里吃"一整篇笔记"，所以多了
        一步从笔记里提炼种子。``effort=off`` 关掉思考链——这一层要的是联想而不是
        推理链，实测关思考更快更稳（思考链会把输出额度吃光，正文反而为空）。
        """
        prompt = (
            "你是「得到大脑」思考技能包里的【发芽】模式。你的唯一任务：把这条笔记"
            "的种子想法，撞到别的领域去，长出新东西。\n\n"
            f"【素材：一篇视频笔记】\n{markdown.strip()}\n\n"
            "按三步走：\n\n"
            "**第一步 · 提炼种子**：从上面这篇笔记里提炼出**一条**最值得发芽的种子"
            "想法（一句话）——它应该是这条笔记真正在讲的那个判断、机制或主张。"
            "只提炼一条，不要罗列多个。\n\n"
            "**第二步 · 跨界碰撞**：围绕这颗种子，去**别的领域**找类比、碰撞、启发。"
            "用这三个问题逼自己：\n"
            "- 这条想法的核心机制 / 结构，在另一个领域有没有对应物？\n"
            "- 如果把这个领域的规律搬到那个领域，会发生什么？\n"
            "- 有没有看似无关的事物，其实和这条想法同构？\n\n"
            "**硬标准**：跨界必须**真有结构上的相似**，不是硬扯谐音梗或表面联想。\n"
            "- 好的发芽：「咦，这两个东西居然是一回事」\n"
            "- 坏的发芽：「这让我想到了……然后就没了」\n"
            "每条都要说清「**为什么像**」。\n\n"
            "**第三步 · 输出新知卡片**，严格用下面这个格式，不要加别的小节：\n\n"
            "# 🌱 新知卡片：[种子想法的一句话概括]\n\n"
            "**种子：** [提炼出的那条种子想法]\n\n"
            "**跨界碰撞：**\n"
            "- [领域A] 里有个东西叫 XX，它和你的想法同构，因为……\n"
            "- [领域B] 的 YY 现象，其实是同一种机制……\n"
            "- （2–4 条，每条都要说清“为什么像”）\n\n"
            "**启发：** 这条想法可以迁移到…… / 反过来想，它可能也意味着……\n"
            "（一句让人停下来想的话）\n\n"
            "---\n\n"
            "**纪律（违反即失败）：**\n"
            "1. 跨界 **2–4 条**，不凑数、不注水；素材少就写少。\n"
            "2. **绝不编造**：不许说「我查了某某资料」这类话——你没查就别说。"
            "卡片里出现的每个出处都必须是真实存在的、公认的知识。\n"
            "3. **凑不出就老实说**：如果这条笔记比较封闭，找不到好的跨界类比，"
            "就只输出「这条笔记比较封闭，暂时找不到好的跨界类比」并说明原因，不要硬扯。\n"
            "4. **卡片要能独立成立**：别人光看这张卡片、不看原笔记也能懂。\n"
            "5. 去 AI 味：不要用「很棒的洞察 / 让我们 / 总之 / 值得注意的是 / "
            "在当今时代」这类套话。\n"
            "6. **只输出卡片本身**（从 `# 🌱 新知卡片：` 这一行开始），"
            "不要前言、不要确认语、不要解释你做了什么，"
            "也不要在卡片结尾加「未联网核验」「仅供参考」「具体以权威资料为准」"
            "这类免责说明或元评论——那不属于卡片内容。"
        )
        return await self._complete(
            prompt,
            max_tokens=3000,
            effort="off",
            budget="requested",
            stage="发芽",
        )

    async def interrogate_report(self, markdown: str) -> str:
        """「拷问」：把视频里的主张当对手，挑漏洞并强制给替代方案。

        规则照搬技能包的 ``interrogate.md``，只做一处适配：那里拷问的是**用户自己的
        方案**，这里拷问的是**视频作者的主张**，属于「他人内容」——技能包明确规定
        此时全程不能说「你说的/你的观点」，要改成「这个视频主张」。

        额外加了一条原版没有的硬要求：**每条判断都要附原片时间点**。笔记里有时间轴，
        系统会把 ``[MM:SS]`` 转成可点击跳回原片的链接——批评必须可验证。
        """
        prompt = (
            "你是「得到大脑」思考技能包里的【拷问】模式。你是用户的**直友**——"
            "站在他这边，但绝不哄他。\n\n"
            f"【素材：一篇视频笔记】\n{markdown.strip()}\n\n"
            "⚠️ 这是**他人内容**（视频作者的观点），不是用户自己的方案。全程**禁止**说"
            "「你说的」「你的观点」，要说「这个视频主张」「作者认为」。\n\n"
            "按下面六步走。**哪一步发现了真问题就重点说，没问题的步骤一句话带过**"
            "（例如「Step 3：前提成立，问题是真的」）。\n\n"
            "**Step 1 · 理解 + 矛盾检测**：先用一两句复述作者的核心主张（证明你读懂了），"
            "然后扫这五类矛盾：\n"
            "- 目的矛盾：嘴上说要 A，论据却在损害 A\n"
            "- 受众矛盾：说给谁听的和实际服务的不是一拨人\n"
            "- 角色矛盾：想同时扮演互相冲突的角色\n"
            "- 时间线矛盾：结论和它自己的论证节奏对不上\n"
            "- **利益矛盾：作者为什么这么说？他在卖什么（课 / 货 / 流量 / 立场）？"
            "信息源是几手的？**\n\n"
            "**Step 2 · 主张逐条审查**：对每个核心主张过这五问，每问给结论"
            "（站得住 / 站不住 / 存疑 + 为什么）：\n"
            "- 证据是数据、案例，还是「研究表明」式的空口断言？\n"
            "- 样本 / 出处是几手的？原始来源是谁？\n"
            "- 有没有反例，或者被省略的对照组？\n"
            "- 结论是相关性还是因果性？有没有把相关说成因果？\n"
            "- 他说的那个对象，能落到一个具体的人 / 场景上吗？\n\n"
            "**Step 3 · 前提挑战**：这是对的问题吗？会不会问错了方向？"
            "如果什么都不做会怎样（可能根本用不上这个结论）？"
            "三看：看对象（到底是谁）、看难点（真正的难点在哪）、"
            "看交付（结论能不能解决那个难点）。\n\n"
            "**Step 4 · 替代方案（强制，唯一不可压缩的一步）**：给 2–3 条不同路径：\n"
            "- 一条**替代解释**：同样的现象，还有哪种更简单的解释？\n"
            "- 一条**最小可行**：如果只能从这个视频拿走一件事用起来，那是什么？\n"
            "- （可选）一条**反向版本**：结论完全反过来时，什么情况下反而成立？\n"
            "每条说清：它解决了原主张的什么漏洞、代价是什么。\n\n"
            "**Step 5 · 认知挑战**：能反悔吗（这个结论错了代价可逆吗）？"
            "什么信号被忽略了？**反过来想：什么会让这个结论失败？**"
            "作者盯的那个指标还有用吗？\n\n"
            "**Step 6 · 时间线扫描**：**只在这个视频确实给了行动建议时才写**，"
            "铺到时间上看落地风险；它若只是观点输出，这一步一句话跳过。\n\n"
            "---\n\n"
            "**关于「原片定位」（本篇特有，很重要）：**\n"
            "笔记里每个小节都带 `[起始-结束]` 时间点。你的**每一条具体判断，都要在句末"
            "附上对应时间点**，格式就用笔记里原有的样子（`[02:14]` 或 `[02:14-02:56]`）"
            "——系统会把它转成可点击跳回原片的链接。**批评必须可验证**：让人点一下就能"
            "回去核对作者的原话。\n\n"
            "**纪律：**\n"
            "1. **12 种逻辑谬误全程挂着这把尺子**，发现就点名：稻草人、滑坡、因果倒置、"
            "假二分、诉诸权威、诉诸群众、循环论证、偷换概念、以偏概全、幸存者偏差、"
            "沉没成本、锚定偏差。\n"
            "2. **不编造**：视频里没有的内容绝不脑补，也不要假装查过资料。\n"
            "3. **禁词**：不出现「加油 / 继续努力 / 你很有想法 / 很棒的洞察 / 总之 / "
            "值得注意的是」这类套话——出现就破功。\n"
            "4. 语气：直接、不绕弯、带锋利，但不刻薄。**结尾把决定权交还给用户**"
            "（「以上是漏洞和替代路径，怎么用是你的事」这个意思，但别写成一模一样的句子）。\n"
            "5. 只输出正文，**从 `# ⚔️ 拷问：` 这一行开始**；不要前言、不要确认语、"
            "不要免责声明。"
        )
        return await self._complete(
            prompt,
            max_tokens=4000,
            effort="off",
            budget="requested",
            stage="拷问",
        )

    async def quiz_report(self, markdown: str) -> str:
        """「自测」：把一篇笔记变成能考自己的题（技能包里没有，是视频笔记特有的闭环）。

        笔记只是**输入**，学会需要**回忆**。这个模式把笔记转成 6–8 道题，答案折叠在
        ``<details>`` 里（Obsidian 和网页端都支持）。每题附原片定位，答不出来时
        点一下就跳回作者讲那段的地方。
        """
        prompt = (
            "你是「得到大脑」思考技能包里的【自测】模式：把一份笔记，变成能考自己的题。\n\n"
            f"【素材：一篇视频笔记】\n{markdown.strip()}\n\n"
            "**出题要求：**\n"
            "1. 一共 **6–8 道题**，覆盖这篇笔记的主要知识点，**不要只考开头那几段**。\n"
            "2. 题型混搭：\n"
            "   - **回忆题**：直接问结论 / 定义 / 数字\n"
            "   - **理解题**：问「为什么」「这两者是什么关系」\n"
            "   - **应用题**：给一个新场景，问这套逻辑会怎么用\n"
            "   - 至少一道**易错题**：考笔记里最容易被记反、最容易想当然的那个点\n"
            "3. 每题**必须附原片定位**：用笔记里原有的时间点格式（如 `[02:14]`），"
            "系统会把它转成可点击跳回原片的链接。定位放在**答案里**，别放在题干里"
            "（题干带时间点等于泄题）。\n"
            "4. 答案要**具体**，不许写「见原文」「参考笔记」这类敷衍话；"
            "答错的人要能看懂自己错在哪。\n"
            "5. **不许出笔记里没有的内容**，不许编造作者没说过的话。\n\n"
            "**输出格式（严格遵守，不要改动结构）：**\n\n"
            "# 🎯 自测：<一句话说明这套题考什么>\n\n"
            "> 先自己答一遍，再点开答案。答不出来的地方，就是该回原片的地方。\n\n"
            "**1. <题干>**\n\n"
            "<details><summary>答案</summary>\n\n"
            "<答案正文> `[MM:SS]`\n\n"
            "</details>\n\n"
            "**2. <题干>**\n\n"
            "<details><summary>答案</summary>\n\n"
            "<答案正文> `[MM:SS]`\n\n"
            "</details>\n\n"
            "（依次到 6–8 题）\n\n"
            "---\n\n"
            "**一句话复盘**：这套题里最容易错的是第 N 题，因为……（**只写这一句**）\n\n"
            "**纪律：**\n"
            "- 只用 `**N. 题干**` 这种加粗编号，**不要用 `##` 标题层级**"
            "（会让侧边大纲被题目塞满）。\n"
            "- 只输出题目与答案，不要前言、不要解释你做了什么、不要免责声明。"
        )
        return await self._complete(
            prompt,
            max_tokens=4000,
            effort="off",
            budget="requested",
            stage="自测",
        )

    async def sprout_aggregate(self, material: str, note_count: int) -> str:
        """「整体发芽」：跨多篇笔记找**共同种子**，再按新知卡片格式跨界。

        这是得到大脑官方「整体发芽」的语义——原话是「综合多条记录发现深层关联」。
        与单篇发芽的唯一区别是多了一步：不是逐篇各发各的芽，而是先找出这几篇
        反复出现的那一颗种子，再拿它去别的领域撞。

        素材太少或找不到共同种子时按技能包纪律**老实说**，不硬凑——此时不输出
        卡片标题，调用方凭这一点判断「本次不落盘」。
        """
        prompt = (
            "你是「得到大脑」思考技能包里的【发芽】模式，这次做的是**整体发芽**："
            f"跨越多篇笔记，找它们共同的那颗种子。\n\n"
            f"【素材：{note_count} 篇视频笔记】\n{material}\n\n"
            "按三步走：\n\n"
            "**第一步 · 跨篇找种子**：这几篇笔记**反复出现的那一颗种子**是什么？"
            "不是罗列每篇讲了什么，而是找出它们共享的那个内核——那个在不同话题下"
            "一再冒头的判断、机制或主张。如果几篇之间只有题材相似、没有真正共同的内核，"
            "老实说没有（见下面的纪律）。\n\n"
            "**第二步 · 跨界碰撞**：拿这颗**共同**种子去别的领域撞。用这三个问题逼自己：\n"
            "- 这颗种子的核心机制 / 结构，在另一个领域有没有对应物？\n"
            "- 如果把这个领域的规律搬到那个领域，会发生什么？\n"
            "- 有没有看似无关的事物，其实和这颗种子同构？\n\n"
            "**硬标准**：跨界必须**真有结构上的相似**，不是硬扯谐音梗或表面联想。\n"
            "- 好的发芽：「咦，这两个东西居然是一回事」\n"
            "- 坏的发芽：「这让我想到了……然后就没了」\n"
            "每条都要说清「**为什么像**」。\n\n"
            "**第三步 · 输出新知卡片**，严格用下面这个格式，不要加别的小节：\n\n"
            "# 🌱 新知卡片：[共同种子的一句话概括]\n\n"
            "**种子：** [这颗共同种子，并指出它是从哪几篇里长出来的]\n\n"
            "**跨界碰撞：**\n"
            "- [领域A] 里有个东西叫 XX，它和这颗种子同构，因为……\n"
            "- [领域B] 的 YY 现象，其实是同一种机制……\n"
            "- （2–4 条，每条都要说清“为什么像”）\n\n"
            "**启发：** 这颗种子能迁移到…… / 反过来想，它可能也意味着……\n"
            "（一句让人停下来想的话）\n\n"
            "---\n\n"
            "**纪律（违反即失败）：**\n"
            "1. 跨界 **2–4 条**，不凑数、不注水。\n"
            "2. **绝不编造**：不许说「我查了某某资料」这类话——你没查就别说。"
            "卡片里出现的每个出处都必须是真实存在的、公认的知识。\n"
            "3. **没有共同种子时，不要输出卡片**：只输出一句话，说明这几篇为什么"
            "没有共同内核（题材相近但内核不同 / 素材太少 / 只有一篇等），"
            "**不要写 `# 🌱 新知卡片：` 这个标题**，也不要硬凑一个共同点。\n"
            "4. **卡片要能独立成立**：别人光看这张卡片、不看那几篇原笔记也能懂。\n"
            "5. 去 AI 味：不要用「很棒的洞察 / 让我们 / 总之 / 值得注意的是 / "
            "在当今时代」这类套话。\n"
            "6. **只输出内容本身**，不要前言、不要确认语、不要解释你做了什么，"
            "也不要加「未联网核验」「仅供参考」这类免责说明。"
        )
        return await self._complete(
            prompt,
            max_tokens=3000,
            effort="off",
            budget="requested",
            stage="整体发芽",
        )

    async def generate_summary(
        self,
        title: str,
        segments: Sequence[TranscriptSegment],
        metadata: dict[str, Any] | None = None,
        style: str = "detailed",
        reasoning_effort: str = "auto",
        progress_callback: ProgressCallback | None = None,
        should_abort: Callable[[], bool] | None = None,
        content_type: str = "general",
    ) -> str:
        if not segments:
            raise ValueError("Cannot summarize an empty transcript")
        if style not in SUMMARY_STYLES:
            raise ValueError(f"Unsupported summary style: {style}")
        if content_type not in CONTENT_TYPES:
            raise ValueError(f"Unsupported content type: {content_type}")
        if reasoning_effort not in {"auto", "off", "high", "max"}:
            raise ValueError(f"Unsupported reasoning effort: {reasoning_effort}")
        if reasoning_effort == "max":
            # 界面上已经没有「最大」这一档（对照见 STYLE_DEFAULT_EFFORT），但存过 max 的旧档案、
            # MCP 与 Skill 仍会把它传进来——按 high 处理。报错等于让老用户的任务直接失败。
            reasoning_effort = "high"
        if should_abort is not None and should_abort():
            raise asyncio.CancelledError("任务已取消")

        await self._report_progress(progress_callback, 2, "正在分析转录内容")
        notes_effort = self._stage_effort(reasoning_effort, style, "notes")
        if style not in CONCISE_LIKE_STYLES and self._is_long_transcript(segments):
            draft = await self._write_long_transcript_notes(
                title,
                segments,
                metadata or {},
                style,
                reasoning_effort,
                progress_callback,
                should_abort,
                content_type,
            )
        else:
            draft = await self._write_condensed_notes(
                title,
                segments,
                metadata or {},
                style,
                notes_effort,
                reasoning_effort,
                progress_callback,
                should_abort,
                content_type,
            )
        await self._report_progress(progress_callback, 91, "完整笔记初稿已生成")
        if style == "detailed":
            analysis_progress = 93
            analysis_stage = "正在补充点评与分析"
            await self._report_progress(progress_callback, analysis_progress, analysis_stage)
            analysis = ""
            reason = ""
            # 点评是初稿之后的可选增强：免费网关的公共并发随时段波动（群测期实测 429），
            # 限流退避一次再试，仍失败就保留初稿正常完成，不能把 96% 的任务判死
            for attempt in range(2):
                if attempt:
                    await self._report_progress(
                        progress_callback,
                        0,
                        f"{analysis_stage} 遇到限流，等待 {SECTION_RATE_LIMIT_BACKOFF_SECONDS:.0f} 秒后重试",
                    )
                    # 不做 try：用户取消必须从这里直接抛出去（asyncio.sleep 可被取消）
                    await asyncio.sleep(SECTION_RATE_LIMIT_BACKOFF_SECONDS)
                try:
                    analysis = await self._complete(
                        self._analysis_prompt(title, draft),
                        max_tokens=3_200,
                        effort=self._stage_effort(reasoning_effort, style, "analysis"),
                        progress_callback=progress_callback,
                        progress=analysis_progress,
                        stage=analysis_stage,
                        should_abort=should_abort,
                    )
                    break
                except Exception as exc:
                    # CancelledError 不是 Exception 子类：用户取消必须原样往外抛
                    reason = f"{type(exc).__name__}: {exc}"
                    if not _looks_rate_limited(exc):
                        break
            if analysis.strip():
                draft = f"{draft.rstrip()}\n\n{analysis.lstrip()}"
            else:
                self.warnings.append(f"点评与分析未生成（{reason}），笔记正文不受影响")
        await self._report_progress(progress_callback, 99, "正在保存笔记")
        cleaned = self._normalize_math_delimiters(self._strip_code_fence(draft))
        return self._normalize_timestamp_delimiters(cleaned)

    async def _write_condensed_notes(
        self,
        title: str,
        segments: Sequence[TranscriptSegment],
        metadata: dict[str, Any],
        style: str,
        notes_effort: str,
        reasoning_effort: str,
        progress_callback: ProgressCallback | None,
        should_abort: Callable[[], bool] | None,
        content_type: str = "general",
    ) -> str:
        """原有管线：逐段压缩成片段笔记 → 分层归并 → 一次成稿。

        继续服务 concise，以及净字数不超过 ``LONG_TRANSCRIPT_CHARACTERS`` 的短片——这个规模
        下逐段压缩后的总量本来就小于 ``MERGE_INPUT_CHARACTERS``，归并根本不会触发，实测质量
        与排版都良好（37 分钟用户重测通过）。**分流阈值**仍守着"不拿已验证的行为去换未验证
        的路径"；改的是切块之前先合并碎行——不合并时 32 字/段的时间戳开销会把中短视频也切成
        好几块，每块都押一次有损压缩调用。
        """
        # 先把碎 ASR 行拼成长行再切块。不合并的话每段都要摊上 32 字时间戳开销，
        # whisper 碎到平均 7.5 字一段时开销吃掉 81% 的预算——7 分钟的视频也会被切成
        # 好几块，每块多一次有损压缩调用，而每一次都是"通道在思考就把整条任务押上"的
        # 赌注（ollama gemma 用户实测：40 分钟切成 4 块，一次压缩被思考吃满即整条判死）。
        # 合并只作用于喂给模型的提示词，落盘的 transcript.md 仍是逐句粒度。
        chunks = chunk_segments(
            merge_segment_lines(segments, SECTION_LINE_MERGE_CHARACTERS),
            max_characters=SUMMARY_CHUNK_CHARACTERS,
        )
        if len(chunks) == 1:
            source = segments_to_prompt(chunks[0])
        else:
            condensed_chunks: list[str] = []
            for index, chunk in enumerate(chunks, start=1):
                if should_abort is not None and should_abort():
                    raise asyncio.CancelledError("任务已取消")
                chunk_progress = 5 + int(52 * (index - 1) / len(chunks))
                chunk_stage = f"正在整理第 {index}/{len(chunks)} 个转录片段"
                await self._report_progress(
                    progress_callback,
                    chunk_progress,
                    chunk_stage,
                )
                try:
                    condensed_chunks.append(
                        await self._complete(
                            self._chunk_prompt(title, index, len(chunks), chunk),
                            max_tokens=self._note_budget_max_tokens(
                                sum(len(segment.text) for segment in chunk)
                            ),
                            effort=notes_effort,
                            progress_callback=progress_callback,
                            progress=chunk_progress,
                            stage=chunk_stage,
                            should_abort=should_abort,
                        )
                    )
                except Exception as exc:
                    # 一次压缩失败不该判死整条任务：这一块的原文直接当作片段材料往下走，
                    # 内容一条不丢，只是成稿读到的是转录而不是压缩稿。
                    # CancelledError 不是 Exception 子类，用户取消照样原样往外抛。
                    self.warnings.append(
                        f"第 {index}/{len(chunks)} 个转录片段未能整理"
                        f"（{type(exc).__name__}: {exc}），已把该片段原文交给后续成稿"
                    )
                    condensed_chunks.append(segments_to_prompt(chunk))
            await self._report_progress(
                progress_callback, 60, f"已整理 {len(chunks)} 个转录片段"
            )
            condensed_chunks = await self._reduce_chunks(
                title,
                condensed_chunks,
                reasoning_effort,
                style,
                progress_callback,
                should_abort,
            )
            source = "\n\n".join(
                f"### 片段 {index}\n{note}"
                for index, note in enumerate(condensed_chunks, start=1)
            )

        max_tokens = {
            "detailed": 4_600,
            "faithful": 4_600,
            "concise": 2_400,
            "keypoints": 2_800,
        }[style]
        coverage_end = format_timestamp(segments[-1].end)
        draft_progress = 78
        draft_stage = "正在生成完整笔记"
        await self._report_progress(progress_callback, draft_progress, draft_stage)
        draft = await self._complete(
            self._note_prompt(title, source, metadata, style, coverage_end, content_type),
            max_tokens=max_tokens,
            effort=notes_effort,
            progress_callback=progress_callback,
            progress=draft_progress,
            stage=draft_stage,
            should_abort=should_abort,
        )
        return await self._ensure_tail_coverage(
            title,
            draft,
            segments,
            style,
            notes_effort,
            progress_callback,
            should_abort,
        )

    @staticmethod
    def _is_long_transcript(segments: Sequence[TranscriptSegment]) -> bool:
        """转录净字数是否超过长视频阈值（时间戳开销不计，见 ``_write_long_transcript_notes``）。"""
        return sum(len(segment.text) for segment in segments) > LONG_TRANSCRIPT_CHARACTERS

    @staticmethod
    def _mechanical_effort(reasoning_effort: str) -> str:
        """续写与概览这类机械阶段的档位。

        ``auto`` 时返回 ``auto``（= 不注入任何思考参数，交给通道默认）而不是 ``off``：
        ``off`` 在非官方 deepseek 兼容网关上会被翻译成 ``reasoning_effort=none``，而
        ModelScope 这类网关只认 ``low/medium/high/xhigh/max``，直接回 400 且那条报错的
        形状（"'reasoning_effort' must be one of ..."）不在参数降级匹配里。
        少关一次思考只是慢一点，任务失败是把内容整段丢掉。
        用户显式选了 off/high/max 时一律照原样尊重。
        """
        return "auto" if reasoning_effort == "auto" else reasoning_effort

    async def _write_long_transcript_notes(
        self,
        title: str,
        segments: Sequence[TranscriptSegment],
        metadata: dict[str, Any] | None,
        style: str,
        reasoning_effort: str,
        progress_callback: ProgressCallback | None,
        should_abort: Callable[[], bool] | None,
        content_type: str = "general",
    ) -> str:
        """长转录管线：切块 → 并发把每块直接写成终稿 → 机械拼接 → 目录与概览。

        与 ``_write_condensed_notes`` 的唯一区别是**正文层没有有损归并**：每块转录都由一段
        笔记直接负责，成稿可见的材料量随视频长度线性增长，而不是被压成与时长无关的
        ``MERGE_INPUT_CHARACTERS`` 常数。之所以必须拆成多次写作请求：3 万字量级的笔记单次
        输出装不下（思考链还要先吃掉几千 token），拆完再压缩反而是丢内容的地方。

        ``_ensure_tail_coverage`` 在这一路不需要：末段的尾部推进已经在段级检查里做过。
        """
        merged = merge_segment_lines(segments, SECTION_LINE_MERGE_CHARACTERS)
        sections = chunk_segments(
            merged,
            max_characters=NOTE_SECTION_CHARACTERS,
            include_timestamp_overhead=False,
        )
        total = len(sections)
        effort = self._stage_effort(reasoning_effort, style, "section")
        mechanical_effort = self._mechanical_effort(reasoning_effort)
        brief = self._context_brief(metadata or {})
        # 并发完成顺序不定，对外回报只允许单调不减；心跳沿用同一个峰值。
        peak = [0]

        async def report(value: int, message: str) -> None:
            peak[0] = max(peak[0], value)
            await self._report_progress(progress_callback, peak[0], message)

        await report(
            4,
            f"转录 {sum(len(segment.text) for segment in segments):,} 字，"
            f"按每段约 {NOTE_SECTION_CHARACTERS:,} 字分成 {total} 段逐段完整记录",
        )

        results: list[str] = [""] * total
        done = 0
        semaphore = asyncio.Semaphore(SECTION_WRITE_CONCURRENCY)

        async def worker(index: int) -> None:
            nonlocal done
            async with semaphore:
                results[index] = await self._write_one_section(
                    title,
                    index + 1,
                    total,
                    sections[index],
                    self._context_tail(sections, index),
                    effort,
                    mechanical_effort,
                    report,
                    should_abort,
                    brief,
                    content_type,
                )
            done += 1
            await report(
                5 + int(60 * done / total), f"已完成 {done}/{total} 段笔记"
            )

        jobs = [asyncio.create_task(worker(index)) for index in range(total)]
        try:
            await asyncio.gather(*jobs)
        finally:
            # 某一段抛出（含用户取消）时把兄弟段一起收掉，不留后台请求。
            for job in jobs:
                if not job.done():
                    job.cancel()

        headings = [self._section_heading(piece) for piece in results]
        overview = await self._write_overview(
            title,
            self._toc_lines(sections, headings),
            mechanical_effort,
            report,
            should_abort,
        )
        await report(74, f"{total} 段笔记已按时间顺序拼接")
        return self._assemble_sections(title, sections, results, headings, overview)

    async def _write_one_section(
        self,
        title: str,
        index: int,
        total: int,
        section: Sequence[TranscriptSegment],
        context_tail: str,
        effort: str,
        mechanical_effort: str,
        report: Callable[[int, str], Awaitable[None]],
        should_abort: Callable[[], bool] | None,
        brief: str = "",
        content_type: str = "general",
    ) -> str:
        """把一段转录直接写成终稿笔记，并保证它推进到自己那段的时间末尾。

        唯一的质检判据是段级尾部推进（本段最大时间戳离区间结尾还差 60 秒以上就算没写完），
        它同时覆盖「被输出额度截断」和「模型自己提前收尾」两种失败，所以不需要读
        ``finish_reason``。整篇笔记做不到「结尾必须带时间戳」这个要求，单段 6,500 字可以——
        这也是拆段顺带换来的可检验性。

        两次尝试都不合格时仍然不丢内容：已写出的部分保留，没覆盖到的原文附在后面。
        """
        stage = f"直写第 {index}/{total} 段"
        max_tokens = self._section_max_tokens(section)
        best = ""
        reason = ""
        rate_limited = False
        for attempt in range(SECTION_ATTEMPTS):
            if rate_limited:
                # 走 _report_progress 而不是直接 await report：回调可能是同步函数
                await self._report_progress(
                    report,
                    0,
                    f"{stage} 遇到限流，等待 {SECTION_RATE_LIMIT_BACKOFF_SECONDS:.0f} 秒后重试",
                )
                # 不做 try：用户取消必须从这里直接抛出去（asyncio.sleep 可被取消）
                await asyncio.sleep(SECTION_RATE_LIMIT_BACKOFF_SECONDS)
            try:
                text = await self._complete(
                    self._section_note_prompt(
                        title,
                        index,
                        total,
                        section,
                        context_tail,
                        brief=brief,
                        retried=attempt > 0,
                        content_type=content_type,
                    ),
                    max_tokens=max_tokens,
                    effort=effort,
                    progress_callback=report,
                    stage=stage,
                    should_abort=should_abort,
                )
                try:
                    text = await self._continue_section(
                        title,
                        index,
                        total,
                        text,
                        section,
                        mechanical_effort,
                        report,
                        stage,
                        should_abort,
                    )
                except Exception as exc:
                    # 续写失败不推翻首写：这一段已经写出来的部分照样要留在笔记里
                    reason = f"补写结尾失败（{type(exc).__name__}: {exc}）"
                    rate_limited = _looks_rate_limited(exc)
            except Exception as exc:
                # CancelledError 不是 Exception 子类：用户取消必须原样往外抛
                reason = f"模型调用失败（{type(exc).__name__}: {exc}）"
                rate_limited = _looks_rate_limited(exc)
                continue
            if self._coverage_gap(section, text) <= TAIL_GAP_SECONDS:
                return text
            reason = reason or "多次整理后仍未写到本段结尾"
            if len(text) > len(best):
                best = text
        uncovered = self._uncovered(section, best)
        self.warnings.append(
            f"第 {index}/{total} 段（{format_timestamp(section[0].start)} 起）{reason}，"
            "已把未覆盖的原文附在笔记里"
        )
        if not uncovered:
            return best
        block = self._raw_block(uncovered, reason)
        return f"{best.rstrip()}\n\n{block}" if best else block

    async def _continue_section(
        self,
        title: str,
        index: int,
        total: int,
        text: str,
        section: Sequence[TranscriptSegment],
        effort: str,
        report: Callable[[int, str], Awaitable[None]],
        stage: str,
        should_abort: Callable[[], bool] | None,
    ) -> str:
        """本段没写到自己区间末尾时接着写，最多 ``SECTION_CONTINUATION_MAX`` 次。

        续写不设 ``TAIL_PATCH_MAX_INPUT_CHARACTERS`` 那样的规模上限，也不需要整体耗时上限：
        剩余材料天然被一段的 ``NOTE_SECTION_CHARACTERS`` 框住，且每次请求都有心跳回报。
        """
        previous_covered: float | None = None
        for _ in range(SECTION_CONTINUATION_MAX):
            if self._coverage_gap(section, text) <= TAIL_GAP_SECONDS:
                return text
            covered = self._max_note_timestamp(text)
            if covered is None or (
                previous_covered is not None and covered <= previous_covered
            ):
                # 没有可用的时间锚点，或续写并没有往前推进：交给外层重试与兜底
                return text
            previous_covered = covered
            remaining = [segment for segment in section if segment.end > covered + 0.5]
            if not remaining:
                return text
            patch = await self._complete(
                self._section_continuation_prompt(
                    title,
                    index,
                    total,
                    text,
                    remaining,
                    format_timestamp(covered),
                    format_timestamp(section[-1].end),
                ),
                max_tokens=self._section_max_tokens(remaining),
                effort=effort,
                progress_callback=report,
                stage=f"{stage}续写",
                should_abort=should_abort,
            )
            if not patch.strip():
                return text
            text = f"{text.rstrip()}\n\n{patch.strip()}"
        return text

    @classmethod
    def _coverage_gap(
        cls, section: Sequence[TranscriptSegment], text: str
    ) -> float:
        """本段区间结尾与笔记里最大时间戳的差距；没有任何时间戳视为无穷大缺口。"""
        covered = cls._max_note_timestamp(text)
        if covered is None:
            return float("inf")
        return section[-1].end - covered

    @classmethod
    def _uncovered(
        cls, section: Sequence[TranscriptSegment], text: str
    ) -> Sequence[TranscriptSegment]:
        covered = cls._max_note_timestamp(text)
        if covered is None:
            return section
        remaining = [segment for segment in section if segment.end > covered + 0.5]
        return remaining

    @staticmethod
    def _section_max_tokens(segments: Sequence[TranscriptSegment]) -> int:
        """每段输出额度随该段转录净字数缩放，不用与时长无关的常数。"""
        characters = sum(len(segment.text) for segment in segments)
        estimate = int(
            characters / SECTION_TOKENS_PER_CHARACTER * SECTION_TOKEN_KEEP_RATIO
        )
        return max(SECTION_MAX_TOKENS_MIN, min(SECTION_MAX_TOKENS_MAX, estimate))

    @staticmethod
    def _note_budget_max_tokens(characters: int) -> int:
        """「整理」类请求（分片压缩、分层归并）的输出额度：正文需要 + 思考链预留。

        入参是材料的净字数。字/token 沿用 ``SECTION_TOKENS_PER_CHARACTER`` 那一套口径；
        保留率比逐段直写低（0.45 vs 0.8），因为这一层的产物只是给成稿用的中间材料。
        """
        content = int(characters / SECTION_TOKENS_PER_CHARACTER * NOTE_BUDGET_KEEP_RATIO)
        estimate = content + NOTE_BUDGET_THINKING_RESERVE_TOKENS
        return max(NOTE_BUDGET_MAX_TOKENS_MIN, min(NOTE_BUDGET_MAX_TOKENS_MAX, estimate))

    @staticmethod
    def _context_tail(sections: Sequence[Sequence[TranscriptSegment]], index: int) -> str:
        """上一段结尾的原话，只用来让模型看懂「接着上面那道题」这类指代。

        并发直写时拿不到上一段的成稿（可能还没写完），所以取的是原文而不是笔记——
        它在切块时就已知，且不会诱导模型去复述上一段。
        """
        if index == 0:
            return ""
        return sections[index - 1][-1].text[-SECTION_CONTEXT_CHARACTERS:]

    @staticmethod
    def _section_heading(text: str) -> str:
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("## "):
                # 目录行前已经印过这一段的起止时间，标题里再带一份就是重复
                return _HEADING_TIMESTAMP_RE.sub("", stripped[3:].strip())
        return ""

    @staticmethod
    def _toc_lines(
        sections: Sequence[Sequence[TranscriptSegment]], headings: Sequence[str]
    ) -> str:
        """目录完全由代码生成：每段的起止时间是已知的，不给模型编时间戳的机会。"""
        lines = []
        for section, heading in zip(sections, headings):
            span = (
                f"{format_timestamp(section[0].start)}-"
                f"{format_timestamp(section[-1].end)}"
            )
            lines.append(f"- `{span}` {heading or '（该段未给出小节标题）'}")
        return "\n".join(lines)

    async def _write_overview(
        self,
        title: str,
        listing: str,
        effort: str,
        report: Callable[[int, str], Awaitable[None]],
        should_abort: Callable[[], bool] | None,
    ) -> str:
        """一次小请求写概览。摘要是唯一允许有损的层，失败只跳过，不影响正文。"""
        stage = "生成内容概览"
        await report(68, "正在写内容概览")
        try:
            return await self._complete(
                self._overview_prompt(title, listing),
                max_tokens=OVERVIEW_MAX_TOKENS,
                effort=effort,
                progress_callback=report,
                stage=stage,
                should_abort=should_abort,
            )
        except Exception as exc:
            self.warnings.append(
                f"内容概览生成失败，已跳过（{type(exc).__name__}: {exc}）"
            )
            return ""

    @classmethod
    def _assemble_sections(
        cls,
        title: str,
        sections: Sequence[Sequence[TranscriptSegment]],
        pieces: Sequence[str],
        headings: Sequence[str],
        overview: str,
    ) -> str:
        parts = [f"# 视频笔记：《{title}》"]
        overview = overview.strip()
        if overview:
            parts.append(overview)
        parts.append("## 本片目录\n\n" + cls._toc_lines(sections, headings))
        for piece in pieces:
            cleaned = cls._strip_document_heading(piece)
            if cleaned:
                parts.append(cleaned)
        return "\n\n".join(parts).strip() + "\n"

    @staticmethod
    def _strip_document_heading(text: str) -> str:
        """去掉模型自作主张补的文档级标题（整篇只应有一个 # 标题，由代码写在最前）。"""
        lines = text.strip().splitlines()
        if lines and lines[0].startswith("# "):
            lines.pop(0)
        return "\n".join(lines).strip()

    @staticmethod
    def _raw_block(uncovered: Sequence[TranscriptSegment], reason: str) -> str:
        return (
            f"{RAW_FALLBACK_HEADING}（{format_timestamp(uncovered[0].start)} – "
            f"{format_timestamp(uncovered[-1].end)}）\n\n"
            f"{reason}，为保证内容不丢失，这里直接附上原始转录：\n\n"
            f"{segments_to_prompt(uncovered)}"
        )

    @staticmethod
    def _section_note_prompt(
        title: str,
        index: int,
        total: int,
        section: Sequence[TranscriptSegment],
        context_tail: str,
        brief: str = "",
        retried: bool = False,
        content_type: str = "general",
    ) -> str:
        start = format_timestamp(section[0].start)
        end = format_timestamp(section[-1].end)
        # 长视频是逐段直写，不塞固定小节结构（会打断按时间顺序的连贯记录），
        # 只把场景的取材侧重带进去。
        scenario = LLMSummarizer._SCENARIO_EMPHASIS.get(content_type, "")
        scenario_line = f"\n- {scenario}" if scenario else ""
        context = (
            f"\n上一段结尾的原话（仅供理解指代，绝对不要写进笔记）：{context_tail}\n"
            if context_tail
            else ""
        )
        retry_hint = (
            "\n- 上一次这一段写到一半就停了，这次必须推进到 " + end
            if retried
            else ""
        )
        brief_block = f"\n{brief}\n" if brief else ""
        return f"""视频标题：{title}
这是全片第 {index}/{total} 段，覆盖时间轴 {start} – {end}。直接把这一段写成最终成稿的笔记正文：不要写提纲、不要预告后文、不要总结全片。
{brief_block}
要求：{scenario_line}
- 完整记录本段的每一个话题、例子、推导过程和数字，本段材料一直推进到 {end}
- 把讲同一件事的连续几句合并成通顺段落，不要一行对应一句口播；长度约为本段材料的五到七成。压缩的是说法，不是话题——每个话题、例子和数字都必须留下
- 每个 ## 或 ### 小标题都写成「## 标题 [起点-终点]」，时间戳放在标题文字之后，区间取该话题在材料里的首尾时间点
- 本段最后一行以接近 {end} 的时间戳开头，用于核对有没有写到本段结尾{retry_hint}
- 时间点只能取自材料，写成 [MM:SS] 或 [起点-终点]，超过一小时请带上小时
- 用 ## 组织小节、### 组织子话题；不要输出 # 开头的文档标题，也不要输出视频元信息
- 数学公式一律写成 Markdown 公式：行内用 $...$，独立公式块用 $$...$$，不要用 \\(...\\) 或 \\[...\\] 包裹
- 忠实于讲者本意，允许结合上下文修正明显的口误与转写错误；不补充外部知识，不做评价
{context}
本段转录：
{segments_to_prompt(section)}"""

    @staticmethod
    def _section_continuation_prompt(
        title: str,
        index: int,
        total: int,
        text: str,
        remaining: Sequence[TranscriptSegment],
        covered_ts: str,
        end: str,
    ) -> str:
        return f"""视频《{title}》笔记第 {index}/{total} 段（覆盖到 {end}）已经写到 {covered_ts}，后半部分还没写完。

请接着已有结尾往下写，只输出剩余内容：
- 不要重复已有内容；需要新小节时直接起一个 ## 或 ### 标题，不要重复整篇结构
- 时间戳格式与已有结尾保持一致，一直写到 {end} 为止
- 数学公式与已有笔记保持一致，用 Markdown 公式：行内 $...$、独立公式块 $$...$$
- 忠实转录内容，不补充外部知识，不做评价

已有结尾（仅供衔接参考）：
{text[-NOTE_TAIL_CHARACTERS:]}

还没整理的转录：
{segments_to_prompt(remaining)}"""

    @staticmethod
    def _overview_prompt(title: str, listing: str) -> str:
        return f"""下面是视频《{title}》笔记的各段标题与时间范围。写 2~3 句话的内容概览，说明这份笔记覆盖了什么。
不要评价，不补充外部知识，只输出概览正文，不要重复下面的标题清单。

{listing}"""

    async def _reduce_chunks(
        self,
        title: str,
        notes: list[str],
        reasoning_effort: str,
        style: str,
        progress_callback: ProgressCallback | None,
        should_abort: Callable[[], bool] | None = None,
    ) -> list[str]:
        level = 0
        while sum(len(note) for note in notes) > MERGE_INPUT_CHARACTERS:
            if should_abort is not None and should_abort():
                raise asyncio.CancelledError("任务已取消")
            groups = self._group_notes(notes, MERGE_INPUT_CHARACTERS)
            if len(groups) >= len(notes):
                break
            level += 1
            merged: list[str] = []
            for index, group in enumerate(groups, start=1):
                if should_abort is not None and should_abort():
                    raise asyncio.CancelledError("任务已取消")
                merge_progress = min(75, 61 + level * 4 + int(4 * index / len(groups)))
                merge_stage = f"正在归并第 {level} 层内容 {index}/{len(groups)} 组"
                await self._report_progress(
                    progress_callback,
                    merge_progress,
                    merge_stage,
                )
                try:
                    merged.append(
                        await self._complete(
                            self._merge_prompt(title, level, index, len(groups), group),
                            max_tokens=self._note_budget_max_tokens(
                                sum(len(note) for note in group)
                            ),
                            effort=self._stage_effort(reasoning_effort, style, "notes"),
                            progress_callback=progress_callback,
                            progress=merge_progress,
                            stage=merge_stage,
                            should_abort=should_abort,
                        )
                    )
                except Exception as exc:
                    # 归并失败就停在当前层，把这一层的材料原样交给成稿：成稿读到的材料
                    # 比 ``MERGE_INPUT_CHARACTERS`` 预算大，但内容一条不丢，也不会让
                    # 已经写了大半的任务死在归并上。不保留半截 ``merged``——同一批 notes
                    # 必须来自同一层，混层会让材料密度前后不一致。
                    self.warnings.append(
                        f"第 {level} 层内容归并失败（{type(exc).__name__}: {exc}），"
                        "已停止归并，未归并的片段笔记直接交给成稿"
                    )
                    return notes
            notes = merged
        return notes

    @staticmethod
    def _group_notes(notes: Sequence[str], max_characters: int) -> list[list[str]]:
        groups: list[list[str]] = []
        current: list[str] = []
        size = 0
        for note in notes:
            if current and size + len(note) > max_characters:
                groups.append(current)
                current = []
                size = 0
            current.append(note)
            size += len(note)
        if current:
            groups.append(current)
        return groups

    async def _ensure_tail_coverage(
        self,
        title: str,
        draft: str,
        segments: Sequence[TranscriptSegment],
        style: str,
        effort: str,
        progress_callback: ProgressCallback | None,
        should_abort: Callable[[], bool] | None = None,
    ) -> str:
        """核对笔记是否推进到转写结尾，尽力弥补生成时丢失的尾部内容。

        - concise 摘要允许只挑要点，不做核对；
        - faithful（详细复原）要求完整覆盖：确实丢尾时自动补写一次结尾；
        - detailed 的笔记允许按需省略时间戳，自动补写容易误伤，只记录告警，并把
          覆盖度报告留在 ``self.coverage`` 里——由用户看着缺口和代价自己决定补不补
          （见 ``tail_coverage`` / ``patch_missing_tail`` 与 ``POST .../patch-tail``）。

        补写受三道护栏约束（缺口上限 / 材料字符上限 / 整体耗时上限）：成稿提示词允许
        “不必每段都加时间戳”，长视频里只标前半段时间戳很常见，此时“末尾时间戳落后”
        并不代表丢尾。没有护栏时会自动发一次覆盖剩余全部转录的 max 思考档请求，几十分钟
        不返回进度，表现为“详细复原卡在生成阶段不动，而详细+点评正常”（用户反馈）。
        """
        self.coverage = None
        if style in CONCISE_LIKE_STYLES or not segments:
            return draft
        coverage = self.tail_coverage(draft, segments)
        self.coverage = coverage
        if coverage is None:
            self.warnings.append(
                "笔记中没有可识别的时间戳，无法自动核对是否覆盖到 "
                f"{format_timestamp(segments[-1].end)}"
            )
            return draft
        if coverage.gap_seconds <= TAIL_GAP_SECONDS:
            return draft
        message = (
            "检测到笔记内容停在 "
            f"{format_timestamp(coverage.note_end)}，未覆盖转写结尾 "
            f"{format_timestamp(coverage.coverage_end)}"
        )
        if style == "detailed":
            self.warnings.append(
                message + "；也可能只是笔记省略了时间戳；建议改用“详细复原”风格或更高推理强度重新生成"
            )
            return draft
        if coverage.missing_segments == 0:
            return draft
        # 护栏一：缺口过大时不是“丢尾”，补写等于把剩余转录重写一遍，只告警。
        if not coverage.patch_available:
            self.warnings.append(
                message
                + f"；缺口约 {coverage.gap_seconds / 60:.0f} 分钟，更像笔记省略了时间戳而不是真的丢尾，"
                "已跳过自动补写（强行补写会把剩余转录重写一遍，长时间出不来结果）"
            )
            return draft
        # 护栏二：材料只保留预算内最靠近结尾的一段，保证补写请求规模有上限。
        _, missing_start = self._tail_patch_material_for(coverage, segments)
        scope_note = "" if coverage.covers_full_gap else "（缺口较大，只补写最靠近结尾的一段）"
        self.warnings.append(
            message + f"，正在自动补写 {missing_start} 之后的内容{scope_note}"
        )
        await self._report_progress(progress_callback, 88, "正在补写笔记缺失的结尾内容")
        # 护栏三：补写本身有整体耗时上限，超时保留原笔记，不停在补写阶段不放。
        outcome = await self.patch_missing_tail(
            title,
            draft,
            segments,
            effort=effort,
            progress_callback=progress_callback,
            should_abort=should_abort,
        )
        if outcome.ok:
            return outcome.text or draft
        if outcome.error_kind == "timeout":
            self.warnings.append(
                f"补写结尾超过 {TAIL_PATCH_TIMEOUT_SECONDS:.0f} 秒仍未完成，已跳过并保留原笔记"
                "（模型响应过慢，可降低推理强度后重新生成）"
            )
            await self._report_progress(progress_callback, 91, "补写结尾超时，已保留原笔记")
            return draft
        # 补写只是"把结尾再往前推一点"的可选增强，它失败绝不能推翻已经写完整的笔记。
        # 以前只 catch TimeoutError，一次「模型未返回正文」会把 78% 处已成稿的任务
        # 判死——用户实测：18 分钟视频「详细复原」失败、「精简摘要」通过，差别正是
        # concise 在函数开头就跳过了补写。
        # CancelledError 不是 Exception 子类，用户取消照样原样往外抛。
        self.warnings.append(
            f"补写结尾失败（{outcome.error}），已跳过并保留原笔记"
        )
        await self._report_progress(progress_callback, 91, "补写结尾失败，已保留原笔记")
        return draft

    def resolve_note_effort(self, reasoning_effort: str, style: str) -> str:
        """「整理成稿」阶段的生效推理档位；按需补写沿用同一套规则。"""
        return self._stage_effort(reasoning_effort, style, "notes")

    @staticmethod
    def tail_coverage(
        draft: str,
        segments: Sequence[TranscriptSegment],
        *,
        material_budget: int = TAIL_PATCH_MAX_INPUT_CHARACTERS,
    ) -> TailCoverage | None:
        """核对笔记推进到哪、缺口多大、补写要花多少预算。

        与 ``_ensure_tail_coverage`` 同源，只算事实不看风格：风格决定的是“自动补不补”，
        而用户手动补写时不受风格限制。刻意做成静态方法——不碰实例、不联网，老任务
        可以随时现算一份覆盖度，不必重建模型客户端。笔记里没有可识别时间戳、或没有
        转录时返回 None。

        ``material_budget`` 要跟真正执行补写的那条路一致，否则报告里的“能不能补全”
        会和点下按钮后的结果对不上（手动补写用 ``..._MANUAL``）。
        """
        if not segments:
            return None
        note_end = LLMSummarizer._max_note_timestamp(draft)
        if note_end is None:
            return None
        coverage_end = float(segments[-1].end)
        gap = coverage_end - note_end
        missing = (
            [segment for segment in segments if segment.start >= note_end - 5]
            if gap > TAIL_GAP_SECONDS
            else []
        )
        material = (
            LLMSummarizer._tail_patch_material(missing, material_budget)
            if missing
            else []
        )
        blocked: str | None = None
        if missing and gap > TAIL_PATCH_MAX_GAP_SECONDS:
            blocked = (
                f"缺口约 {gap / 60:.0f} 分钟，补写等于把剩余转录重写一遍，"
                "长时间出不来结果，因此不提供补写"
            )
        return TailCoverage(
            note_end=note_end,
            coverage_end=coverage_end,
            gap_seconds=gap,
            missing_segments=len(missing),
            missing_characters=sum(len(segment.text) for segment in missing),
            material_segments=len(material),
            material_characters=sum(len(segment.text) for segment in material),
            patch_max_tokens=LLMSummarizer._note_budget_max_tokens(
                sum(len(segment.text) for segment in material)
            ),
            patch_available=bool(missing) and blocked is None,
            patch_blocked_reason=blocked,
        )

    @staticmethod
    def _tail_patch_material_for(
        coverage: TailCoverage,
        segments: Sequence[TranscriptSegment],
        budget: int = TAIL_PATCH_MAX_INPUT_CHARACTERS,
    ) -> tuple[Sequence[TranscriptSegment], str]:
        """（送进补写请求的材料，补写起点时间戳）——材料受预算上限约束。"""
        note_end = coverage.note_end or 0.0
        missing = [segment for segment in segments if segment.start >= note_end - 5]
        material = LLMSummarizer._tail_patch_material(missing, budget)
        return material, format_timestamp(material[0].start)

    async def patch_missing_tail(
        self,
        title: str,
        draft: str,
        segments: Sequence[TranscriptSegment],
        *,
        effort: str,
        material_budget: int = TAIL_PATCH_MAX_INPUT_CHARACTERS,
        progress_callback: ProgressCallback | None = None,
        should_abort: Callable[[], bool] | None = None,
    ) -> TailPatchOutcome:
        """补写笔记缺失的结尾（faithful 自动补写与用户手动补写都走这里）。

        只做补写本身：不写盘、不改任务状态；失败原因原样返回，由调用方决定怎么呈现。
        ``material_budget`` 由调用方定：自动补写用保守值，用户手动补写可以放宽。
        """
        coverage = self.tail_coverage(draft, segments, material_budget=material_budget)
        if coverage is None:
            return TailPatchOutcome(
                error_kind="unavailable",
                error="笔记里没有可识别的时间戳，无法定位缺口",
            )
        if not coverage.patch_available:
            return TailPatchOutcome(
                error_kind="unavailable",
                error=coverage.patch_blocked_reason or "当前笔记不需要补写",
            )
        material, missing_start = self._tail_patch_material_for(
            coverage, segments, material_budget
        )
        try:
            patch = await asyncio.wait_for(
                self._complete(
                    self._tail_patch_prompt(
                        title,
                        draft,
                        material,
                        missing_start,
                        format_timestamp(coverage.coverage_end),
                    ),
                    max_tokens=coverage.patch_max_tokens,
                    effort=effort,
                    progress_callback=progress_callback,
                    progress=88,
                    stage="补写结尾",
                    should_abort=should_abort,
                ),
                TAIL_PATCH_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            return TailPatchOutcome(
                missing_start=missing_start,
                error_kind="timeout",
                error=f"超过 {TAIL_PATCH_TIMEOUT_SECONDS:.0f} 秒仍未完成（模型响应过慢，可降低推理强度再试）",
            )
        except Exception as exc:
            return TailPatchOutcome(
                missing_start=missing_start,
                error_kind="error",
                error=f"{type(exc).__name__}: {exc}",
            )
        body = (patch or "").strip()
        if not body:
            return TailPatchOutcome(
                missing_start=missing_start,
                error_kind="empty",
                error="模型没有返回正文（思考链可能吃满了输出额度）",
            )
        return TailPatchOutcome(
            text=self._append_tail_patch(draft, body, missing_start),
            body=body,
            missing_start=missing_start,
        )

    @staticmethod
    def _append_tail_patch(draft: str, body: str, missing_start: str) -> str:
        """把补写片段接在成稿后面；补写已有笔记时要另按页脚位置插入（见 main.py）。"""
        return f"{draft.rstrip()}\n\n### 补遗（{missing_start} 之后）\n\n{body}"

    @staticmethod
    def _tail_patch_material(
        segments: Sequence[TranscriptSegment],
        budget: int = TAIL_PATCH_MAX_INPUT_CHARACTERS,
    ) -> Sequence[TranscriptSegment]:
        """从缺口尾部往前取材料，保证补写请求的输入规模有上限。

        ``budget`` 是给用户手动补写放宽的入口（见 ``TAIL_PATCH_MAX_INPUT_CHARACTERS_MANUAL``）。
        """
        if len(segments) == 1:
            return segments
        kept: list[TranscriptSegment] = []
        used = 0
        for segment in reversed(segments):
            # 与 segments_to_prompt 一致：每行另有时间戳前缀与换行开销。
            line_characters = len(segment.text) + 33
            if kept and used + line_characters > budget:
                break
            kept.append(segment)
            used += line_characters
        kept.reverse()
        return kept if len(kept) < len(segments) else segments

    @staticmethod
    def _parse_clock(value: str) -> float:
        """把 [HH:MM:SS] / [MM:SS] 转成秒；两段且分钟超过 99 时按累计分钟处理。"""
        parts = value.split(":")
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        return int(parts[0]) * 60 + float(parts[1])

    @staticmethod
    def _max_note_timestamp(text: str) -> float | None:
        seconds: list[float] = []
        for match in _NOTE_TIMESTAMP_RE.finditer(text):
            seconds.append(LLMSummarizer._parse_clock(match.group(1)))
            if match.group(2):
                seconds.append(LLMSummarizer._parse_clock(match.group(2)))
        return max(seconds) if seconds else None

    @staticmethod
    def _tail_patch_prompt(
        title: str,
        draft: str,
        missing_segments: Sequence[TranscriptSegment],
        missing_start: str,
        coverage_end: str,
    ) -> str:
        material = segments_to_prompt(missing_segments)
        note_tail = draft[-NOTE_TAIL_CHARACTERS:]
        return f"""视频《{title}》的笔记在 {missing_start} 之前已经写好，但遗漏了视频最后一段内容。

请根据下面的结尾部分转录补写这一部分的笔记：
- 与已有笔记保持一致的结构、语气和时间戳格式（[MM:SS] 或 [起点-终点]）
- 数学公式与已有笔记保持一致，用 Markdown 公式：行内 $...$、独立公式块 $$...$$
- 忠实转录内容，不补充外部知识，不做评价
- 只输出补写的 Markdown 正文，不要重复已有笔记，不要输出标题或解释

已有笔记的结尾（仅供衔接参考）：
{note_tail}

缺失部分的转录（{missing_start} – {coverage_end}）：
{material}"""

    @staticmethod
    async def _report_progress(
        callback: ProgressCallback | None, progress: int, message: str
    ) -> None:
        if callback is None:
            return
        result = callback(progress, message)
        if inspect.isawaitable(result):
            await result

    def _stage_effort(self, selected: str, style: str, stage: str) -> str:
        """解析单个生成阶段实际生效的推理强度。

        - 用户显式选择（off/high/max）永远优先；
        - auto 在 DeepSeek 兼容通道（含 custom 中识别出的 DeepSeek 模型）按
          笔记风格给默认档：detailed/faithful → max，concise → high；
          但 ``section``（长视频的逐段直写）一律关闭思考——那一段材料只有几千字，
          是“照着原文整理成稿”的局部任务，思考链实测只会吃掉输出额度与时间
          （每段 1 万多字思考换 1 千字正文，28 分钟视频总耗时 44 分钟）；
        - 其余通道返回 auto，保持模型默认，不注入私有参数。
        """
        if selected in {"off", "high", "max"}:
            return selected
        if self._uses_deepseek_compatibility():
            if stage == "section":
                return SECTION_EFFORT
            return STYLE_DEFAULT_EFFORT
        return "auto"

    def describe_effort(self, reasoning_effort: str, style: str) -> str:
        """返回用于任务日志的推理设置说明。

        这里报的是**实际生效的档位**而不是用户选了什么：以前写死一句"DeepSeek 通道默认
        关思考"，默认值一改这行日志就成了假话，而它是用户在任务记录里唯一能看到的依据。
        """
        rejected = (
            "；该通道不支持 " + "/".join(sorted(self.rejected_params))
            if self.rejected_params
            else ""
        )
        selected = "high" if reasoning_effort == "max" else reasoning_effort
        if selected != "auto":
            return f"{selected}{rejected}"
        notes = self._stage_effort("auto", style, "notes")
        if notes == "auto":
            return "auto（使用模型默认）"
        section = self._stage_effort("auto", style, "section")
        resolved = notes if notes == section else f"{notes}，长视频逐段直写为 {section}"
        return f"auto（本机按 {resolved} 档{rejected}）"

    def _param_cache_key(self) -> tuple[str, str, str]:
        return (
            str(getattr(self, "model_type", "")),
            str(getattr(self, "model", "")).lower(),
            str(getattr(self, "base_url", "")).lower(),
        )

    @property
    def rejected_params(self) -> set[str]:
        """该配置已确认不被接受的思考参数（同一进程内按配置共用 + 环境变量声明）。"""
        cached = getattr(self, "_rejected_params", None)
        if cached is None:
            cached = set(_REJECTED_REASONING_PARAMS.get(self._param_cache_key(), ()))
            cached |= _rejected_params_from_env()
            self._rejected_params = cached
        return cached

    @property
    def thinking_seen(self) -> bool:
        """这条配置是否已实测到模型在返回思考链（同一进程内按配置共用）。"""
        return self._param_cache_key() in _THINKING_SEEN_CHANNELS

    def _controls_thinking(self) -> bool:
        """本程序是否知道怎么在这条通道上开关思考。

        不知道就别劝用户去改档位：custom / ollama 上换档既不会下发任何思考参数，也不会
        改变 ``_build_request`` 的额度（它的膨胀条件是 DeepSeek 兼容），照做只会原地重投。
        """
        return self._uses_deepseek_compatibility() or self.model_type in {
            "glm",
            "qwen",
            "openai",
        }

    def _build_request(
        self, prompt: str, max_tokens: int, effort: str, budget: str = "auto"
    ) -> dict[str, Any]:
        """budget: auto（思考档抬高输出预算）/ requested（只用调用方额度）/ omit（不下发）。

        思考链计入输出 token，所以默认要给足额度；但第三方网关的上限常常比官方低
        （拒绝时一般报 param=max_tokens），因此留出两级降级，不让任务卡在参数上。
        """
        request: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是视频笔记编辑。忠实理解语境与作者意图，允许结合上下文"
                        "保守修正明显的口误、笔误和语音转写错误。"
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        }
        token_key = (
            "max_completion_tokens"
            if self.model_type == "openai" and str(self.model).startswith("gpt-5")
            else "max_tokens"
        )
        thinking_budget = (
            max(max_tokens * MAX_EFFORT_BUDGET_MULTIPLIER, DEEPSEEK_HIGH_TOKEN_BUDGET)
            if self._uses_deepseek_compatibility() and effort == "max"
            else DEEPSEEK_HIGH_TOKEN_BUDGET
            if self._uses_deepseek_compatibility() and effort != "off"
            else max_tokens
        )
        if self.thinking_seen and not self._uses_deepseek_compatibility():
            # 实测到这条通道在返回思考链，就把预留加在最终额度上，而不是去改各阶段传进来的
            # max_tokens——那个值表达的是"正文要写多长"，语义不该被通道行为污染。
            thinking_budget += THINKING_RESERVE_TOKENS
        if budget != "omit":
            request[token_key] = thinking_budget if budget == "auto" else max_tokens
        self._apply_reasoning(request, effort)
        return request

    async def _complete(
        self,
        prompt: str,
        max_tokens: int,
        effort: str = "auto",
        retry_empty: bool = True,
        progress_callback: ProgressCallback | None = None,
        progress: int = 0,
        stage: str = "模型调用",
        should_abort: Callable[[], bool] | None = None,
        param_attempts: int = MAX_PARAM_RETRIES,
        budget: str = "auto",
        connection_retries: int = 1,
    ) -> str:
        if should_abort is not None and should_abort():
            raise asyncio.CancelledError("任务已取消")

        request = self._build_request(prompt, max_tokens, effort, budget)
        try:
            # 流式建连：既能逐块读取，也能在取消时立即中止本次请求
            stream = await self.client.chat.completions.create(**request, stream=True)
        except Exception as exc:
            # 通道不认我们下发的参数时，按下一级写法重试，而不是直接判任务失败
            rejected = self._rejected_request_params(exc, request)
            if not rejected or param_attempts <= 0:
                raise
            budget, note = self._degrade_rejected(rejected, budget)
            warning = f"{stage}：该通道不支持 {'/'.join(rejected)} 参数，{note}后重试"
            self._warn_once(warning)
            await self._report_progress(progress_callback, progress, warning)
            return await self._complete(
                prompt,
                max_tokens,
                effort,
                retry_empty=retry_empty,
                progress_callback=progress_callback,
                progress=progress,
                stage=stage,
                should_abort=should_abort,
                param_attempts=param_attempts - 1,
                budget=budget,
            )

        parts: list[str] = []
        content_characters = 0
        reasoning_characters = 0
        next_beat = time.monotonic() + LLM_HEARTBEAT_SECONDS
        finish_reason = "unknown"
        try:
            async for chunk in stream:
                if should_abort is not None and should_abort():
                    raise asyncio.CancelledError("任务已取消")
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                delta = choice.delta
                content = getattr(delta, "content", None) if delta is not None else None
                if content:
                    parts.append(content)
                    content_characters += len(content)
                # 不按"没有正文才算思考"分叉：有的通道把 reasoning_content 和 content
                # 塞在同一个块里，那样会一个都不计，心跳与"关思考没生效"的判读都会失真
                if delta is not None:
                    reasoning_characters += self._reasoning_characters(delta)
                if getattr(choice, "finish_reason", None):
                    finish_reason = choice.finish_reason
                # 心跳：深度思考阶段正文前只有 reasoning 增量，若不回报进度，
                # 前端与“真卡死”无法区分（本次故障的表现就是“卡住不动”）。
                if progress_callback is not None and time.monotonic() >= next_beat:
                    next_beat = time.monotonic() + LLM_HEARTBEAT_SECONDS
                    await self._report_progress(
                        progress_callback,
                        progress,
                        self._stream_status_message(
                            stage, content_characters, reasoning_characters
                        ),
                    )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # 长流中途被网关掐断（实测 peer closed connection ...）：正文一个字
            # 都没拿到时自动重投一次。失败就是整条任务报废、已烧的思考全白费，
            # 重投最坏也只是再花一次，不会更糟；已收到部分正文时不重投（没法续）。
            if (
                not parts
                and connection_retries > 0
                and _looks_disconnected(exc)
            ):
                warning = (
                    f"{stage}：与模型接口的连接中断"
                    f"（{type(exc).__name__}: {exc}），自动重试一次"
                )
                self.warnings.append(warning)
                await self._report_progress(progress_callback, progress, warning)
                return await self._complete(
                    prompt,
                    max_tokens,
                    effort,
                    retry_empty=retry_empty,
                    progress_callback=progress_callback,
                    progress=progress,
                    stage=stage,
                    should_abort=should_abort,
                    param_attempts=param_attempts,
                    budget=budget,
                    connection_retries=connection_retries - 1,
                )
            raise
        finally:
            close = getattr(stream, "close", None)
            if close is not None:
                closed = close()
                if inspect.isawaitable(closed):
                    await closed

        content = "".join(parts)
        # 把"我们下发了什么"与"通道实际给了什么"并排记下来：静默忽略关思考请求、或把
        # max_tokens 悄悄砍小，都不会回 400，参数降级阶梯因此永远不触发，光看
        # finish_reason=length 猜不到额度是谁缩的。
        if reasoning_characters:
            # 收到过思考增量就是实测证据：本进程内这条通道后续每次请求都自动补一份
            # 思考预留（见 _build_request）。第一次可能仍然不够，但不会一直不够。
            _THINKING_SEEN_CHANNELS.add(self._param_cache_key())
        if effort == "off" and reasoning_characters:
            # 不带阶段名与字数：这是通道级的事实，一次任务说一句就够（逐段直写会调几十次）
            hint = (
                "可在界面上改选「高」或「最大」档"
                if self._controls_thinking()
                else "本程序对这条通道没有思考开关，改档位也不会变"
            )
            self._warn_once(
                "该通道没有响应「关闭深度思考」的请求，仍在返回思考链；"
                f"这类通道上正文偏长时容易被思考吃满输出额度，{hint}。"
            )
        if not content:
            outcome = (
                f"下发 max_tokens={self._sent_token_budget(request)}、思考档={effort}，"
                f"实际收到 思考 {reasoning_characters} 字 / 正文 0 字，"
                f"finish_reason={finish_reason}"
            )
            if retry_empty and effort != "off" and self._can_disable_thinking():
                warning = (
                    f"{stage}：模型未返回正文（{outcome}），"
                    "多为思考链吃满输出额度后流被截断，已关闭深度思考并重试"
                )
                self.warnings.append(warning)
                await self._report_progress(progress_callback, progress, warning)
                return await self._complete(
                    prompt,
                    max_tokens,
                    "off",
                    retry_empty=False,
                    progress_callback=progress_callback,
                    progress=progress,
                    stage=stage,
                    should_abort=should_abort,
                    param_attempts=param_attempts,
                    budget=budget,
                )
            # 已经关过思考却仍只收到思考 ⇒ 这条通道不认我们的关闭请求，再劝"关闭深度
            # 思考"就是把人往死胡同里带（1.4.2 群测实测：off 档下发后仍累计 6473 字
            # 思考、4600 额度吃满，正文一个字没出）。
            if not self._controls_thinking():
                # custom / ollama：换档位既不下发思考参数也不改变额度，劝人改档等于骗他重投
                advice = (
                    "本程序没有这条通道的思考开关，思考链只能占正文的额度，改推理档位不会"
                    "有任何变化；可改用「精简摘要」、缩短视频，或换一个输出额度更大的模型。"
                )
            elif effort == "off":
                advice = (
                    "该通道没有响应关思考的请求，请改用「高」或「最大」档给正文留出额度，"
                    "或缩短转录、更换模型。"
                )
            else:
                advice = "可尝试关闭深度思考、缩短转录或更换模型。"
            raise RuntimeError(f"模型未返回正文（{outcome}）。{advice}")
        return utf8_safe(content).strip()

    @staticmethod
    def _sent_token_budget(request: dict[str, Any]) -> Any:
        """本次实际下发的输出额度（OpenAI 新模型用的是另一个键名）。"""
        return request.get("max_tokens") or request.get("max_completion_tokens") or "未下发"

    @staticmethod
    def _reasoning_characters(delta: Any) -> int:
        """取一个流式块里 reasoning/思考链增量的长度（不同通道字段名不一致）。"""
        values = [getattr(delta, name, None) for name in ("reasoning_content", "reasoning")]
        extra = getattr(delta, "model_extra", None)
        if isinstance(extra, dict):
            values.extend(extra.get(name) for name in ("reasoning_content", "reasoning"))
        return max((len(value) for value in values if isinstance(value, str)), default=0)

    @staticmethod
    def _stream_status_message(
        stage: str, content_characters: int, reasoning_characters: int
    ) -> str:
        if content_characters:
            return f"{stage}：模型已输出 {content_characters} 字"
        if reasoning_characters:
            return f"{stage}：模型正在深度思考（已累计 {reasoning_characters} 字）"
        return f"{stage}：等待模型返回首个数据块"

    def _apply_reasoning(self, request: dict[str, Any], effort: str) -> None:
        # auto 含义统一：不注入任何思考参数，交给通道默认行为
        # （generate_summary 会先把用户的 auto 按风格解析成 high/max）
        if effort == "auto":
            return
        enabled = effort != "off"
        if self._uses_deepseek_compatibility():
            # 开思考只用标准写法（flat reasoning_effort → 嵌套 reasoning.effort），
            # 私有 thinking 仅用于官方 DeepSeek 的“关掉思考”：那是它唯一能关思考的写法。
            # 很多兼容网关不认 thinking，包括“选了 DeepSeek 预设但填了代理 Base URL”
            # 的情况（v1.2.3 用户反馈 400 unsupported_parameter "thinking"）。
            if enabled:
                if not self._send_reasoning_effort(
                    request, "max" if effort == "max" else "high"
                ):
                    self._warn_once("该通道不支持思考强度参数，已按模型默认思考设置生成")
            elif self._is_official_deepseek() and "thinking" not in self.rejected_params:
                request["extra_body"] = {"thinking": {"type": "disabled"}}
            elif not self._send_reasoning_effort(request, "none"):
                self._warn_once("该通道不支持关闭深度思考，已按模型默认思考设置生成")
        elif self.model_type == "glm":
            request["extra_body"] = {"thinking": {"type": "enabled" if enabled else "disabled"}}
        elif self.model_type == "qwen":
            request["extra_body"] = {"enable_thinking": enabled}
        elif self.model_type == "openai":
            request["reasoning_effort"] = {
                "off": "none",
                "max": "xhigh",
            }.get(effort, effort)
        elif enabled:
            self.warnings.append(
                f"{self.model_type} 未配置通用推理强度映射，已使用模型默认设置"
            )

    def _warn_once(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)

    def _send_reasoning_effort(self, request: dict[str, Any], value: str) -> bool:
        """按兼容性阶梯下发思考档位，返回是否找到了可用写法。

        先试通用的 flat `reasoning_effort`；被拒后改试嵌套 `reasoning.effort`（同一条
        400 提示里就并列给了这两种写法）。两种写法不会同时下发。
        """
        rejected = self.rejected_params
        if "reasoning_effort" not in rejected:
            request["reasoning_effort"] = value
            return True
        if "reasoning" not in rejected:
            # 嵌套写法不是 openai SDK 的具名参数，必须走 extra_body（会被合并进 JSON body）
            extra = request.setdefault("extra_body", {})
            if isinstance(extra, dict):
                extra["reasoning"] = {"effort": value}
            else:
                request["reasoning_effort"] = value
            return True
        return False

    def _uses_deepseek_compatibility(self) -> bool:
        provider = str(getattr(self, "model_type", "")).lower()
        model = str(getattr(self, "model", "")).lower()
        base_url = str(getattr(self, "base_url", "")).lower()
        return provider == "deepseek" or "deepseek" in model or "deepseek" in base_url

    def _is_official_deepseek(self) -> bool:
        """只有官方 DeepSeek API 认私有 thinking 参数。"""
        if str(getattr(self, "model_type", "")).lower() == "deepseek":
            return True
        host = (urlparse(str(self.base_url)).hostname or "").lower()
        return host == "api.deepseek.com" or host.endswith(".deepseek.com")

    def _rejected_request_params(
        self, exc: Exception, request: dict[str, Any]
    ) -> tuple[str, ...]:
        """从报错里认出“本次确实下发过、但这条通道不认的思考/额度参数”。

        两种形状都要认：否认参数名（``"thinking" is not supported``），以及只否认我们给的
        取值（``'reasoning_effort' must be one of: 'low', ...``）。后者按参数不可用处理——
        会多花两次请求把 flat → 嵌套的阶梯试完，之后按 (provider, model, base_url) 记住。

        不能简单“看消息里提到了哪个名字”：拒绝提示会同时推荐另一个参数
        （"thinking" is not supported ... use "reasoning_effort"），所以只取服务端
        标出的 param 字段，或引号包住且紧跟拒绝措辞的那个名字。
        """
        text = str(exc)
        lowered = text.lower()
        if not any(
            hint in lowered
            for hint in (
                "unsupported_parameter",
                "not supported",
                "unsupported parameter",
                "unknown parameter",
                "unrecognized",
                "must be one of",
                "invalid_value",
                "invalid value",
            )
        ):
            return ()
        names: set[str] = set()
        body = getattr(exc, "body", None)
        if isinstance(body, dict):
            error = body.get("error")
            if isinstance(error, dict) and error.get("param"):
                names.add(str(error["param"]).strip().lower())
        for pattern in (_UNSUPPORTED_PARAM_RE, _REJECTED_VALUE_RE):
            names.update(match.group(1).lower() for match in pattern.finditer(text))
        sent = set(request)
        extra_body = request.get("extra_body")
        if isinstance(extra_body, dict):
            sent.update(extra_body)
        return tuple(sorted(names & sent & set(_DEGRADABLE_PARAM_NAMES)))

    def _degrade_rejected(
        self, rejected: tuple[str, ...], budget: str
    ) -> tuple[str, str]:
        """记下被拒的思考参数、选下一级输出额度，返回 (额度档, 提示语)。"""
        notes: list[str] = []
        reasoning = [name for name in rejected if name in REASONING_PARAM_NAMES]
        if reasoning:
            self.rejected_params.update(reasoning)
            _REJECTED_REASONING_PARAMS.setdefault(
                self._param_cache_key(), set()
            ).update(reasoning)
            notes.append("已改用下一级思考写法")
        if any(name in TOKEN_BUDGET_PARAM_NAMES for name in rejected):
            budget = "requested" if budget == "auto" else "omit"
            notes.append(
                "已把输出额度上限降回笔记长度"
                if budget == "requested"
                else "已不再下发输出额度上限"
            )
        return budget, ("，".join(notes) if notes else "已去掉该参数")

    def _can_disable_thinking(self) -> bool:
        """空正文重试只在“确实有办法关掉思考”的通道上做，否则只是白跑一次请求。"""
        if self._uses_deepseek_compatibility():
            if self._is_official_deepseek() and "thinking" not in self.rejected_params:
                return True
            rejected = self.rejected_params
            return "reasoning_effort" not in rejected or "reasoning" not in rejected
        return self.model_type in {"openai", "glm", "qwen"}

    @staticmethod
    def _chunk_prompt(
        title: str,
        index: int,
        total: int,
        segments: Sequence[TranscriptSegment],
    ) -> str:
        first_ts = format_timestamp(segments[0].start)
        last_ts = format_timestamp(segments[-1].end)
        return f"""视频标题：{title}
这是第 {index}/{total} 个连续片段。

请忠实压缩这个片段，供后续撰写完整笔记使用。保留重要事实、例子、论证关系和原有时间点；
本片段覆盖时间轴 {first_ts} – {last_ts}，区间末尾的话题同样重要，压缩时不要丢掉结尾内容；
结合上下文理解表达意图，不做评价，不补充外部知识。

转录：
{segments_to_prompt(segments)}"""

    @staticmethod
    def _merge_prompt(
        title: str,
        level: int,
        index: int,
        total: int,
        notes: Sequence[str],
    ) -> str:
        material = "\n\n".join(notes)
        return f"""视频标题：{title}
这是长视频内容的第 {level} 层归并，第 {index}/{total} 组。

请合并以下连续片段笔记，去除重复但保留观点、依据、例子、时间点和前后关系。
不要评价，不补充外部知识，只输出供最终成稿使用的连续材料。

片段笔记：
{material}"""

    @staticmethod
    def _context_brief(metadata: dict[str, Any]) -> str:
        """把视频简介压成一小块背景资料；没有简介时返回空串。

        字幕稿本身往往不点名「这是谁的课」，简介通常一眼可辨。只允许用于
        理解与校对；热评多为玩梗（误导风险大于信息量），刻意不采集。
        """
        description = str(metadata.get("description") or "").strip()
        if not description:
            return ""
        return (
            "视频背景资料（来自平台简介，用于理解主题背景、识别这是哪门课程"
            "或哪个系列、核对术语与专有名词写法；不要把简介内容当成视频内容"
            "写进笔记）：\n"
            f"简介：{description[:BRIEF_DESCRIPTION_CHARACTERS]}"
        )

    # 内容场景：管"写什么结构"，与风格正交；general 返回空串，行为与历史版本一致。
    # 结构版用于 detailed/faithful（正文按场景小节组织）；侧重版用于 keypoints/concise——
    # 这两类结构已由风格定死，场景只影响"从哪些内容里取材"。
    _SCENARIO_STRUCTURE = {
        "tutorial": (
            "这是教程/操作类内容，正文按这个结构组织，没有内容的小节省略，不要为凑结构编造：\n"
            "## 前置准备 —— 需要的环境、基础或材料\n"
            "## 操作步骤 —— 编号步骤，每步给出关键动作与参数，参数与命令原样保留\n"
            "## 易错点 —— 常见错误与规避办法\n"
            "## 验证结果 —— 怎样确认操作做对了"
        ),
        "knowledge": (
            "这是知识/原理类内容，正文按这个结构组织，没有内容的小节省略：\n"
            "## 核心概念 —— 一句话说清定义\n"
            "## 原理机制 —— 为什么是这样\n"
            "## 关系框架 —— 概念之间的联系与层级\n"
            "## 适用边界 —— 什么情况下成立、什么情况下不成立"
        ),
        "learning": (
            "这是以学会为目标的学习内容，正文按这个结构组织，没有内容的小节省略：\n"
            "## 学习目标 —— 看完应该能做什么\n"
            "## 知识点清单 —— 按重要性排列，重点条目标【重点】\n"
            "## 记忆线索 —— 类比、口诀或便于记住的说法\n"
            "## 自测 —— 1 到 3 道检验理解的问题，不要照抄原文"
        ),
    }
    _SCENARIO_EMPHASIS = {
        "tutorial": "这是教程/操作类内容：步骤、关键参数、易错点属于核心内容，不要略过。",
        "knowledge": "这是知识/原理类内容：概念定义、机制、成立条件属于核心内容，不要略过。",
        "learning": "这是以学会为目标的内容：该掌握的重点与易混淆点属于核心内容，不要略过。",
    }

    @staticmethod
    def _scenario_block(content_type: str, style: str = "detailed") -> str:
        """内容场景的取材/结构要求；general（或未知值）返回空串，保持历史行为。"""
        if content_type not in CONTENT_TYPES or content_type == "general":
            return ""
        if style in CONCISE_LIKE_STYLES:
            return LLMSummarizer._SCENARIO_EMPHASIS.get(content_type, "")
        return LLMSummarizer._SCENARIO_STRUCTURE.get(content_type, "")

    @staticmethod
    def _note_prompt(
        title: str,
        source: str,
        metadata: dict[str, Any],
        style: str = "detailed",
        coverage_end: str | None = None,
        content_type: str = "general",
    ) -> str:
        owner = metadata.get("owner") or "未知"
        published_at = metadata.get("published_at") or "未知"
        duration = metadata.get("duration_text") or "未知"
        view_count = metadata.get("view_count") or 0
        like_count = metadata.get("like_count") or 0
        source_label = metadata.get("transcript_source") or "语音转写"
        scenario = LLMSummarizer._scenario_block(content_type, style)
        scenario_block = f"\n{scenario}\n" if scenario else ""
        if style == "keypoints":
            task = (
                "写一份「要点提炼」笔记：只输出视频的核心观点与结论，不展开论证、不复述过程。"
                "结构固定为三块——「## 核心观点」（编号列表，每条一句话结论）、"
                "「## 结论清单」（作者最终主张或落点，编号列表）、"
                "「## 关键时间点」（核心观点对应的时间锚点速查）。"
            )
            timestamp_hint = (
                "每条核心观点句末都要带 [起点-终点] 时间锚点，时间点只能取自材料；"
                "不要为不存在的点编造时间。"
            )
            coverage_hint = ""
        elif style == "concise":
            task = (
                "写一份精炼的 Markdown 摘要，说明视频谈了什么、作者的主要观点，"
                "并列出少量最有用的时间点。"
            )
            timestamp_hint = ""
            coverage_hint = ""
        else:
            task = (
                "写一份翔实、自然的 Markdown 视频笔记。完整复原内容脉络、具体例子、"
                "核心观点及其依据，并在有帮助时加入关键时间点。不要加入外部知识或评价。"
            )
            timestamp_hint = (
                "每个 ## 或 ### 小标题都写成「## 标题 [起点-终点]」，区间取该话题在材料里的"
                "首尾时间点；标题之外的段落仅在有助于定位时另加 [MM:SS]，不必每段都加；"
                "最后一个小节的区间终点必须接近材料末尾，以便核对是否写到视频最后。"
            )
            coverage_hint = (
                f"材料内容一直推进到时间点 {coverage_end}；笔记必须覆盖到该时间点，"
                "收尾前先自查是否漏掉了材料末尾的话题。"
                if coverage_end
                else ""
            )
        headline = (
            f"# 视频要点：《{title}》" if style == "keypoints" else f"# 视频笔记：《{title}》"
        )
        count_hint = (
            "核心观点条数按视频体量自适应：短视频 3-8 条，长视频 8-15 条，宁缺毋滥，"
            "观点之间不要重复。"
            if style == "keypoints"
            else ""
        )
        if style in CONCISE_LIKE_STYLES:
            # 这两类的结构已由风格本身定死（清单 / 短摘要），再叠一句"按结构组织"只会互相打架；
            # 场景对它们的作用已经写在 scenario_block 的取材侧重里。
            structure_hint = ""
        elif scenario:
            structure_hint = (
                "严格按上面给出的结构组织小节，小节标题就用上面给定的小节名；没有内容的小节省略，"
                "不要为凑结构编造。"
            )
        else:
            structure_hint = "结构按内容自然组织，不必凑固定模板。"
        metadata_hint = (
            "如果有助于读者快速了解视频，可在标题下自然带出材料中明确的作者、发布时间、"
            "时长或播放信息；缺失的信息省略，不要猜测，也不必为了满足格式强行补齐。"
            "应用会在标题下补充一行可验证的元信息，正文无需重复同一组信息。"
        )
        brief = LLMSummarizer._context_brief(metadata)
        brief_block = f"\n{brief}\n" if brief else ""
        return f"""{task}
{scenario_block}
视频标题：{title}
作者：{owner}
发布时间：{published_at}
时长：{duration}
播放：{view_count}
点赞：{like_count}
文字来源：{source_label}
{brief_block}
以准确理解语境和作者立场为先。{metadata_hint}时间点只能取自材料。{coverage_hint}{timestamp_hint}可在上下文支持时直接修正明显的口误、
笔误或转写错误；只有歧义会影响结论且无法可靠判断时，才在正文采用最可能的解释，并在文末用
Markdown 脚注集中说明。不要在正文反复插入“原文如此”或“疑为转写错误”。
{count_hint}{structure_hint}数学公式一律用 Markdown 公式书写：行内 $...$、独立
公式块 $$...$$，不要用 \\(...\\) 或 \\[...\\] 包裹。标题使用：{headline}

材料：
{source}"""

    @staticmethod
    def _analysis_prompt(title: str, draft: str) -> str:
        return f"""请针对这份视频笔记所对应的视频内容本身，补写一段独立的点评与分析，只输出可直接追加到笔记后的 Markdown。

点评对象是视频里的内容，而不是笔记的写法：作者的核心观点与立场是否成立、论据与例证是否充分、论证逻辑是否严密、结论是否可靠，以及内容的信息价值、适用场景和值得进一步思考的问题。可以补充有助于理解的稳定背景知识，但必须标明“模型补充，未联网核验”，不得把它写成视频原话或已核实事实。语气中性自然，服务于理解内容，避免复述已有笔记；不要评价笔记的组织结构、文笔或排版。

标题：{title}

已有笔记：
{draft}"""

    @staticmethod
    def _normalize_timestamp_delimiters(text: str) -> str:
        """把时间区间里的长破折号统一成半角连字符。"""
        return _TIMESTAMP_DASH_RE.sub(r"\1-\2", text)

    @classmethod
    def _normalize_math_delimiters(cls, text: str) -> str:
        """把 \\(...\\) 与 \\[...\\] 统一成 Markdown 通用的 $...$ / $$...$$。

        提示词已要求模型直接用 $ 系定界符，这里兜底收尾——提示词拦不住的漏网
        之鱼如果留着，在多数渲染器里就是一堆裸反斜杠。代码块内是字面文本，跳过。
        """
        parts = text.split("```")
        for index in range(0, len(parts), 2):
            segment = parts[index]
            segment = _MATH_DISPLAY_RE.sub(
                lambda match: f"$$\n{match.group(1).strip()}\n$$", segment
            )
            segment = _MATH_INLINE_RE.sub(
                lambda match: f"${match.group(1).strip()}$", segment
            )
            parts[index] = segment
        return "```".join(parts)

    @staticmethod
    def _strip_code_fence(value: str) -> str:
        if value.startswith("```markdown") and value.endswith("```"):
            return value[len("```markdown") : -3].strip()
        if value.startswith("```") and value.endswith("```"):
            return value[3:-3].strip()
        return value
