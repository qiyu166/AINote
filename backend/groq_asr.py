"""Groq 云端语音转写（whisper-large-v3 / whisper-large-v3-turbo）。

为什么需要它：本机跑大模型在这台机器上不可行——到 HuggingFace 的下载只有约
32KB/s，``base`` 要 75 分钟、``large-v3`` 约 3GB 根本下不动；而本地 ``tiny``
的中文准确率不可用（实测同一条音频："闲鱼"→"鲜魚"、"眼力见"→"眼力件"、
"海鲜自助"→"海蝕自助"、"是人家公司年会"→"时间加工四年会"）。

Groq 的 LPU 推理约 70× 实时、**零模型下载**，中文质量与 tiny 不是一个量级
（同一条 199.7 秒音频：2.9 秒返回，错别字从十余处降到 1~2 处）。

限制与对策：免费档单文件上限 25MB（社区也有 >30MB 静默失败的报告），所以统一
按 ``CHUNK_SECONDS`` 切片成 16kHz 单声道 WAV（约 19.2MB/片），片内时间戳整体
偏移后合并，对外仍是一条完整转录。

注意：Groq 按地区封锁（中国大陆直连时所有路径含根路径均 403），需要代理才可用；
代理未开启时这里会给出可读的失败原因，而不是让它变成一个语焉不详的任务失败。
"""
from __future__ import annotations

import asyncio
import inspect
import os
import tempfile
import time
import wave
from pathlib import Path
from typing import Any, Callable

import httpx

from .transcript import TranscriptSegment

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_TRANSCRIBE_URL = f"{GROQ_BASE_URL}/audio/transcriptions"

# 对外暴露的模型 id -> Groq 上的实际模型名
GROQ_ASR_MODELS: dict[str, str] = {
    "groq-whisper-large-v3-turbo": "whisper-large-v3-turbo",
    "groq-whisper-large-v3": "whisper-large-v3",
}

SAMPLE_RATE = 16000
# 16kHz / 16bit 单声道 = 32KB/s；10 分钟 ≈ 19.2MB，对 25MB 上限留足余量
CHUNK_SECONDS = 600
MAX_UPLOAD_BYTES = 24 * 1024 * 1024
REQUEST_TIMEOUT_SECONDS = 600.0
# 正式上传前做一次零成本的鉴权预检（列模型不消耗额度）
KEY_PROBE_URL = f"{GROQ_BASE_URL}/models"
# 分片上传的重试次数：连接被中途重置是常见瞬态，重试往往能直接扛过去
UPLOAD_ATTEMPTS = 3

ProgressCallback = Callable[[float, float], Any]

# Groq 的 verbose_json 返回语言"名称"（实测中文返回 "Chinese"），而 faster-whisper
# 返回 ISO 代码（"zh"）。下游有拿 language 做字幕轨语言提示的地方，两套值混用会
# 让提示失真，所以这里统一成 ISO 代码。
_LANGUAGE_CODES = {
    "chinese": "zh",
    "english": "en",
    "japanese": "ja",
    "korean": "ko",
    "russian": "ru",
    "french": "fr",
    "german": "de",
    "spanish": "es",
    "portuguese": "pt",
    "italian": "it",
    "arabic": "ar",
    "cantonese": "yue",
}


def _to_iso_language(raw: str) -> str:
    value = (raw or "").strip()
    if not value:
        return "unknown"
    lowered = value.lower()
    if lowered in _LANGUAGE_CODES:
        return _LANGUAGE_CODES[lowered]
    if len(value) <= 3:  # 已经是 "zh" / "en" 这类代码
        return lowered
    return lowered


def is_groq_model(model_name: str) -> bool:
    return model_name in GROQ_ASR_MODELS


def resolve_api_key(store: Any) -> str:
    """取 Groq Key，缺 Key 时抛出可直接照做的中文错误。

    优先级：
    1. 环境变量 ``GROQ_API_KEY``（Docker / CI 场景最直接）
    2. 本机密钥库中为 ``https://api.groq.com/openai/v1`` 保存的那把
       （前端「模型配置」页选 Groq 录入后保存到本机）

    ``store`` 由调用方注入（main.py 持有唯一的 ``ConfigStore`` 实例），
    避免本模块反向依赖 main 造成循环导入。
    """
    from .llm_summarizer import normalize_endpoint_host

    env_key = (os.environ.get("GROQ_API_KEY") or "").strip()
    if env_key:
        return env_key

    resolved = store.resolve_stored_key(model_type="groq", base_url=GROQ_BASE_URL)
    secret = str(resolved.get("api_key") or "")
    if secret:
        return secret

    host = normalize_endpoint_host(GROQ_BASE_URL, "groq") or GROQ_BASE_URL
    raise RuntimeError(
        f"未找到 Groq API Key：请在「模型配置」页选择 Groq（接口地址 {host}）"
        "填写 Key 并保存到本机，或设置环境变量 GROQ_API_KEY。"
        "（Groq 在中国大陆直连不可用，需先开启代理。）"
    )


def _encode_wav(samples: Any, target: Path) -> Path:
    """float32 [-1,1] 单声道样本 -> 16bit PCM WAV（Groq 支持 wav）。"""
    import numpy as np

    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(target), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(SAMPLE_RATE)
        fh.writeframes(pcm.tobytes())
    return target


def _slice_wavs(
    media_path: Path, work_dir: Path, cancel_event: Any = None
) -> list[tuple[Path, float]]:
    """把媒体解码成 16kHz 单声道并按 ``CHUNK_SECONDS`` 切片。

    返回 ``[(wav 路径, 该片起始秒), ...]``。复用 faster-whisper 的解码器（本项目
    既有依赖），不再引入 ffmpeg 命令行依赖。
    """
    from faster_whisper import decode_audio

    audio = decode_audio(str(media_path), sampling_rate=SAMPLE_RATE)
    total_samples = len(audio)
    step = CHUNK_SECONDS * SAMPLE_RATE
    pieces: list[tuple[Path, float]] = []
    offset = 0
    index = 0
    while offset < total_samples:
        if cancel_event is not None and cancel_event.is_set():
            raise asyncio.CancelledError("任务已取消")
        target = _encode_wav(audio[offset: offset + step], work_dir / f"chunk_{index:03d}.wav")
        pieces.append((target, offset / SAMPLE_RATE))
        offset += step
        index += 1
    return pieces


def _explain_http_error(status: int, body: str) -> str:
    """把 Groq 的失败翻译成能指导下一步的中文。"""
    text = (body or "").strip()[:200]
    if status == 403 and "Forbidden" in text and len(text) <= 60:
        return (
            "Groq 返回 403 且响应体无细节，通常是**地区封锁**（中国大陆直连时对该域名"
            "所有路径一律 403，与 Key 无关）。请开启代理后重试。"
        )
    if status == 401:
        return "Groq 拒绝了这个 API Key（401），请到「模型配置」页重新填写并保存。"
    if status == 413:
        return "音频分片超过 Groq 的单文件上限（413），请调小 CHUNK_SECONDS 后重试。"
    if status == 429:
        return "Groq 限流（429），稍后重试；或改用 whisper-large-v3-turbo（配额更宽松）。"
    return f"Groq 转写失败 HTTP {status}：{text}"


def _explain_transport_error(exc: Exception | None) -> str:
    """连接层失败的文案。**永不返回空串**——httpx 的 ReadError/WriteError 消息就是空的。

    实测：同一个 7MB 分片，服务端提前拒绝时 curl 能拿到干净的 400/401，
    httpx 却抛一个 *消息为空* 的 ``ReadError``，把真实状态码整个吞掉。
    """
    if exc is None:
        return "Groq 上传失败：未知连接错误"
    name = type(exc).__name__
    detail = str(exc).strip()
    head = f"Groq 连接失败（{name}）" + (f"：{detail}" if detail else "")
    if isinstance(exc, (httpx.ReadError, httpx.WriteError, httpx.RemoteProtocolError)):
        return (
            f"{head}。上传途中连接被中断，通常是 Groq 提前拒绝了这次请求"
            "（Key 失效、账号限流或地区限制），也可能是网络/代理不稳定；已自动重试。"
        )
    if isinstance(exc, httpx.TimeoutException):
        return f"{head}。请求超时，请检查网络或代理后重试。"
    return f"{head}。请检查网络/代理后重试。"


async def _probe_api_key(client: httpx.AsyncClient, api_key: str) -> None:
    """正式上传前做一次零成本鉴权预检，把「Key 失效 / 地区封锁 / 限流」变成可读文案。

    为什么需要：真实分片是 7–19MB，而 httpx 在服务端**提前拒绝**时会抛一个消息为空的
    ``ReadError``，原本清清楚楚的 401/403/429 就这样没了。先用一个小请求问一句，
    这类问题就能在动手上传之前说清楚。
    """
    try:
        response = await client.get(
            KEY_PROBE_URL, headers={"Authorization": f"Bearer {api_key}"}, timeout=15.0
        )
    except httpx.TransportError:
        return  # 网络层问题交给正式上传去报，别在这里抢先误报
    if response.status_code != 200:
        raise RuntimeError(_explain_http_error(response.status_code, response.text))


async def _post_chunk(
    client: httpx.AsyncClient,
    wav_path: Path,
    groq_model: str,
    api_key: str,
    cancel_event: Any,
) -> httpx.Response:
    """上传一个分片；连接被中途重置时重试，全部失败则抛出可读的错误。"""
    last_error: Exception | None = None
    for attempt in range(UPLOAD_ATTEMPTS):
        if cancel_event is not None and cancel_event.is_set():
            raise asyncio.CancelledError("任务已取消")
        try:
            with wav_path.open("rb") as fh:
                return await client.post(
                    GROQ_TRANSCRIBE_URL,
                    headers={"Authorization": f"Bearer {api_key}"},
                    files={"file": (wav_path.name, fh, "audio/wav")},
                    data={
                        "model": groq_model,
                        "response_format": "verbose_json",
                        # 不锁 language：让 Whisper 自检，中英混排的内容也不会被带偏
                        "temperature": "0",
                    },
                )
        except httpx.TransportError as exc:
            last_error = exc
            if attempt + 1 < UPLOAD_ATTEMPTS:
                await asyncio.sleep(1.5 * (attempt + 1))
    raise RuntimeError(_explain_transport_error(last_error)) from last_error


async def transcribe(
    media_path: Path,
    model_name: str,
    api_key: str,
    cancel_event: Any = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """按 Groq 的方式转写，返回结构与 ``whisper_asr`` 对齐，可直接替换。"""
    groq_model = GROQ_ASR_MODELS.get(model_name)
    if groq_model is None:
        raise ValueError(f"不是 Groq 转写模型：{model_name}")
    if not api_key:
        raise RuntimeError("Groq 转写需要 API Key")

    started = time.monotonic()
    segments: list[TranscriptSegment] = []
    languages: list[str] = []
    reported_duration = 0.0

    with tempfile.TemporaryDirectory(prefix="groq_asr_") as tmp:
        work_dir = Path(tmp)
        pieces = await asyncio.to_thread(_slice_wavs, media_path, work_dir, cancel_event)
        if not pieces:
            raise RuntimeError("音频切分后没有可转写的内容（音轨可能为空）")

        total_bytes = sum(path.stat().st_size for path, _ in pieces)
        oversized = [path.name for path, _ in pieces if path.stat().st_size > MAX_UPLOAD_BYTES]
        if oversized:
            raise RuntimeError(
                f"音频分片超过 Groq 单文件上限：{', '.join(oversized)}，请调小 CHUNK_SECONDS"
            )

        chunks_done = 0
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS, follow_redirects=True) as client:
            # 动手传 7–19MB 之前，先用一个零成本小请求把「Key/地区/限流」这类
            # 会被提前拒绝、又会被 httpx 吞成空 ReadError 的问题问清楚
            await _probe_api_key(client, api_key)
            for wav_path, offset_seconds in pieces:
                if cancel_event is not None and cancel_event.is_set():
                    raise asyncio.CancelledError("任务已取消")
                response = await _post_chunk(
                    client, wav_path, groq_model, api_key, cancel_event
                )
                if response.status_code != 200:
                    raise RuntimeError(_explain_http_error(response.status_code, response.text))

                payload = response.json()
                reported_duration = max(
                    reported_duration, float(payload.get("duration") or 0) + offset_seconds
                )
                if payload.get("language"):
                    languages.append(str(payload["language"]))
                for item in payload.get("segments") or []:
                    text = str(item.get("text") or "").strip()
                    if not text:
                        continue
                    segments.append(
                        TranscriptSegment(
                            start=round(float(item.get("start") or 0) + offset_seconds, 2),
                            end=round(float(item.get("end") or 0) + offset_seconds, 2),
                            text=text,
                        )
                    )
                chunks_done += 1
                if progress_callback is not None:
                    # 上游 transcribe_progress_reporter 返回的是同步函数；这里两种都兼容，
                    # 免得把「回调返回值不可 await」变成一个莫名其妙的 NoneType 报错。
                    outcome = progress_callback(
                        chunks_done * CHUNK_SECONDS, len(pieces) * CHUNK_SECONDS
                    )
                    if inspect.isawaitable(outcome):
                        await outcome

    if not segments:
        raise RuntimeError("Groq 没有返回任何转写分段")

    elapsed = round(time.monotonic() - started, 1)
    # 以实际音频长度为准：Groq 报的 duration 是分片时长，合并不便，这里用末段时间兜底
    duration = reported_duration or segments[-1].end
    language = _to_iso_language(max(set(languages), key=languages.count) if languages else "")

    return {
        "text": " ".join(item.text for item in segments),
        "segments": segments,
        "language": language,
        "language_probability": 1.0,
        "device": "groq",
        "model": groq_model,
        "requested_model": model_name,
        "duration": float(duration or 0),
        "duration_after_vad": float(duration or 0),
        "fallback_note": (
            f"云端转写：Groq {groq_model}，{len(pieces)} 个分片，"
            f"上传合计 {total_bytes / 1048576:.1f}MB，用时 {elapsed} 秒"
        ),
        "model_load_seconds": 0.0,
        "transcribe_seconds": elapsed,
    }
