from __future__ import annotations

import asyncio
import importlib.util
import io
import ipaddress
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
import zipfile
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Sequence
from urllib.parse import parse_qs, quote, urlencode, urlsplit, urlunsplit

import aiofiles
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException


class NoCacheStaticFiles(StaticFiles):
    """前端静态资源每次都重新校验，避免浏览器缓存旧版本。

    HTML 入口页更是直接 no-store：旧 HTML 与新版 script.js 混用会导致
    页面元素对不上、按钮全部"点了没反应"，因此入口页不允许任何缓存。
    """

    def file_response(self, *args: Any, **kwargs: Any) -> FileResponse:
        response = super().file_response(*args, **kwargs)
        if response.headers.get("content-type", "").startswith("text/html"):
            response.headers["Cache-Control"] = "no-store"
        else:
            response.headers["Cache-Control"] = "no-cache"
        return response
from pydantic import BaseModel, Field, SecretStr

from . import paraformer_asr, secret_box
from . import asr_process
from . import groq_asr
from .config_store import (
    BiliCredentialsUnavailable,
    DouyinCredentialsUnavailable,
    LLM_KEYS_FILE,
    ConfigStore,
)
from .llm_summarizer import (
    LONG_TRANSCRIPT_CHARACTERS,
    TAIL_GAP_SECONDS,
    TAIL_PATCH_MAX_INPUT_CHARACTERS_MANUAL,
    LLMSummarizer,
    default_base_url,
    env_proxy_summary,
    mask_proxy_target,
    normalize_endpoint_host,
    utf8_safe,
)
from .bili_login import BiliLoginManager
from .douyin_login import DouyinLoginManager
from .transcript import (
    TranscriptSegment,
    format_timestamp,
    segments_to_prompt,
    transcript_quality,
)
from .video_processor import (
    FRAMES_DIR_NAME,
    VideoProcessor,
    VideoSource,
    bilibili_page_url,
    merge_bilibili_pages,
    normalize_video_input,
)
from .paraformer_asr import ParaformerTranscriber
from .whisper_asr import WhisperTranscriber


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")
WORKSPACE_DIR = Path(
    os.getenv("AINOTE_WORKSPACE", str(BASE_DIR / "workspace"))
).resolve()
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
WHISPER_CACHE_DIR = Path(
    os.getenv("WHISPER_CACHE_DIR", str(WORKSPACE_DIR / "_model_cache"))
).resolve()
# workspace/ 下两个面向人的产物目录。转录稿单独一个目录而不是混进 notes/：先转写、后生成
# 笔记的同一视频会各留一份，分开摆一眼看得清，不用靠【转录】后缀在一堆文件里认。
NOTES_DIR_NAME = "notes"
TRANSCRIPTS_DIR_NAME = "transcripts"
# 成品归档根目录。默认为空 = 沿用 workspace/<folder>/（截图靠"与任务目录同级"的相对
# 路径就地引用）。设了就把成品集中到项目外的知识库目录，此时归档会把截图复制到笔记
# 旁边——否则 ../<task_id>/frames/ 会指向不存在的位置，图全丢。
NOTES_OUTPUT_DIR = (os.getenv("NOTES_OUTPUT_DIR") or "").strip()
FRONTEND_DIR = BASE_DIR / "frontend"
APP_ICON_PATH = BASE_DIR / "sources" / "icon.png"
FAVICON_PATH = BASE_DIR / "sources" / "icon.ico"
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "2048"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
# 超过该体积的本地视频上传后自动提取音频（16kHz m4a）并删除原视频，
# 保留与在线视频一致的轻量工作目录；勾选“视频截图”时保留原视频。
LARGE_UPLOAD_EXTRACT_MB = int(os.getenv("LARGE_UPLOAD_EXTRACT_MB", "300"))
LARGE_UPLOAD_EXTRACT_BYTES = LARGE_UPLOAD_EXTRACT_MB * 1024 * 1024
MAX_CONCURRENT_TASKS = max(1, int(os.getenv("MAX_CONCURRENT_TASKS", "1")))
TASK_HISTORY_LIMIT = max(10, int(os.getenv("TASK_HISTORY_LIMIT", "100")))
# 「最近任务」的分页：首屏给一页，网页端点一次「加载更多」就多要一页。
TASK_PAGE_SIZE = 20
TASK_PAGE_MAX = 200
ALLOWED_MEDIA_SUFFIXES = {
    ".mp3",
    ".m4a",
    ".wav",
    ".flac",
    ".aac",
    ".mp4",
    ".mkv",
    ".mov",
    ".webm",
    ".avi",
}
WHISPER_MODELS = {"tiny", "base", "small", "medium", "large-v3", "turbo", "belle-turbo-zh"}
# Groq 是云端转写（whisper-large-v3 系列），不进 WHISPER_MODELS：那一组会被当成
# 本地 faster-whisper 的缓存清单展示，混进去会出现「点下载却没有对应本地模型」。
ASR_MODELS = WHISPER_MODELS | {"paraformer-zh"} | set(groq_asr.GROQ_ASR_MODELS)
DOUYIN_HINT = (
    "抖音链接解析失败。程序会在本机浏览器里打开作品页再取数据（抖音风控要求"
    "浏览器上下文），公开作品无需登录；若作品仅登录可见，请在「抖音访问凭据」里"
    "扫码登录后重试；也可以把视频保存到本地后用「上传本地视频」。"
)
# 只有这两步的失败才该套上面那句「抖音解析失败」；其余步骤（转写、生成笔记…）
# 与抖音无关，一律透传原错误，免得用户去重投一条本来没问题的链接。
DOUYIN_HINT_STEPS = frozenset({"读取视频信息", "准备音频"})
NOTE_STEP_NAME = "生成笔记"
API_KEY_PATTERN = re.compile(r"sk-[A-Za-z0-9_\-]{4,}")
# 网关报错常把整个请求头或 JSON 回显出来；只认 sk- 前缀会漏掉 Bearer 与非 sk- 格式的凭据。
# 字段名用 \b 框住，否则 "tokenizer.json" 会被当成 "token" + 值咬掉。
SECRET_ECHO_PATTERN = re.compile(
    r"\b(api[_-]?key|authorization|access[_-]?token|refresh[_-]?token|bearer|sessdata|bili_jct)"
    r"\b([\"'’：:=\s]{1,4})([A-Za-z0-9_\-\.]{8,})",
    re.IGNORECASE,
)
# 1.4.1 之前笔记里写的是 ./images/，而磁盘上一直是 frames/；两种都要认，
# 否则历史任务的笔记重新导出时仍然带不回截图。
NOTE_IMAGE_REF_RE = re.compile(
    rf"(!\[[^\]]*\]\()\./(?:{FRAMES_DIR_NAME}|images)/([^)]+)\)"
)

app = FastAPI(title="AINote API", version="1.4.5")


def is_loopback_client(host: str | None) -> bool:
    if not host or host == "testclient":
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return host.lower() == "localhost"
    if address.is_loopback:
        return True
    return bool(address.version == 6 and address.ipv4_mapped and address.ipv4_mapped.is_loopback)


# 允许的 Host 头主机名。只认回环字面量：DNS rebinding 攻击正是靠"域名解析到
# 127.0.0.1、但 Host 头仍是攻击者域名"来绕过单纯的回环 IP 校验——浏览器把
# rebind 后的域名当同源，于是能读到响应。校验 Host 头是这类攻击的收口点。
LOOPBACK_HOSTNAMES = {"127.0.0.1", "localhost", "::1", "[::1]"}
# Starlette TestClient 的默认 Host 头（httpx 用 "testserver"）；与
# is_loopback_client 放行的 "testclient" 对应，都是测试客户端的哨兵值。
TEST_HOSTNAMES = {"testserver", "testclient"}


def is_loopback_host_header(value: str | None) -> bool:
    """Host 头必须是回环字面量（允许带端口）。

    空 Host 放行：HTTP/1.0 与少数本机工具（如某些 MCP 客户端）不带 Host 头，
    而它们本来就走回环 IP 校验，收严到"缺头即拒"会误伤正常调用。
    """
    if value is None:
        return True
    text = value.strip()
    if not text:
        return True
    # 去掉端口：IPv6 形如 [::1]:8000，IPv4/主机名形如 127.0.0.1:8000
    if text.startswith("["):
        closing = text.find("]")
        if closing == -1:
            return False
        hostname = text[: closing + 1].lower()
        remainder = text[closing + 1 :]
        if remainder and not remainder.startswith(":"):
            return False
    else:
        hostname, _, port = text.partition(":")
        hostname = hostname.lower()
        if port and not port.isdigit():
            return False
    return hostname in LOOPBACK_HOSTNAMES or hostname in TEST_HOSTNAMES


SECURITY_HEADERS = {
    # style-src 放开 inline style：KaTeX 产物的上下标/矩阵定位全靠 inline style 属性。
    # 模型产出的 HTML 仍被前端 DOMPurify 剥掉 style（FORBID_ATTR），放行的只有
    # 本机 KaTeX 渲染结果。注意 CSP 的 hash 与 'unsafe-inline' 并存时后者会被
    # 忽略，必须整体替换；此处与 frontend/index.html 的 meta 是两份独立生效的
    # CSP，改一处就要同步改另一处。
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; connect-src 'self'; object-src 'none'; "
        "base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
}


class LocalSecurityMiddleware:
    """纯 ASGI 中间件：只允许本机回环客户端访问，并为响应附加安全头。

    不用 Starlette 的 BaseHTTPMiddleware（@app.middleware("http")）：
    它对流式响应（如 /mcp/sse 的 SSE 长连接）会在转发响应体时遇到第二次
    http.response.start 而触发 AssertionError，导致 MCP 会话请求直接崩溃；
    这里直接包装 ASGI 应用，流式响应可以原样透传。
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        client = scope.get("client")
        client_host = client[0] if isinstance(client, (tuple, list)) and client else None
        # Host 头必须也是回环字面量，否则视为 DNS rebinding：域名解析到 127.0.0.1
        # 时客户端地址是回环、Host 却是攻击者域名，只查地址挡不住。
        host_header = None
        for key, value in scope.get("headers") or []:
            if key == b"host":
                host_header = value.decode("latin-1", "replace")
                break
        if not is_loopback_client(client_host) or not is_loopback_host_header(host_header):
            body = json.dumps(
                {"detail": "AINote 仅允许本机访问"}, ensure_ascii=False
            ).encode("utf-8")
            await send(
                {
                    "type": "http.response.start",
                    "status": 403,
                    "headers": [
                        (b"content-type", b"application/json; charset=utf-8"),
                        (b"content-length", str(len(body)).encode("latin-1")),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": body})
            return

        async def wrapped_send(message: dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                headers = message.get("headers", [])
                existing = {key.lower() for key, _value in headers}
                for key, value in SECURITY_HEADERS.items():
                    if key.lower() not in existing:
                        headers.append((key.encode("latin-1"), value.encode("latin-1")))
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, wrapped_send)


app.add_middleware(LocalSecurityMiddleware)


_STARTED_MONO = time.monotonic()
_LAST_BACKEND_ERRORS: deque[str] = deque(maxlen=5)


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """没兜住的异常一律回 JSON，而不是 uvicorn 那句纯文本 Internal Server Error。

    前端 `extractErrorMessage` 只在响应是 JSON 时才取得到原因，裸 500 会退化成
    "测试请求失败（HTTP 500）"——群里报障时既分不清是后端崩了、还是端口被别的
    服务占着，也拿不到任何线索。traceback 由 Starlette 原样继续上抛，仍会进
    `_app.log`；最近几条（脱敏后）留在 `_LAST_BACKEND_ERRORS`，`/api/diagnostics`
    会带出来当悬案线索。
    """
    try:
        reason = utf8_safe(f"{type(exc).__name__}: {exc}")[:300]
        reason = redact_secrets(reason)
        _LAST_BACKEND_ERRORS.append(f"{request.url.path}: {reason}")
    except Exception:
        # 兜错误的兜底自己也不能再炸：炸了就把话说到"有异常但构造不出说明"为止
        reason = "后端内部错误（构造错误说明时又出了岔子，详情见本机 _app.log）"
    return JSONResponse(
        status_code=500,
        content={"detail": f"后端内部错误：{reason}（详情见本机 _app.log）"},
    )


FRONTEND_MISSING_HTML = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>AINote 界面没加载出来</title></head>
<body style="font-family:system-ui,sans-serif;max-width:34em;margin:14vh auto;padding:0 1.2em;line-height:1.9">
<h1>界面没加载出来</h1>
<p><strong>这样做就好：</strong>在右下角任务栏找到 AINote 图标，右键选「退出」，
再双击打开程序。找不到那个图标的话，直接再双击一次程序也行。</p>
<p>要是反复出现，请把程序目录里 <code>workspace\\_app.log</code> 发给作者，
并说一下你地址栏里的端口号（这台机器上是 __PORT__）。</p>
<p style="color:#666">程序本身在正常运行，只是界面文件没读到——不是端口被别的软件占了。</p>
</body></html>
"""


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> Response:
    """从浏览器地址栏打开的 404 说人话；/api 与 /mcp 仍按原样回 JSON。

    打包版是 onefile，运行时把前端解压到临时目录。那个目录一旦被安全软件或磁盘清理
    删掉，`/api/*` 全部正常而 `/` 只剩一句 `{"detail":"Not Found"}`——连排查的人都会
    误判成端口问题，用户只能靠重启软件碰运气（重启会重新解压，所以"重启就好了"）。
    """
    if exc.status_code == 404 and not request.url.path.startswith(("/api", "/mcp")):
        port = (request.scope.get("server") or (None, None))[1]
        return HTMLResponse(
            status_code=404,
            content=FRONTEND_MISSING_HTML.replace("__PORT__", str(port or "未知")),
        )
    return JSONResponse(
        status_code=exc.status_code, content={"detail": exc.detail}, headers=exc.headers
    )


video_processor = VideoProcessor(WORKSPACE_DIR)
transcriber = WhisperTranscriber(WHISPER_CACHE_DIR)
paraformer_transcriber = ParaformerTranscriber(WHISPER_CACHE_DIR)


async def run_transcription(
    media_path: Path,
    model_name: str,
    use_gpu: bool,
    title: str | None,
    cancel_event: Any,
    progress_callback: Any = None,
) -> dict:
    """转写入口：打包版交给工人子进程，源码版在原进程里跑。

    CTranslate2 / sherpa-onnx 在打包环境里出过一次访问违例（崩的是整个进程，
    托盘和界面一起没，日志一个字都没留下），所以正式产物里的转写必须死在外头。
    """
    if asr_process.use_child_process() and not groq_asr.is_groq_model(model_name):
        return await asr_process.run_in_child(
            media_path, model_name, use_gpu, title, cancel_event, progress_callback
        )
    return await run_transcription_local(
        media_path, model_name, use_gpu, title, cancel_event, progress_callback
    )


async def run_transcription_local(
    media_path: Path,
    model_name: str,
    use_gpu: bool,
    title: str | None,
    cancel_event: Any,
    progress_callback: Any = None,
) -> dict:
    """按所选模型路由转写引擎：Groq 走云端，paraformer-zh 走 sherpa-onnx，其余走 faster-whisper。

    progress_callback(已转写秒数, 总秒数) 由 whisper 与 Groq 路径支持（paraformer
    一次性出结果，且 RTF ≈ 0.05 本来就不需要心跳）。
    """
    if groq_asr.is_groq_model(model_name):
        return await groq_asr.transcribe(
            media_path,
            model_name,
            groq_asr.resolve_api_key(config_store),
            cancel_event=cancel_event,
            progress_callback=progress_callback,
        )
    if model_name == "paraformer-zh":
        return await paraformer_transcriber.transcribe(
            media_path, model_name, use_gpu, initial_prompt=title, cancel_event=cancel_event
        )
    return await transcriber.transcribe(
        media_path,
        model_name,
        use_gpu,
        initial_prompt=title,
        cancel_event=cancel_event,
        progress_callback=progress_callback,
    )


def transcribe_progress_reporter(
    task: dict[str, Any], model_name: str, page_label: str = ""
) -> Any:
    """转写心跳回调：工作线程里只经 call_soon_threadsafe 回主循环改任务状态。

    set_progress 会写任务字典并落盘运行时状态，不能在转写线程里直接调。
    """
    loop = asyncio.get_running_loop()

    def _clock(seconds: float) -> str:
        seconds = max(0, int(seconds))
        return f"{seconds // 60:02d}:{seconds % 60:02d}"

    def report(position: float, duration: float) -> None:
        total = _clock(duration) if duration > 0 else "未知时长"
        message = (
            f"正在转写{page_label}音频（{model_name}）："
            f"已推进到 {_clock(position)} / {total}"
        )
        loop.call_soon_threadsafe(set_progress, task, 4, "语音转写", 38, message)

    return report


bili_login_manager = BiliLoginManager(WORKSPACE_DIR)
douyin_login_manager = DouyinLoginManager(WORKSPACE_DIR)
config_store = ConfigStore(WORKSPACE_DIR)
tasks: dict[str, dict[str, Any]] = {}
running_jobs: dict[str, asyncio.Task[None]] = {}
task_slots = asyncio.Semaphore(MAX_CONCURRENT_TASKS)


class BilibiliCookie(BaseModel):
    sessdata: str = ""
    bili_jct: str = ""
    buvid3: str = ""


class LLMConfigPayload(BaseModel):
    model_type: str = "deepseek"
    api_key: SecretStr
    base_url: str | None = None
    model: str | None = None
    label: str | None = None


class LLMKeyPayload(BaseModel):
    """按接口地址保存一把 API Key：provider + base_url 决定它属于哪个端点。"""

    api_key: SecretStr
    provider: str = "deepseek"
    base_url: str | None = None
    label: str | None = None
    model: str | None = None


class LLMTestPayload(BaseModel):
    """测试连接的可选覆盖参数。

    字段不再与已保存配置逐项混合（那会把一个端点的 Key 配到另一个端点上），
    只有 API Key 会在地址完全一致时复用本机已存的那一把。
    """

    model_type: str | None = None
    api_key: SecretStr | None = None
    base_url: str | None = None
    model: str | None = None


class BiliCredentialsPayload(BaseModel):
    sessdata: str = ""
    bili_jct: str = ""
    buvid3: str = ""


class DouyinCredentialsPayload(BaseModel):
    """手动粘贴的抖音 cookie（名称→值）。字段随平台变化，不逐字段建模。"""

    cookies: dict[str, str] = {}


class LLMConfig(BaseModel):
    model_type: str = "deepseek"
    api_key: SecretStr = SecretStr("")
    base_url: str | None = None
    model: str | None = None
    custom_base_url: str | None = None
    custom_model_name: str | None = None


class SproutRequest(BaseModel):
    """单篇「思考」：对整篇笔记跑一个模式，产出一份独立笔记。

    ``mode`` 决定跑哪个模式（发芽 / 拷问 / 自测），默认发芽以兼容旧调用。
    ``markdown`` 是整篇笔记正文（不按段落切片）；``title`` / ``note_path`` 用于
    把产物落成独立笔记并与原笔记用 Obsidian 双链互连。
    """

    mode: Literal["sprout", "interrogate", "quiz"] = "sprout"
    markdown: str
    title: str = ""
    note_path: str = ""
    llm_config: LLMConfig


class SproutAggregateRequest(BaseModel):
    """「整体发芽」：跨多篇历史笔记找共同种子。

    只传任务 ID——正文由后端自己取（优先内存里的 result.markdown，回退读任务
    目录的 notes.md），避免把好几篇正文在前后端之间来回搬。
    """

    task_ids: list[str]
    llm_config: LLMConfig


class SummarizeRequest(BaseModel):
    video_url: str = ""
    upload_task_id: str | None = None
    resume_task_id: str | None = None
    processing_mode: Literal["reuse", "restart"] = "reuse"
    # transcript：只做「视频 → 带时间轴转录」，不调用大模型，产物到 transcript.json 为止
    output: Literal["note", "transcript"] = "note"
    summary_style: Literal["detailed", "faithful", "concise", "keypoints"] = "detailed"
    # 内容场景：决定笔记按什么结构组织（教程/知识/学习），与风格正交；general = 历史行为
    content_type: Literal["general", "tutorial", "knowledge", "learning"] = "general"
    reasoning_effort: Literal["auto", "off", "high", "max"] = "auto"
    prefer_subtitles: bool = True
    include_screenshots: bool = False
    screenshot_interval: int = Field(default=30, ge=5, le=300)
    whisper_model: str = "base"
    use_gpu: bool = False
    # B 站分 P 显式选择（页码从 1 起）；None/空 = 按 URL ?p=N 规则（缺省全部）
    bilibili_pages: list[int] | None = None
    bilibili_cookie: BilibiliCookie | None = None
    # 抖音会话 cookie（名称→值）：ttwid/msToken/sessionid/Uifid ... 整份透传
    douyin_cookie: dict[str, str] | None = None
    llm_config: LLMConfig


class TranscribeRequest(BaseModel):
    """`POST /api/transcribe` 的请求体：刻意没有任何大模型字段。"""

    video_url: str = ""
    upload_task_id: str | None = None
    resume_task_id: str | None = None
    processing_mode: Literal["reuse", "restart"] = "reuse"
    prefer_subtitles: bool = True
    whisper_model: str = "base"
    use_gpu: bool = False
    # B 站分 P 显式选择（页码从 1 起）；None/空 = 按 URL ?p=N 规则（缺省全部）
    bilibili_pages: list[int] | None = None
    bilibili_cookie: BilibiliCookie | None = None
    # 抖音会话 cookie（名称→值）：ttwid/msToken/sessionid/Uifid ... 整份透传
    douyin_cookie: dict[str, str] | None = None


class BiliPagesRequest(BaseModel):
    """`POST /api/bili-pages` 的请求体：提交前预览 B 站分 P 清单。"""

    video_url: str
    bilibili_cookie: BilibiliCookie | None = None


class PatchTailRequest(BaseModel):
    """`POST /api/task/{id}/patch-tail` 的请求体：补写笔记缺失的结尾。

    不重跑整篇，只补缺口，因此只需要模型配置：风格、标题、来源、转录一律从任务
    记录里取，避免前端把当时的设置再抄一遍（抄错就补出风格不一致的段落）。
    推理档位默认沿用任务记录，允许显式覆盖——补写是"接着写一段"，通常关掉思考更快。
    """

    llm_config: LLMConfig
    reasoning_effort: Literal["auto", "off", "high", "max"] | None = None


def new_task(status: str = "pending", task_id: str | None = None) -> dict[str, Any]:
    return {
        "status": status,
        "step": 0,
        "step_name": "初始化",
        "progress_message": "等待任务开始",
        "progress": 0,
        "logs": [],
        "result": None,
        "error": None,
        "advisory": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "elapsed_seconds": 0.0,
        "_task_id": task_id,
        "_cancel_requested": False,
        "_cancel_event": threading.Event(),
        "_started_monotonic": time.monotonic(),
    }


def task_elapsed_seconds(task: dict[str, Any]) -> float:
    if task.get("finished_at"):
        return float(task.get("elapsed_seconds") or 0)
    started = task.get("_started_monotonic")
    return max(0.0, time.monotonic() - float(started)) if started is not None else 0.0


def finish_task_timing(task: dict[str, Any]) -> float:
    elapsed = task_elapsed_seconds(task)
    task["elapsed_seconds"] = elapsed
    task["finished_at"] = datetime.now(timezone.utc).isoformat()
    return elapsed


def set_progress(
    task: dict[str, Any], step: int, name: str, progress: int, message: str
) -> None:
    task.update(step=step, step_name=name, progress=progress, progress_message=message)
    task["logs"].append(message)
    task_id = task.get("_task_id")
    if task_id:
        persist_task_runtime(str(task_id))


def task_directory(task_id: str) -> Path:
    candidate = (WORKSPACE_DIR / task_id).resolve()
    if candidate.parent != WORKSPACE_DIR:
        raise ValueError("非法任务 ID")
    return candidate


def persist_task_runtime(task_id: str) -> None:
    """Persist non-secret runtime state atomically beside task artifacts."""
    task = tasks.get(task_id)
    if not task:
        return
    directory = task_directory(task_id)
    if not directory.is_dir():
        return
    path = directory / "task.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except (OSError, ValueError, TypeError):
        payload = {"task_id": task_id}
    result = task.get("result")
    result_metadata = None
    if isinstance(result, dict):
        result_metadata = {key: value for key, value in result.items() if key != "markdown"}
    payload["runtime"] = {
        "status": task.get("status"),
        "step": task.get("step", 0),
        "step_name": task.get("step_name", ""),
        "progress": task.get("progress", 0),
        "logs": list(task.get("logs", []))[-200:],
        "error": task.get("error"),
        "created_at": task.get("created_at"),
        "finished_at": task.get("finished_at"),
        "elapsed_seconds": task_elapsed_seconds(task),
        "result": result_metadata,
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def raise_if_cancel_requested(task: dict[str, Any]) -> None:
    if task.get("_cancel_requested"):
        raise asyncio.CancelledError


# 任务结果系统通知：由启动器把托盘气泡接进来（打包模式才有效），
# backend 不直接依赖 pystray，避免引入循环依赖。
_task_notify_hook: Callable[[str, str], None] | None = None


def register_task_notify(callback: Callable[[str, str], None]) -> None:
    """注册通知回调：callback(title, message)。"""
    global _task_notify_hook
    _task_notify_hook = callback


def notify_task(title: str, message: str) -> None:
    """发送系统通知；异步执行且绝不影响任务主流程。"""
    callback = _task_notify_hook
    if callback is None:
        return

    def _run() -> None:
        try:
            callback(title, message)
        except Exception:
            pass

    threading.Thread(target=_run, daemon=True).start()


def normalize_source_url(value: str) -> str:
    parsed = urlsplit(value.strip())
    hostname = (parsed.hostname or "").lower()
    netloc = hostname
    if parsed.port:
        netloc = f"{hostname}:{parsed.port}"
    path = parsed.path.rstrip("/")
    query = "" if hostname == "bilibili.com" or hostname.endswith(".bilibili.com") else parsed.query
    return urlunsplit((parsed.scheme.lower(), netloc, path, query, ""))


def source_url_from_task(task_dir: Path) -> str | None:
    manifest_path = task_dir / "task.json"
    if manifest_path.is_file():
        try:
            value = json.loads(manifest_path.read_text(encoding="utf-8")).get("source_url")
            if value:
                return str(value)
        except (OSError, ValueError, TypeError):
            pass

    notes_path = task_dir / "notes.md"
    if notes_path.is_file():
        try:
            for line in reversed(notes_path.read_text(encoding="utf-8").splitlines()):
                if line.startswith("来源："):
                    return line.removeprefix("来源：").strip()
        except OSError:
            pass
    return None


def find_reusable_task(source_url: str) -> str | None:
    normalized_source = normalize_source_url(source_url)
    candidates = sorted(
        (path for path in WORKSPACE_DIR.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        if not (candidate / "transcript.json").is_file():
            continue
        existing_source = source_url_from_task(candidate)
        if existing_source and normalize_source_url(existing_source) == normalized_source:
            return candidate.name
    return None


def load_transcript_result(task_dir: Path) -> dict[str, Any] | None:
    path = task_dir / "transcript.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        segments = [
            TranscriptSegment(float(item["start"]), float(item["end"]), str(item["text"]))
            for item in payload.get("segments", [])
            if str(item.get("text", "")).strip()
        ]
    except (OSError, ValueError, TypeError, KeyError):
        return None
    if not segments:
        return None
    return {
        "segments": segments,
        "language": payload.get("language") or "unknown",
        "source": payload.get("source") or "reused_transcript",
        "quality": payload.get("quality"),
    }


def load_reused_video_info(
    task_dir: Path,
    transcript_result: dict[str, Any],
    source_url: str | None,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {}
    manifest_path = task_dir / "task.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            pass

    title = str(manifest.get("title") or "").strip()
    if not title:
        notes_path = task_dir / "notes.md"
        try:
            first_line = notes_path.read_text(encoding="utf-8").splitlines()[0]
            if first_line.startswith("# 视频笔记：《") and first_line.endswith("》"):
                title = first_line.removeprefix("# 视频笔记：《")[:-1]
        except (OSError, IndexError):
            pass
    if not title:
        title = f"已恢复任务 {task_dir.name}"

    segments: list[TranscriptSegment] = transcript_result["segments"]
    duration = float(manifest.get("duration") or max(segment.end for segment in segments))
    if source_url:
        source = video_processor.detect_source(source_url).value
    else:
        source = VideoSource.LOCAL.value
    return {
        "title": title,
        "source": source,
        "duration": duration,
        "owner": str(manifest.get("owner") or ""),
        "upload_date": str(manifest.get("upload_date") or ""),
        "timestamp": manifest.get("timestamp") or 0,
        "view_count": manifest.get("view_count") or 0,
        "like_count": manifest.get("like_count") or 0,
        "bili_pages": manifest.get("bili_pages"),
        "bili_total_pages": manifest.get("bili_total_pages"),
        "description": "",
    }


def update_task_manifest(task_id: str, info: dict[str, Any]) -> None:
    path = task_directory(task_id) / "task.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        payload = {"task_id": task_id}
    payload.update(
        title=info.get("title"),
        owner=info.get("owner"),
        duration=info.get("duration"),
        source=info.get("source"),
        upload_date=info.get("upload_date"),
        timestamp=info.get("timestamp"),
        view_count=info.get("view_count"),
        like_count=info.get("like_count"),
    )
    # 分 P 信息只在 B 站多 P 时才有，缺键的旧任务重开时不能把它写空
    for key in ("bili_pages", "bili_total_pages"):
        if info.get(key) is not None:
            payload[key] = info[key]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_task_manifest(task_id: str) -> dict[str, Any]:
    path = task_directory(task_id) / "task.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def write_task_manifest(
    task_id: str,
    request: SummarizeRequest,
    reused_task_id: str | None,
    origin: str,
) -> None:
    uploaded_filename = tasks.get(task_id, {}).get("uploaded_filename")
    payload = {
        "task_id": task_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_url": request.video_url if request.video_url and not uploaded_filename else None,
        "normalized_source_url": (
            normalize_source_url(request.video_url)
            if request.video_url and not uploaded_filename
            else None
        ),
        "uploaded_filename": uploaded_filename,
        "processing_mode": request.processing_mode,
        "output": request.output,
        # origin=agent 的纯转录任务不进网页端历史（见 list_recent_tasks）
        "origin": origin,
        "summary_style": request.summary_style,
        "content_type": getattr(request, "content_type", "general") or "general",
        "reasoning_effort": request.reasoning_effort,
        "reused_task_id": reused_task_id,
        # 分 P 勾选随任务落盘：同一视频重新处理时，/bili-pages 用它预勾上次的范围
        "bilibili_pages": getattr(request, "bilibili_pages", None),
    }
    (task_directory(task_id) / "task.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def write_upload_manifest(task_id: str, filename: str) -> None:
    payload = {
        "task_id": task_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_url": None,
        "uploaded_filename": filename,
    }
    (task_directory(task_id) / "task.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


@app.post("/api/summarize")
async def start_summarize(request: SummarizeRequest) -> dict[str, str | None]:
    return await submit_video_task(request, origin="web")


@app.post("/api/transcribe")
async def start_transcribe(request: TranscribeRequest) -> dict[str, str | None]:
    """只做「视频 → 带时间轴转录」，全程不调用大模型，本机没有 API Key 也能用。

    给自带模型的 agent 用：拿原料自己组织笔记，不必先给本机配一把 Key。
    复用同一套任务机制（进度、取消、并发闸门、产物复用），产物停在
    `transcript.json` / `transcript.md`；正文用 `GET /api/task/{id}/transcript` 取。
    """
    return await submit_video_task(
        SummarizeRequest(
            **request.model_dump(), output="transcript", llm_config=LLMConfig()
        ),
        origin="agent",
    )


async def submit_video_task(
    request: SummarizeRequest, origin: str
) -> dict[str, str | None]:
    if request.whisper_model not in ASR_MODELS:
        raise HTTPException(status_code=422, detail="不支持的转写模型")
    uploaded_task = None
    if request.upload_task_id:
        # 本地上传的媒体位置记在上传任务上；此时 video_url 是后端目录里的文件路径，
        # 不是链接，不能进链接校验（否则必 422）。
        uploaded_task = tasks.get(request.upload_task_id)
        if not uploaded_task or uploaded_task.get("status") != "uploaded":
            raise HTTPException(status_code=400, detail="上传任务不存在或已经处理")
        request.video_url = ""
    elif request.video_url:
        normalized = normalize_video_input(request.video_url)
        if not normalized:
            raise HTTPException(
                status_code=422,
                detail="无法识别视频链接：支持 http(s) 链接，或包含 B 站/抖音链接的分享文本",
            )
        request.video_url = normalized
    reused_task_id = request.resume_task_id
    if request.processing_mode == "restart":
        reused_task_id = None
    elif not reused_task_id and request.video_url:
        reused_task_id = find_reusable_task(request.video_url)

    if reused_task_id:
        resume_dir = task_directory(reused_task_id)
        if not resume_dir.is_dir():
            raise HTTPException(status_code=404, detail="要继续的任务目录不存在")

    task_id = request.upload_task_id or str(uuid.uuid4())
    if uploaded_task is not None:
        tasks[task_id] = new_task(task_id=task_id)
        tasks[task_id]["uploaded_file_path"] = uploaded_task.get("uploaded_file_path")
        tasks[task_id]["uploaded_filename"] = uploaded_task.get("uploaded_filename")
    else:
        if request.video_url:
            try:
                video_processor.detect_source(request.video_url)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
        elif not reused_task_id:
            raise HTTPException(status_code=422, detail="请提供视频链接、上传文件或可继续的任务")
        tasks[task_id] = new_task(task_id=task_id)
        task_directory(task_id).mkdir(parents=True, exist_ok=False)

    tasks[task_id]["resume_task_id"] = reused_task_id
    tasks[task_id]["output"] = request.output
    tasks[task_id]["_origin"] = origin
    if reused_task_id:
        old_task = tasks.get(reused_task_id, {})
        if old_task.get("uploaded_file_path"):
            tasks[task_id]["uploaded_file_path"] = old_task["uploaded_file_path"]
        old_filename = old_task.get("uploaded_filename") or read_task_manifest(
            reused_task_id
        ).get("uploaded_filename")
        if old_filename:
            tasks[task_id]["uploaded_filename"] = old_filename
        tasks[task_id]["logs"].append(f"将从任务 {reused_task_id} 复用已有产物")
    write_task_manifest(task_id, request, reused_task_id, origin)
    persist_task_runtime(task_id)

    job = asyncio.create_task(run_queued_video_task(task_id, request))
    running_jobs[task_id] = job
    job.add_done_callback(lambda _job, current_id=task_id: running_jobs.pop(current_id, None))
    return {"task_id": task_id, "reused_task_id": reused_task_id}


def task_status_payload(task_id: str, task: dict[str, Any]) -> dict[str, Any]:
    payload = {key: value for key, value in task.items() if not key.startswith("_")}
    payload["elapsed_seconds"] = task_elapsed_seconds(task)
    manifest = read_task_manifest(task_id)
    source_url = str(manifest.get("source_url") or "")
    payload["source_url"] = source_url if source_url.startswith(("http://", "https://")) else None
    payload["uploaded_filename"] = payload.get("uploaded_filename") or manifest.get(
        "uploaded_filename"
    )
    result = task.get("result") if isinstance(task.get("result"), dict) else {}
    payload["download_name"] = download_stem(
        str(
            result.get("title")
            or manifest.get("title")
            or manifest.get("uploaded_filename")
            or ""
        ),
        manifest,
        str(task.get("output") or manifest.get("output") or "note"),
    )
    return payload


@app.get("/api/tasks")
async def list_recent_tasks(limit: int = TASK_PAGE_SIZE) -> dict[str, Any]:
    # agent 经 /api/transcribe 提交的纯转录任务不进网页端历史：它没有 result.markdown，
    # 而且批量取原料不应该把用户的笔记历史挤出列表（过滤在截断之前）。
    # 网页端自己点的「仅转录」是用户要回看的成品，照常进历史（origin=web）。
    def is_visible(task: dict[str, Any]) -> bool:
        if task.get("output") != "transcript":
            return True
        return task.get("_origin") == "web"

    # 分页只在这里截断：网页端点「加载更多」就把 limit 调大，一次多给一页。
    # 同时返回 total，前端才知道还有没有更多可加载（否则只能靠"给满一页就猜还有"）。
    ordered_all = sorted(
        (item for item in tasks.items() if is_visible(item[1])),
        key=lambda item: str(item[1].get("created_at") or ""),
        reverse=True,
    )
    # limit 尊重调用方给的值（只挡住 0/负数与过大的值）：给 5 就回 5，
    # 悄悄回一页满的会让调试的人怀疑人生。前端「加载更多」永远按页递增。
    size = max(1, min(int(limit or TASK_PAGE_SIZE), TASK_PAGE_MAX))
    ordered = ordered_all[:size]
    summaries: list[dict[str, Any]] = []
    for task_id, task in ordered:
        manifest = read_task_manifest(task_id)
        result = task.get("result") if isinstance(task.get("result"), dict) else {}
        summaries.append(
            {
                "task_id": task_id,
                "status": task.get("status"),
                "step_name": task.get("step_name"),
                "progress": task.get("progress", 0),
                "output": task.get("output") or manifest.get("output") or "note",
                "title": result.get("title") or manifest.get("title") or manifest.get("uploaded_filename") or "未命名任务",
                "created_at": task.get("created_at"),
                "finished_at": task.get("finished_at"),
                "elapsed_seconds": task_elapsed_seconds(task),
            }
        )
    return {"tasks": summaries, "total": len(ordered_all), "limit": size}


@app.get("/api/task/{task_id}")
async def get_task_status(task_id: str) -> dict[str, Any]:
    task = tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在或已被清理")
    return task_status_payload(task_id, task)


@app.get("/api/task/{task_id}/transcript")
async def get_task_transcript(
    task_id: str, output_format: Literal["json", "markdown"] = "json"
) -> dict[str, Any]:
    """取该任务的带时间轴转录（`segments` 的时间单位是秒）。

    只看转录产物在不在，不看任务状态：即便笔记在第 6 步失败，已完成的转录
    照样能取走，不必重跑。`markdown` 返回人读版整篇文本，`json` 返回分段。
    """
    try:
        directory = task_directory(task_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="非法任务 ID") from exc
    json_path = directory / "transcript.json"
    if not json_path.is_file():
        raise HTTPException(
            status_code=409, detail="该任务还没有转录产物，可先调用 POST /api/transcribe"
        )
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise HTTPException(status_code=500, detail="转录文件无法读取") from None

    task = tasks.get(task_id) or {}
    result = task.get("result") if isinstance(task.get("result"), dict) else {}
    manifest = read_task_manifest(task_id)
    title = str(
        result.get("title") or manifest.get("title") or manifest.get("uploaded_filename") or ""
    )

    if output_format == "markdown":
        markdown_path = directory / "transcript.md"
        if not markdown_path.is_file():
            raise HTTPException(status_code=409, detail="转录 Markdown 不存在")
        return {
            "task_id": task_id,
            "title": title,
            "format": "markdown",
            "text": markdown_path.read_text(encoding="utf-8"),
        }

    segments = payload.get("segments") or []
    return {
        "task_id": task_id,
        "title": title,
        "format": "json",
        "language": payload.get("language"),
        "transcript_source": payload.get("source"),
        "transcript_quality": payload.get("quality"),
        "segment_count": len(segments),
        "segments": segments,
    }


@app.get("/api/whisper-models")
async def whisper_models_status() -> dict[str, Any]:
    """返回各 Whisper 模型在本机的缓存状态（供前端标注下拉框）。"""
    models = [
        {
            "id": model_id,
            "status": transcriber._model_cache_status(model_id),
        }
        for model_id in sorted(WHISPER_MODELS)
    ]
    models.append({"id": "paraformer-zh", "status": paraformer_transcriber.cache_status()})
    return {"models": models, "manual_dir": str(WHISPER_CACHE_DIR / "manual")}


class WhisperManualFolderPayload(BaseModel):
    model: str
    open: bool = True


def _open_in_file_manager(path: Path) -> bool:
    """用系统文件管理器打开目录；失败时返回 False（前端展示路径兜底）。"""
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
        return True
    except Exception:
        return False


PARAFORMER_IMPORT_LABELS = {
    "asr": "转写主模型",
    "punc": "标点模型（可以不放：缺了会自动降级为无标点转写）",
    "vad": "语音活动检测",
}


@app.post("/api/whisper-models/manual-folder")
async def open_whisper_manual_folder(payload: WhisperManualFolderPayload) -> dict[str, Any]:
    """创建并返回手动导入模型的引导（大模型网络下载失败时的替代方案）。

    open=false 只给引导信息，供前端先展示步骤、用户确认后再开文件夹。
    一条模型可能由多个组件目录组成（paraformer-zh 是 asr / punc / vad 三份，
    而且和 faster-whisper 系根本不是一个缓存目录），所以引导统一返回 folders 列表。
    """
    if payload.model not in ASR_MODELS:
        raise HTTPException(status_code=422, detail="不支持的转写模型")

    if payload.model == "paraformer-zh":
        folders = []
        for component, spec in paraformer_asr.MODEL_SPECS.items():
            directory = paraformer_transcriber._component_dir(component)
            directory.mkdir(parents=True, exist_ok=True)
            folders.append(
                {
                    "label": PARAFORMER_IMPORT_LABELS[component],
                    "path": str(directory),
                    "files": list(spec["files"]),
                    "download_url": f"https://hf-mirror.com/{spec['repo']}/tree/main",
                }
            )
        open_path = paraformer_transcriber.download_root
    else:
        manual_dir = WHISPER_CACHE_DIR / "manual" / payload.model
        manual_dir.mkdir(parents=True, exist_ok=True)
        folders = [
            {
                "label": "模型文件",
                "path": str(manual_dir),
                "files": transcriber.manual_import_files(payload.model),
                "download_url": f"https://hf-mirror.com/{transcriber._model_repo(payload.model)}/tree/main",
            }
        ]
        open_path = manual_dir

    return {
        "folders": folders,
        "open_path": str(open_path),
        "opened": _open_in_file_manager(open_path) if payload.open else False,
    }


def _key_storage_info() -> dict[str, Any]:
    return {"algorithm": secret_box.storage_backend(), "secure": secret_box.is_secure_storage()}


def resolve_llm_credentials(*, model_type: str, base_url: str | None, api_key: str) -> tuple[str, str]:
    """显式 Key 永远优先；否则只按接口地址复用本机已存 Key。

    地址不一致就直接失败，不会"先拿本机某把 Key 试试"——那正是把 DeepSeek 的
    Key 发给第三方网关的路径。返回 ``(api_key, 来源说明)``，来源说明进任务日志。
    """
    explicit = (api_key or "").strip()
    if explicit:
        return explicit, "本次请求提供"
    resolved = config_store.resolve_stored_key(model_type=model_type, base_url=base_url or "")
    secret = str(resolved.get("api_key") or "")
    if secret:
        entry = resolved.get("entry") or {}
        where = str(entry.get("label") or resolved.get("host") or "")
        masked = str(entry.get("api_key_masked") or "")
        return secret, f"本机已存 {masked}（{where}）"
    raise RuntimeError(_missing_key_message(model_type, base_url, resolved))


def _missing_key_message(model_type: str, base_url: str | None, resolved: dict[str, Any]) -> str:
    target = (base_url or "").strip() or default_base_url(model_type) or "未指定接口地址"
    reason = str(resolved.get("reason") or "")
    detail = {
        "undecryptable": (
            f"为 {target} 保存的 Key 无法解密（可能来自其他机器或其他 Windows 账户），"
            "请重新填写并保存"
        ),
        "corrupt_keys_file": (
            f"本机密钥文件 {LLM_KEYS_FILE} 已损坏，请在模型配置页清除本机配置后重新保存"
        ),
        "unknown_endpoint": f"接口地址无法识别（{target}），不会复用任何本机 Key",
    }.get(reason) or f"本机没有为 {target} 保存 Key"
    return (
        f"未提供 API Key，{detail}。"
        "请在模型配置页填写 API Key 并「保存到本机」，或在请求中直接带上 api_key。"
    )


@app.get("/api/llm-keys")
async def get_llm_keys(base_url: str = "", provider: str = "") -> dict[str, Any]:
    """列出本机保存的接口 Key（只有掩码）；带 base_url 时顺带回答"这个地址存了没有"。"""
    if config_store.keys_are_corrupt():
        return {"storage": _key_storage_info(), "error": "corrupt_keys_file", "entries": []}
    payload: dict[str, Any] = {"storage": _key_storage_info(), "entries": config_store.list_keys()}
    if base_url or provider:
        resolved = config_store.resolve_stored_key(model_type=provider, base_url=base_url)
        entry = resolved.get("entry")
        key_state = "none"
        if entry:
            key_state = "saved" if resolved.get("api_key") else "undecryptable"
        payload["match"] = {
            "saved": bool(entry),
            "key_state": key_state,
            "endpoint": normalize_endpoint_host(base_url, provider),
            **(entry or {}),
        }
    return payload


@app.put("/api/llm-keys")
async def put_llm_key(payload: LLMKeyPayload) -> dict[str, Any]:
    """显式保存：把某个接口地址的 API Key 加密写入本机。清除只能走 DELETE。"""
    try:
        entry = config_store.put_key(
            provider=payload.provider,
            base_url=payload.base_url or "",
            api_key=payload.api_key.get_secret_value(),
            label=payload.label,
            model=payload.model,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except secret_box.SecretBoxError as error:
        raise HTTPException(status_code=500, detail=f"本机加密保存失败：{error}") from error
    return {"saved": True, **entry}


@app.delete("/api/llm-keys")
async def delete_llm_key(base_url: str = "", provider: str = "") -> dict[str, Any]:
    return {"removed": config_store.delete_key(base_url=base_url, model_type=provider)}


@app.get("/api/llm-config")
async def get_llm_config() -> dict[str, Any]:
    """旧接口：报告本机最近保存的一条配置（含掩码，不含 Key 原文），供 MCP 免传参复用。"""
    entry = config_store.most_recent_entry()
    if not entry or not entry["has_key"]:
        return {"saved": False}
    return {
        "saved": True,
        "model_type": entry["provider"],
        "model": entry["model"],
        "base_url": entry["base_url"] or None,
        "api_key_masked": entry["api_key_masked"] or None,
    }


@app.post("/api/llm-config")
async def save_llm_config(payload: LLMConfigPayload) -> dict[str, Any]:
    """旧接口：等价于按接口地址保存 Key（MCP 的 save_llm_config 仍走这里）。"""
    try:
        entry = config_store.put_key(
            provider=payload.model_type,
            base_url=payload.base_url or "",
            api_key=payload.api_key.get_secret_value(),
            label=payload.label,
            model=payload.model,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except secret_box.SecretBoxError as error:
        raise HTTPException(status_code=500, detail=f"本机加密保存失败：{error}") from error
    return {
        "saved": True,
        "model_type": entry["provider"],
        "model": entry["model"],
        "label": entry["label"],
    }


@app.post("/api/llm-test")
async def test_llm_connection(
    payload: LLMTestPayload | None = None,
) -> dict[str, Any]:
    """用（可选覆盖的）LLM 配置做一次轻量连通测试。"""
    body = payload or LLMTestPayload()
    model_type = body.model_type or "deepseek"
    base_url = body.base_url or default_base_url(model_type) or None
    try:
        api_key, key_source = resolve_llm_credentials(
            model_type=model_type,
            base_url=base_url,
            api_key=body.api_key.get_secret_value() if body.api_key else "",
        )
    except RuntimeError as error:
        return {"ok": False, "error": str(error)}
    except Exception as error:
        # 密钥档案读不了（磁盘被拖垮 / DPAPI 失灵 / 文件损坏）不该裸 500：
        # 1.4.2 悬案里前端只剩「（HTTP 500）」，就是这类异常漏出去了
        return {
            "ok": False,
            "error": utf8_safe(f"读取本机保存的 Key 失败：{type(error).__name__}: {error}"),
        }
    try:
        summarizer = LLMSummarizer(
            model_type=model_type,
            api_key=api_key,
            base_url=base_url,
            model=body.model,
        )
        ok, message, latency = await summarizer.test_connection()
        return {
            "ok": ok,
            "message": message,
            "latency_ms": int(round(latency * 1000)),
            "model": body.model or summarizer.model,
            "base_url": summarizer.base_url,
            "key_source": key_source,
        }
    except Exception as exc:
        return {"ok": False, "error": utf8_safe(f"初始化失败：{exc}")}


@app.post("/api/knowledge/sprout")
async def knowledge_sprout(request: SproutRequest) -> dict[str, Any]:
    """对整篇笔记跑一个「思考」模式：发芽 / 拷问 / 自测，并落成独立笔记。

    与生成笔记共用同一套 LLM 配置解析与脱敏，但不进任务队列：它是同步交互，
    直接返回正文。Key 来源与生成笔记一致（显式 Key 优先，其次复用本机已存）。

    产物写成 ``<原笔记名><模式后缀>.md``，放在原笔记同目录（即 Obsidian 库），
    并在正文里写 ``[[原笔记名]]``——Obsidian 的反向链接会自动把两者连起来。
    """
    material = request.markdown.strip()
    if not material:
        raise HTTPException(status_code=400, detail="笔记内容为空，无法思考")
    mode = request.mode
    spec = THINK_MODES[mode]
    config = request.llm_config
    base_url = config.base_url or config.custom_base_url
    model = config.model or config.custom_model_name
    try:
        api_key, key_source = resolve_llm_credentials(
            model_type=config.model_type,
            base_url=base_url,
            api_key=config.api_key.get_secret_value(),
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    summarizer = LLMSummarizer(
        model_type=config.model_type,
        api_key=api_key,
        base_url=base_url,
        model=model,
    )
    runner = {
        "sprout": summarizer.sprout_report,
        "interrogate": summarizer.interrogate_report,
        "quiz": summarizer.quiz_report,
    }[mode]
    try:
        text = await runner(material)
    except Exception as exc:
        detail = redact_secrets(utf8_safe(str(exc)), api_key)
        raise HTTPException(status_code=502, detail=f"{spec['name']}失败：{detail}") from exc

    report_path, note_stem = write_think_report(text, request.title, request.note_path, mode)
    linked = (
        link_sprout_from_note(
            request.note_path, report_path.stem, label=spec["link_label"]
        )
        if report_path
        else False
    )
    return {
        "ok": True,
        "mode": mode,
        "mode_name": spec["name"],
        "text": text,
        "report_path": str(report_path) if report_path else "",
        "note_stem": note_stem,
        "linked": linked,
        "key_source": key_source,
    }


@app.post("/api/knowledge/sprout-aggregate")
async def knowledge_sprout_aggregate(request: SproutAggregateRequest) -> dict[str, Any]:
    """整体发芽：跨多篇历史笔记找**共同种子**，产出一张新知卡片写进知识库。

    与单篇发芽同一套 LLM 解析 / 脱敏 / 落盘 / 双链逻辑，区别只在素材是好几篇，
    而且多一步「先找共同种子」。这几篇没有共同内核时模型不给卡片标题，这时
    **不落盘**，只把话说清楚返回给前端。
    """
    if not request.task_ids:
        raise HTTPException(status_code=400, detail="请先勾选要做整体发芽的笔记")
    config = request.llm_config
    base_url = config.base_url or config.custom_base_url
    model = config.model or config.custom_model_name
    try:
        api_key, key_source = resolve_llm_credentials(
            model_type=config.model_type,
            base_url=base_url,
            api_key=config.api_key.get_secret_value(),
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    notes = collect_aggregate_notes(request.task_ids)
    if not notes:
        raise HTTPException(
            status_code=400,
            detail="勾选的任务里没有可用的笔记正文（字幕稿没有正文，无法发芽）",
        )

    summarizer = LLMSummarizer(
        model_type=config.model_type,
        api_key=api_key,
        base_url=base_url,
        model=model,
    )
    try:
        text = await summarizer.sprout_aggregate(
            build_aggregate_material(notes), len(notes)
        )
    except Exception as exc:
        detail = redact_secrets(utf8_safe(str(exc)), api_key)
        raise HTTPException(status_code=502, detail=f"整体发芽失败：{detail}") from exc

    report_path, seed = write_aggregate_sprout_report(text, notes)
    return {
        "ok": True,
        "text": text,
        "report_path": str(report_path) if report_path else "",
        "seed": seed,
        "used_count": len(notes),
        "linked_count": sum(1 for note in notes if note.get("stem")),
        "notes": [
            {
                "task_id": note["task_id"],
                "title": note["title"],
                "in_vault": bool(note.get("stem")),
            }
            for note in notes
        ],
        "key_source": key_source,
    }


@app.delete("/api/llm-config")
async def clear_llm_config() -> dict[str, Any]:
    """旧接口：只清除它刚报告的那一条，不会一次清空全部本机 Key。"""
    entry = config_store.most_recent_entry()
    removed = bool(entry) and config_store.delete_key(
        base_url=entry["base_url"], model_type=entry["provider"]
    )
    return {"saved": False, "removed": removed}


@app.get("/api/bili-credentials")
async def get_bili_credentials() -> dict[str, Any]:
    """Return saved Bilibili credential status without exposing cookie values.

    走 ``bili_credentials_status()``：只用保存时写死的掩码，全程不解密，
    所以换机器解不开时这个端点仍然有得报。
    """
    return config_store.bili_credentials_status()


@app.post("/api/bili-credentials")
async def save_bili_credentials(payload: BiliCredentialsPayload) -> dict[str, Any]:
    try:
        config_store.save_bili_credentials(
            {"sessdata": payload.sessdata, "bili_jct": payload.bili_jct, "buvid3": payload.buvid3}
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except secret_box.SecretBoxError as error:
        raise HTTPException(status_code=500, detail=f"本机加密保存失败：{error}") from error
    return {"saved": True}


@app.delete("/api/bili-credentials")
async def clear_bili_credentials() -> dict[str, bool]:
    config_store.clear_bili_credentials()
    return {"saved": False}


@app.get("/api/douyin-credentials")
async def get_douyin_credentials() -> dict[str, Any]:
    """抖音凭据状态：只报字段名与数量，不返回 cookie 值。"""
    return config_store.douyin_credentials_status()


@app.post("/api/douyin-credentials")
async def save_douyin_credentials(payload: DouyinCredentialsPayload) -> dict[str, Any]:
    try:
        config_store.save_douyin_credentials(payload.cookies)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except secret_box.SecretBoxError as error:
        raise HTTPException(status_code=500, detail=f"本机加密保存失败：{error}") from error
    return {"saved": True}


@app.delete("/api/douyin-credentials")
async def clear_douyin_credentials() -> dict[str, bool]:
    config_store.clear_douyin_credentials()
    return {"saved": False}


def previous_bili_pages(normalized_url: str) -> list[int] | None:
    """该视频最近一次任务显式勾选的分 P（新任务在前扫描，没有则 None）。

    重新处理多分 P 视频时，面板默认沿用上次的范围，不用重新勾一遍；
    只认显式下发过的 bilibili_pages（None 表示当时的「全部」，不预勾）。
    """
    for candidate in sorted(
        (path for path in WORKSPACE_DIR.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    ):
        manifest_path = candidate / "task.json"
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if manifest.get("normalized_source_url") != normalized_url:
            continue
        pages = manifest.get("bilibili_pages")
        if isinstance(pages, list) and pages:
            return [int(page) for page in pages]
    return None


@app.post("/api/bili-pages")
async def preview_bilibili_pages(request: BiliPagesRequest) -> dict[str, Any]:
    """提交任务前预览 B 站分 P 清单，让用户显式勾选要转写的分 P。

    只读 view 接口，不触发下载；凭据可选（公开接口一般无需登录，
    携带凭据可降低风控概率）。front 端拿 url 的 ?p= 自行决定默认勾选。
    """
    url = normalize_video_input(request.video_url) or request.video_url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="请先粘贴 B 站视频链接")
    try:
        source = video_processor.detect_source(url)
    except ValueError:
        source = None
    if source != VideoSource.BILIBILI:
        raise HTTPException(status_code=400, detail="分 P 预览仅支持 B 站视频链接")
    if request.bilibili_cookie:
        cookie: dict[str, str] | None = request.bilibili_cookie.model_dump()
    else:
        try:
            cookie = config_store.load_bili_credentials()
        except BiliCredentialsUnavailable:
            cookie = None
    video = await video_processor.bilibili_video_pages(url, cookie)
    if video is None:
        raise HTTPException(
            status_code=400, detail="未能从链接解析出 B 站视频（BV/av 号）"
        )
    if not video.pages:
        raise HTTPException(
            status_code=502,
            detail="B 站接口未返回分 P 信息（网络或风控异常），可稍后重试或填写凭据",
        )
    return {
        "title": video.title,
        "total_pages": len(video.pages),
        "pages": [
            {"page": page.page, "part": page.part, "duration": page.duration}
            for page in video.pages
        ],
        # 该视频上次任务显式勾选的分 P；前端拿它做默认勾选（优先于 ?p=N）
        "previous_pages": previous_bili_pages(normalize_source_url(url)),
    }


@app.get("/api/health")
async def health_check() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "AINote",
        "version": app.version,
        # portable/dev 互不复用对方实例，避免 dev 服务占用端口导致打包版不驻留托盘
        "mode": "portable" if getattr(sys, "frozen", False) else "dev",
        # 本机保存的 API Key 是加密还是明文回退，答疑时看一眼健康接口即可确认
        "llm_key_storage": secret_box.storage_backend(),
        # 前端上传上限跟随本机配置，不再自己写死一个数字
        "max_upload_mb": MAX_UPLOAD_MB,
        "dependencies": {
            "yt_dlp": importlib.util.find_spec("yt_dlp") is not None,
            "faster_whisper": importlib.util.find_spec("faster_whisper") is not None,
            "openai": importlib.util.find_spec("openai") is not None,
        },
    }


@app.get("/api/update/check")
async def check_app_update() -> dict[str, Any]:
    """更新检查已按需移除：本机 GitHub 被 /etc/hosts 屏蔽，该端点只会白等超时。

    保留端点本身并直接回「无更新」，是为了不把前端的旧调用变成 404 噪音。
    """
    return {"update_available": False, "disabled": True, "latest": None}


def _tcp_reachable(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _http_probe(url: str, timeout: float = 4.0, *, use_proxy: bool) -> dict[str, Any]:
    """只看"HTTP 层有没有应答"的探测：任何状态码都算通，应答体截一小段当证据。"""
    try:
        if use_proxy:
            opener = urllib.request.build_opener()
        else:
            # 本机回环与直连对照都必须绕开代理：urllib 在 Windows 会读注册表系统代理
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        request = urllib.request.Request(url, headers={"User-Agent": "AINote-diagnostics"})
        with opener.open(request, timeout=timeout) as response:
            body = response.read(240)
            return {
                "ok": True,
                "status": response.status,
                "body": utf8_safe(body.decode("utf-8", "replace")),
            }
    except urllib.error.HTTPError as exc:
        # 收得到状态码说明网络层是通的（403/404 也算）
        return {"ok": True, "status": exc.code, "body": ""}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {utf8_safe(str(exc))}"[:200]}


def _resolve_host(host: str) -> dict[str, Any]:
    try:
        infos = socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {utf8_safe(str(exc))}"[:160]}
    return {"ok": True, "addresses": sorted({info[4][0] for info in infos})[:3]}


def _winsock_probe() -> dict[str, Any]:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return {"ok": True, "port": sock.getsockname()[1]}
    except OSError as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _wininet_proxy_snapshot() -> dict[str, Any]:
    """Windows 注册表里的系统代理：浏览器和 urllib 认它，httpx 不认——分开报。"""
    if os.name != "nt":
        return {"available": False}
    try:
        import winreg
    except ImportError:
        return {"available": False}
    try:
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Internet Settings",
        ) as key:
            enabled = bool(winreg.QueryValueEx(key, "ProxyEnable")[0])
            server = str(winreg.QueryValueEx(key, "ProxyServer")[0] or "")
            try:
                override = str(winreg.QueryValueEx(key, "ProxyOverride")[0] or "")
            except FileNotFoundError:
                override = ""
    except FileNotFoundError:
        return {"available": True, "enabled": False}
    except OSError as exc:
        return {"available": True, "error": utf8_safe(str(exc))[:120]}
    if not enabled or not server:
        return {"available": True, "enabled": False}
    # ProxyServer 可能是 "host:port"，也可能按协议写成 "http=…;https=…"
    target = server
    if "=" in server:
        for part in server.split(";"):
            if part.lower().startswith(("https=", "http=")):
                target = part.split("=", 1)[1]
                break
    host, _, port_text = target.rpartition(":")
    port = int(port_text) if port_text.isdigit() else 80
    return {
        "available": True,
        "enabled": True,
        "server": mask_proxy_target(target),
        "override": override[:160],
        "listening": _tcp_reachable(host or target, port, 1.0),
    }


def _count_free_cdp_ports(start: int = 9333, count: int = 20) -> int:
    free = 0
    for port in range(start, start + count):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(("127.0.0.1", port))
                free += 1
        except OSError:
            continue
    return free


@app.get("/api/diagnostics")
async def run_diagnostics(request: Request) -> dict[str, Any]:
    """本机自检：现场回答"测试连接/扫码登录 500"这类悬案能查的问题。

    只读、零密钥：响应里永远不出现 Key 明文、cookie、代理地址的账号段。
    """
    server_port = request.url.port or (request.scope.get("server") or (None, None))[1]
    checks: dict[str, Any] = {}
    hints: list[str] = []

    checks["service"] = {
        "pid": os.getpid(),
        "version": app.version,
        "mode": "portable" if getattr(sys, "frozen", False) else "dev",
        "port": server_port,
        "uptime_s": int(time.monotonic() - _STARTED_MONO),
    }

    loopback = await asyncio.wait_for(
        asyncio.to_thread(
            _http_probe, f"http://127.0.0.1:{server_port}/api/health", 3.0, use_proxy=False
        ),
        timeout=6,
    )
    identity = None
    try:
        identity = json.loads(str(loopback.get("body") or "null"))
    except ValueError:
        pass
    is_ours = isinstance(identity, dict) and identity.get("service") == "AINote"
    checks["loopback_http"] = {**loopback, "is_ours": is_ours}
    if loopback.get("ok") and not is_ours:
        hints.append(
            f"127.0.0.1:{server_port} 上应答的不是本程序（开头是：{str(loopback.get('body') or '')[:60]}）"
            "——端口被别的软件占着，或本机代理在截本机请求"
        )
    elif not loopback.get("ok"):
        hints.append("本机 HTTP 服务没答上话——后端可能刚被搞挂，看 workspace\\_app.log")

    checks["winsock"] = await asyncio.to_thread(_winsock_probe)

    registry = await asyncio.to_thread(_wininet_proxy_snapshot)
    checks["system_proxy"] = {"env": env_proxy_summary(), "registry": registry}
    if registry.get("enabled") and registry.get("listening") is False:
        hints.append(
            f"Windows 系统代理指向 {registry.get('server')}，但那个地址没有进程在听"
            "——代理/加速器多半僵死了：关掉它或重启电脑，软件不用重装"
        )

    dns_checks = {
        host: await asyncio.wait_for(asyncio.to_thread(_resolve_host, host), timeout=8)
        for host in ("api.deepseek.com", "passport.bilibili.com")
    }
    checks["dns"] = dns_checks
    if all(not entry.get("ok") for entry in dns_checks.values()):
        hints.append("两个域名都解析失败——DNS 或网络本身有问题：换手机热点试一次，不行再重启电脑")

    direct = await asyncio.wait_for(
        asyncio.to_thread(_http_probe, "https://api.deepseek.com", 5.0, use_proxy=False),
        timeout=9,
    )
    via_proxy = await asyncio.wait_for(
        asyncio.to_thread(_http_probe, "https://api.deepseek.com", 5.0, use_proxy=True),
        timeout=9,
    )
    checks["outbound"] = {"direct": direct, "with_system_proxy": via_proxy}
    if direct.get("ok") and not via_proxy.get("ok"):
        hints.append("直连通、带系统代理出不去——就是本机代理/加速器的问题，关掉它即可，不用重启电脑")
    if not direct.get("ok"):
        hints.append("不走代理也出不去——到 api.deepseek.com 整条路不通（网卡/防火墙/驱动）；重启电脑仍不行就换网络")

    checks["cdp_ports_free"] = await asyncio.to_thread(_count_free_cdp_ports)
    try:
        checks["llm_key_storage"] = {
            "backend": secret_box.storage_backend(),
            "corrupt": bool(config_store.keys_are_corrupt()),
        }
    except Exception as exc:
        checks["llm_key_storage"] = {"error": f"{type(exc).__name__}: {utf8_safe(str(exc))}"[:120]}

    recent = list(_LAST_BACKEND_ERRORS)
    checks["recent_backend_errors"] = recent
    if recent:
        hints.append("后端近期抛过未兜住的异常（见 recent_backend_errors，对照 _app.log 的 Traceback）")

    if not hints:
        hints.append("各项自检没发现异常——把这份报告整段发给作者/群里，比一句 500 有用得多")
    return {"ok": True, "checks": checks, "hints": hints}


@app.post("/api/bili-login/start")
async def bili_login_start() -> dict[str, Any]:
    return await bili_login_manager.start()


@app.get("/api/bili-login/status")
async def bili_login_status() -> dict[str, Any]:
    return await bili_login_manager.status()


@app.post("/api/bili-login/cancel")
async def bili_login_cancel() -> dict[str, Any]:
    return await bili_login_manager.cancel()


@app.post("/api/douyin-login/start")
async def douyin_login_start() -> dict[str, Any]:
    return await douyin_login_manager.start()


@app.get("/api/douyin-login/status")
async def douyin_login_status() -> dict[str, Any]:
    return await douyin_login_manager.status()


@app.post("/api/douyin-login/cancel")
async def douyin_login_cancel() -> dict[str, Any]:
    return await douyin_login_manager.cancel()


# 视频容器后缀（可能带音轨、需要时提取）；纯音频后缀直接使用
VIDEO_CONTAINER_SUFFIXES = {".mp4", ".mkv", ".mov", ".webm", ".avi"}


@app.post("/api/upload")
async def upload_video(
    file: UploadFile = File(...),
    include_screenshots: str | None = Form(default=None),
) -> dict[str, str]:
    filename = Path(file.filename or "upload.bin").name
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_MEDIA_SUFFIXES:
        raise HTTPException(status_code=415, detail="不支持的媒体文件类型")

    task_id = str(uuid.uuid4())
    task_dir = WORKSPACE_DIR / task_id
    task_dir.mkdir(parents=True, exist_ok=False)
    file_path = task_dir / f"input{suffix}"
    wants_screenshots = (include_screenshots or "").strip().lower() in {
        "1", "true", "yes"
    }
    written = 0
    try:
        async with aiofiles.open(file_path, "wb") as destination:
            while chunk := await file.read(1024 * 1024):
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"上传文件超过大小限制（上限 {MAX_UPLOAD_MB} MB）",
                    )
                await destination.write(chunk)
    except Exception:
        shutil.rmtree(task_dir, ignore_errors=True)
        raise
    finally:
        await file.close()

    log_lines: list[str] = [f"文件已上传：{filename}"]
    # 大视频且不需要截图时：提取音频、删除原视频，减少工作目录体积
    if (
        written >= LARGE_UPLOAD_EXTRACT_BYTES
        and suffix in VIDEO_CONTAINER_SUFFIXES
        and not wants_screenshots
    ):
        audio_path = task_dir / "input.m4a"
        try:
            extracted = await asyncio.to_thread(
                video_processor.extract_audio_track, file_path, audio_path
            )
        except Exception:
            extracted = None
        if extracted is not None and extracted.is_file():
            original_size = written
            file_path.unlink(missing_ok=True)
            file_path = audio_path
            log_lines.append(
                f"大视频已提取音频：{format_bytes_size(original_size)} → "
                f"{format_bytes_size(audio_path.stat().st_size)}（原视频已清理；"
                "如需截图请取消勾选该任务后重新上传）"
            )
        else:
            log_lines.append(
                "视频较大，未能提取音频，将直接用原文件转写（不影响使用）"
            )

    tasks[task_id] = new_task(status="uploaded", task_id=task_id)
    tasks[task_id].update(
        uploaded_file_path=str(file_path),
        uploaded_filename=filename,
        logs=log_lines,
    )
    write_upload_manifest(task_id, filename)
    persist_task_runtime(task_id)
    return {"task_id": task_id, "file_path": str(file_path), "filename": filename}


@app.get("/api/image/{task_id}/{filename}")
async def get_image(task_id: str, filename: str) -> FileResponse:
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    if Path(filename).name != filename:
        raise HTTPException(status_code=400, detail="非法文件名")
    task_dir = (WORKSPACE_DIR / task_id).resolve()
    if task_dir.parent != WORKSPACE_DIR:
        raise HTTPException(status_code=400, detail="非法任务 ID")
    expected_parent = (task_dir / FRAMES_DIR_NAME).resolve()
    image_path = (expected_parent / filename).resolve()
    if image_path.parent != expected_parent or not image_path.is_file():
        raise HTTPException(status_code=404, detail="图片不存在")
    return FileResponse(image_path)


@app.get("/api/download/{task_id}")
async def download_summary_markdown(task_id: str) -> Response:
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    if task["status"] != "completed":
        raise HTTPException(status_code=409, detail="任务尚未完成")

    task_dir = WORKSPACE_DIR / task_id
    output_path = task_dir / "notes.md"
    if not output_path.is_file():
        # 纯转录任务按设计不产出 notes.md，能下载的就是带时间轴的转录
        output_path = task_dir / "transcript.md"
    if not output_path.is_file():
        raise HTTPException(status_code=404, detail="Markdown 笔记不存在")

    title = (task.get("result") or {}).get("title", "video-notes")
    safe_name = download_stem(
        str(title), read_task_manifest(task_id), str(task.get("output") or "note")
    )
    try:
        note_text = output_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"读取笔记失败：{exc}")

    bundle = await asyncio.to_thread(
        build_note_bundle, note_text, task_dir, f"{safe_name}.md"
    )
    if bundle is not None:
        encoded_name = quote(f"{safe_name}.zip")
        return Response(
            content=bundle,
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename*=UTF-8''{encoded_name}"},
        )

    encoded_name = quote(f"{safe_name}.md")
    return FileResponse(
        output_path,
        media_type="text/markdown; charset=utf-8",
        filename=f"{safe_name}.md",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{encoded_name}"},
    )


def build_note_bundle(note_text: str, task_dir: Path, note_filename: str) -> bytes | None:
    """把笔记和它引用的截图打成一个包；正文里没有可解析的截图时返回 None。"""
    frames_dir = task_dir / FRAMES_DIR_NAME
    names: list[str] = []
    for match in NOTE_IMAGE_REF_RE.finditer(note_text):
        name = Path(match.group(2)).name
        if name not in names and (frames_dir / name).is_file():
            names.append(name)
    if not names:
        return None
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
        archive.writestr(note_filename, rewrite_note_image_refs(note_text, "./"))
        for name in names:
            archive.writestr(
                f"{FRAMES_DIR_NAME}/{name}", (frames_dir / name).read_bytes()
            )
    return buffer.getvalue()


@app.post("/api/task/{task_id}/cancel")
async def cancel_task(task_id: str) -> dict[str, Any]:
    task = tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    if task["status"] in {"completed", "failed", "cancelled"}:
        raise HTTPException(status_code=409, detail="任务已经结束")

    first_request = not task.get("_cancel_requested")
    task["_cancel_requested"] = True
    cancel_event = task.get("_cancel_event")
    if cancel_event is not None:
        cancel_event.set()
    task.update(status="cancelling", step_name="取消中", error=None)
    task["logs"].append("已收到取消请求；正在尽快停止当前步骤")
    persist_task_runtime(task_id)
    if first_request:
        # 取消的唯一收口：直接掐掉 asyncio 任务，让 CancelledError 落在当前 await 点。
        # should_abort / cancel_event 只是让后台线程早点收工的优化，新增阶段忘了接线
        # 也不会再让取消"失效"（v1.2.3 的 412 回退把没接线的读取信息/查找字幕阶段
        # 拉长，表现为取消不动）。这一句之后不能再有 await，否则会覆盖处理器写的终态。
        job = running_jobs.get(task_id)
        if job is not None and not job.done():
            job.cancel()
    return {
        "cancelled": False,
        "status": task["status"],
        "elapsed_seconds": task_elapsed_seconds(task),
    }


@app.get("/api/task/{task_id}/coverage")
async def task_coverage(task_id: str) -> dict[str, Any]:
    """按需算一份笔记覆盖度。

    覆盖度是纯计算、不联网，所以这里不需要模型配置：改动之前生成的老任务结果里
    没有这个字段，换机器/换模型的旧任务也照样能看出“笔记缺了结尾”。
    """
    try:
        directory = task_directory(task_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="非法任务 ID") from exc
    notes_path = directory / "notes.md"
    transcript = load_transcript_result(directory)
    if not notes_path.is_file() or not transcript:
        return {"coverage": None}
    try:
        draft = notes_path.read_text(encoding="utf-8")
    except OSError:
        return {"coverage": None}
    coverage = LLMSummarizer.tail_coverage(
        draft,
        transcript["segments"],
        material_budget=TAIL_PATCH_MAX_INPUT_CHARACTERS_MANUAL,
    )
    return {"coverage": coverage.to_dict() if coverage is not None else None}


@app.post("/api/task/{task_id}/patch-tail")
async def patch_task_tail(task_id: str, request: PatchTailRequest) -> dict[str, Any]:
    """补写笔记缺失的结尾：只补缺口，不重跑整篇。

    要不要补由用户决定（覆盖度随任务结果推给前端，代价一并摆出来），这一端点只在
    用户按下按钮时执行。成功时同时更新任务目录的 notes.md 与已归档的那一份——归档走
    原地覆盖，因为 ``archive_note()`` 遇重名会加序号，复用它只会凭空多出一份重复笔记。
    """
    try:
        directory = task_directory(task_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="非法任务 ID") from exc
    notes_path = directory / "notes.md"
    if not notes_path.is_file():
        raise HTTPException(status_code=409, detail="该任务没有笔记产物，无法补写")
    transcript = load_transcript_result(directory)
    if not transcript:
        raise HTTPException(status_code=409, detail="该任务没有转录产物，无法核对笔记缺口")
    manifest = read_task_manifest(task_id)
    try:
        draft = notes_path.read_text(encoding="utf-8")
    except OSError:
        raise HTTPException(status_code=500, detail="笔记文件无法读取") from None

    config = request.llm_config
    base_url = config.base_url or config.custom_base_url
    model = config.model or config.custom_model_name
    try:
        api_key, _ = resolve_llm_credentials(
            model_type=config.model_type,
            base_url=base_url,
            api_key=config.api_key.get_secret_value(),
        )
        summarizer = LLMSummarizer(
            model_type=config.model_type,
            api_key=api_key,
            base_url=base_url,
            model=model,
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    style = str(manifest.get("summary_style") or "detailed")
    selected_effort = request.reasoning_effort or str(
        manifest.get("reasoning_effort") or "auto"
    )
    coverage = summarizer.tail_coverage(
        draft,
        transcript["segments"],
        material_budget=TAIL_PATCH_MAX_INPUT_CHARACTERS_MANUAL,
    )
    if coverage is None:
        raise HTTPException(status_code=409, detail="笔记里没有可识别的时间戳，无法定位缺口")
    if not coverage.patch_available:
        raise HTTPException(
            status_code=409,
            detail=coverage.patch_blocked_reason or "这篇笔记没有需要补写的缺口",
        )

    outcome = await summarizer.patch_missing_tail(
        str(manifest.get("title") or ""),
        draft,
        transcript["segments"],
        effort=summarizer.resolve_note_effort(selected_effort, style),
        material_budget=TAIL_PATCH_MAX_INPUT_CHARACTERS_MANUAL,
    )
    if not outcome.ok:
        raise HTTPException(status_code=502, detail=f"补写失败：{outcome.error}")

    body = outcome.body or ""
    # 补写片段里的时间点也要能点；只处理新写的这一段，不动既有正文。多分 P 笔记
    # 与生成流程保持一致，保持纯文本（单条链接定位不到正确分 P）。
    pages = manifest.get("bilibili_pages") or []
    source_url = manifest.get("normalized_source_url") or manifest.get("source_url")
    if source_url and len(pages) <= 1:
        body = linkify_timestamps(body, str(source_url))

    patched = insert_tail_patch(draft, body, outcome.missing_start)
    write_text_atomically(notes_path, patched)

    runtime_result = (manifest.get("runtime") or {}).get("result") or {}
    archived_raw = str(runtime_result.get("archived_path") or "")
    overwritten = False
    if archived_raw:
        try:
            archived_file = Path(archived_raw)
            if archived_file.is_file():
                # 归档那份的截图引用与任务目录不同（../<task_id>/ 或 <笔记名>.assets/），
                # 必须在它自己的正文上拼接，不能拿任务目录的正文覆盖过去。
                archived_text = archived_file.read_text(encoding="utf-8")
                write_text_atomically(
                    archived_file,
                    insert_tail_patch(archived_text, body, outcome.missing_start),
                )
                overwritten = True
        except OSError:
            overwritten = False

    new_coverage = summarizer.tail_coverage(
        patched,
        transcript["segments"],
        material_budget=TAIL_PATCH_MAX_INPUT_CHARACTERS_MANUAL,
    )
    remaining = ""
    if new_coverage is not None and new_coverage.gap_seconds > TAIL_GAP_SECONDS:
        remaining = f"；仍缺约 {new_coverage.gap_seconds / 60:.1f} 分钟"
    log_line = redact_secrets(
        f"已补写结尾（{outcome.missing_start} 之后）"
        + ("，任务目录与归档文件均已更新" if overwritten else "，已更新任务目录")
        + remaining,
        api_key,
    )
    record_patch_result(
        task_id,
        patched,
        new_coverage.to_dict() if new_coverage is not None else None,
        log_line,
    )
    return {
        "ok": True,
        "markdown": patched,
        "coverage": new_coverage.to_dict() if new_coverage is not None else None,
        "archived_path": archived_raw or None,
        "archived_overwritten": overwritten,
        "log": log_line,
    }


@app.delete("/api/task/{task_id}")
async def delete_task(task_id: str) -> dict[str, bool]:
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    job = running_jobs.get(task_id)
    if job and not job.done():
        raise HTTPException(status_code=409, detail="任务仍在运行，请先取消并等待其停止")
    running_jobs.pop(task_id, None)
    await video_processor.cleanup(task_id)
    tasks.pop(task_id, None)
    return {"deleted": True}


def _format_page_nums(pages) -> str:
    """把分 P 对象 / 页码列表格式化为 “1、2、3”。"""
    nums = sorted(int(page.page if hasattr(page, "page") else page) for page in pages)
    return "、".join(str(num) for num in nums)


def append_asr_notes(task: dict[str, Any], whisper_result: dict[str, Any]) -> None:
    """转写阶段的降级说明与耗时入日志（单文件与 B 站多分 P 共用）。"""
    note = whisper_result.get("fallback_note")
    if note and note not in task["logs"]:
        task["logs"].append(note)
    load_seconds = float(whisper_result.get("model_load_seconds") or 0)
    if load_seconds >= 1:
        task["logs"].append(
            f"模型加载用时 {load_seconds} 秒，转写用时 "
            f"{whisper_result.get('transcribe_seconds')} 秒"
        )


async def run_queued_video_task(task_id: str, request: SummarizeRequest) -> None:
    task = tasks[task_id]
    task.update(status="queued", step_name="等待执行", progress=0)
    task["logs"].append(
        f"任务已进入队列；当前最多同时处理 {MAX_CONCURRENT_TASKS} 个任务"
    )
    persist_task_runtime(task_id)
    try:
        async with task_slots:
            raise_if_cancel_requested(task)
            await process_video_task(task_id, request)
    except asyncio.CancelledError:
        finish_task_timing(task)
        task.update(status="cancelled", step_name="已取消", error=None)
        task["logs"].append("任务已取消；已生成的中间文件将保留")
        persist_task_runtime(task_id)


async def process_video_task(task_id: str, request: SummarizeRequest) -> None:
    task = tasks[task_id]
    task["status"] = "processing"
    persist_task_runtime(task_id)

    def should_abort() -> bool:
        return bool(task.get("_cancel_requested"))

    # 终态通知会用 title 组文案：早期失败（读凭据、取元数据）时 title 还没赋值，
    # 少了这个初值 except 分支自己会 NameError，任务就只改了内存、task.json 留在 processing
    title = request.video_url or "上传的视频"
    # B 站视频页被风控时（Issue #1）处理链会回退到开放接口直连，回退说明经这里进任务日志
    fallback_notes: list[str] = []
    # except 分支靠它决定能不能套用平台文案，而 detect_source 自己就可能抛错（非 http 链接）
    source_kind = VideoSource.LOCAL
    # 通道报错里可能整段回显本次用的明文 Key，落盘前要换成掩码；
    # 放在 try 之外初始化，因为早期失败根本没走到 resolve_llm_credentials
    used_api_key = ""

    def note_api_fallback() -> None:
        for note in fallback_notes:
            if note not in task["logs"]:
                task["logs"].append(note)

    try:
        raise_if_cancel_requested(task)
        bili_cookie = (
            request.bilibili_cookie.model_dump()
            if request.bilibili_cookie
            else config_store.load_bili_credentials()
        )
        resume_task_id = task.get("resume_task_id")
        resume_dir = task_directory(resume_task_id) if resume_task_id else None
        source_url = request.video_url or (source_url_from_task(resume_dir) if resume_dir else None)
        uploaded_path = task.get("uploaded_file_path")
        if not uploaded_path and resume_dir:
            uploaded_media = next(
                (path for path in resume_dir.glob("input.*") if path.is_file()), None
            )
            uploaded_path = str(uploaded_media) if uploaded_media else None
        is_local = bool(uploaded_path) or not source_url
        source_kind = (
            video_processor.detect_source(source_url) if source_url else VideoSource.LOCAL
        )
        # 平台凭据按来源分流：抖音走自己的会话 cookie（B 站凭据对它没用，反之亦然）
        if source_kind == VideoSource.DOUYIN:
            try:
                cookie = (
                    request.douyin_cookie
                    if request.douyin_cookie
                    else config_store.load_douyin_credentials()
                )
            except DouyinCredentialsUnavailable as error:
                fallback_notes.append(str(error))
                cookie = None
        else:
            cookie = bili_cookie
        transcript_result = load_transcript_result(resume_dir) if resume_dir else None
        bili_pages_to_transcribe: list = []

        if transcript_result:
            set_progress(task, 1, "恢复任务信息", 8, "正在读取已有任务信息")
            info = load_reused_video_info(resume_dir, transcript_result, source_url)
        else:
            media_input = uploaded_path or source_url or ""
            set_progress(task, 1, "读取视频信息", 8, "正在读取视频信息")
            info = await video_processor.get_video_info(
                media_input, cookie, allow_local=is_local, notes=fallback_notes
            )
            note_api_fallback()
        raise_if_cancel_requested(task)
        if is_local:
            # 上传落地文件统一叫 input.<ext>，路径推出来的标题就是占位名 input
            original_name = str(task.get("uploaded_filename") or "").strip()
            if original_name:
                info["title"] = Path(original_name).stem
        title = info["title"]
        update_task_manifest(task_id, info)
        task["logs"].append(f"标题：{title}")
        if info.get("duration"):
            task["logs"].append(f"时长：{format_duration(info['duration'])}")

        if transcript_result:
            set_progress(task, 4, "复用转录", 48, "已读取已有转录，跳过字幕、音频和 Whisper")
            task["logs"].append(
                f"复用任务 {resume_task_id} 的 {len(transcript_result['segments'])} 个转录分段"
            )
        elif request.prefer_subtitles and not is_local:
            set_progress(task, 2, "查找平台字幕", 16, "正在查找平台字幕")
            subtitle = await video_processor.fetch_subtitles(source_url or "", info, cookie)
            if subtitle:
                transcript_result = {
                    "segments": subtitle.segments,
                    "language": subtitle.language,
                    "source": subtitle.source,
                }
                task["logs"].append(
                    f"已使用平台字幕：{subtitle.language}，共 {len(subtitle.segments)} 段"
                )
            else:
                is_bilibili = (
                    video_processor.detect_source(source_url or "")
                    == VideoSource.BILIBILI
                )
                subtitle_outcome = (
                    await video_processor.fetch_bilibili_subtitles(
                        source_url or "", cookie, only_pages=request.bilibili_pages or None
                    )
                    if is_bilibili
                    else None
                )
                if subtitle_outcome is not None:
                    bili_pages_to_transcribe = list(subtitle_outcome.pages_to_transcribe)
                    if subtitle_outcome.title:
                        info["title"] = subtitle_outcome.title
                    if subtitle_outcome.pages:
                        info["duration"] = sum(
                            page.duration for page in subtitle_outcome.pages
                        ) or info.get("duration")
                    if subtitle_outcome.total_pages > 1:
                        # 记下了本次覆盖哪几个分 P，产物文件名才带得出 P 号与分 P 名
                        info["bili_pages"] = [
                            {"page": page.page, "part": page.part}
                            for page in subtitle_outcome.pages
                        ]
                        info["bili_total_pages"] = subtitle_outcome.total_pages
                        task["logs"].append(
                            f"检测到 B 站分 P 视频：共 {subtitle_outcome.total_pages} 个分 P，"
                            f"本次处理 {_format_page_nums(subtitle_outcome.pages)}"
                        )
                        update_task_manifest(task_id, info)
                if subtitle_outcome is not None and subtitle_outcome.result:
                    transcript_result = {
                        "segments": subtitle_outcome.result.segments,
                        "language": subtitle_outcome.result.language,
                        "source": subtitle_outcome.result.source,
                    }
                    task["logs"].append(
                        f"已使用 B 站 AI 字幕：{subtitle_outcome.result.language}，"
                        f"共 {len(subtitle_outcome.result.segments)} 段"
                    )
                    if bili_pages_to_transcribe:
                        task["logs"].append(
                            "分 P "
                            f"{_format_page_nums(bili_pages_to_transcribe)}"
                            " 无 AI 字幕，将通过语音转写补齐"
                        )
                elif subtitle_outcome is not None:
                    reason = subtitle_outcome.reason
                    if reason == "credentials_missing":
                        task["logs"].append(
                            "未读取到可用的 B 站字幕轨；AI 字幕需要填写当前账号的 "
                            "SESSDATA 等访问凭据"
                        )
                    elif reason == "no_track":
                        task["logs"].append(
                            "当前账号已登录，但该视频没有可用的 AI 字幕轨"
                        )
                    elif reason == "error":
                        task["logs"].append(
                            "读取 B 站 AI 字幕失败（网络或接口异常，"
                            f"{subtitle_outcome.detail}）"
                        )
                    elif reason == "empty":
                        task["logs"].append("B 站字幕轨存在但内容为空")
                    if bili_pages_to_transcribe:
                        task["logs"].append(
                            f"分 P {_format_page_nums(bili_pages_to_transcribe)} 将通过语音转写"
                        )
                    task["logs"].append("未找到可用平台字幕，将进行语音转写")
            title = info["title"]

        raise_if_cancel_requested(task)
        if transcript_result is None or bili_pages_to_transcribe:
            set_progress(task, 3, "准备音频", 25, "正在准备音频")
            if bili_pages_to_transcribe:
                # 多分 P：对没有 AI 字幕的分 P 逐个下载音频并转写，再按顺序合并
                whisper_by_page: dict[int, dict] = {}
                for page in bili_pages_to_transcribe:
                    raise_if_cancel_requested(task)
                    set_progress(
                        task, 3, "准备音频", 25, f"正在下载分 P {page.page} 音频"
                    )
                    media_path = await video_processor.download_audio(
                        bilibili_page_url(source_url or "", page.page),
                        task_id,
                        cookie,
                        media_name=f"audio_p{page.page}",
                        should_abort=should_abort,
                        notes=fallback_notes,
                    )
                    note_api_fallback()
                    raise_if_cancel_requested(task)
                    set_progress(
                        task,
                        4,
                        "语音转写",
                        38,
                        f"正在转写分 P {page.page}（{request.whisper_model} 模型）",
                    )
                    whisper_result = await run_transcription(
                        media_path,
                        request.whisper_model,
                        request.use_gpu,
                        title,
                        task.get("_cancel_event"),
                        progress_callback=transcribe_progress_reporter(
                            task, request.whisper_model, f"分 P {page.page} "
                        ),
                    )
                    raise_if_cancel_requested(task)
                    whisper_by_page[page.page] = whisper_result
                    task["logs"].append(
                        f"分 P {page.page} 语音转写完成：{len(whisper_result['segments'])} 段，"
                        f"设备 {whisper_result['device']}"
                    )
                    append_asr_notes(task, whisper_result)
                    if whisper_result["model"] != whisper_result["requested_model"]:
                        task["logs"].append(
                            f"所选 Whisper {whisper_result['requested_model']} 未缓存或无法加载，"
                            f"已自动降级为本机缓存的 {whisper_result['model']}"
                        )
                if subtitle_outcome is not None and subtitle_outcome.pages:
                    merged_segments, language = merge_bilibili_pages(
                        subtitle_outcome.pages,
                        dict(subtitle_outcome.subtitle_by_page),
                        whisper_by_page,
                    )
                else:
                    merged_segments = [
                        seg
                        for whisper in whisper_by_page.values()
                        for seg in (whisper.get("segments") or [])
                    ]
                    language = next(
                        (
                            str(whisper.get("language") or "zh")
                            for whisper in whisper_by_page.values()
                        ),
                        "zh",
                    )
                transcript_result = {
                    "segments": merged_segments,
                    "language": language,
                    "source": "bilibili_multi_page",
                }
                if not info.get("duration"):
                    info["duration"] = sum(
                        float(whisper.get("duration") or 0)
                        for whisper in whisper_by_page.values()
                    )
                task["logs"].append(f"分 P 转录合并完成：共 {len(merged_segments)} 段")
            elif transcript_result is None:
                set_progress(task, 3, "准备音频", 25, "正在准备音频")
                reused_audio = None
                if resume_dir:
                    reused_audio = next(
                        (path for path in resume_dir.glob("audio.*") if path.is_file()), None
                    )
                if reused_audio:
                    media_path = reused_audio
                    task["logs"].append(f"复用已有音频：{reused_audio.name}")
                elif is_local:
                    media_path = Path(uploaded_path)
                else:
                    media_path = await video_processor.download_audio(
                        source_url or "", task_id, cookie, should_abort=should_abort,
                        notes=fallback_notes,
                    )
                raise_if_cancel_requested(task)
                note_api_fallback()
                set_progress(
                    task,
                    4,
                    "语音转写",
                    38,
                    (
                        f"正在上传音频到 Groq 转写（{request.whisper_model}，云端无需下载模型）"
                        if groq_asr.is_groq_model(request.whisper_model)
                        else f"正在加载 {request.whisper_model}（未缓存时可能下载模型）"
                    ),
                )
                whisper_result = await run_transcription(
                    media_path,
                    request.whisper_model,
                    request.use_gpu,
                    title,
                    task.get("_cancel_event"),
                    progress_callback=transcribe_progress_reporter(
                        task, request.whisper_model
                    ),
                )
                raise_if_cancel_requested(task)
                if groq_asr.is_groq_model(request.whisper_model):
                    asr_source = "groq"
                elif request.whisper_model == "paraformer-zh":
                    asr_source = "paraformer"
                else:
                    asr_source = "faster_whisper"
                transcript_result = {
                    "segments": whisper_result["segments"],
                    "language": whisper_result["language"],
                    "source": asr_source,
                }
                if not info.get("duration"):
                    info["duration"] = whisper_result["duration"]
                task["logs"].append(
                    f"语音转写完成：{len(whisper_result['segments'])} 段，设备 {whisper_result['device']}"
                )
                append_asr_notes(task, whisper_result)
                # Groq 的 requested_model 是前端模型 id、model 是 Groq 上的真名，两者必然
                # 不同；那条「未缓存已降级」的告警只适用于本地 faster-whisper 的缓存回退。
                if (
                    whisper_result["model"] != whisper_result["requested_model"]
                    and not groq_asr.is_groq_model(request.whisper_model)
                ):
                    task["logs"].append(
                        f"所选 Whisper {whisper_result['requested_model']} 未缓存或无法加载，"
                        f"已自动降级为本机缓存的 {whisper_result['model']}"
                    )

        segments: list[TranscriptSegment] = transcript_result["segments"]
        if not segments:
            raise RuntimeError("没有从视频中获得可总结的文字")
        quality = transcript_quality(segments, float(info.get("duration") or 0))
        transcript_result["quality"] = quality
        task["logs"].append(
            f"文字质量检查：{quality['characters']} 字，语音覆盖 {quality['speech_coverage']:.0%}"
        )
        await write_transcript_files(task_id, segments, transcript_result)
        if (
            transcript_result["source"] in {"faster_whisper", "paraformer"}
            and quality["insufficient"]
        ):
            stage = "调用大模型" if request.output == "note" else "返回转录"
            raise RuntimeError(
                f"可识别语音覆盖过低，已在{stage}前停止。源视频可能被静音、替换、"
                "受版权或审核限制，或主要内容并非语音；请先检查原视频。"
            )

        if request.output == "transcript":
            # 正文只留在 transcript.json / transcript.md：result 会被 wait_for_task
            # 每 2 秒轮询一次并由 persist_task_runtime 反复落盘，不能放大文本
            elapsed = finish_task_timing(task)
            # 归档读的是刚落盘的那份 transcript.md（与下载端点同一个文件），不在这里
            # 重新拼一遍格式——两处各拼一次迟早会对不上。
            try:
                transcript_markdown = (
                    WORKSPACE_DIR / task_id / "transcript.md"
                ).read_text(encoding="utf-8")
            except OSError:
                transcript_markdown = ""
            archived_path = (
                archive_note(
                    download_stem(title, info, "transcript"),
                    transcript_markdown,
                    folder=TRANSCRIPTS_DIR_NAME,
                )
                if transcript_markdown
                else None
            )
            if archived_path:
                task["logs"].append(f"转录稿已归档：{archived_path}")
            task["result"] = {
                "title": title,
                "output": "transcript",
                "source": VideoSource.LOCAL.value if is_local else info.get("source", "other"),
                "duration": info.get("duration", 0),
                "segment_count": len(segments),
                "characters": quality["characters"],
                "transcript_source": transcript_result["source"],
                "transcript_language": transcript_result["language"],
                "transcript_quality": quality,
                "output_directory": str(WORKSPACE_DIR / task_id),
                "archived_path": str(archived_path) if archived_path else None,
                "processing_seconds": elapsed,
            }
            task.update(
                status="completed", step=7, step_name="转录完成", progress=100, error=None
            )
            task["logs"].append(
                f"转录完成：{len(segments)} 段，可用 GET /api/task/{task_id}/transcript 取正文"
            )
            notify_task("转录完成", f"已生成带时间轴转录：{title}")
            persist_task_runtime(task_id)
            return

        screenshots: list[Path] = []
        if request.include_screenshots:
            set_progress(task, 5, "提取截图", 50, "正在提取低清预览截图")
            video_path = (
                Path(uploaded_path)
                if is_local
                else await video_processor.download_preview_video(
                    source_url or "", task_id, cookie, should_abort=should_abort,
                    notes=fallback_notes,
                )
            )
            note_api_fallback()
            screenshots = await video_processor.extract_frames(
                video_path, task_id, request.screenshot_interval, should_abort
            )
            raise_if_cancel_requested(task)
            task["logs"].append(f"已提取 {len(screenshots)} 张截图")

        transcript_characters = int(quality.get("characters") or 0)
        duration_seconds = float(info.get("duration") or 0)
        advisory = ""
        if (
            transcript_characters > LONG_TRANSCRIPT_CHARACTERS
            and request.summary_style not in {"concise", "keypoints"}
        ):
            advisory = (
                "这段内容很长，笔记会按时间顺序逐段完整记录（不做预先压缩），"
                "因此 Token 消耗与耗时都随时长增长，输出量与口播字数同量级。"
                "3 小时量级的课可能要几十分钟才出稿（限流的免费通道更慢），"
                "挂着等即可，任何时候都能取消。想省一些可以改用「精简摘要」。"
            )
        elif transcript_characters >= 9_000 or duration_seconds >= 1_800:
            advisory = (
                "这段内容较长，将通过多轮整理生成完整笔记，耗时和 Token 消耗会相应增加。"
                "上下文容量较大、指令理解能力较强的模型通常更稳定；免费或轻量模型可能出现遗漏或截断。"
            )
        if advisory:
            task["advisory"] = advisory
            task["logs"].append(advisory)
        set_progress(task, 6, "生成笔记", 55, "正在规划笔记生成流程")
        config = request.llm_config
        base_url = config.base_url or config.custom_base_url
        model = config.model or config.custom_model_name
        api_key, key_source = resolve_llm_credentials(
            model_type=config.model_type,
            base_url=base_url,
            api_key=config.api_key.get_secret_value(),
        )
        used_api_key = api_key
        summarizer = LLMSummarizer(
            model_type=config.model_type,
            api_key=api_key,
            base_url=base_url,
            model=model,
        )
        task["logs"].append(
            f"调用模型：Provider={config.model_type}，Model={model}，"
            f"Base URL={base_url}，Key={key_source}"
        )
        task["logs"].append(
            f"推理设置：{summarizer.describe_effort(request.reasoning_effort, request.summary_style)}"
        )

        async def report_llm_progress(progress: int, message: str) -> None:
            raise_if_cancel_requested(task)
            mapped = 55 + round(max(0, min(100, progress)) * 0.44)
            set_progress(task, 6, "生成笔记", mapped, message)

        summary = await summarizer.generate_summary(
            title,
            segments,
            {
                "owner": info.get("owner"),
                "published_at": format_publish_time(info),
                "duration_text": format_duration(info.get("duration", 0)),
                "view_count": info.get("view_count") or 0,
                "like_count": info.get("like_count") or 0,
                "transcript_source": transcript_result["source"],
                "description": (info.get("description") or "")[:600],
            },
            style=request.summary_style,
            reasoning_effort=request.reasoning_effort,
            content_type=request.content_type,
            progress_callback=report_llm_progress,
            should_abort=should_abort,
        )
        raise_if_cancel_requested(task)
        for warning in getattr(summarizer, "warnings", []):
            # 告警会把网关异常原文嵌进来，落盘前必须过一遍脱敏
            cleaned = redact_secrets(warning, api_key)
            if cleaned not in task["logs"]:
                task["logs"].append(cleaned)
        # 笔记标题不再沿用原视频标题：让模型从成稿正文里拟一个短标题，再拼上风格标签
        # （如「大模型记忆机制（详细）」）。失败一律降级回视频标题，绝不因此让任务失败。
        note_title = composite_note_title(title, request.summary_style) or title
        try:
            short_title = await summarizer.generate_note_title(
                summary, request.summary_style, should_abort=should_abort
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            short_title = ""
            task["logs"].append(
                f"短标题生成失败，沿用视频标题（{type(exc).__name__}）"
            )
        composed = composite_note_title(short_title, request.summary_style)
        if composed:
            note_title = composed
            task["logs"].append(f"笔记标题：{note_title}")
        elif not short_title:
            task["logs"].append("未能拟出短标题，沿用视频标题")
        summary = add_note_header_metadata(summary, note_title, info)
        if screenshots:
            summary += "\n\n## 视频截图\n\n" + "\n\n".join(
                f"![截图 {index}](./{FRAMES_DIR_NAME}/{path.name})"
                for index, path in enumerate(screenshots, start=1)
            )
        summary = append_note_footer(
            summary,
            source_url,
            config.model_type,
            model,
            request.summary_style,
            request.content_type,
        )
        # 多分 P 合并后的时间轴按选中分 P 顺序累积，单条链接定位不到正确分 P，
        # 此时保持纯文本。同时看两个信号：旧版 manifest 可能缺 bili_total_pages。
        if _timestamp_page_count(info.get("bili_total_pages")) > 1 or len(
            info.get("bili_pages") or []
        ) > 1:
            task["logs"].append(
                "多分 P 笔记：时间点保留为纯文本（单条链接无法定位到正确分 P）"
            )
        else:
            summary = linkify_timestamps(summary, source_url)

        task_dir = WORKSPACE_DIR / task_id
        write_text_atomically(task_dir / "notes.md", summary)
        archived_path = archive_note(download_stem(note_title, info), summary, task_id)
        if archived_path:
            task["logs"].append(f"笔记已归档：{archived_path}")
        elapsed = finish_task_timing(task)
        task["result"] = {
            "title": note_title,
            "markdown": summary,
            "source": VideoSource.LOCAL.value if is_local else info.get("source", "other"),
            "duration": info.get("duration", 0),
            "screenshot_count": len(screenshots),
            "transcript_source": transcript_result["source"],
            "transcript_language": transcript_result["language"],
            "segment_count": len(segments),
            "transcript_quality": quality,
            "output_directory": str(task_dir),
            "archived_path": str(archived_path) if archived_path else None,
            "processing_seconds": elapsed,
            # 覆盖度核对结果：前端据此提示“笔记没写到结尾”，并把补写的代价摆给用户看，
            # 由用户决定补不补（详见 POST /api/task/{id}/patch-tail）
            "coverage": (
                summarizer.coverage.to_dict()
                if getattr(summarizer, "coverage", None) is not None
                else None
            ),
        }
        task.update(
            status="completed", step=7, step_name="完成", progress=100, error=None
        )
        task["logs"].append("视频笔记生成完成")
        notify_task("任务完成", f"视频笔记已生成：{note_title}")
        persist_task_runtime(task_id)
    except asyncio.CancelledError:
        finish_task_timing(task)
        task.update(status="cancelled", step_name="已取消", error=None)
        if not task["logs"] or "用户已取消任务" not in task["logs"][-1]:
            task["logs"].append("任务已取消；已生成的中间文件将保留")
        persist_task_runtime(task_id)
    except Exception as exc:
        finish_task_timing(task)
        raw = raw_error_text(exc)
        # 失败文案有三条出口（error 字段、任务日志、系统通知），未知异常时
        # friendly_* 会原样返回底层文本，网关回显的明文 Key 就会跟着落进 task.json
        message = redact_secrets(
            friendly_task_error(
                raw, source=source_kind, step_name=task.get("step_name", "")
            ),
            used_api_key,
        )
        task.update(status="failed", error=message)
        task["logs"].append(f"处理失败：{message}")
        if message != raw:
            # 文案一旦被替换掉，真实的返回码就没人能看到了——这次误诊就是这么来的
            task["logs"].append(f"底层错误：{scrub_technical_error(raw, used_api_key)}")
        notify_task("任务失败", f"{title}\n{message}")
        persist_task_runtime(task_id)


async def write_transcript_files(
    task_id: str, segments: list[TranscriptSegment], metadata: dict[str, Any]
) -> None:
    """转录文件先写 .tmp 再改名。

    取消能落在任意 await 点（见 cancel_task），而「复用转录」以 transcript.json
    是否存在为准——半截文件会比没有文件更糟。
    """
    task_dir = WORKSPACE_DIR / task_id
    json_payload = {
        "language": metadata["language"],
        "source": metadata["source"],
        "quality": metadata.get("quality"),
        "segments": [segment.to_dict() for segment in segments],
    }
    await _write_atomically(
        task_dir / "transcript.json",
        (json.dumps(json_payload, ensure_ascii=False, indent=2),),
    )
    await _write_atomically(
        task_dir / "transcript.md",
        ("# 带时间戳转录\n\n", segments_to_prompt(segments), "\n"),
    )


async def _write_atomically(path: Path, chunks: tuple[str, ...]) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    async with aiofiles.open(temporary, "w", encoding="utf-8") as file:
        for chunk in chunks:
            await file.write(chunk)
    os.replace(temporary, path)


def write_text_atomically(path: Path, content: str) -> None:
    """同步版原子写：先写 .tmp 再 os.replace 顶上，用于归档到知识库的成品。

    ``Path.write_text`` 是"先截断再写"，中途失败会留下半截文件；这些文件躺在
    用户的 Obsidian 库里，写坏了没有第二份可以恢复。
    """
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def friendly_task_error(
    message: str,
    *,
    source: VideoSource | None = None,
    step_name: str = "",
) -> str:
    """把已知的平台限制错误转成对用户友好的提示。

    平台文案只在对应平台的任务里出现，模型通道的错误不套平台说法：v1.4.0 之前
    任何含 403 的异常都说成「抖音媒体地址已过期」，而实际触发它的是 B 站任务里
    模型接口返回的 403，用户据此重投了三次没问题的链接。
    """
    lowered = message.lower()
    if "certificate_verify_failed" in lowered or "certificate verify failed" in lowered:
        return (
            "Whisper 模型下载时 TLS 证书校验失败。程序已尝试备用下载源；"
            "如果仍失败，请检查系统时间、网络代理或企业根证书，"
            "并通过 HF_ENDPOINT 指定能正常验证证书的镜像后重试。"
        )
    if step_name == NOTE_STEP_NAME:
        return friendly_llm_error(lowered, message)
    if source == VideoSource.DOUYIN and step_name in DOUYIN_HINT_STEPS:
        # 抖音提示只对「取视频信息 / 准备音频」这两步成立——那才是浏览器通道与媒体
        # 地址相关的地方。转写、生成笔记等后续步骤的失败跟抖音无关，必须原样透传，
        # 否则用户会拿着「链接解析失败」去重投一条本来没问题的链接（线上已发生）。
        return DOUYIN_HINT
    if source == VideoSource.BILIBILI and any(
        marker in lowered for marker in ("403", "412", "forbidden", "precondition")
    ):
        return (
            "B 站拒绝了本次请求（风控或访问凭据失效）。请在「B 站访问凭据」里重新扫码导入 "
            "SESSDATA 后重投链接；也可以稍等几分钟再试。"
        )
    return message


# 各家对"上下文窗口不够"的说法不一样（ollama 两种、OpenAI/vLLM 一种、LM Studio 一种），
# 但都带这些字样。刻意不含 "too many tokens"：OpenAI 限流的原文就是
# `Too many tokens per minute`，按它判会把限流说成窗口不够。
_CONTEXT_LIMIT_MARKERS = (
    "context_length_exceeded",
    "context length",
    "context window",
    "prompt is too long",
    "num_ctx",
    "maximum capacity of",
)


def friendly_llm_error(lowered: str, message: str) -> str:
    """生成笔记阶段的失败来自模型通道，重投视频链接不会有任何改变。"""
    # 放在额度/429 之前：下面几条判的是裸子串 "429"/"403"，而本类报错里的数字是
    # token 数（`requested 9403 tokens` 就含 "403"），先判会把窗口不够说成权限问题。
    if any(marker in lowered for marker in _CONTEXT_LIMIT_MARKERS):
        return (
            "模型的上下文窗口装不下本次请求（这是模型侧的容量上限，不是本程序的笔记长度设置）。"
            "本地模型（ollama / LM Studio / llama.cpp）请把上下文调到 16K 以上再重投："
            "ollama 设 OLLAMA_CONTEXT_LENGTH=16384 后重启服务，或在 Modelfile 里写 "
            "`num_ctx 16384`。在线通道多为该模型或网关的上限较低，"
            "可换上下文更大的模型，或改用「精简摘要」减少单次送入的材料。"
        )
    if "insufficient_quota" in lowered or "quota" in lowered or "余额" in message:
        return (
            "模型通道的额度或余额不足，笔记生成中断。"
            "请在「模型配置」里换一个通道（Base URL 与 Key），或给当前通道续额后重投。"
        )
    if "429" in lowered or "rate limit" in lowered or "too many requests" in lowered:
        return (
            "模型通道限流。可以稍后重投，或改用付费通道；"
            "长视频换用「精简摘要」能明显减少请求数。"
        )
    if "401" in lowered or "invalid_api_key" in lowered or "incorrect api key" in lowered:
        return (
            "模型接口拒绝了本机保存的 Key（401）。"
            "请在「模型配置」里重新填写该 Base URL 对应的 Key 后重投。"
        )
    if "403" in lowered or "permission" in lowered or "forbidden" in lowered:
        return (
            "模型接口拒绝了本次请求（403）：本机保存的 Key 可能没有该模型的权限，"
            "或 Base URL 与模型名不匹配。请核对「模型配置」后重投，"
            "任务日志里「调用模型」一行就是本次实际使用的通道。"
        )
    if "404" in lowered or "not found" in lowered or "does not exist" in lowered:
        return "模型接口找不到该模型（404）。请检查「模型配置」里的模型名与 Base URL。"
    if "timeout" in lowered or "timed out" in lowered or "connection" in lowered:
        return "与模型接口的连接中断，多为网络抖动或通道限流。可以直接重投任务，或换一个通道。"
    return message


def redact_secrets(text: str, secret: str = "") -> str:
    """把文本里的凭据换成掩码。

    比正则更可靠的一层是 `secret`：调用方手里就有本次用的明文 Key，
    整串替换不依赖网关把回显写成什么格式。
    """
    cleaned = API_KEY_PATTERN.sub("sk-****", text)
    cleaned = SECRET_ECHO_PATTERN.sub(
        lambda match: f"{match.group(1)}{match.group(2)}****", cleaned
    )
    if secret and len(secret) >= 8:
        cleaned = cleaned.replace(secret, secret_box.mask_secret(secret))
    return cleaned


def raw_error_text(exc: BaseException) -> str:
    """异常的可读文本；``str()`` 为空时退回类型名。

    httpx 的 ``ReadError`` / ``WriteError`` / 各类 ``TimeoutException`` 在服务端
    提前拒绝时会抛**消息为空**的异常。直接用 ``str(exc)`` 会让日志里的「底层错误」
    变成空白——线上那次「底层错误：」后面什么都没有，就是这么来的。
    """
    text = str(exc).strip()
    return text or f"{type(exc).__name__}（该异常未附带文本消息）"


def scrub_technical_error(raw: str, secret: str = "") -> str:
    """异常文本进日志前的脱敏与截断：通道报错可能整段回显 JSON。"""
    return redact_secrets(" ".join(raw.split()), secret)[:400]


def format_duration(seconds: float) -> str:
    if not seconds:
        return "未知"
    return format_timestamp(float(seconds))


def format_bytes_size(num_bytes: float) -> str:
    """把字节数格式化为人类可读大小（MB / GB）。"""
    size = max(0.0, float(num_bytes))
    if size >= 1024 * 1024 * 1024:
        return f"{size / (1024 ** 3):.2f} GB"
    return f"{size / (1024 ** 2):.2f} MB"


def format_publish_time(info: dict[str, Any]) -> str:
    """优先使用 yt-dlp 的发布日期；没有日期时再使用 Unix 时间戳。"""
    raw_date = str(info.get("upload_date") or info.get("release_date") or "").strip()
    if len(raw_date) == 8 and raw_date.isdigit():
        return f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:]}"
    raw_timestamp = info.get("timestamp") or info.get("release_timestamp")
    try:
        timestamp = float(raw_timestamp)
    except (TypeError, ValueError):
        return ""
    if timestamp <= 0:
        return ""
    try:
        return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")
    except (OverflowError, OSError, ValueError):
        return ""


def add_note_header_metadata(summary: str, title: str, info: dict[str, Any]) -> str:
    """用代码写死 H1，并在其下放置可验证的视频元信息。

    H1 一律由这里写：``title`` 现在是 generate_note_title 产出的「短标题 + 风格
    标签」，模型在正文里顺手写的 H1 必须让位，否则标题会随模型发挥漂移，
    下载名与归档名也跟着变。
    """
    value = summary.strip()
    lines = value.splitlines()
    if lines and lines[0].lstrip().startswith("#"):
        body = "\n".join(lines[1:]).lstrip()
    else:
        body = value
    heading = f"# 视频笔记：《{title}》"

    body = strip_model_metadata_block(body)

    metadata: list[str] = []
    owner = str(info.get("owner") or "").strip()
    published_at = format_publish_time(info)
    duration = format_duration(info.get("duration", 0))
    view_count = int(info.get("view_count") or 0)
    like_count = int(info.get("like_count") or 0)
    if owner:
        metadata.append(f"UP主：{owner}")
    if published_at:
        metadata.append(f"发布时间：{published_at}")
    if duration != "未知":
        metadata.append(f"时长：{duration}")
    if view_count > 0:
        metadata.append(f"播放：{view_count:,}")
    if like_count > 0:
        metadata.append(f"点赞：{like_count:,}")
    if not metadata:
        return value or heading
    metadata_block = "> " + " · ".join(metadata)
    return f"{heading}\n\n{metadata_block}" + (f"\n\n{body}" if body else "")


def strip_model_metadata_block(body: str) -> str:
    """移除模型紧跟标题生成的重复元信息，保留正文内容。

    提示词已经把作者、时间、时长和文字来源作为参考资料传给模型，
    但模型仍可能输出一行 ``作者：…｜发布时间：…``。这些字段由代码
    确定性写入标题下方，因此只清理标题后的重复元信息块。
    """
    lines = body.splitlines()
    index = 0
    while index < len(lines) and not lines[index].strip():
        index += 1
    start = index
    metadata_labels = ("作者", "UP主", "发布时间", "时长", "播放", "点赞", "文字来源")
    removed = 0
    while index < len(lines):
        line = lines[index].strip().lstrip("> ")
        label_count = sum(f"{label}：" in line or f"{label}:" in line for label in metadata_labels)
        if label_count < 2:
            break
        removed += 1
        index += 1
        while index < len(lines) and not lines[index].strip():
            index += 1
    if not removed:
        return body
    return "\n".join(lines[index:]).lstrip()


def summary_style_label(style: str) -> str:
    return {
        "detailed": "详细笔记 + 点评分析",
        "faithful": "详细复原（仅视频内容）",
        "concise": "精简摘要",
        "keypoints": "要点提炼",
    }.get(style, style)


def short_style_label(style: str) -> str:
    """笔记标题里用的短标签（完整标签见 summary_style_label，留给页脚用）。"""
    return {
        "detailed": "详细",
        "faithful": "详细复原",
        "concise": "简要摘述",
        "keypoints": "要点提炼",
    }.get(style, style)


def composite_note_title(short_title: str, style: str) -> str:
    """拼出最终标题：``短标题（详细）``。短标题为空时返回空串，调用方据此降级。"""
    base = (short_title or "").strip()
    if not base:
        return ""
    return f"{base}（{short_style_label(style)}）"


def content_type_label(content_type: str) -> str:
    """内容场景的中文标签；general（或未知值）返回空串，页脚据此决定是否输出该行。"""
    return {
        "tutorial": "教程",
        "knowledge": "知识",
        "learning": "学习",
    }.get(content_type, "")


def append_note_footer(
    summary: str,
    source_url: str | None,
    model_type: str,
    model: str | None,
    style: str,
    content_type: str = "general",
) -> str:
    """确定性追加来源与生成配置，避免不同风格遗漏关键追溯信息。"""
    source = source_url.strip() if source_url else ""
    source_line = source or "本地文件（未提供视频链接）"
    model_line = model.strip() if model else "未指定模型"
    lines = [
        "---",
        f"来源：{source_line}",
        f"生成模型：{model_type} · {model_line}",
        f"笔记风格：{summary_style_label(style)}",
    ]
    scene = content_type_label(content_type)
    if scene:
        lines.append(f"内容场景：{scene}")
    footer = "\n".join(lines)
    return f"{summary.rstrip()}\n\n{footer}\n"


# 页脚是 append_note_footer 确定性写出来的，可以按结构定位：补写正文要插在它前面，
# 排在“来源／生成模型”之后会读起来像附注。
_NOTE_FOOTER_RE = re.compile(
    r"^---\n来源：.*\n生成模型：.*\n笔记风格：.*(?:\n内容场景：.*)?\n?\Z", re.MULTILINE
)


def insert_tail_patch(draft: str, body: str, missing_start: str) -> str:
    """把补写片段插到笔记页脚之前；没有页脚（用户改过）就接在末尾。"""
    section = f"### 补遗（{missing_start} 之后）\n\n{body.strip()}"
    match = _NOTE_FOOTER_RE.search(draft)
    if not match:
        return f"{draft.rstrip()}\n\n{section}\n"
    head = draft[: match.start()].rstrip()
    tail = draft[match.start() :].strip("\n")
    return f"{head}\n\n{section}\n\n{tail}\n"


def record_patch_result(
    task_id: str, markdown: str, coverage: dict[str, Any] | None, log_line: str
) -> None:
    """补写完成后更新任务状态：内存里有任务就顺带落盘，没有就直接改 task.json。

    服务重启后旧任务只在磁盘上，这时不能依赖 ``tasks``（``persist_task_runtime``
    以它为唯一来源），否则用户补写完旧笔记、界面刷新一看还是残的。
    """
    task = tasks.get(task_id)
    if task is not None and isinstance(task.get("result"), dict):
        task["result"]["markdown"] = markdown
        if coverage is not None:
            task["result"]["coverage"] = coverage
        if log_line not in task.get("logs", []):
            task.setdefault("logs", []).append(log_line)
        persist_task_runtime(task_id)
        return
    path = task_directory(task_id) / "task.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return
    runtime = payload.get("runtime")
    if not isinstance(runtime, dict):
        return
    result = runtime.get("result")
    if isinstance(result, dict) and coverage is not None:
        result["coverage"] = coverage
    logs = runtime.get("logs")
    if isinstance(logs, list) and log_line not in logs:
        logs.append(log_line)
    write_text_atomically(path, json.dumps(payload, ensure_ascii=False, indent=2))


# 笔记里的时间点写法：[MM:SS] 或 [MM:SS-MM:SS]（也容忍 [H:MM:SS] 与全角连接符）。
# 外层 (?<!\[) 与末尾 (?!\() 一起保证幂等：链接化后写成 [MM:SS](url)，
# 时间戳后面紧跟 "("，不会再被二次包裹；前面有 "[[" 的旧写法同样不会再被包一层。
_TIMESTAMP_PATTERN = re.compile(
    r"(?<!\[)\[(\d{1,3}:\d{2}(?::\d{2})?)(?:\s*[-–—~]\s*(\d{1,3}:\d{2}(?::\d{2})?))?\](?!\()"
)

# 代码块与行内代码里的方括号不是时间点，必须原样保留。按这段切分后，
# 奇数下标是代码片段、偶数下标是正文，只处理正文。
_CODE_SEGMENT_PATTERN = re.compile(r"(```.*?```|~~~.*?~~~|`[^`\n]*`)", re.DOTALL)


def timestamp_seconds(value: str) -> int | None:
    """把 MM:SS / HH:MM:SS 转成秒数；格式或取值范围不合法时返回 None。"""
    try:
        parts = [int(part) for part in value.split(":")]
    except ValueError:
        return None
    if len(parts) == 2:
        minutes, seconds = parts
        return None if seconds > 59 else minutes * 60 + seconds
    if len(parts) == 3:
        hours, minutes, seconds = parts
        if minutes > 59 or seconds > 59:
            return None
        return hours * 3600 + minutes * 60 + seconds
    return None


def _youtube_video_id(parsed: Any) -> str | None:
    """兼容 watch?v=、shorts、live、embed、v 五种链接形态。"""
    video_id = (parse_qs(parsed.query).get("v") or [""])[-1].strip()
    if video_id:
        return video_id
    for prefix in ("/shorts/", "/live/", "/embed/", "/v/"):
        if parsed.path.startswith(prefix):
            return parsed.path[len(prefix):].strip("/") or None
    return None


def seek_url(source_url: str | None, seconds: int) -> str | None:
    """把秒数变成平台播放器的定位链接；来源无可跳转地址时返回 None。

    B 站用 ?t=秒，YouTube 用 &t=秒s。链接一律丢掉分享追踪参数（trackid /
    spm_id_from / vd_source 等），分 P 链接保留 ?p=N。
    """
    value = (source_url or "").strip()
    if not value or seconds < 0:
        return None
    parsed = urlsplit(value)
    host = (parsed.hostname or "").lower()
    if not host or not parsed.path:
        return None

    if host == "bilibili.com" or host.endswith(".bilibili.com"):
        query = {"t": str(seconds)}
        page = (parse_qs(parsed.query).get("p") or [""])[-1]
        if page.isdigit() and int(page) > 1:
            query["p"] = str(int(page))
        return urlunsplit(("https", "www.bilibili.com", parsed.path, urlencode(query), ""))

    if host in {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com"}:
        video_id = _youtube_video_id(parsed)
        if not video_id:
            return None
        return urlunsplit(
            ("https", "www.youtube.com", "/watch", urlencode({"v": video_id, "t": f"{seconds}s"}), "")
        )

    if host == "youtu.be":
        video_id = parsed.path.strip("/")
        if not video_id:
            return None
        return urlunsplit(("https", "youtu.be", f"/{video_id}", urlencode({"t": f"{seconds}s"}), ""))

    return None


def linkify_timestamps(markdown: str, source_url: str | None) -> str:
    """把笔记里的时间点换成可点击跳转到原片对应位置的链接。

    区间取起点（[02:29-02:57] 跳到 02:29）。写成 ``[MM:SS](url)``——只留一层方括号：
    它既是 Markdown 链接语法，也让网址彻底不显示，渲染出来就是可点击的时段本身。
    旧写法 ``[[MM:SS]](url)`` 多包了一层，在 Obsidian 里会先被当成 ``[[双链]]`` 吃掉，
    外层链接失效、网址裸露成一大串，网页端（marked）却看不出问题。

    来源是本地文件等没有可跳转地址的情形原样返回；代码块与行内代码里的时间点
    也保持原样。
    """
    if not markdown or not source_url:
        return markdown

    def replace(match: re.Match[str]) -> str:
        seconds = timestamp_seconds(match.group(1))
        if seconds is None:
            return match.group(0)
        url = seek_url(source_url, seconds)
        if not url:
            return match.group(0)
        return f"{match.group(0)}({url})"

    parts = _CODE_SEGMENT_PATTERN.split(markdown)
    for index in range(0, len(parts), 2):
        parts[index] = _TIMESTAMP_PATTERN.sub(replace, parts[index])
    return "".join(parts)


def _timestamp_page_count(value: Any) -> int:
    """把 manifest 里的分 P 计数安全转成 int；脏数据当作 0，绝不因此崩掉任务。"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


SAFE_NAME_LIMIT = 80


def _safe_filename(value: str) -> str:
    """把标题清洗为安全的文件名：只去掉 Windows 非法字符，保留中文标点。"""
    invalid = set('\\/:*?"<>|')
    cleaned = "".join(char for char in value if char not in invalid and ord(char) >= 32)
    return (cleaned.strip() or "video-notes")[:SAFE_NAME_LIMIT]


def download_stem(title: str, meta: dict[str, Any], output: str = "note") -> str:
    """产物文件名主干（不含扩展名），Markdown 下载、压缩包、归档和网页端共用。

    B 站分 P 与纯转录都要带进名字：同一视频逐 P 跑出来的笔记、以及同一视频
    先转写后生成的两份产物，否则在下载目录里只剩一串相同或时间戳文件名。
    分 P 标记与转录正文里用的「【P3 分P名】」保持同一写法。
    """
    base = str(title or "").strip() or "video-notes"
    suffix = ""
    pages = [page for page in (meta.get("bili_pages") or []) if isinstance(page, dict)]
    total = int(meta.get("bili_total_pages") or 0)
    if total > 1 and pages:
        if len(pages) == 1:
            page = pages[0]
            part = str(page.get("part") or "").strip()
            suffix = f"【P{page.get('page')} {part}】" if part else f"【P{page.get('page')}】"
        elif len(pages) < total:
            suffix = f"【P{_format_page_nums([int(page['page']) for page in pages])}】"
    if output == "transcript":
        suffix += "【转录】"
    if suffix and len(base) + len(suffix) > SAFE_NAME_LIMIT:
        base = base[: max(1, SAFE_NAME_LIMIT - len(suffix))]
    return _safe_filename(f"{base}{suffix}")


def rewrite_note_image_refs(content: str, prefix: str) -> str:
    return NOTE_IMAGE_REF_RE.sub(
        lambda match: f"{match.group(1)}{prefix}{FRAMES_DIR_NAME}/{match.group(2)})",
        content,
    )


def notes_output_root(folder: str) -> Path:
    """成品（笔记 / 转录稿）的归档根目录。

    未配置 NOTES_OUTPUT_DIR：workspace/<folder>/，与历史行为一致。
    已配置：笔记直接放根目录，转录稿放 <根>/transcripts/ —— 两者仍分开摆。
    """
    if not NOTES_OUTPUT_DIR:
        return WORKSPACE_DIR / folder
    base = Path(NOTES_OUTPUT_DIR).expanduser()
    return base if folder == NOTES_DIR_NAME else base / folder


def resolve_note_in_vault(note_path: str) -> Path | None:
    """把客户端传来的 note_path 收口到知识库目录内。不在库内一律返回 None。

    落盘（写报告）与回写（补双链）都按这个判断走：note_path 是请求体里的字符串，
    不校验就等于允许调用方指定任意可写目录——写报告按 note_path.parent 落盘、
    补双链对任意已存在文件追加，两条路都成了任意文件写。
    """
    text = (note_path or "").strip()
    if not text:
        return None
    try:
        candidate = Path(text).expanduser().resolve()
    except (OSError, RuntimeError):
        return None
    for folder in (NOTES_DIR_NAME, TRANSCRIPTS_DIR_NAME):
        try:
            if candidate.parent == notes_output_root(folder).resolve():
                return candidate
        except (OSError, RuntimeError):
            continue
    return None


SPROUT_REPORT_SUFFIX = " · 发芽报告"

# 单篇「思考」的三个模式。每个模式的产物都是独立一篇笔记，区别只在这几处文案：
#   suffix         文件名后缀（发芽沿用旧后缀，已有笔记里的双链才不会变死链）
#   fallback_title 模型没给标题时的兜底
#   footer         文末那行来源说明
#   link_label     追加到原笔记末尾的回链标签
#   name           API / 前端用的中文名
THINK_MODES: dict[str, dict[str, str]] = {
    "sprout": {
        "suffix": SPROUT_REPORT_SUFFIX,
        "fallback_title": "# 🌱 新知卡片",
        "footer": "由 AINote 发芽生成（按「得到大脑」发芽模式：种子 → 跨界碰撞 → 启发）",
        "link_label": "发芽报告",
        "name": "发芽",
    },
    "interrogate": {
        "suffix": " · 拷问",
        "fallback_title": "# ⚔️ 拷问",
        "footer": "由 AINote 拷问生成（按「得到大脑」拷问模式：挑漏洞 + 强制替代方案，"
        "每条判断都附了原片定位）",
        "link_label": "拷问",
        "name": "拷问",
    },
    "quiz": {
        "suffix": " · 自测",
        "fallback_title": "# 🎯 自测",
        "footer": "由 AINote 自测生成（题目与答案，答案里附了原片定位）",
        "link_label": "自测",
        "name": "自测",
    },
}


def write_think_report(
    report: str, title: str, note_path: str, mode: str = "sprout"
) -> tuple[Path | None, str]:
    """把「思考」产物（发芽 / 拷问 / 自测）落成一篇独立笔记，并与原笔记双链互连。

    报告里写 ``[[原笔记名]]``：Obsidian 会自动在原笔记的「反向链接」面板里列出它，
    这就是两个文件被连起来的方式（不需要改动原笔记也能看到）。

    模型输出的第一行就是它的标题（``# 🌱 新知卡片：…`` / ``# ⚔️ 拷问：…`` /
    ``# 🎯 自测：…``），这里只负责把「原笔记」双链插到标题下面，不再自己造标题。

    返回 ``(报告路径, 原笔记名)``；写盘失败返回 ``(None, "")``，不影响本次结果本身。
    """
    spec = THINK_MODES.get(mode) or THINK_MODES["sprout"]
    try:
        note_file = resolve_note_in_vault(note_path)
        if note_file is not None and not note_file.is_file():
            note_file = None
        stem = note_file.stem if note_file else _safe_filename(title or "视频笔记")
        archive_root = note_file.parent if note_file else notes_output_root(NOTES_DIR_NAME)
        archive_root.mkdir(parents=True, exist_ok=True)
        heading = title.strip() or stem
        lines = report.strip().splitlines()
        if lines and lines[0].lstrip().startswith("#"):
            card_title, body = lines[0].strip(), lines[1:]
        else:
            card_title, body = f"{spec['fallback_title']}：《{heading}》", lines
        # 模型常在标题后留一个空行，这里吃掉，免得标题和双链之间空两行
        while body and not body[0].strip():
            body.pop(0)
        content = "\n".join(
            [
                card_title,
                "",
                f"> 原笔记：[[{stem}]]",
                "",
                *body,
                "",
                "---",
                spec["footer"],
                "",
            ]
        )
        target = archive_root / f"{stem}{spec['suffix']}.md"
        write_text_atomically(target, content)
        return target, stem
    except OSError:
        return None, ""


def write_sprout_report(
    report: str, title: str, note_path: str
) -> tuple[Path | None, str]:
    """发芽模式的落盘（历史调用点与测试仍在用这个名字）。"""
    return write_think_report(report, title, note_path, "sprout")


def link_sprout_from_note(
    note_path: str, report_stem: str, label: str = "发芽报告"
) -> bool:
    """在原笔记末尾补一行指向发芽产物的双链，让双向跳转都能点。

    报告 → 原笔记的链接（写在报告里）已经能让原笔记的反向链接面板显示报告；
    这一行是反向的显式出口。幂等：笔记里已经提到过报告名就不再重复追加。

    原笔记是用户唯一的一份数据，因此这里先用临时文件写完再 ``os.replace`` 顶上，
    并且先留一份 ``.bak``——直接 ``write_text`` 是"先截断再写"，写到一半崩溃/断电
    就把笔记本身弄没了，而这不是本程序有权造成的结果。
    """
    if not report_stem:
        return False
    note_file = resolve_note_in_vault(note_path)
    if note_file is None or not note_file.is_file():
        return False
    try:
        text = note_file.read_text(encoding="utf-8")
        if report_stem in text:
            return True
        backup = note_file.with_name(note_file.name + ".bak")
        try:
            shutil.copy2(note_file, backup)
        except OSError:
            pass
        payload = f"{text.rstrip()}\n\n🌱 {label}：[[{report_stem}]]\n"
        temporary = note_file.with_name(note_file.name + ".tmp")
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, note_file)
        return True
    except OSError:
        return False


AGGREGATE_SPROUT_MAX_NOTES = 8
AGGREGATE_SPROUT_NOTE_CHARS = 1500


def _note_stem_for_task(task_id: str, title: str) -> tuple[str | None, Path | None]:
    """这篇任务的笔记是不是在知识库里？是的话返回 (文件名主干, 绝对路径)。

    归档路径可能指向任意目录（历史里混着当初跑测试、落在 /tmp 的任务），所以只有
    真的落在当前知识库根目录下才算「可双链」，其余只当素材、不生成死链。
    """
    task = tasks.get(task_id) or {}
    result = task.get("result") if isinstance(task.get("result"), dict) else {}
    root = notes_output_root(NOTES_DIR_NAME)
    archived = str(result.get("archived_path") or "").strip()
    if archived:
        path = Path(archived).expanduser()
        try:
            if path.resolve().parent == root.resolve() and path.is_file():
                return path.stem, path
        except OSError:
            pass
    # 回退：按标题猜知识库里的文件名（老任务没有 archived_path）
    guess = root / f"{_safe_filename(title)}.md"
    if guess.is_file():
        return guess.stem, guess
    return None, None


def collect_aggregate_notes(task_ids: Sequence[str]) -> list[dict[str, Any]]:
    """把勾选的历史任务整理成整体发芽的素材。

    三件事：① 正文优先取内存里的 result.markdown，回退读任务目录的 notes.md
    （服务重启后内存里可能已经没有了）；② 同一篇笔记只保留一次——同一条视频
    跑过多遍时标题和归档名都一样，重复素材会把「共同种子」带偏；
    ③ 逐篇截断，避免八篇长笔记把上下文撑爆。
    """
    collected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for task_id in task_ids:
        if len(collected) >= AGGREGATE_SPROUT_MAX_NOTES:
            break
        try:
            directory = task_directory(task_id)
        except ValueError:
            continue
        task = tasks.get(task_id) or {}
        result = task.get("result") if isinstance(task.get("result"), dict) else {}
        markdown = str(result.get("markdown") or "").strip()
        if not markdown:
            try:
                markdown = (directory / "notes.md").read_text(encoding="utf-8").strip()
            except OSError:
                markdown = ""
        if not markdown:
            continue
        manifest = read_task_manifest(task_id)
        title = (
            str(result.get("title") or "").strip()
            or str(manifest.get("title") or "").strip()
            or "未命名笔记"
        )
        stem, path = _note_stem_for_task(task_id, title)
        key = stem or f"title:{title}"
        if key in seen:
            continue
        seen.add(key)
        collected.append(
            {
                "task_id": task_id,
                "title": title,
                "markdown": markdown[:AGGREGATE_SPROUT_NOTE_CHARS],
                "stem": stem,
                "path": str(path) if path else "",
            }
        )
    return collected


def build_aggregate_material(notes: Sequence[dict[str, Any]]) -> str:
    """把多篇笔记拼成给模型看的素材（分篇编号带标题，避免串味）。"""
    return "\n\n".join(
        f"【第 {index} 篇 · {note['title']}】\n{note['markdown']}"
        for index, note in enumerate(notes, start=1)
    )


def aggregate_report_stem(notes: Sequence[dict[str, Any]]) -> str:
    """整体发芽卡片的文件名主干——**按选中的笔记定，不按种子概括定**。

    种子是模型每次现写的，同一批笔记两次能给出一句措辞完全不同的概括。要是拿它
    当文件名，覆盖就永远不生效：文件越堆越多，每篇原笔记末尾的回链也跟着一条条
    加下去（实测同一批笔记跑四次长出四份文件、四条回链）。按笔记定名，同一批
    笔记反复发芽才会稳定覆盖同一份，回链也保持一条。
    """
    stems = sorted({note["stem"] for note in notes if note.get("stem")})
    if not stems:
        stems = [_safe_filename(str(notes[0]["title"])) if notes else "未命名"]
    head = stems[0] if len(stems) == 1 else f"{stems[0]}等{len(stems)}篇"
    return f"整体发芽 · {_safe_filename(head)}"


def write_aggregate_sprout_report(
    report: str, notes: Sequence[dict[str, Any]]
) -> tuple[Path | None, str]:
    """把整体发芽的新知卡片落盘，并与每一篇原笔记双链互连。

    只有真产出卡片（首行是 ``# 🌱 新知卡片：``）才落盘——模型判断「这几篇没有
    共同种子」时不会输出那个标题，此时不生成文件，免得往知识库里塞一篇空东西。

    返回 ``(报告路径, 种子概括)``；不落盘时返回 ``(None, "")``。
    """
    lines = report.strip().splitlines()
    if not lines or not lines[0].lstrip().startswith("#"):
        return None, ""
    card_title = lines[0].strip()
    seed = card_title.split("：", 1)[-1].strip() if "：" in card_title else card_title
    body = lines[1:]
    while body and not body[0].strip():
        body.pop(0)

    linked = [note for note in notes if note.get("stem")]
    origin = "｜".join(f"[[{note['stem']}]]" for note in linked) or "（均不在知识库目录）"
    try:
        root = notes_output_root(NOTES_DIR_NAME)
        root.mkdir(parents=True, exist_ok=True)
        target = root / f"{aggregate_report_stem(notes)}.md"
        write_text_atomically(
            target,
            "\n".join(
                [
                    card_title,
                    "",
                    f"> 原笔记（{len(linked)} 篇）：{origin}",
                    "",
                    *body,
                    "",
                    "---",
                    "由 AINote 整体发芽生成"
                    "（按「得到大脑」发芽模式：跨篇种子 → 跨界碰撞 → 启发）",
                    "",
                ]
            ),
        )
        for note in linked:
            link_sprout_from_note(note["path"], target.stem, label="整体发芽")
        return target, seed
    except OSError:
        return None, ""


def _relocate_note_images(
    content: str, archive_root: Path, stem: str, task_id: str
) -> str:
    """按归档位置决定截图引用怎么改写（详见 archive_note 的说明）。"""
    if archive_root.is_relative_to(WORKSPACE_DIR):
        # 归档与任务目录同级：就地引用，不复制文件
        return rewrite_note_image_refs(content, f"../{task_id}/")
    frames_src = WORKSPACE_DIR / task_id / FRAMES_DIR_NAME
    if not frames_src.is_dir():
        return content
    assets_dir = archive_root / f"{stem}.assets"
    try:
        assets_dir.mkdir(parents=True, exist_ok=True)
        for frame in sorted(frames_src.iterdir()):
            if frame.is_file():
                shutil.copy2(frame, assets_dir / frame.name)
    except OSError:
        # 复制失败退回原引用：正文照常归档，比整篇丢掉强
        return content
    return NOTE_IMAGE_REF_RE.sub(
        lambda match: f"{match.group(1)}./{assets_dir.name}/{match.group(2)})",
        content,
    )


def archive_note(
    title: str,
    content: str,
    task_id: str | None = None,
    folder: str = NOTES_DIR_NAME,
) -> Path | None:
    """把产物按标题命名归档（重名自动加序号），便于集中回顾。

    归档位置默认 workspace/<folder>/；设 NOTES_OUTPUT_DIR 就改到项目外的知识库目录。

    截图引用有两种处理：归档在 workspace 内时改写成 ../<task_id>/frames/ 就地引用、
    不复制文件；归档在 workspace 外时把该任务的 frames/ 复制到 <笔记名>.assets/，
    改写成 ./<笔记名>.assets/，让知识库自包含（删掉任务目录、换机器都不丢图）。

    返回归档路径；写入失败时返回 None（不影响任务本身）。
    """
    archive_root = notes_output_root(folder)
    try:
        archive_root.mkdir(parents=True, exist_ok=True)
        safe = _safe_filename(title)
        candidate = archive_root / f"{safe}.md"
        index = 2
        while candidate.exists():
            candidate = archive_root / f"{safe}-{index}.md"
            index += 1
        if task_id:
            content = _relocate_note_images(content, archive_root, candidate.stem, task_id)
        # 用独占创建收口"检查是否存在 → 写入"之间的窗口：两个任务同时归档同名笔记时，
        # candidate.exists() 都返回 False，后写的那个会把先写的覆盖掉。
        while True:
            try:
                with open(candidate, "x", encoding="utf-8") as handle:
                    handle.write(content)
                return candidate
            except FileExistsError:
                candidate = archive_root / f"{safe}-{index}.md"
                index += 1
    except OSError:
        return None


def restore_tasks_from_workspace() -> None:
    """Rebuild recent task status from manifests after a service restart."""
    try:
        candidates = sorted(
            (
                path
                for path in WORKSPACE_DIR.iterdir()
                if path.is_dir() and (path / "task.json").is_file()
            ),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )[:TASK_HISTORY_LIMIT]
    except OSError:
        return

    for task_dir in candidates:
        try:
            manifest = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        task_id = task_dir.name
        runtime = manifest.get("runtime") if isinstance(manifest.get("runtime"), dict) else {}
        # 纯转录任务按设计不产出 notes.md，完成与否只能看 transcript.json
        transcript_only = manifest.get("output") == "transcript"
        notes_path = task_dir / "notes.md"
        uploaded_media = next(
            (path for path in task_dir.glob("input.*") if path.is_file()), None
        )
        status = str(runtime.get("status") or "")
        interrupted = status in {"pending", "queued", "processing", "cancelling"}
        if notes_path.is_file() or (
            transcript_only and (task_dir / "transcript.json").is_file()
        ):
            status = "completed"
        elif interrupted:
            status = "failed"
        elif status not in {"failed", "cancelled", "uploaded"}:
            status = "uploaded" if uploaded_media else "failed"

        task = new_task(status=status, task_id=task_id)
        task["_started_monotonic"] = None
        task.update(
            step=int(runtime.get("step") or 0),
            step_name=str(runtime.get("step_name") or "已恢复"),
            progress=int(runtime.get("progress") or (100 if status == "completed" else 0)),
            logs=list(runtime.get("logs") or []),
            error=runtime.get("error"),
            created_at=runtime.get("created_at") or manifest.get("created_at"),
            finished_at=runtime.get("finished_at"),
            elapsed_seconds=float(runtime.get("elapsed_seconds") or 0),
            output=str(manifest.get("output") or "note"),
            # 旧版 manifest 没有 origin：按 agent 处理，维持这类任务此前不进历史的表现
            _origin=str(manifest.get("origin") or "agent"),
        )
        if interrupted:
            task.update(
                step_name="服务已重启",
                error="任务因服务重启而中断；已有转录和音频仍可复用",
                finished_at=datetime.now(timezone.utc).isoformat(),
            )
            task["logs"].append("服务重启时任务尚未结束，已标记为中断")
        if status == "uploaded" and uploaded_media:
            task["uploaded_file_path"] = str(uploaded_media)
            task["uploaded_filename"] = manifest.get("uploaded_filename") or uploaded_media.name
        if status == "completed" and transcript_only:
            # 纯转录任务没有 notes.md，正文留在 transcript.json 里
            result = dict(runtime.get("result") or {})
            result.update(
                title=result.get("title") or manifest.get("title") or "",
                output="transcript",
                output_directory=str(task_dir),
                processing_seconds=task["elapsed_seconds"],
            )
            task["result"] = result
            task["finished_at"] = task["finished_at"] or datetime.now(timezone.utc).isoformat()
        elif status == "completed":
            try:
                markdown = notes_path.read_text(encoding="utf-8")
            except OSError:
                continue
            result = dict(runtime.get("result") or {})
            result.update(
                title=result.get("title") or manifest.get("title") or "video-notes",
                markdown=markdown,
                output_directory=str(task_dir),
                processing_seconds=task["elapsed_seconds"],
            )
            task["result"] = result
            task["finished_at"] = task["finished_at"] or datetime.now(timezone.utc).isoformat()
        tasks[task_id] = task
        if interrupted:
            persist_task_runtime(task_id)


restore_tasks_from_workspace()


@app.get("/icon.png", include_in_schema=False)
async def app_icon() -> FileResponse:
    if not APP_ICON_PATH.is_file():
        raise HTTPException(status_code=404, detail="应用图标不存在")
    return FileResponse(APP_ICON_PATH, media_type="image/png")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> FileResponse:
    if not FAVICON_PATH.is_file():
        raise HTTPException(status_code=404, detail="应用图标不存在")
    return FileResponse(FAVICON_PATH, media_type="image/x-icon")


if FRONTEND_DIR.is_dir():
    app.mount(
        "/",
        NoCacheStaticFiles(directory=FRONTEND_DIR, html=True),
        name="frontend",
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    os.environ.setdefault("AINOTE_BACKEND_URL", f"http://127.0.0.1:{port}")
    uvicorn.run(
        "backend.main:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=port,
        reload=os.getenv("RELOAD", "false").lower() == "true",
    )
