"""抖音 Web 视频详情客户端。

**本机实测结论（决定了这里的实现）**：

1. 抖音 ``/aweme/v1/web/aweme/detail/`` 由 ``ArgusSecurityPlugin`` 把关；
2. 用 a_bogus 签名直调 → ``403 ... Sign Invalid``（签名反而把请求弄坏了）；
3. 带浏览器全套 cookie 的 Python 请求（**不带签名**）→ 仍 ``403 ... Uifid Not Found``；
4. 只有**在真实浏览器页面里用 JS ``fetch`` 发、且不带任何签名**时才成功
   （实测拿回完整 ``aweme_detail``）。

原因是 ``Uifid`` 并不是一个普通 cookie，而是抖音前端 SDK 在 XHR/fetch 上注入的
**请求头**——只有页面内的 JS 请求才会带上它。因此真正干活的是
``backend/douyin_browser.py``（浏览器里 in-page fetch）；本模块负责
「构造请求参数 / 解析响应 / 把结果翻译成统一字典」。

``signed_detail_url`` 与 ``backend/douyin_sign/`` 保留作参考与将来备用，但
**默认请求路径不使用签名**。
"""
from __future__ import annotations

import gzip
import json
import random
import re
import string
import time
import urllib.error
import urllib.request
from typing import Any

# 注意：签名算法依赖 gmssl，这里刻意**延迟导入**——否则缺 gmssl 时
# `import video_processor` 会直接失败，整个应用起不来（连 B 站/本地视频都用不了）。
# 延迟到真正要用签名时才 import，缺依赖只影响抖音这一条路，并抛出可读的错误。

# 与 f2 的 conf 保持一致：Edge 浏览器 UA + 匹配的指纹。
DOUYIN_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36 Edg/130.0.0.0"
)
DOUYIN_REFERER = "https://www.douyin.com/"
DETAIL_ENDPOINT = "https://www.douyin.com/aweme/v1/web/aweme/detail/"

# msToken 申请接口与参数（照抄 f2 conf/conf.yaml 的 douyin.msToken 段）。
MSTOKEN_URL = "https://mssdk.bytedance.com/web/report"
MSTOKEN_MAGIC = 538969122
MSTOKEN_VERSION = 1
MSTOKEN_DATA_TYPE = 8
MSTOKEN_STR_DATA = (
    "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"
)

_AWEME_ID_PATTERNS = (
    re.compile(r"/video/(\d{8,})"),
    re.compile(r"[?&]modal_id=(\d{8,})"),
    re.compile(r"/share/video/(\d{8,})"),
)


def extract_aweme_id(url: str) -> str | None:
    """从抖音链接直接取作品 ID（``/video/<id>`` 或 ``?modal_id=<id>``）。

    覆盖「网页版复制的个人主页 + modal_id」这种形态：链接本身不带 ``/video/``，
    作品 ID 藏在 ``modal_id`` 查询参数里。取不到时由调用方跟随短链重定向后再试。
    """
    for pattern in _AWEME_ID_PATTERNS:
        match = pattern.search(url or "")
        if match:
            return match.group(1)
    return None


def generate_false_ms_token() -> str:
    """申请不到真 msToken 时的兜底随机串（f2 同款：126 位 + ``==``）。"""
    alphabet = string.ascii_letters + string.digits + "_-"
    return "".join(random.choices(alphabet, k=126)) + "=="


def generate_ms_token(user_agent: str = DOUYIN_USER_AGENT, timeout: int = 15) -> str:
    """向 mssdk 申请真实 msToken；任何失败都返回空串，由调用方退化成随机串。"""
    payload = json.dumps(
        {
            "magic": MSTOKEN_MAGIC,
            "version": MSTOKEN_VERSION,
            "dataType": MSTOKEN_DATA_TYPE,
            "strData": MSTOKEN_STR_DATA,
            "tspFromClient": int(time.time() * 1000),
        }
    )
    request = urllib.request.Request(
        MSTOKEN_URL,
        data=payload.encode("utf-8"),
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "User-Agent": user_agent,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            cookies = response.headers.get_all("Set-Cookie") or []
    except Exception:
        return ""
    for cookie in cookies:
        if cookie.startswith("msToken="):
            value = cookie.split(";", 1)[0].split("=", 1)[1].strip()
            # f2 只接受 120/128 长度；其它长度视为无效
            return value if len(value) in (120, 128) else ""
    return ""


def build_detail_params(aweme_id: str, ms_token: str) -> dict[str, Any]:
    """构造 detail 接口查询参数。

    字段与顺序照搬 f2 的 ``BaseRequestModel`` + ``PostDetail.aweme_id``：
    a_bogus 是把这串参数按该顺序拼接后签名的，顺序变了签名就不匹配。
    """
    return {
        "device_platform": "webapp",
        "aid": "6383",
        "channel": "channel_pc_web",
        "pc_client_type": 1,
        "publish_video_strategy_type": 2,
        "pc_libra_divert": "Windows",
        "version_code": "290100",
        "version_name": "29.1.0",
        "cookie_enabled": "true",
        "screen_width": 1920,
        "screen_height": 1080,
        "browser_language": "zh-CN",
        "browser_platform": "Win32",
        "browser_name": "Edge",
        "browser_version": "130.0.0.0",
        "browser_online": "true",
        "engine_name": "Blink",
        "engine_version": "130.0.0.0",
        "os_name": "Windows",
        "os_version": "10",
        "cpu_core_num": 12,
        "device_memory": 8,
        "platform": "PC",
        "downlink": 10,
        "effective_type": "4g",
        "round_trip_time": 100,
        "msToken": ms_token,
        "aweme_id": aweme_id,
    }


def build_detail_url(params: dict[str, Any]) -> str:
    """把参数拼成 ``k=v&...`` 的 detail 请求 URL（**不带签名**）。

    实测：带 a_bogus 会被判 ``Sign Invalid``，不带签名反而正常返回。
    """
    param_str = "&".join(f"{key}={value}" for key, value in params.items())
    return f"{DETAIL_ENDPOINT}?{param_str}"


def signed_detail_url(
    params: dict[str, Any], user_agent: str = DOUYIN_USER_AGENT
) -> str:
    """带 a_bogus 签名的变体。

    ⚠️ 实测该变体会被抖音判 ``403 Sign Invalid``，**默认不要用**；保留它是为了
    记录签名算法、以及将来接口要求变化时备用（对应 ``backend/douyin_sign/``）。
    """
    try:
        from .douyin_sign import ABogus, BrowserFingerprintGenerator
    except ImportError as exc:  # 缺 gmssl：只让抖音这条路报错，不影响其它来源
        raise RuntimeError(
            "抖音签名需要 gmssl 依赖，请先执行 pip install -r backend/requirements.txt"
        ) from exc
    param_str = "&".join(f"{key}={value}" for key, value in params.items())
    fingerprint = BrowserFingerprintGenerator.generate_fingerprint("Edge")
    a_bogus = ABogus(fp=fingerprint, user_agent=user_agent).generate_abogus(
        param_str, ""
    )[1]
    return f"{build_detail_url(params)}&a_bogus={a_bogus}"


def _get_json(
    url: str,
    cookie_header: str,
    user_agent: str,
    referer: str = DOUYIN_REFERER,
    timeout: int = 20,
) -> tuple[int, Any]:
    """GET 并解析 JSON（自动解 gzip）。HTTP 错误也返回状态码 + 解释体，不抛。"""
    headers = {"User-Agent": user_agent, "Referer": referer}
    if cookie_header:
        headers["Cookie"] = cookie_header
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            if response.headers.get("Content-Encoding") == "gzip":
                raw = gzip.decompress(raw)
            return response.status, json.loads(raw.decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read().decode("utf-8", "replace")
        except Exception:
            body = ""
        return exc.code, {"_error": body[:400], "_reason": str(exc.reason)}
    except Exception as exc:
        return 0, {"_error": f"{type(exc).__name__}: {exc}"}


def _url_list(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return []
    return [str(item) for item in (value.get("url_list") or []) if item]


def info_from_aweme_detail(detail: dict[str, Any]) -> dict[str, Any]:
    """把 detail 接口的 ``aweme_detail`` 翻成本项目统一的 info 字典（纯函数）。"""
    video = detail.get("video") or {}
    author = detail.get("author") or {}
    statistics = detail.get("statistics") or {}
    video_urls = _url_list(video.get("play_addr"))
    audio_urls = _url_list((detail.get("music") or {}).get("play_url"))
    # playwm -> play 去掉水印，与旧的分享页解析保持一致
    video_url = next(
        (value.replace("playwm", "play") for value in video_urls if value), None
    )
    audio_url = next((value for value in audio_urls if value), None)
    covers = (
        _url_list(video.get("origin_cover"))
        or _url_list(video.get("cover"))
        or [""]
    )
    return {
        "title": detail.get("desc") or "Untitled video",
        "source": "douyin",
        "duration": round(float(video.get("duration") or 0) / 1000, 3),
        "owner": author.get("nickname") or author.get("unique_id") or "",
        "upload_date": "",
        "timestamp": detail.get("create_time") or 0,
        "description": (detail.get("desc") or "")[:1_000],
        "thumbnail": covers[0],
        "view_count": int(statistics.get("play_count") or 0),
        "like_count": int(statistics.get("digg_count") or 0),
        "subtitles": {},
        "automatic_captions": {},
        "_douyin_video_url": video_url,
        "_douyin_audio_url": audio_url,
        "_douyin_aweme_id": str(detail.get("aweme_id") or ""),
    }


def fetch_video_detail(
    aweme_id: str,
    cookie_header: str = "",
    user_agent: str = DOUYIN_USER_AGENT,
) -> dict[str, Any]:
    """按 aweme_id 取作品详情，返回统一 info 字典（**不带签名**的 Python 直调）。

    实测：这条路即使带上浏览器全套 cookie 也会被 ``Uifid Not Found`` 拦下
    （Uifid 是页面 JS 注入的请求头，见模块 docstring），所以它只是快速探测 /
    将来接口放宽时的备用；真正干活的是 ``douyin_browser`` 的 in-page fetch。
    """
    ms_token = generate_ms_token(user_agent) or generate_false_ms_token()
    params = build_detail_params(aweme_id, ms_token)
    url = build_detail_url(params)
    status, payload = _get_json(url, cookie_header, user_agent)
    if status != 200:
        body = str(payload.get("_error") or "") if isinstance(payload, dict) else ""
        lowered = body.lower()
        if "uifid" in lowered or "argussecurity" in lowered:
            raise RuntimeError(
                "抖音风控拦截（ArgusSecurityPlugin/Uifid）：Uifid 是浏览器 JS 注入的"
                "请求头，Python 直调拿不到；请走浏览器通道"
            )
        raise RuntimeError(f"抖音 detail 接口返回 HTTP {status}：{body[:200]}")
    if not isinstance(payload, dict):
        raise RuntimeError("抖音 detail 接口返回的不是 JSON")
    code = payload.get("status_code")
    if code not in (0, None):
        raise RuntimeError(f"抖音 detail 接口 status_code={code}（作品可能已删除或不可见）")
    detail = payload.get("aweme_detail")
    if not isinstance(detail, dict):
        raise RuntimeError("抖音 detail 接口未返回 aweme_detail（作品可能已删除或不可见）")
    return info_from_aweme_detail(detail)
