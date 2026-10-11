"""GitHub Release 更新检查：网页端启动自检（/api/update/check）与托盘菜单共用。

版本比较与接口地址在这里只有一份。launcher 的 ``latest_release_info`` 委托到
``fetch_latest_release``——它的测试钉住了 urlopen 的调用点（launcher 命名空间）
和 timeout=5，所以委托时不能再包一层自己的网络调用。
"""
from __future__ import annotations

import json
import urllib.request
from typing import Any

GITHUB_LATEST_RELEASE_API = "https://api.github.com/repos/like-attract/video-to-note/releases/latest"
DEFAULT_RELEASE_URL = "https://github.com/like-attract/video-to-note/releases/latest"


def version_tuple(value: str) -> tuple[int, ...]:
    """把 v1.2.3 或 1.2.3-rc1 转成可比较的版本元组。"""
    text = str(value or "").strip().lstrip("vV")
    parts: list[int] = []
    for part in text.split("."):
        digits = ""
        for character in part:
            if not character.isdigit():
                break
            digits += character
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts or [0])


def fetch_latest_release(current_version: str, timeout: float = 5.0) -> dict[str, Any]:
    """查询 GitHub 最新 Release；网络/解析失败抛异常，由调用方决定怎么呈现。"""
    request = urllib.request.Request(
        GITHUB_LATEST_RELEASE_API,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": f"VideoToNo/{current_version}",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    tag = str(payload.get("tag_name") or "").strip()
    if not tag:
        raise RuntimeError("GitHub 未返回有效版本号")
    return {
        "tag_name": tag,
        "html_url": str(payload.get("html_url") or DEFAULT_RELEASE_URL),
        "update_available": version_tuple(tag) > version_tuple(current_version),
        "notes": str(payload.get("body") or "")[:600],
    }


def check_summary(current_version: str, timeout: float = 5.0) -> dict[str, Any]:
    """端点用的安全封装：失败不抛，返回 ok=False 让前端静默。"""
    try:
        info = fetch_latest_release(current_version, timeout)
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}".encode("utf-8", "replace").decode("utf-8", "replace")
        return {"ok": False, "error": reason[:200], "current_version": current_version}
    return {
        "ok": True,
        "current_version": current_version,
        "latest_version": info["tag_name"],
        "update_available": info["update_available"],
        "release_url": info["html_url"],
        "notes": info["notes"],
    }
