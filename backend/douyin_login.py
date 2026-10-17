"""抖音扫码登录：复用 B 站那套「独立浏览器 + CDP 读 cookie」思路。

和 ``bili_login.py`` 的差别只有三处：

1. 打开的是抖音首页（扫码入口在页面里），不是 passport 登录页；
2. cookie 字段更多（``ttwid`` / ``msToken`` / ``sessionid`` / ``Uifid`` ...），
   按 ``DOUYIN_COOKIE_NAMES`` 白名单取；
3. 「已登录」判定为出现了 ``sessionid`` / ``sessionid_ss`` / ``sid_guard``。

浏览器配置存 ``_douyin_profile``，与 B 站的 ``_bili_profile`` 互不干扰。
底层那套跨平台杀进程逻辑直接复用 ``BiliLoginManager._kill_profile_browsers``，
不再抄一遍。
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import websockets

from .bili_login import BiliLoginManager, find_browser, free_cdp_port
from .video_processor import DOUYIN_COOKIE_NAMES

LOGIN_URL = "https://www.douyin.com/"
SESSION_TTL_SECONDS = 300
# 出现任一个即视为「已登录」（这些字段只有登录后才有）
LOGIN_COOKIE_NAMES = ("sessionid", "sessionid_ss", "sid_guard")


def _no_proxy_opener() -> urllib.request.OpenerDirector:
    """CDP 的本机 HTTP 调用绝不能挂系统代理（与 bili_login 同因）。"""
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


_CDP_OPENER = _no_proxy_opener()
_kill_profile_browsers = BiliLoginManager._kill_profile_browsers


@dataclass
class DouyinLoginSession:
    browser_path: str
    profile_dir: Path
    cdp_port: int
    started_at: float
    process: subprocess.Popen[Any] | None = field(default=None)
    state: str = "waiting"  # waiting | ready | failed | timeout
    message: str = ""
    cookies: dict[str, str] | None = None


class DouyinLoginManager:
    def __init__(self, profile_root: Path) -> None:
        self.profile_root = profile_root
        self.session: DouyinLoginSession | None = None

    async def start(self) -> dict[str, Any]:
        existing = self.session
        if existing and self._alive() and existing.state != "timeout":
            return self._describe()
        self._terminate_browser()

        browser = await asyncio.to_thread(find_browser)
        if not browser:
            return {"ok": False, "error": "未找到 Edge 或 Chrome，请手动填写凭据"}
        port = free_cdp_port()
        if port is None:
            return {"ok": False, "error": "没有可用的调试端口，请稍后重试"}

        profile_dir = self.profile_root / "_douyin_profile"
        profile_dir.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(_kill_profile_browsers, profile_dir)
        args = [
            browser,
            f"--user-data-dir={profile_dir}",
            f"--remote-debugging-port={port}",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-sync",
            "--disable-extensions",
            "--disable-component-update",
            "--disable-background-networking",
            "--window-size=430,700",
            LOGIN_URL,
        ]
        cookies = None
        for retry in range(2):
            try:
                process = subprocess.Popen(args)
            except OSError as exc:
                return {"ok": False, "error": f"启动浏览器失败：{exc}"}

            self.session = DouyinLoginSession(
                browser_path=browser,
                profile_dir=profile_dir,
                cdp_port=port,
                started_at=time.time(),
                process=process,
            )
            cookies = None
            for _ in range(30):
                if process.poll() is not None:
                    break
                cookies = await self._fetch_cookies()
                if cookies:
                    break
                await asyncio.sleep(0.25)
            if cookies:
                break
            if process.poll() is None:
                # 窗口还活着：用户还没扫码，正常状态，绝不重开窗口
                break
            if retry == 0:
                await asyncio.to_thread(_kill_profile_browsers, profile_dir)
                await asyncio.sleep(1)
        if cookies:
            self.session.cookies = cookies
            self.session.state = "ready"
            self.session.message = "已从上次登录恢复"
            asyncio.create_task(self.cancel())
        return self._describe()

    def _alive(self) -> bool:
        session = self.session
        if not session or not session.process:
            return False
        return session.process.poll() is None

    def _terminate_browser(self) -> None:
        session = self.session
        if session and session.process and session.process.poll() is None:
            try:
                session.process.terminate()
            except OSError:
                pass
            _kill_profile_browsers(session.profile_dir)
        self.session = None

    def _describe(self) -> dict[str, Any]:
        session = self.session
        if not session:
            return {
                "ok": False,
                "state": "idle",
                "cookies": None,
                "message": "无进行中的登录会话",
            }
        return {
            "ok": True,
            "state": session.state,
            "cookies": session.cookies,
            "message": session.message,
            "browser": os.path.basename(session.browser_path),
        }

    async def status(self) -> dict[str, Any]:
        session = self.session
        if not session:
            return self._describe()
        if session.cookies:
            return self._describe()
        if not self._alive():
            session.state = "failed"
            session.message = "登录窗口已被关闭"
            return self._describe()
        if time.time() - session.started_at > SESSION_TTL_SECONDS:
            message = f"登录超时（{SESSION_TTL_SECONDS} 秒）"
            self._terminate_browser()
            return {"ok": False, "state": "timeout", "cookies": None, "message": message}

        try:
            cookies = await asyncio.wait_for(self._fetch_cookies(), timeout=5)
        except asyncio.TimeoutError:
            return self._describe()
        except Exception as exc:
            session.state = "failed"
            session.message = f"读取登录状态失败：{exc}"
            return self._describe()

        if cookies:
            session.cookies = cookies
            session.state = "ready"
            session.message = "登录成功"
            asyncio.create_task(self.cancel())
        return self._describe()

    async def _fetch_cookies(self) -> dict[str, str]:
        session = self.session
        if not session:
            return {}
        try:
            with _CDP_OPENER.open(
                f"http://127.0.0.1:{session.cdp_port}/json/version", timeout=2
            ) as response:
                version = json.loads(response.read().decode("utf-8"))
        except Exception:
            return {}
        cookies: list[dict[str, Any]] = []
        try:
            async with websockets.connect(version["webSocketDebuggerUrl"], proxy=None) as ws:
                await ws.send(json.dumps({"id": 1, "method": "Storage.getCookies"}))
                while True:
                    message = json.loads(await asyncio.wait_for(ws.recv(), timeout=4))
                    if message.get("id") == 1:
                        cookies = (message.get("result") or {}).get("cookies", [])
                        break
        except Exception:
            return {}
        wanted: dict[str, str] = {}
        for cookie in cookies:
            domain = cookie.get("domain", "")
            name = cookie.get("name", "")
            if ("douyin.com" in domain or "iesdouyin.com" in domain) and name in DOUYIN_COOKIE_NAMES:
                wanted[name] = cookie.get("value", "")
        # 只有出现登录态字段才算「登录成功」，否则只是匿名访问首页
        if not any(wanted.get(name) for name in LOGIN_COOKIE_NAMES):
            return {}
        return wanted

    async def cancel(self) -> dict[str, Any]:
        session = self.session
        if not session:
            return {"ok": True, "message": "无进行中的登录会话"}
        process = session.process
        if process and process.poll() is None:
            try:
                with _CDP_OPENER.open(
                    f"http://127.0.0.1:{session.cdp_port}/json/version", timeout=2
                ) as response:
                    version = json.loads(response.read().decode("utf-8"))
                async with websockets.connect(version["webSocketDebuggerUrl"], proxy=None) as ws:
                    await ws.send(json.dumps({"id": 1, "method": "Browser.close"}))
                    await asyncio.sleep(0.3)
            except Exception:
                pass
            if process.poll() is None:
                try:
                    process.terminate()
                except OSError:
                    pass
        _kill_profile_browsers(session.profile_dir)
        if not session.cookies:
            self.session = None
        return {"ok": True, "message": "登录会话已结束"}
