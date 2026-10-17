"""抖音浏览器抓取：在真实浏览器页面里用 JS ``fetch`` 请求 detail 接口。

**为什么只能用这条路（全部是本机实测结论）**：

- 脚本化 HTTP 打抖音首页，拿到的是 JS VM 挑战页（200，只有 ``_$jsvmprt`` 脚本、
  不设 cookie）；
- 带 a_bogus 签名直调 detail 接口 → ``403 Sign Invalid``（签名反而把请求弄坏）；
- 带浏览器**全套 cookie** 的 Python 请求（不带签名）→ 仍 ``403 Uifid Not Found``；
- 只有**在页面内用 JS fetch、且不带签名**时才成功（实测拿回完整 ``aweme_detail``）。

根因：``Uifid`` 不是普通 cookie，而是抖音前端 SDK 在 XHR/fetch 上注入的**请求头**，
只有页面里的 JS 请求才带得出来。

流程：无头 Chrome（复用 ``douyin_login`` 的 ``_douyin_profile``）→ 打开作品页 →
等 ``UIFID`` cookie 出现（实测约 15–25 秒）→ 在页面里 fetch detail 接口 → 读回 JSON。

**排障设计**：浏览器启动失败时，这里会把「退出码 + Chrome 自己的 stderr」带进异常，
并依次尝试 ``--headless=new`` / ``--headless`` 两种无头方式。以前只报一句
「CDP 未就绪」，把真正的死因（profile 被占、沙箱禁止 Chrome 写 Crashpad 等）全吞了。
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import IO, Any, Iterable

from websockets.sync.client import connect

from . import douyin_web
from .bili_login import BiliLoginManager, find_browser, free_cdp_port

_kill_profile_browsers = BiliLoginManager._kill_profile_browsers

HOME_URL = "https://www.douyin.com/"
PROFILE_DIR_NAME = "_douyin_profile"
# 实测 UIFID 出现需要十几到二十几秒；给足余量，但拿到就立刻继续
DEFAULT_WAIT_UIFID_SECONDS = 45.0
# 浏览器从启动到开放 CDP 端口的上限；正常机器 1–3 秒，慢机器/首启留足余量
DEFAULT_READY_TIMEOUT_SECONDS = 45.0
# Chrome 111+ 默认拒绝带 Origin 的 CDP WebSocket 连接，必须放行
_REMOTE_ALLOW_ORIGINS = "--remote-allow-origins=*"
# 依次尝试的无头方式；某个 Chrome 版本不认 --headless=new 时退到旧写法
HEADLESS_FLAG_SETS: tuple[tuple[str, ...], ...] = (("--headless=new",), ("--headless",))
# Chrome 启动时必然刷屏、但与本问题无关的噪音行
_STDERR_NOISE = ("cvdisplaylink", "trying to load the allocator multiple times")

# CDP 的本机 HTTP 探测一律走「明确不挂任何代理」的 opener。
# bili_login.py 踩过这个坑：系统代理指向一个僵死的本地代理进程时，
# 连 127.0.0.1 上的 CDP 都会被劫走，表现同样是"端口永远不就绪"。
_CDP_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class DouyinBrowserError(RuntimeError):
    """浏览器抓取通道的失败（没浏览器 / CDP 起不来 / 超时 / 响应非 JSON）。"""


def profile_dir_for(root: Path) -> Path:
    """登录与抓取共用的浏览器 profile 目录（与 DouyinLoginManager 一致）。"""
    return Path(root) / PROFILE_DIR_NAME


# ---------------------------------------------------------------- 小工具


def _read_json_url(url: str, timeout: float = 2.0) -> Any:
    with _CDP_OPENER.open(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _page_target(port: int) -> str | None:
    """探一次 CDP：拿到 page 目标的 websocket 地址就返回，否则 None。"""
    try:
        targets = _read_json_url(f"http://127.0.0.1:{port}/json/list")
    except Exception:
        return None
    for target in targets or []:
        if target.get("type") == "page" and target.get("webSocketDebuggerUrl"):
            return str(target["webSocketDebuggerUrl"])
    return None


def _stderr_excerpt(stderr: IO[bytes] | None, limit: int = 400) -> str:
    """读出 Chrome stderr 里真正有用的那几行（噪音行丢掉）。"""
    if stderr is None:
        return ""
    try:
        stderr.seek(0)
        raw = stderr.read().decode("utf-8", "replace")
    except Exception:
        return ""
    lines = [
        line.strip()
        for line in raw.splitlines()
        if line.strip() and not any(noise in line.lower() for noise in _STDERR_NOISE)
    ]
    if not lines:
        return ""
    return "｜Chrome 输出：" + " / ".join(lines[:4])[:limit]


def _sandbox_hint(stderr: IO[bytes] | None) -> str:
    """Chrome 因写不了自己的 Crashpad 目录而秒退时，给出确切处置办法。

    实测（本机逐项验证过）：Chrome 无法写
    ``~/Library/Application Support/Google/Chrome/Crashpad/settings.dat`` 时会直接
    退出（信号 -5），且 ``--disable-crash-reporter`` / ``--disable-breakpad`` /
    ``--disable-crashpad-for-testing`` / ``--crash-dumps-dir`` 这些开关**一个都救不回来**。
    唯一解是让应用本身离开沙箱（例如被 AI/DSH 会话或沙箱内的 shell 拉起时就会中招）。
    """
    if stderr is None:
        return ""
    try:
        stderr.seek(0)
        text = stderr.read().decode("utf-8", "replace").lower()
    except Exception:
        return ""
    if "crashpad" in text and (
        "operation not permitted" in text or "permission denied" in text
    ):
        return (
            "。真因：应用运行在受限的沙箱环境里，Chrome 写不了自己的配置目录"
            "（~/Library/Application Support/Google/Chrome/Crashpad）。"
            "请从【普通终端】启动应用（不要从 AI/DSH 会话或沙箱内的 shell 启动）后重试；"
            "实测任何 Chrome 启动开关都绕不过这一步。"
        )
    return ""


def _profile_pids(profile_dir: Path) -> list[int]:
    """还占着这个 profile 的浏览器进程（非 Windows；Windows 走 bili_login 那套）。"""
    marker = str(profile_dir)
    try:
        result = subprocess.run(
            ["ps", "-axo", "pid=,command="], capture_output=True, text=True, timeout=10
        )
    except Exception:
        return []
    pids: list[int] = []
    for line in result.stdout.splitlines():
        parts = line.strip().split(" ", 1)
        if len(parts) != 2:
            continue
        pid_text, command = parts
        if marker in command and ("Chrome" in command or "Edge" in command):
            try:
                pids.append(int(pid_text))
            except ValueError:
                continue
    return pids


def _prepare_profile(profile_dir: Path) -> None:
    """清掉占用该 profile 的浏览器进程与残留单例锁。

    这里**必须等旧实例真正退出**：``_kill_profile_browsers`` 是发 SIGKILL 就返回，
    而被杀的 Chrome 还要一会儿才释放 profile 单例；此时若立刻起新实例，新实例会以为
    「已有实例在用这个 profile」而**直接退出**，症状就是 CDP 端口永远不开放。
    残留的 ``SingletonLock`` 同理。
    """
    _kill_profile_browsers(profile_dir)
    if os.name == "nt":
        time.sleep(1.0)
    else:
        deadline = time.time() + 5.0
        while time.time() < deadline and _profile_pids(profile_dir):
            time.sleep(0.3)
    for name in ("SingletonLock", "SingletonCookie", "SingletonSocket"):
        try:
            (profile_dir / name).unlink(missing_ok=True)
        except OSError:
            pass


def _launch(
    browser: str, profile_dir: Path, port: int, flags: Iterable[str]
) -> tuple[subprocess.Popen[bytes], IO[bytes]]:
    """起浏览器，stderr 落到临时文件（避免管道写满把 Chrome 卡死）。"""
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
        _REMOTE_ALLOW_ORIGINS,
        "--window-size=900,700",
        *flags,
        "about:blank",
    ]
    stderr = tempfile.TemporaryFile()
    process = subprocess.Popen(  # noqa: S603 - 只执行本机已知浏览器
        args, stdout=subprocess.DEVNULL, stderr=stderr
    )
    return process, stderr


def _wait_cdp(
    port: int, process: subprocess.Popen[bytes], stderr: IO[bytes], timeout: float
) -> str:
    """等 CDP 端口可用；浏览器中途退出就立刻带着退出码和 stderr 报错。"""
    deadline = time.time() + max(1.0, timeout)
    while time.time() < deadline:
        if process.poll() is not None:
            raise DouyinBrowserError(
                f"浏览器启动后立刻退出（退出码 {process.returncode}）"
                f"{_stderr_excerpt(stderr)}{_sandbox_hint(stderr)}"
            )
        ws_url = _page_target(port)
        if ws_url:
            return ws_url
        time.sleep(0.4)
    raise DouyinBrowserError(
        f"浏览器 {timeout:.0f} 秒内没有开放调试端口 {port}"
        f"{_stderr_excerpt(stderr)}{_sandbox_hint(stderr)}"
    )


class _CdpConnection:
    """极简同步 CDP 封装：够用就好（Page / Runtime 两个域）。"""

    def __init__(self, ws_url: str) -> None:
        self._ws = connect(ws_url, proxy=None)
        self._id = 0

    def call(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self._id += 1
        message_id = self._id
        self._ws.send(
            json.dumps({"id": message_id, "method": method, "params": params or {}})
        )
        while True:
            # 事件消息（Page.loadEventFired 等）没有 id，直接跳过
            message = json.loads(self._ws.recv(timeout=60))
            if message.get("id") == message_id:
                return message

    def _evaluate(self, expression: str, *, await_promise: bool = False) -> Any:
        result = self.call(
            "Runtime.evaluate",
            {
                "expression": expression,
                "returnByValue": True,
                "awaitPromise": await_promise,
            },
        )
        return ((result.get("result") or {}).get("result") or {}).get("value")

    def document_cookie(self) -> str:
        return str(self._evaluate("document.cookie") or "")

    def has_uifid(self) -> bool:
        names = {
            part.split("=", 1)[0].strip()
            for part in self.document_cookie().split(";")
            if "=" in part
        }
        return "UIFID" in names

    def fetch_text(self, url: str) -> str:
        """在页面上下文里 fetch，返回响应文本（失败返回 ``ERR:`` 前缀串）。"""
        expression = (
            f"fetch({json.dumps(url)},{{credentials:'include'}})"
            ".then(r=>r.text()).catch(e=>'ERR:'+e)"
        )
        return str(self._evaluate(expression, await_promise=True) or "")

    def close(self) -> None:
        try:
            self._ws.close()
        except Exception:
            pass


# ---------------------------------------------------------------- 主流程


def _fetch_with_flags(
    browser: str,
    profile_dir: Path,
    port: int,
    aweme_id: str,
    flags: tuple[str, ...],
    *,
    ready_timeout: float,
    wait_uifid_seconds: float,
    fetch_attempts: int,
    timeout: float,
) -> dict[str, Any]:
    process, stderr = _launch(browser, profile_dir, port, flags)
    try:
        ws_url = _wait_cdp(port, process, stderr, ready_timeout)
        cdp = _CdpConnection(ws_url)
        try:
            cdp.call("Page.enable")
            cdp.call("Runtime.enable")
            # 先打开作品页：让前端 SDK 跑起来、把 UIFID 等风控 cookie 种上
            cdp.call("Page.navigate", {"url": f"{HOME_URL}video/{aweme_id}"})
            uifid_deadline = time.time() + max(0.0, wait_uifid_seconds)
            while time.time() < uifid_deadline:
                if cdp.has_uifid():
                    break
                time.sleep(1.0)

            # 关键：不带任何签名，就在页面里 fetch
            url = douyin_web.build_detail_url(
                douyin_web.build_detail_params(
                    aweme_id,
                    douyin_web.generate_ms_token()
                    or douyin_web.generate_false_ms_token(),
                )
            )
            last_text = ""
            fetch_deadline = time.time() + timeout
            for _ in range(max(1, fetch_attempts)):
                if time.time() > fetch_deadline:
                    break
                last_text = cdp.fetch_text(url).strip()
                if last_text.startswith("{"):
                    try:
                        return json.loads(last_text)
                    except ValueError as exc:
                        raise DouyinBrowserError("浏览器取到的内容不是合法 JSON") from exc
                if not cdp.has_uifid():
                    # 还没种上 UIFID，再等等重试
                    time.sleep(3.0)
            raise DouyinBrowserError(
                f"浏览器未取到 JSON（{(last_text[:140] or '空响应')}）"
            )
        finally:
            cdp.close()
    finally:
        try:
            process.terminate()
            process.wait(timeout=10)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass
        _kill_profile_browsers(profile_dir)
        try:
            stderr.close()
        except Exception:
            pass


def fetch_detail_payload(
    aweme_id: str,
    root: Path,
    *,
    timeout: float = 180.0,
    headless: bool = True,
    wait_uifid_seconds: float = DEFAULT_WAIT_UIFID_SECONDS,
    fetch_attempts: int = 3,
    ready_timeout: float = DEFAULT_READY_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """在浏览器页面里取 detail 接口 JSON，返回原始 payload。

    失败抛 ``DouyinBrowserError``，消息里带上每种无头方式的真实死因。
    """
    browser = find_browser()
    if not browser:
        raise DouyinBrowserError("未找到 Edge 或 Chrome，无法在浏览器里抓取抖音")

    profile_dir = profile_dir_for(root)
    profile_dir.mkdir(parents=True, exist_ok=True)
    _prepare_profile(profile_dir)

    flag_sets: tuple[tuple[str, ...], ...] = (
        HEADLESS_FLAG_SETS if headless else ((),)
    )
    failures: list[str] = []
    for flags in flag_sets:
        port = free_cdp_port()
        if port is None:
            failures.append("没有可用的调试端口（9333–9342 全被占用）")
            break
        label = " ".join(flags) or "(有窗口模式)"
        try:
            return _fetch_with_flags(
                browser,
                profile_dir,
                port,
                aweme_id,
                flags,
                ready_timeout=ready_timeout,
                wait_uifid_seconds=wait_uifid_seconds,
                fetch_attempts=fetch_attempts,
                timeout=timeout,
            )
        except DouyinBrowserError as exc:
            failures.append(f"{label}: {exc}")
        except Exception as exc:  # noqa: BLE001 - 汇总所有失败原因给上层
            failures.append(f"{label}: {type(exc).__name__}: {exc}")
        _prepare_profile(profile_dir)

    raise DouyinBrowserError("；".join(failures))
