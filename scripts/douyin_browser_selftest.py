"""抖音浏览器通道自检：一条命令定位「浏览器起不来」的真正原因。

用法（在项目根目录）：

    .venv/bin/python scripts/douyin_browser_selftest.py
    .venv/bin/python scripts/douyin_browser_selftest.py 7693866327673900025

它按 ``backend/douyin_browser.py`` 完全相同的参数去起浏览器，但把每一步
（找到哪个浏览器 / 用哪种无头方式 / CDP 是否就绪 / Chrome 自己的 stderr）
都打印出来。报障时把整段输出贴出来即可。
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.bili_login import find_browser, free_cdp_port  # noqa: E402
from backend.douyin_browser import (  # noqa: E402
    HEADLESS_FLAG_SETS,
    PROFILE_DIR_NAME,
    _page_target,
    _stderr_excerpt,
)

DEFAULT_AWEME_ID = "7693866327673900025"
DETAIL = "https://www.douyin.com/aweme/v1/web/aweme/detail/"


def probe(browser: str, profile_dir: Path, flags: tuple[str, ...], ready_timeout: float = 30.0) -> str:
    port = free_cdp_port()
    if port is None:
        return "✗ 没有可用调试端口（9333–9342 全被占用）"
    label = " ".join(flags) or "(有窗口模式)"
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
        "--remote-allow-origins=*",
        "--window-size=900,700",
        *flags,
        "about:blank",
    ]
    stderr = tempfile.TemporaryFile()
    started = time.time()
    process = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=stderr)
    outcome = ""
    try:
        deadline = time.time() + ready_timeout
        while time.time() < deadline:
            if process.poll() is not None:
                outcome = (
                    f"✗ [{label}] 进程立刻退出（退出码 {process.returncode}）"
                    f"{_stderr_excerpt(stderr)}"
                )
                break
            if _page_target(port):
                outcome = f"✓ [{label}] CDP 就绪，用时 {time.time() - started:.1f}s，端口 {port}"
                break
            time.sleep(0.4)
        else:
            outcome = (
                f"✗ [{label}] {ready_timeout:.0f}s 内未开放端口 {port}"
                f"{_stderr_excerpt(stderr)}"
            )
    finally:
        try:
            process.terminate()
            process.wait(timeout=10)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass
        try:
            stderr.close()
        except Exception:
            pass
    return outcome


def main() -> int:
    aweme_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_AWEME_ID
    print("=" * 68)
    print("抖音浏览器通道自检")
    print("=" * 68)
    browser = find_browser()
    print(f"浏览器      : {browser or '未找到 Edge / Chrome'}")
    if not browser:
        print("=> 结论：机器上没有 Chrome/Edge，浏览器通道不可用。")
        return 1
    profile_dir = ROOT / "workspace" / PROFILE_DIR_NAME
    profile_dir.mkdir(parents=True, exist_ok=True)
    print(f"profile 目录: {profile_dir}")
    print(f"profile 存在: {profile_dir.is_dir()}")
    print()

    ok_any = False
    for flags in HEADLESS_FLAG_SETS:
        result = probe(browser, profile_dir, flags)
        print(result)
        ok_any = ok_any or result.startswith("✓")
    print()

    # 能否访问抖音首页（网络层）
    try:
        with urllib.request.urlopen("https://www.douyin.com/", timeout=10) as response:
            print(f"抖音首页连通: ✓ HTTP {response.status}")
    except Exception as exc:  # noqa: BLE001
        print(f"抖音首页连通: ✗ {type(exc).__name__}: {exc}")

    if not ok_any:
        print()
        print("=> 结论：浏览器 CDP 起不来。请把上面『Chrome 输出』那段原样贴出来。")
        return 1

    print()
    print("=> 结论：CDP 通道正常，继续做一次真实抓取…")
    from backend.douyin_browser import fetch_detail_payload  # noqa: PLC0415

    started = time.time()
    try:
        payload = fetch_detail_payload(aweme_id, ROOT / "workspace")
        detail = payload.get("aweme_detail") or {}
        print(f"✓ 抓取成功（{time.time() - started:.1f}s）")
        print(f"  文案: {(detail.get('desc') or '')[:60]}")
        print(f"  作者: {(detail.get('author') or {}).get('nickname')}")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"✗ 抓取失败（{time.time() - started:.1f}s）: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
