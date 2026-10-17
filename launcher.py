"""AINote 启动器。

开发模式（`python launcher.py`）：
  - 控制台窗口显示地址，Ctrl+C 退出

打包模式（PyInstaller onefile，--noconsole）：
  - 无控制台窗口，服务启动后前台驻留（Ctrl+C 或结束进程退出）
  - 日志写入工作目录的 _app.log

注：托盘图标与「检查更新」已按需移除（本机用源码模式跑 uvicorn，launcher 不在链路上）。
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path
from typing import Any

VERSION = "1.4.5"
DEFAULT_PORT = 8000
PORT_SCAN_RANGE = 20
START_TIMEOUT_SECONDS = 60
APP_NAME = f"AINote v{VERSION}"


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def base_dir() -> Path:
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def configure_runtime_dirs() -> None:
    """设置工作目录：打包模式默认放在 exe 旁边（便携），允许环境变量覆盖。"""
    if is_frozen():
        exe_dir = base_dir()
        default_workspace = exe_dir / "workspace"
        try:
            default_workspace.mkdir(parents=True, exist_ok=True)
            probe = default_workspace / ".write_test"
            probe.write_text("", encoding="utf-8")
            probe.unlink()
        except OSError:
            # exe 所在目录不可写时（如 Program Files），退回用户目录
            local_app_data = os.environ.get("LOCALAPPDATA")
            fallback_root = Path(local_app_data) if local_app_data else Path.home() / ".ainote"
            default_workspace = fallback_root / "AINote" / "workspace"
            default_workspace.mkdir(parents=True, exist_ok=True)
    else:
        default_workspace = base_dir() / "workspace"
        default_workspace.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("AINOTE_WORKSPACE", str(default_workspace))
    # 环境变量覆盖的路径也要确保存在：打包版写日志文件前依赖此目录，缺失会启动崩溃
    Path(os.environ["AINOTE_WORKSPACE"]).mkdir(parents=True, exist_ok=True)


RUN_MODE = "portable" if is_frozen() else "dev"


def server_alive(
    url: str,
    mode: str | None = None,
    version: str | None = None,
) -> bool:
    """健康检查；可同时隔离运行模式和应用版本。

    新旧便携版不能只按 ``portable`` 复用，否则旧版服务会继续提供旧前端，
    造成页面脚本、HTML 和后端版本混用。
    """
    try:
        with urllib.request.urlopen(f"{url}/api/health", timeout=1.5) as response:
            if response.status != 200:
                return False
            payload = json.loads(response.read().decode("utf-8"))
            if not (payload.get("status") == "ok" and payload.get("service") == "AINote"):
                return False
            if mode is not None and payload.get("mode") != mode:
                return False
            return version is None or payload.get("version") == version
    except Exception:
        return False


def port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def find_available_port() -> int:
    for port in range(DEFAULT_PORT, DEFAULT_PORT + PORT_SCAN_RANGE):
        url = f"http://127.0.0.1:{port}"
        if server_alive(url, mode=RUN_MODE, version=VERSION):
            return port  # 同模式实例已在运行，直接复用
        if port_free(port):
            return port
    raise RuntimeError(f"端口 {DEFAULT_PORT}-{DEFAULT_PORT + PORT_SCAN_RANGE - 1} 均不可用，请稍后重试")


# --------------------------------------------------------------------------
# 服务启动（两种模式共用）
# --------------------------------------------------------------------------

def start_server(port: int) -> tuple[Any, threading.Thread]:
    """在线程中启动 uvicorn，返回 (server, thread)。"""
    import uvicorn

    # 后端自调用端点时需要知道自己挂在哪个端口（ASR 子进程与静态资源都读它）
    os.environ.setdefault("AINOTE_BACKEND_URL", f"http://127.0.0.1:{port}")
    from backend.main import app

    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="info",
        access_log=False,
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    return server, thread


def wait_healthy(
    url: str,
    timeout: float = START_TIMEOUT_SECONDS,
    mode: str | None = None,
    version: str | None = None,
) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if server_alive(url, mode=mode, version=version):
            return True
        time.sleep(0.2)
    return False


# --------------------------------------------------------------------------
# 开发模式（控制台）
# --------------------------------------------------------------------------

def print_banner(url: str, workspace: Path) -> None:
    width = 58
    print("=" * width)
    print("  AINote v" + VERSION + "（开发模式）")
    print(f"  界面地址: {url}")
    print(f"  工作目录: {workspace}")
    print("  关闭本窗口或按 Ctrl+C 退出。")
    print("=" * width, flush=True)


def run_console(url: str, workspace: Path, port: int) -> int:
    print_banner(url, workspace)
    print("  正在启动服务，请稍候…", flush=True)
    try:
        server, _ = start_server(port)
        if not wait_healthy(url, mode=RUN_MODE, version=VERSION):
            print(f"警告: {START_TIMEOUT_SECONDS} 秒内未就绪，请手动打开 {url}", flush=True)
        else:
            webbrowser.open(url)
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n已退出。再见！")
        server.should_exit = True
    except Exception as error:
        print(f"\n启动失败：{error}")
        input("按回车键退出…")
        return 1
    return 0


# --------------------------------------------------------------------------
# 打包模式（系统托盘）
# --------------------------------------------------------------------------

def setup_file_logging(workspace: Path) -> Path:
    """noconsole 模式下把 stdout/stderr 重定向到日志文件。"""
    log_path = workspace / "_app.log"
    log_file = open(log_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = log_file
    sys.stderr = log_file
    return log_path


def show_error(message: str) -> None:
    """启动失败提示：Windows 弹消息框，其他平台写 stderr。"""
    if sys.platform == "win32":
        import ctypes

        ctypes.windll.user32.MessageBoxW(None, message, f"{APP_NAME} 启动失败", 0x10)
    else:
        print(message, file=sys.stderr, flush=True)


def open_in_shell(path: Path) -> None:
    if sys.platform == "win32":
        os.startfile(path)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


# --------------------------------------------------------------------------
# 入口
# --------------------------------------------------------------------------

def main() -> int:
    if "--asr-child" in sys.argv[1:]:
        # 转写工人：只干活，不起服务、不进托盘、不重定向 stdout（主程序在用管道读它）
        configure_runtime_dirs()
        from backend.asr_process import child_main

        return child_main()
    configure_runtime_dirs()
    workspace = Path(os.environ["AINOTE_WORKSPACE"])
    no_browser = os.environ.get("AINOTE_NO_BROWSER") == "1"

    try:
        port = find_available_port()
    except RuntimeError as error:
        show_error(str(error))
        return 1

    url = f"http://127.0.0.1:{port}"

    if server_alive(url, mode=RUN_MODE, version=VERSION):
        # 同模式服务已在运行，直接打开界面
        if not no_browser:
            webbrowser.open(url)
        return 0

    if not is_frozen():
        return run_console(url, workspace, port)

    # ---- 打包模式 ----
    log_path = setup_file_logging(workspace)
    try:
        server, _ = start_server(port)
        if not wait_healthy(url, mode=RUN_MODE, version=VERSION):
            server.should_exit = True
            show_error(f"服务启动超时，请查看日志：{log_path}")
            return 1
        if not no_browser:
            webbrowser.open(url)
        # 无托盘：前台驻留到进程被结束（关闭窗口 / Ctrl+C / 结束进程）
        print(f"{APP_NAME} 已启动: {url}（关闭窗口或 Ctrl+C 退出）", flush=True)
        try:
            while not getattr(server, "should_exit", False):
                time.sleep(1)
        except KeyboardInterrupt:
            server.should_exit = True
        return 0
    except Exception as error:
        show_error(f"启动失败：{error}\n日志：{log_path}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
