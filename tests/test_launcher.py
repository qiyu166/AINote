"""启动器测试。

注：「托盘图标 / 检查更新」已按需移除（本机用源码模式跑 uvicorn，launcher 不在链路上），
对应的 version_tuple / latest_release_info / check_for_updates 测试一并删掉。
"""
import launcher


def test_server_alive_requires_matching_version(monkeypatch) -> None:
    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"status":"ok","service":"AINote","mode":"portable","version":"1.1.7"}'

    monkeypatch.setattr(launcher.urllib.request, "urlopen", lambda *args, **kwargs: FakeResponse())

    assert launcher.server_alive("http://127.0.0.1:8000", mode="portable", version="1.1.7")
    assert not launcher.server_alive("http://127.0.0.1:8000", mode="portable", version="1.1.8")


def test_configure_runtime_dirs_creates_custom_workspace(monkeypatch, tmp_path) -> None:
    """环境变量指定的 workdir 不存在时也要自动创建（打包版写日志前依赖此目录）。"""
    custom = tmp_path / "custom-ws"
    assert not custom.exists()
    monkeypatch.setenv("AINOTE_WORKSPACE", str(custom))
    monkeypatch.setattr(launcher, "is_frozen", lambda: False)
    launcher.configure_runtime_dirs()
    assert custom.is_dir()


def test_tray_and_update_check_are_gone() -> None:
    """按需移除的模块不得回潮：缺了它们后 launcher 仍要能导入。"""
    for name in ("run_tray", "tray_icon_image", "check_for_updates", "latest_release_info"):
        assert not hasattr(launcher, name), f"{name} 已移除，不该再出现"
