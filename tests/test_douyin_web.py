"""抖音 web 解析（a_bogus 签名 + detail 接口）的单元测试：全部离线。"""
from __future__ import annotations

from pathlib import Path

import pytest

from backend import douyin_browser, douyin_web
from backend.video_processor import DOUYIN_COOKIE_NAMES, VideoProcessor

USER_SELF_LINK = (
    "https://www.douyin.com/user/self?from_tab_name=main"
    "&modal_id=7693866327673900025&showTab=like"
)


def _detail_fixture() -> dict:
    return {
        "aweme_id": "7693866327673900025",
        "desc": "测试抖音视频",
        "create_time": 1_700_000_000,
        "video": {
            "duration": 12_500,
            "play_addr": {
                "url_list": ["https://v.douyinvod.com/playwm/demo.mp4"]
            },
            "origin_cover": {"url_list": ["https://cover.example/a.jpg"]},
        },
        "author": {"nickname": "作者"},
        "music": {"play_url": {"url_list": ["https://music.example/a.mp3"]}},
        "statistics": {"play_count": 42, "digg_count": 7},
    }


def test_extract_aweme_id_covers_video_and_modal_forms() -> None:
    assert (
        douyin_web.extract_aweme_id("https://www.douyin.com/video/7300000000000000000")
        == "7300000000000000000"
    )
    # 网页版个人主页 + modal_id：作品 ID 藏在查询参数里
    assert douyin_web.extract_aweme_id(USER_SELF_LINK) == "7693866327673900025"
    assert douyin_web.extract_aweme_id("https://www.douyin.com/user/self") is None
    assert douyin_web.extract_aweme_id("") is None


def test_info_from_aweme_detail_maps_all_fields() -> None:
    info = douyin_web.info_from_aweme_detail(_detail_fixture())
    assert info["source"] == "douyin"
    assert info["title"] == "测试抖音视频"
    assert info["owner"] == "作者"
    assert info["duration"] == 12.5
    assert info["view_count"] == 42
    assert info["like_count"] == 7
    assert info["thumbnail"] == "https://cover.example/a.jpg"
    # playwm -> play 去水印，与旧的分享页解析行为一致
    assert info["_douyin_video_url"] == "https://v.douyinvod.com/play/demo.mp4"
    assert info["_douyin_audio_url"] == "https://music.example/a.mp3"
    assert info["_douyin_aweme_id"] == "7693866327673900025"


def test_info_from_aweme_detail_tolerates_missing_sections() -> None:
    info = douyin_web.info_from_aweme_detail({"aweme_id": "1", "desc": "只有文案"})
    assert info["title"] == "只有文案"
    assert info["owner"] == ""
    assert info["_douyin_video_url"] is None


def test_build_detail_params_order_puts_aweme_id_last() -> None:
    params = douyin_web.build_detail_params("123", "tok")
    keys = list(params)
    # a_bogus 是按这个顺序拼串签名的，末两位必须是 msToken、aweme_id
    assert keys[-2:] == ["msToken", "aweme_id"]
    assert params["aweme_id"] == "123"
    assert params["device_platform"] == "webapp"


def test_signed_detail_url_carries_a_bogus() -> None:
    params = douyin_web.build_detail_params("123", "tok")
    url = douyin_web.signed_detail_url(params)
    assert url.startswith(douyin_web.DETAIL_ENDPOINT + "?")
    assert "aweme_id=123" in url
    assert "&a_bogus=" in url
    signature = url.split("&a_bogus=", 1)[1]
    assert len(signature) > 20


def test_fetch_video_detail_surfaces_uifid_block(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(douyin_web, "generate_ms_token", lambda *a, **k: "t" * 128)
    monkeypatch.setattr(
        douyin_web,
        "_get_json",
        lambda *a, **k: (403, {"_error": "Blocked by ArgusSecurityPlugin Uifid Not Found"}),
    )
    with pytest.raises(RuntimeError, match="Uifid"):
        douyin_web.fetch_video_detail("123")


def test_fetch_video_detail_parses_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(douyin_web, "generate_ms_token", lambda *a, **k: "")
    payload = {"status_code": 0, "aweme_detail": _detail_fixture()}
    monkeypatch.setattr(douyin_web, "_get_json", lambda *a, **k: (200, payload))
    info = douyin_web.fetch_video_detail("123")
    assert info["title"] == "测试抖音视频"


def test_fetch_video_detail_rejects_missing_aweme_detail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(douyin_web, "generate_ms_token", lambda *a, **k: "")
    monkeypatch.setattr(douyin_web, "_get_json", lambda *a, **k: (200, {"status_code": 0}))
    with pytest.raises(RuntimeError, match="aweme_detail"):
        douyin_web.fetch_video_detail("123")


def test_generate_false_ms_token_shape() -> None:
    token = douyin_web.generate_false_ms_token()
    assert len(token) == 128
    assert token.endswith("==")


def test_douyin_cookie_header_includes_uifid(tmp_path: Path) -> None:
    processor = VideoProcessor(tmp_path)
    header = processor._cookie_header(
        {"Uifid": "uif-1", "ttwid": "tt-1", "sessionid": "s-1"}, douyin=True
    )
    assert "Uifid=uif-1" in header
    assert "ttwid=tt-1" in header
    assert "sessionid=s-1" in header
    # 风控拦的就是「缺 Uifid」，白名单里必须有它
    assert "Uifid" in DOUYIN_COOKIE_NAMES


def test_build_detail_url_carries_no_signature() -> None:
    """实测带 a_bogus 会被判 Sign Invalid；正式请求必须不带签名。"""
    params = douyin_web.build_detail_params("123", "tok")
    url = douyin_web.build_detail_url(params)
    assert url.startswith(douyin_web.DETAIL_ENDPOINT + "?")
    assert "aweme_id=123" in url
    assert "a_bogus" not in url


def test_extract_douyin_uses_browser_channel_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """浏览器通道是唯一实测能过的路，必须排在 Python 直调前面。"""
    processor = VideoProcessor(tmp_path)
    seen: dict[str, str] = {}

    def fake_browser(aweme_id: str, root: Path, **kwargs: object) -> dict:
        seen["aweme_id"] = aweme_id
        seen["root"] = str(root)
        return {"status_code": 0, "aweme_detail": _detail_fixture()}

    def must_not_call(*args: object, **kwargs: object) -> dict:
        raise AssertionError("浏览器通道成功后不该再走 Python 直调")

    monkeypatch.setattr(douyin_browser, "fetch_detail_payload", fake_browser)
    monkeypatch.setattr(douyin_web, "fetch_video_detail", must_not_call)
    info = processor._extract_douyin_share_info(
        "https://www.douyin.com/video/7300000000000000000",
        {"Uifid": "u-1", "sessionid": "s-1"},
    )
    assert info["title"] == "测试抖音视频"
    assert info["_douyin_video_url"] == "https://v.douyinvod.com/play/demo.mp4"
    assert seen["aweme_id"] == "7300000000000000000"
    # 浏览器 profile 必须落在同一个工作目录下（与扫码登录共用）
    assert seen["root"] == str(tmp_path)


def test_extract_douyin_handles_user_self_modal_id_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    processor = VideoProcessor(tmp_path)
    seen: dict[str, str] = {}

    def fake_browser(aweme_id: str, root: Path, **kwargs: object) -> dict:
        seen["aweme_id"] = aweme_id
        return {"status_code": 0, "aweme_detail": _detail_fixture()}

    monkeypatch.setattr(douyin_browser, "fetch_detail_payload", fake_browser)
    info = processor._extract_douyin_share_info(USER_SELF_LINK, None)
    assert info["title"] == "测试抖音视频"
    assert seen["aweme_id"] == "7693866327673900025"


def test_extract_douyin_falls_back_to_api_when_browser_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """浏览器起不来（没装浏览器等）时，仍要退回 Python 直调。"""
    processor = VideoProcessor(tmp_path)
    seen: dict[str, str] = {}

    def browser_boom(*args: object, **kwargs: object) -> dict:
        raise douyin_browser.DouyinBrowserError("未找到 Edge 或 Chrome")

    def fake_fetch(aweme_id: str, cookie_header: str = "", user_agent: str = "") -> dict:
        seen["aweme_id"] = aweme_id
        seen["cookie"] = cookie_header
        return {"title": "接口直调标题", "source": "douyin"}

    monkeypatch.setattr(douyin_browser, "fetch_detail_payload", browser_boom)
    monkeypatch.setattr(douyin_web, "fetch_video_detail", fake_fetch)
    info = processor._extract_douyin_share_info(
        "https://www.douyin.com/video/7300000000000000000",
        {"Uifid": "u-1", "sessionid": "s-1"},
    )
    assert info["title"] == "接口直调标题"
    assert seen["aweme_id"] == "7300000000000000000"
    assert "Uifid=u-1" in seen["cookie"]


def test_extract_douyin_falls_back_to_share_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    processor = VideoProcessor(tmp_path)

    def boom(*args: object, **kwargs: object) -> dict:
        raise RuntimeError("风控")

    monkeypatch.setattr(douyin_web, "fetch_video_detail", boom)
    monkeypatch.setattr(douyin_browser, "fetch_detail_payload", boom)
    monkeypatch.setattr(
        processor,
        "_extract_douyin_share_page",
        lambda url, cookie_header: {"title": "分享页标题", "source": "douyin"},
    )
    info = processor._extract_douyin_share_info(
        "https://www.douyin.com/video/7300000000000000000", None
    )
    assert info["title"] == "分享页标题"


def test_extract_douyin_reports_every_channel_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    processor = VideoProcessor(tmp_path)

    def boom(*args: object, **kwargs: object) -> dict:
        raise RuntimeError("通道挂了")

    monkeypatch.setattr(douyin_web, "fetch_video_detail", boom)
    monkeypatch.setattr(douyin_browser, "fetch_detail_payload", boom)
    monkeypatch.setattr(
        processor,
        "_extract_douyin_share_page",
        lambda url, cookie_header: (_ for _ in ()).throw(RuntimeError("分享页也挂了")),
    )
    with pytest.raises(RuntimeError) as excinfo:
        processor._extract_douyin_share_info(
            "https://www.douyin.com/video/7300000000000000000", None
        )
    message = str(excinfo.value)
    assert "浏览器通道失败" in message
    assert "接口直调失败" in message
    assert "分享页也挂了" in message


def test_browser_fetch_reports_missing_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(douyin_browser, "find_browser", lambda: None)
    with pytest.raises(douyin_browser.DouyinBrowserError, match="未找到"):
        douyin_browser.fetch_detail_payload("123", Path("/tmp/does-not-matter"))


def test_browser_profile_dir_matches_login_profile(tmp_path: Path) -> None:
    """登录用的 profile 与抓取用的必须是同一个，否则登录态白扫。"""
    assert douyin_browser.profile_dir_for(tmp_path) == tmp_path / "_douyin_profile"


def test_sandbox_hint_explains_crashpad_denial() -> None:
    """回归：应用被沙箱拉起时，Chrome 写不了 Crashpad 目录会以 -5 秒退。
    报错必须给出「从普通终端启动」这个唯一解法，而不是让用户反复重投。"""
    import io

    stderr = io.BytesIO(
        b"[1017/185125.5:ERROR:file_io_posix.cc:208] open /Users/x/Library/Application "
        b"Support/Google/Chrome/Crashpad/settings.dat: Operation not permitted (1)\n"
    )
    hint = douyin_browser._sandbox_hint(stderr)
    assert "沙箱" in hint
    assert "普通终端" in hint
    # 无关的 stderr 不加戏
    assert douyin_browser._sandbox_hint(io.BytesIO(b"some unrelated error")) == ""
    assert douyin_browser._sandbox_hint(None) == ""
