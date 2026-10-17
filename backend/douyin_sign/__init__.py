"""抖音 Web 签名算法（vendored，非本项目原创）。

来源：Johnserf-Seed/f2 的 ``f2/utils/abogus.py``（Apache License 2.0，作者
JohnserfSeed，https://github.com/Johnserf-Seed/f2 ）。这里**原样搬运**，未改动
算法实现；算法若需跟进上游更新，对照上游同一文件重新同步即可。

为什么不用 ``pip install f2``：f2 的依赖约束是 ``websockets<13.0``，与本项目
``websockets>=13,<16`` 硬冲突，而且 ``backend/bili_login.py`` 用到了 websockets
14+ 才有的 ``connect(proxy=...)`` 参数——整包装入会把 websockets 降到 12.0，
直接打崩 B 站扫码登录（已实测并回滚，见 ASSUMPTIONS.md H11）。只取算法本体，
唯一新增依赖是纯 Python 的 ``gmssl``。
"""
from .abogus import ABogus, BrowserFingerprintGenerator

__all__ = ["ABogus", "BrowserFingerprintGenerator"]
