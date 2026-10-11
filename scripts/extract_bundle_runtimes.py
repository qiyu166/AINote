"""从 onefile 产物里抽出 VC++ 运行时 DLL，交给调用方判版本。

为什么要有这一步：PyInstaller 把构建机 System32 的 msvcp140.dll / vcruntime140.dll 原样
收进包里，而**包内 DLL 在搜索顺序上盖过用户自己的 System32** —— 构建机的运行时一旦是老的，
每一个用户拿到的包都跟着老，且在他们机器上以同样的方式崩。

2026-10 群测那次 faster-whisper 与 sherpa-onnx 两个引擎都报 0xC0000005，出错模块是
MSVCP140.dll 14.0.24215.1：本机 System32 的 base 正是 2016 年那一版，而同一个包里的
msvcp140_1.dll 是 14.50。`_1` 这个组件是 VS2017 15.3 才引入的，base 比它老一整条产品线
是不被支持的组合。对照之下 CI 正式版收到的是 14.40 —— 所以这是构建机状态漏进了产物。

版本判断放在 build_exe.ps1 里做（PowerShell 读 VersionInfo 是一行的事），这里只负责
"从包里把文件拿出来"，不依赖 shell 也不猜路径。
"""

from __future__ import annotations

import argparse
import pathlib
import sys

from PyInstaller.archive.readers import CArchiveReader

RUNTIME_DLLS = ("msvcp140.dll", "vcruntime140.dll")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", help="onefile 产物 exe 的路径")
    parser.add_argument("--out", required=True, help="抽出文件放哪儿")
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    reader = CArchiveReader(args.archive)
    wanted = {name for name in reader.toc if name.lower() in RUNTIME_DLLS}
    if not wanted:
        print("包里找不到 %s，检查 PyInstaller 是否收到运行时" % " / ".join(RUNTIME_DLLS), file=sys.stderr)
        return 2

    for name in sorted(wanted):
        data = reader.extract(name)
        if not data.startswith(b"MZ"):
            print("%s 抽出来不是 PE 文件" % name, file=sys.stderr)
            return 2
        (out_dir / pathlib.Path(name).name).write_bytes(data)
        print(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
