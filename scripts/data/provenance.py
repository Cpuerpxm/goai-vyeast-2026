"""口径指纹：让「结果 / 代码 / 图 / 预测」能被核验为同一个实例。

为什么要重写（2026-08-24 复赛整改 L1-3）：

旧版 `authoritative_results.code_fingerprint()` 把 `git rev-parse HEAD` 的结果写成
`_git_head`。但 `GOAI_VYEAST/` 不是独立仓库，它躺在本地大仓库 `E:/Claude_Code` 里，
于是记下来的是 `b0a309dce` —— 一个**公开仓库解析不了**的提交号。复现核验的人拿到
`https://github.com/Cpuerpxm/goai-vyeast-2026` 会发现这个 SHA 根本不存在。
而且 CODE_FILES 只列了 8 个文件，`loco_response.py` 的哈希在表里是 `None`。

现在改成三层，都可以被只拿到公开仓库的人独立复算：

1. `tree_digest()` —— `scripts/` 下全部 `*.py` 的内容摘要（路径排序后逐个 SHA-256
   再汇总）。它不依赖 git，谁 clone 下来跑一次就能比对。
2. `release_info()` —— 读项目根的 `RELEASE.json`（发布时由 `scripts/release.py` 写），
   里面是公开仓库的 tag 名与 commit SHA。没发布过就明写「未发布」，不猜。
3. `env_info()` —— Python 与关键库版本，供依赖披露那一项直接引用。

自检：python provenance.py
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from typing import Dict, List

_HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_ROOT = os.path.dirname(_HERE)
PROJECT_ROOT = os.path.dirname(SCRIPTS_ROOT)
RELEASE_FILE = os.path.join(PROJECT_ROOT, "RELEASE.json")

_SKIP_DIRS = {"__pycache__", ".ipynb_checkpoints"}


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def source_files() -> List[str]:
    """`scripts/` 下全部 .py 的相对路径，排序后返回（换行统一，跨平台一致）。"""
    out = []
    for root, dirs, files in os.walk(SCRIPTS_ROOT):
        dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIRS)
        for f in sorted(files):
            if f.endswith(".py"):
                rel = os.path.relpath(os.path.join(root, f), SCRIPTS_ROOT)
                out.append(rel.replace("\\", "/"))
    return sorted(out)


def file_hashes(rels: List[str] | None = None) -> Dict[str, str]:
    """每个源码文件的 SHA-256 前 16 位。"""
    rels = rels if rels is not None else source_files()
    out = {}
    for rel in rels:
        p = os.path.join(SCRIPTS_ROOT, rel)
        out[rel] = _sha256(p)[:16] if os.path.exists(p) else "(缺失)"
    return out


def tree_digest() -> str:
    """全部源码的单一摘要。任何一个字节变了，这个值就变。"""
    h = hashlib.sha256()
    for rel in source_files():
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(_sha256(os.path.join(SCRIPTS_ROOT, rel)).encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()[:16]


def release_info() -> dict:
    """公开仓库的发布标识。没有就明写未发布，不去猜、也不写本地大仓库的 HEAD。"""
    if not os.path.exists(RELEASE_FILE):
        return {
            "_status": "未发布",
            "_note": "项目根无 RELEASE.json。发布走 scripts/release.py，"
                     "它会把公开仓库的 tag 与 commit 写在这里。"
                     "本地 GOAI_VYEAST/ 不是独立 git 仓库，绝不能拿外层大仓库的 "
                     "HEAD 冒充——那个 SHA 在公开仓库里不存在。",
        }
    d = json.load(open(RELEASE_FILE, encoding="utf-8"))
    d["_status"] = "已发布"
    cur = tree_digest()
    d["tree_digest_now"] = cur
    d["tree_digest_matches_release"] = (d.get("tree_digest") == cur)
    return d


def env_info() -> dict:
    out = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for mod in ("numpy", "pandas", "scipy", "sklearn", "matplotlib", "rdkit"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "(无版本号)")
        except Exception:
            out[mod] = "(未安装)"
    return out


def stamp(extra: dict | None = None) -> dict:
    """一次性拿全：源码摘要 + 逐文件哈希 + 发布标识 + 环境。"""
    s = {
        "tree_digest": tree_digest(),
        "n_source_files": len(source_files()),
        "files": file_hashes(),
        "release": release_info(),
        "env": env_info(),
    }
    if extra:
        s.update(extra)
    return s


def _selftest() -> None:
    ok = 0
    fs = source_files()
    assert "models/design.py" in fs and "data/split_guard.py" in fs, fs[:5]
    ok += 1
    assert all(not r.startswith("..") for r in fs); ok += 1
    h = file_hashes(["models/design.py", "does/not/exist.py"])
    assert len(h["models/design.py"]) == 16 and h["does/not/exist.py"] == "(缺失)"
    ok += 1
    d1, d2 = tree_digest(), tree_digest()
    assert d1 == d2 and len(d1) == 16, "tree_digest 必须确定性"; ok += 1
    r = release_info()
    assert r["_status"] in ("已发布", "未发布"); ok += 1
    e = env_info()
    assert e["numpy"] != "(未安装)"; ok += 1
    st = stamp({"x": 1})
    assert st["x"] == 1 and st["tree_digest"] == d1; ok += 1
    print(f"[provenance] selftest {ok} 项全部通过")
    print(f"  tree_digest = {d1}   源码 {len(fs)} 个文件   发布状态 {r['_status']}")


if __name__ == "__main__":
    _selftest()
