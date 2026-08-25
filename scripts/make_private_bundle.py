#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""随提交物交给组委会、但不进公开仓库的那几个文件，打成一个包。

为什么需要这个包
----------------
公开仓库按《选手参赛协议》第八条不含赛事实体名，所以化合物名单、
名称到结构的对照表、以及公开版占位符的还原表都被挡在仓库外。
但**评委要复现就必须有这些东西**，光在 README 里写一句"随提交物交付"
不构成交付。这个脚本把它们做成一个有清单、有校验值的实体。

放进来的三样，各解决一个具体问题
--------------------------------
compound_smiles.csv   54 个化合物的结构。原本设计成由 PubChem 公开转储
                      重建，2026-08-25 发现 NCBI 的 Extras/ 目录是滚动更新的，
                      我方 08-05 用的那一版已经取不回来了（详见脚本
                      setup_external.py 顶部注释）。所以改为直接交付。
compound_aliases.json 4 个在 PubChem 同义词表里查不到的化合物名的别名。
                      没有它只能重建 50/54。
entity_alias.json     公开仓库里 STRAIN_A / COMPOUND_01 这类占位符与真实
                      代号的对照。评委拿它把公开日志与报告读成真实实体名。

用法：python scripts/make_private_bundle.py [--tag semifinal-v3]
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import paths  # noqa: E402

OUT_ROOT = os.path.join(paths.PROJECT_ROOT, "_submission_private")

#: (源路径, 包内文件名, 一句话说明)
ITEMS = [
    ("data/external/compound_smiles.csv", "compound_smiles.csv",
     "54 个化合物的 SMILES 结构表。放回 data/external/ 即可，"
     "setup_external.py 会核对 SHA-256 确认拿对了文件。"),
    ("data/external/compound_aliases.json", "compound_aliases.json",
     "4 条化合物别名映射。只有在需要从 PubChem 重建结构表时才用得上。"),
    ("data/external/entity_alias.json", "entity_alias.json",
     "公开仓库占位符（STRAIN_A / COMPOUND_01）到真实代号的对照表，"
     "共 6 株 + 54 个化合物。只用于读懂公开日志，不参与任何计算。"),
]


def sha256(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="semifinal-v3", help="与公开仓库 tag 对齐")
    args = ap.parse_args()

    name = f"goai_vyeast_private_{args.tag}"
    dest = os.path.join(OUT_ROOT, name)
    if os.path.isdir(dest):
        shutil.rmtree(dest)
    os.makedirs(dest)

    print("=" * 78)
    print(f"私有交付包 · {name}")
    print("=" * 78)

    rows, missing = [], []
    for src_rel, arc, note in ITEMS:
        src = os.path.join(paths.PROJECT_ROOT, src_rel)
        if not os.path.exists(src):
            missing.append(src_rel)
            print(f"  ❗缺 {src_rel}")
            continue
        shutil.copy2(src, os.path.join(dest, arc))
        digest = sha256(src)
        rows.append((arc, os.path.getsize(src), digest, note))
        print(f"  [收] {arc:<24} {os.path.getsize(src):>7} 字节  {digest[:16]}…")

    if missing:
        print("\n❗以下文件缺失，包不完整，拒绝出包：")
        for m in missing:
            print("   ", m)
        return 1

    lines = [
        "# GOAI 虚拟酵母扰动蛋白质组预测 · 随提交物交付的私有文件",
        "",
        f"对应公开仓库 tag：`{args.tag}`",
        "",
        "## 这些文件为什么不在公开仓库里",
        "",
        "赛事数据受《选手参赛协议》第八条约束，不得再分发。下面三个文件都含",
        "赛事实体名（化合物名单或菌株代号），所以公开仓库把它们排除在外",
        "（见仓库根 `.gitignore`）。但复现需要它们，故随提交物一并交给组委会。",
        "",
        "## 三个文件",
        "",
        "| 文件 | 字节 | SHA-256 | 作用 |",
        "|---|---|---|---|",
    ]
    for arc, size, digest, note in rows:
        lines.append(f"| `{arc}` | {size} | `{digest[:32]}…` | {note.splitlines()[0]} |")
    lines += [
        "",
        "## 怎么用",
        "",
        "```bash",
        "# 1) 把三个文件都放进 data/external/",
        "cp compound_smiles.csv compound_aliases.json entity_alias.json <仓库>/data/external/",
        "",
        "# 2) 确认拿对了文件（会逐项核对 SHA-256）",
        "python scripts/setup_external.py --check",
        "",
        "# 3) 剩下的按仓库 README 走",
        "python scripts/setup_external.py     # 只剩 SNP 矩阵需要联网下载",
        "python scripts/run_all.py",
        "```",
        "",
        "## 一处需要说明的地方",
        "",
        "`compound_smiles.csv` 原本的设计是**不交付、由本机从 PubChem 公开转储",
        "重建**，这样任何人都能独立生成。2026-08-25 做干净环境复现测试时发现",
        "这条路走不通：NCBI 的 `Compound/Extras/` 目录是滚动更新的，不是归档",
        "快照。我方 2026-08-05 取回的 `CID-Synonym-filtered.gz` 是 964,716,803",
        "字节，08-25 上游已变成 968,456,680 字节，`Last-Modified` 就是当天早上。",
        "也就是说**无法从那个地址重新取回我方用过的那一版**。",
        "",
        "脚本里的 SHA-256 闸门当场拒绝继续，而不是拿一份不同的输入接着跑——",
        "这正是它该有的行为。修法是把这份派生表直接交付并登记校验值，",
        "复现因此不再依赖一个会变的上游。`setup_external.py` 保留了从 PubChem",
        "重建的回退路径，但会明确提示重建结果与我方报告的数字不保证一致。",
        "",
        "完整依赖披露见公开仓库 `docs/22_复赛_依赖披露.md`。",
        "",
    ]
    io.open(os.path.join(dest, "README.md"), "w", encoding="utf-8",
            newline="\n").write("\n".join(lines))

    io.open(os.path.join(dest, "SHA256SUMS.txt"), "w", encoding="utf-8",
            newline="\n").write(
        "".join(f"{d}  {a}\n" for a, _, d, _ in rows))

    manifest = {
        "bundle": name, "public_repo_tag": args.tag,
        "files": [{"name": a, "bytes": s, "sha256": d} for a, s, d, _ in rows],
        "note": "随提交物交给组委会；不进公开仓库；不得再分发",
    }
    io.open(os.path.join(dest, "MANIFEST.json"), "w", encoding="utf-8",
            newline="\n").write(json.dumps(manifest, ensure_ascii=False, indent=2))

    zpath = dest + ".zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(dest):
            for f in sorted(files):
                fp = os.path.join(root, f)
                z.write(fp, os.path.join(name, os.path.relpath(fp, dest)))

    print(f"\n  目录 {dest}")
    print(f"  压缩 {zpath}  {os.path.getsize(zpath)} 字节")
    print(f"  清单 README.md · MANIFEST.json · SHA256SUMS.txt")
    print("\n  ⚠ 这个包不进 git。交给组委会，不要公开分发。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
