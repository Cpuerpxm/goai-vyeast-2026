"""会诊包泄漏扫描：打包前强制过一遍，确认没有把受协议约束的内容带出去。

CLAUDE.md R1 禁止把**蛋白丰度数值 / 完整化合物名单 / 菌株代号**发给任何外部
服务（含 GPT Pro）。人工目检不可靠——名单有 54 项，报告有上千行。
这里直接从本机数据读出受保护词表，逐文件扫。

判定：
  compound  命中的化合物名（>3 个不同名即视为「名单」级泄漏）
  strain    菌株代号（数据集里的匿名字母代号，任意一个都算）
  abundance 疑似原始丰度数值（5 位以上整数连排，或科学计数法密集出现）

用法：
    python pkg_leak_scan.py <pkg_dir>
退出码 0 = 干净；1 = 有泄漏（打包必须中止）
"""
from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import loader

# 少量代表例是既有惯例允许的（见上一个包的 MANIFEST「只给机制类别分布与少量代表例」）
COMPOUND_ROSTER_THRESHOLD = 3
TEXT_EXT = {".md", ".txt", ".py", ".csv", ".json", ".yml", ".yaml"}


def protected_terms() -> tuple[set[str], set[str]]:
    comp, strain = set(), set()
    for w in ("train_val", "test"):
        m = loader.load_metadata(w)
        comp |= {str(x) for x in m["perturbation_no_concentration"].unique()}
        strain |= {str(x) for x in m["Strains"].unique()}
    comp -= {"Water", "DMSO", "Quality Control"}     # 非化合物，且是手册公开口径
    return comp, strain


# 盐 / 水合物后缀。台账登记全名，正文常只写母体名，比对时两种都要试。
_SALT = ("hydrochloride", "dihydrochloride", "hydrobromide", "sulfate", "sulphate",
         "phosphate", "maleate", "citrate", "mesylate", "tosylate", "acetate",
         "tartrate", "fumarate", "succinate", "isethionate", "hyclate", "besylate",
         "sodium salt", "potassium salt", "monohydrate", "dihydrate", "trihydrate",
         "hydrate", "anhydrous")


def _salt_stem(name: str) -> str:
    """去掉盐/水合物后缀，返回小写母体名。"""
    t = str(name).strip().lower()
    changed = True
    while changed:
        changed = False
        for suf in _SALT:
            if t.endswith(" " + suf):
                t = t[: -(len(suf) + 1)].strip()
                changed = True
    return t


def scan_file(path: str, comp: set[str], strain: set[str]) -> dict:
    try:
        txt = open(path, "r", encoding="utf-8", errors="ignore").read()
    except OSError:
        return {}
    low = txt.lower()
    # 化合物名要连去掉盐/水合物后缀的词干一起找。台账里登记的是
    # "Dyclonine hydrochloride"，而正文往往只写 "dyclonine"——
    # 只比全名的话，词干形式会整个漏过去（2026-08-07 实测漏检）。
    hits_c = set()
    for c in comp:
        if not c:
            continue
        stem = _salt_stem(c)
        if c.lower() in low or (len(stem) >= 5 and stem in low):
            hits_c.add(c)
    hits_c = sorted(hits_c)

    # 菌株代号必须全词匹配，否则三字母代号会撞进普通英文。
    # ❗不能用 \b：Python 里中文属于 \w，所以"菌株XYZ只出现在"这种写法
    # 在 株|B 处根本不存在词边界，正则永远匹配不上——而本项目的文档
    # 恰恰通篇是中文夹英文代号。改用「两侧不是 ASCII 字母数字」判定。
    hits_s = sorted({s for s in strain
                     if re.search(rf"(?<![A-Za-z0-9]){re.escape(s)}(?![A-Za-z0-9])", txt)})
    # 先剔除长 hex 串（SHA-256 摘要等），否则其中的连续数字会被当成丰度值误报
    num_txt = re.sub(r"\b[0-9a-fA-F]{32,}\b", " ", txt)
    big = re.findall(r"(?<![\d.])\d{5,}(?![\d.])", num_txt)
    sci = re.findall(r"\d\.\d+e[+-]?0?[5-9]", num_txt, flags=re.I)
    return {"compound": hits_c, "strain": hits_s,
            "big_ints": len(big), "sci": len(sci)}


def main() -> None:
    if len(sys.argv) < 2:
        print("用法: python pkg_leak_scan.py <pkg_dir>")
        sys.exit(2)
    root = sys.argv[1]
    # 防误用：本工具管的是「要发出去的包」。本机内部文档（docs/、_handoff/、results/）
    # 含化合物名与菌株代号是**正常的**——那是我们正在分析的数据本身，不出本机。
    # 对内部目录跑本工具只会产出成片的假警报（2026-08-06 实际踩过）。
    # 提交给组委会的文档同样不受限：组委会是数据方，不是第三方。
    abs_root = os.path.abspath(root)
    if "_pkg_for_gpt_pro" not in abs_root.replace("\\", "/") and "--force" not in sys.argv:
        print(f"❗本工具用于扫描**外发给第三方**的会诊包，目标应在 _pkg_for_gpt_pro/ 下。")
        print(f"   当前目标：{abs_root}")
        print("   本机内部文档含化合物名/菌株代号属正常，不应用本工具检查。")
        print("   若确要强制扫描，加 --force。")
        sys.exit(2)
    comp, strain = protected_terms()
    print(f"[扫描] 受保护词表：{len(comp)} 个化合物名 / {len(strain)} 个菌株代号")
    print(f"[扫描] 目标目录：{root}\n")

    bad = False
    n_files = 0
    for dirpath, _, files in os.walk(root):
        for fn in sorted(files):
            if os.path.splitext(fn)[1].lower() not in TEXT_EXT:
                continue
            p = os.path.join(dirpath, fn)
            n_files += 1
            r = scan_file(p, comp, strain)
            rel = os.path.relpath(p, root)
            probs = []
            if len(r["compound"]) >= COMPOUND_ROSTER_THRESHOLD:
                probs.append(f"化合物名 {len(r['compound'])} 个（名单级）: "
                             + ", ".join(r["compound"][:6])
                             + ("…" if len(r["compound"]) > 6 else ""))
            elif r["compound"]:
                print(f"  [注意] {rel}: 含 {len(r['compound'])} 个化合物名"
                      f"（未达名单阈值 {COMPOUND_ROSTER_THRESHOLD}，按「少量代表例」放行）: "
                      + ", ".join(r["compound"]))
            if r["strain"]:
                probs.append("菌株代号: " + ", ".join(r["strain"]))
            if r["big_ints"] > 20:
                probs.append(f"5 位以上整数 {r['big_ints']} 处（疑似原始丰度）")
            if r["sci"] > 20:
                probs.append(f"科学计数法大数 {r['sci']} 处（疑似原始丰度）")
            if probs:
                bad = True
                print(f"  ❌ {rel}")
                for x in probs:
                    print(f"       {x}")

    print(f"\n扫描文件 {n_files} 个")
    if bad:
        print("❌ 存在泄漏，打包中止。请脱敏后重扫。")
        sys.exit(1)
    print("✅ 未发现受保护内容，可以打包。")


if __name__ == "__main__":
    main()
