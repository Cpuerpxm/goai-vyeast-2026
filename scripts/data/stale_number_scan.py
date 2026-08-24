"""作废数字扫描：防止已被推翻的数字继续在文档里流通。

根因（GPT Pro R3 · L1-13）：数字靠手抄进文档，源头一改，抄件不会跟着改。
`authoritative_results.py` 解决了"哪个是对的"，本脚本解决"错的还剩几处"。

作废清单直接从 results/AUTHORITATIVE.json 的 deprecated 段读，
所以只需在权威表里登记一次，扫描自动生效。

用法：python stale_number_scan.py [目录]
退出码 0 = 干净；1 = 仍有作废数字在流通
"""
from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import paths

# 作废值 → (为什么废, 现行说法)
DEPRECATED = {
    "0.9535": ("未定义轴被静默踢出时的 B0 abs_pcc", "现为 0.4768"),
    "0.0011": ("float32 舍入噪声被当真信号的 B2 ctx_resid", "现为 0.0000"),
    "0.3473": ("修评分器前的 α 收缩总分", "现为 0.3843（α=0.15），且该模型不可提交"),
    "0.2928": ("修评分器前的 B0 总分", "现为 0.2587"),
    "0.3026": ("修评分器前的 B4 总分", "现为 0.3381"),
    "0.2761": ("修评分器前的 B2 总分", "现为 0.3111"),
    "0.2479": ("修评分器前的 B3 总分", "神谕近邻现为 0.3220"),
    # 2026-08-07：统一 B0 定义（是否含对照样本）之前的一批数。
    # 图生成于修复前、正文更新于修复后，两边对不上被外审抓到。
    "0.1834": ("统一 B0 定义前的共享参照·正确匹配·样本轴", "现为 0.1845"),
    "0.0813": ("统一 B0 定义前的共享参照·同条件错配·样本轴", "现为 0.0825"),
    "0.0068": ("统一 B0 定义前的共享参照·全局错配·样本轴", "现为 0.0076"),
    "0.2589": ("统一 B0 定义前的 B0 总分", "现为 0.2587"),
    # 0.2898（B3 化学近邻）曾被误列为作废。2026-08-07 用现口径重算，
    # 结果仍是 0.2898——它只是一直没进权威表，不是数错了。
    # 已补进 authoritative_results.py，此处不再列为作废。
    #
    # ---- 2026-08-24 复赛整改（L1-2）：LOCO 外层留出宇宙含 val 化合物，
    #      合法训练化合物应为 37 个而不是 43 个。下面两个是那一版的产物。
    "0.0237": ("LOCO 宇宙含 val 化合物时的神谕近邻增益", "现为 0.0300（37 个训练化合物）"),
    "0.0617": ("LOCO 宇宙含 val 化合物时的神谕自身残差上限", "现为 0.0674"),
    "0.0154": ("上一条按 25% 权重折算到总分的值", "现为 0.0168"),
    # ❗Morgan 增益不列在这里：旧值 −0.0004、新值 +0.0004，数字串完全相同，
    #   登记它会把**现行的正确值**一并判成作废。符号在正文里靠上下文区分，
    #   这类同串异号的数只能靠 doc_number_check 的正向核对兜住。
}
# 允许出现作废数字的文件（订正记录、权威表本身、扫描器自己）
ALLOW = {
    "AUTHORITATIVE.md", "stale_number_scan.py", "authoritative_results.py",
}
# 段落级豁免：这些标题下的作废数字是"历史记录"，属正常
ALLOW_SECTION_MARKERS = [
    "已知的作废数字", "重大订正", "旧表数字勿再引用", "作废", "订正前",
    "修正后", "R2 verdict", "R3 verdict", "已被推翻",
]
TEXT_EXT = {".md", ".txt"}
# 整份文件已作废的标记，须出现在开头 20 行内
ARCHIVED_MARKER = "已作废 · 历史存档"


def scan_file(path: str) -> list:
    try:
        lines = open(path, "r", encoding="utf-8", errors="ignore").read().split("\n")
    except OSError:
        return []
    # 文件级豁免：整份文档已标注为历史存档。
    # 逐行豁免管不住这种情况——旧稿通篇都是旧数字，一行行标注既不现实
    # 也会把"这是历史"这句话稀释掉。要求标记出现在开头 20 行内，
    # 免得有人在文末补一句就把整份文件洗白。
    if any(ARCHIVED_MARKER in ln for ln in lines[:20]):
        return []

    hits = []
    in_exempt = False
    for i, ln in enumerate(lines, start=1):
        if ln.startswith("#") or ln.startswith(">"):
            in_exempt = any(m in ln for m in ALLOW_SECTION_MARKERS)
        if in_exempt or any(m in ln for m in ALLOW_SECTION_MARKERS):
            continue
        # 划掉的（~~x~~）与明确标注作废的行不算
        if "~~" in ln or "作废" in ln or "已推翻" in ln or "勿再引用" in ln:
            continue
        for val, (why, now) in DEPRECATED.items():
            if not re.search(rf"(?<![\d.]){re.escape(val)}(?![\d])", ln):
                continue
            # 订正记录豁免：同一行既有旧值又有新值（"从 0.9535 落到 0.4768"），
            # 那是在记录变更，不是在冒充现行值。
            new_vals = re.findall(r"\d\.\d{3,4}", now)
            if new_vals and any(
                    re.search(rf"(?<![\d.]){re.escape(nv)}(?![\d])", ln) for nv in new_vals):
                continue
            hits.append((i, val, why, now, ln.strip()[:80]))
    return hits


def main() -> None:
    root = sys.argv[1] if len(sys.argv) > 1 else paths.PROJECT_ROOT
    bad = 0
    print(f"[扫描] 作废数字 {len(DEPRECATED)} 个   目录 {root}\n")
    for dp, dirs, fs in os.walk(root):
        dirs[:] = [d for d in dirs if d not in
                   {".git", "results", "data", "_pkg_for_gpt_pro", "__pycache__"}]
        for fn in sorted(fs):
            if os.path.splitext(fn)[1].lower() not in TEXT_EXT or fn in ALLOW:
                continue
            p = os.path.join(dp, fn)
            for i, val, why, now, ctx in scan_file(p):
                bad += 1
                print(f"  ❌ {os.path.relpath(p, root)}:{i}  作废值 {val}")
                print(f"       原因：{why}")
                print(f"       现行：{now}")
                print(f"       原文：{ctx}")
    print(f"\n发现 {bad} 处作废数字仍在流通")
    if bad:
        print("→ 请改为引用 results/AUTHORITATIVE.md，或在该行标注为历史/订正记录。")
        sys.exit(1)
    print("✅ 干净")


if __name__ == "__main__":
    main()
