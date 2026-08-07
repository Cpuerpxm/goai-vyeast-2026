"""文档数值一致性检查：**文档里出现的每个指标数，都必须有登记的出处。**

三道防线的第三道：
  authoritative_results.py  → 定义"哪个数是对的"
  stale_number_scan.py      → 查"已作废的数是否还在流通"
  本脚本                     → 查"文档里的数是否都能追到出处"

2026-08-07 改为**反向扫描**。改之前的版本只做正向核对——遍历权威表，看每项
是否出现在文档里；文档中多出来的数它一个都看不见，而且 `bad_total` 恒为 0，
**永远不会失败**。于是文中若干来自 LOCO、来自单元测试、来自实体清点的数
从未被任何一道防线覆盖，而报告却写着"全部对上"。

现在的判据反过来：把文档里长得像指标的数全抓出来，逐个要求它能在
权威表或显式登记表里找到匹配。找不到就是 FAIL，必须要么补进权威表、
要么在 EXTERNAL 里写明出处。

用法：python doc_number_check.py [文档路径 ...]
退出码 0 = 全部有出处；1 = 有无出处的数
"""
from __future__ import annotations

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import paths

AUTH_JSON = os.path.join(paths.RESULTS, "AUTHORITATIVE.json")
DEFAULT_DOCS = [
    os.path.join(paths.PROJECT_ROOT, "docs", "12_初赛提交稿_定稿.md"),
]

# 不由权威表产生、但确有出处的数。键是数值，值是出处说明。
# 往这里加数**必须**同时写清出处；写不出出处的数不该出现在交付文档里。
EXTERNAL: dict[float, str] = {
    61: "评分器单元测试项数 · scripts/scorer/test_metrics.py 运行输出",
    28: "对照匹配单元测试项数 · scripts/data/test_control_match.py 运行输出",
    8: "整化合物留出折数 · scripts/models/loco_response.py --folds 默认值",
    2231: "测试集中未见菌株的行数 · scripts/models/predict_test.py 运行输出",
    20.51: "训练集无观测蛋白列的确定性回退值 · predict_test.py 运行输出",
    5: "shuffled-label 重复次数 · loco_response.py --n-perm 默认值",
    16: "端到端口径选出的 K₀ · scripts/models/select_k0.py",
    96: "插补口径选出的 K₀ · scripts/models/select_k0.py",
    42: "赛题层面响应矩阵秩硬上限 · CLAUDE.md R5",
    36: "本方案响应矩阵秩上限（37 个训练化合物中心化后）· scripts/models/lowrank.py",
    0.15: "对照收缩基线的 α · authoritative_results.py 网格搜索最优",
    0.131: "同条件重复样本缺失不一致率中位数 · scripts/audit/diagnose_missing.py",
    0.351: "缺失指示回归中丰度代理的伪 R² · diagnose_missing.py",
    0.004: "缺失指示回归中技术+生物因子的伪 R² · diagnose_missing.py",
    0.501: "未见菌株占测试集比例 · scripts/audit/perturb_inventory.py",
    0.598: "S2+S3 占测试集比例 · perturb_inventory.py",
    0.05: "匹配对照后响应空间技术变量 η² 下界 · scripts/audit/diagnose_batch.py",
    0.27: "匹配对照后响应空间技术变量 η² 上界 · diagnose_batch.py",
    0.272: "制备孔位在响应空间的残留 η² · diagnose_batch.py",
    4454: "测试样本数 · metadata_test.csv 行数",
    8958: "train_val 样本数 · 权威表 data.n_samples_train_val（此处冗余登记）",
}

# 文档里合法出现、但不是"指标数"的整数：年份、节号、权重百分比等。
# 这些不进反向扫描，否则噪声淹没信号。
INT_WHITELIST_PATTERNS = [
    r"^(19|20)\d{2}$",            # 年份
    r"^[1-9]$",                   # 个位数（节号、条目号）
    r"^(20|25|100)$",             # 评分权重百分比与 100%
]


def _norm(s: str) -> str:
    """归一化负号。

    中文排版用 Unicode 减号 U+2212「−」，Python 格式化出来是 ASCII「-」。
    不归一化的话**每一个负数都会误报**——2026-08-06 实际踩过（Spearman −0.859）。
    """
    for ch in ("−", "–", "—", "－"):
        s = s.replace(ch, "-")
    return s


def walk_numbers(o, path: str = "") -> list[tuple[str, float]]:
    """把权威表里所有数值叶子摊平成 (路径, 值)。"""
    out: list[tuple[str, float]] = []
    if isinstance(o, dict):
        for k, v in o.items():
            if str(k).startswith("_"):
                continue
            out += walk_numbers(v, f"{path}/{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            out += walk_numbers(v, f"{path}[{i}]")
    elif isinstance(o, bool):
        pass
    elif isinstance(o, (int, float)):
        out.append((path.lstrip("/"), float(o)))
    return out


def forms(val: float) -> set[str]:
    """一个数在文档里可能被写成的所有样子。"""
    f = set()
    if float(val).is_integer() and abs(val) < 1e7:
        n = int(val)
        f.add(str(n))
        f.add(f"{n:,}")
    # 六位小数是给"全精度点估计"那类引用留的（正文里写 0.469387 − 0.440649
    # 来说明四舍五入差从哪来），不列进去会被误判成查无出处。
    for p in (2, 3, 4, 5, 6):
        f.add(f"{val:.{p}f}")
        f.add(f"{val:+.{p}f}")
    for p in (1, 2):
        f.add(f"{val * 100:.{p}f}%")
    return {_norm(x) for x in f}


def derived(reg: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """由权威值算出来的量，也算有出处。

    正文引用的 +0.0287 是两个表内值之差，本身不是表里的叶子。
    与其为它单开一个字段（要重跑二十分钟），不如在核对时按同样的定义算一遍——
    这样它永远跟着权威表走，不可能变旧。
    """
    idx = dict(reg)
    out = []
    a = idx.get("models/c_free/逐蛋白 ridge(满秩)/total")
    b = idx.get("models/c_free/低秩 K0=16 + ridge/total")
    if a is not None and b is not None:
        out.append(("派生·低秩相对满秩的总分增益（全精度）", b - a))
        out.append(("派生·同上但按表内四位小数相减", round(b, 4) - round(a, 4)))
    return out


# 候选数：三位及以上小数、带正负号的小数、百分数、四位以上整数
CAND_RE = re.compile(
    r"[+\-−]?\d{1,3}(?:,\d{3})+"          # 千分位整数 8,958
    r"|[+\-−]?\d+\.\d{1,2}%"              # 百分数
    r"|[+\-−]?\d+\.\d{3,}"                # 三位及以上小数
    r"|\b\d{4,}\b"                        # 四位以上整数
)


# 参考文献行：卷号、页码、年份都是数，但没有一个是"指标"。
# 它们的正确性由 Crossref 核验，不归本脚本管。
# ⚠ 一条文献常跨两三行，只认首行会漏掉续行上的页码，所以要认续行的签名。
# 该签名必须与排版无关：先前用的是「**期刊名** 年,」，依赖加粗；
# 2026-08-07 按要求去掉正文全部加粗后，这条规则当场失配、误报两个页码。
# 现改用文献本身的结构特征——年份后紧跟逗号再跟卷号（"2022, 144"），
# 这在正文里几乎不会出现，且不随字体样式变化。
CITATION_RE = re.compile(
    r"et al\.|DOI:\s*10\.|^\s*\d+\.\s+[A-Z][a-z]+ [A-Z]\."
    r"|(?:19|20)\d{2}\s*[,，]\s*\d")
# 版本号与 DOI：先抹掉，否则 "10.0.26200" 会被切出个 "0.26200"
BLANK_RE = re.compile(r"10\.\d{4,}/\S+|\b\d+(?:\.\d+){2,}\b")


def candidates(txt: str) -> list[tuple[str, int]]:
    """抓出文档里像"指标数"的数，返回 (原文, 行号)。"""
    out = []
    for ln, line in enumerate(txt.splitlines(), start=1):
        if line.lstrip().startswith(("![", "|:", "```")):
            continue
        if CITATION_RE.search(line):
            continue
        line = BLANK_RE.sub(" ", line)
        for m in CAND_RE.finditer(line):
            out.append((m.group(0), ln))
    return out


def main() -> None:
    if not os.path.exists(AUTH_JSON):
        print(f"❗缺 {AUTH_JSON}，先跑 scripts/scorer/authoritative_results.py")
        sys.exit(2)
    d = json.load(open(AUTH_JSON, encoding="utf-8"))
    reg = walk_numbers(d)
    reg += derived(reg)

    # 数值样子 → 出处
    index: dict[str, str] = {}
    for name, val in reg:
        for f in forms(val):
            index.setdefault(f, f"权威表 {name}")
    for val, src in EXTERNAL.items():
        for f in forms(val):
            index.setdefault(f, src)

    fp = d.get("code_fingerprint", {})
    print(f"[反向核对] 注册 {len(reg)} 项权威值 + {len(EXTERNAL)} 项登记外部值"
          f"  → {len(index)} 种写法   git HEAD {fp.get('_git_head', '?')}")
    if "_missing" in d.get("loco", {}):
        print(f"  ⚠ 权威表未收录 LOCO：{d['loco']['_missing']}")

    docs = sys.argv[1:] or DEFAULT_DOCS
    bad_total = 0
    for doc in docs:
        if not os.path.exists(doc):
            print(f"  ⚠ 跳过（不存在）{doc}")
            continue
        txt = open(doc, encoding="utf-8", errors="ignore").read()
        rel = os.path.relpath(doc, paths.PROJECT_ROOT)
        unknown: dict[str, list[int]] = {}
        seen = 0
        for raw, ln in candidates(txt):
            tok = _norm(raw)
            if any(re.match(p, tok) for p in INT_WHITELIST_PATTERNS):
                continue
            seen += 1
            if tok in index:
                continue
            # 带符号的数，去掉正号再试一次（文中 +0.0287，表里 0.0287）
            if tok.lstrip("+") in index:
                continue
            unknown.setdefault(tok, []).append(ln)
        if unknown:
            bad_total += len(unknown)
            print(f"\n  {rel}：{seen} 个候选数中 **{len(unknown)} 个查无出处** ❌")
            for tok, lns in sorted(unknown.items()):
                loc = "、".join(f"L{x}" for x in lns[:6])
                print(f"     · {tok:<14} {loc}")
            print("     处理方式：补进权威表重跑，或在本脚本 EXTERNAL 里登记出处。"
                  "写不出出处的数应从文档中删除。")
        else:
            print(f"\n  {rel}：{seen} 个候选数全部可追出处 ✅")

    print()
    if bad_total:
        print(f"FAIL：合计 {bad_total} 个数查无出处。")
    else:
        print("PASS：文档中所有候选数值均可追到权威表或登记出处。")
    sys.exit(1 if bad_total else 0)


if __name__ == "__main__":
    main()
