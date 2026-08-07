"""按内容重排 markdown 表格的分隔行宽度，使转 DOCX 后列宽合理。

pandoc 把分隔行的破折号数当作列的相对宽度，直接写进 Word 的表格网格。
若照抄手写的等宽分隔行，长标签列会占掉大半页宽，而数字列被挤到放不下
一个 "0.4768" —— 转出来是 "0." 换行 "4768"，读起来像坏掉的数据。

分配规则分两步，缺一不可：
  1. 每列先拿到 hard_min = 该列最长**不可断 token** 的宽度。
     数字、英文单词不能从中间折行，这是硬下限；中文可以逐字折，下限很低。
  2. 余量按 (期望宽 − hard_min) 的比例分给各列。

只改分隔行，不动任何单元格内容。

用法：python fix_table_widths.py <md 路径> [--budget N]
"""
from __future__ import annotations

import argparse
import io
import re
import unicodedata

SEP = re.compile(r"^\s*\|[\s:|-]+\|\s*$")
# 逗号要算进 token：Word 不会在 "1,065" 中间断行，
# 但漏掉逗号会把它切成 "1" 和 "065"，硬下限算成 3，格子给窄了。
TOKEN = re.compile(r"[0-9A-Za-z_.,+\-−%/]+")


def dwidth(s: str) -> int:
    """显示宽度：中日韩全角算 2 列。"""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def hard_min(text: str) -> int:
    """最长不可断 token 的宽度。中文没有这个约束，故只看数字与拉丁串。"""
    return max((len(t) for t in TOKEN.findall(text)), default=0)


def rebuild_sep(header: str, sep: str, body: list[str], budget: int) -> str:
    align = cells(sep)
    ncol = len(cells(header))
    want = [3] * ncol
    floor = [3] * ncol
    for row in [header] + body:
        cs = cells(row)
        for c in range(min(ncol, len(cs))):
            txt = re.sub(r"[*`]", "", cs[c])          # 去掉不占宽的标记
            want[c] = max(want[c], dwidth(txt))
            # 分隔单位与实际点宽不是 1:1。budget 个单位铺满约 420pt 的正文宽，
            # 即每单位约 4.6pt；而 12pt Arial 的一个数字约 6.7pt，合 1.5 个单位。
            # 再加单元格左右内边距约 3 个单位。早先按 +1 估，结果 "+0.0287"
            # 拿到的格子只有 36pt，被折成 "+0.028" / "7"。
            floor[c] = max(floor[c], round(hard_min(txt) * 1.5) + 3)
    floor = [min(f, 20) for f in floor]               # 单个 token 再长也不霸占整页
    base = sum(floor)
    extra = max(0, budget - base)
    span = [max(0, want[c] - floor[c]) for c in range(ncol)]
    tot = sum(span)
    alloc = [floor[c] + (round(extra * span[c] / tot) if tot else 0) for c in range(ncol)]

    # 单列封顶。首列往往是最长的标签列，span 最大，会把余量几乎全部吃掉——
    # 实测出现过首列占 68% 页宽、而数字列窄到 "0.4768" 被折成两行的情况。
    # 中文标签可以逐字折行，宽一点窄一点只影响行数；数字折行是读不懂的。
    if ncol >= 3:
        # 封顶随列数收紧。正文 12pt 下，一个 "0.4768" 约需 40pt、
        # 表头 "abs_pcc" 约需 44pt，而 A4 正文宽约 420pt：
        # 5 列时若标签列吃掉 42%，剩下四列每列只有 60pt 出头，
        # 扣掉单元格内边距就放不下表头，实测被折成 "abs_pc" / "c"。
        cap = int(budget * (0.42 if ncol <= 4 else 0.30))
        for c in range(ncol):
            if alloc[c] > cap:
                back = alloc[c] - cap
                alloc[c] = cap
                others = [k for k in range(ncol) if k != c]
                for k in others:                       # 退回的宽度平分给其余列
                    alloc[k] += back // len(others)

    parts = []
    for c in range(ncol):
        n = max(3, alloc[c])
        a = align[c] if c < len(align) else "---"
        left, right = a.startswith(":"), a.endswith(":")
        if right and not left:
            parts.append("-" * n + ":")
        elif left:
            parts.append(":" + "-" * n + (":" if right else ""))
        else:
            parts.append("-" * n)
    return "|" + "|".join(parts) + "|"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--budget", type=int, default=92,
                    help="一行分隔符的总宽预算；A4 正文宽约容纳这么多")
    args = ap.parse_args()

    lines = io.open(args.path, encoding="utf-8").read().split("\n")
    out, i, changed = [], 0, 0
    while i < len(lines):
        if (lines[i].lstrip().startswith("|") and i + 1 < len(lines)
                and SEP.match(lines[i + 1])):
            j = i
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                j += 1
            body = lines[i + 2:j]
            new = rebuild_sep(lines[i], lines[i + 1], body, args.budget)
            changed += new != lines[i + 1]
            out.append(lines[i]); out.append(new); out.extend(body)
            i = j
        else:
            out.append(lines[i]); i += 1
    io.open(args.path, "w", encoding="utf-8", newline="\n").write("\n".join(out))
    print(f"[表宽] 重排 {changed} 张表   预算 {args.budget} 列")


if __name__ == "__main__":
    main()
