"""公开仓库脱敏：把赛事菌株代号与化合物名换成稳定占位符。

为什么公开仓库要脱敏，而给顾问的会诊包不用（CLAUDE.md R1）：
判据是**用途**。会诊是「开发…所需范围内」的技术把关，落在协议允许的范围里；
而推 GitHub 公开是面向不特定第三方分发数据集的一部分，性质不同。
初赛那版公开仓库就是这么处理的（日志里是 `STRAIN_A` 等占位符），本次保持一致。

占位符必须**跨版本稳定**，否则读者拿新旧日志对不上号。映射存在
`data/external/entity_alias.json`（已 gitignore，与 `compound_aliases.json` 同类）；
文件不存在时按当前数据生成一份并落盘，之后一律沿用。

初赛公开仓库用过的菌株映射已经固化在 `_LEGACY_STRAIN_ALIAS` 里，首次生成时直接采用，
新出现的实体才往后排。

用法：
    python desensitize.py <目录>            # 就地脱敏（发布流程会调）
    python desensitize.py <目录> --dry-run   # 只报会改什么
    python desensitize.py --show-map        # 打印当前映射
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import loader, paths

ALIAS_FILE = os.path.join(paths.DATA_EXTERNAL, "entity_alias.json")
TEXT_EXT = {".md", ".txt", ".json", ".csv", ".py", ".yml", ".yaml"}

#: 初赛公开仓库（commit 99893cb）实际用过的菌株占位符，按当时日志逐条核对得出。
#: 新版必须沿用，不能重新编号——否则「STRAIN_D」在新旧两版里指的不是同一株。
_LEGACY_STRAIN_ALIAS_ORDER = ["A", "B", "C", "E", "D"]

CONTROL_LABELS = {"Water", "DMSO", "Quality Control"}


def build_alias(force: bool = False) -> dict:
    """生成或读取占位符映射。已有文件一律沿用，只往后追加新实体。"""
    cur = {"strains": {}, "compounds": {}}
    if os.path.exists(ALIAS_FILE) and not force:
        cur.update(json.load(open(ALIAS_FILE, encoding="utf-8")))

    strains, comps = set(), set()
    tv_strains = []
    for which in ("train_val", "test"):
        m = loader.load_metadata(which)
        vals = {str(x) for x in m["Strains"].unique()}
        strains |= vals
        if which == "train_val":
            tv_strains = sorted(vals)
        comps |= {str(x) for x in m["perturbation_no_concentration"].unique()}
    comps -= CONTROL_LABELS          # 对照名是手册公开口径，不算实体

    if not cur["strains"]:
        # 首次生成：先把初赛公开仓库用过的 5 株按当时顺序固化。
        # ❗只能拿 **train_val 的菌株**去对，不能拿 train_val+test 的并集——
        # 初赛日志里只出现过 train_val 那 5 株，多出来的 test 独有菌株若插进这个
        # 序列，会把「STRAIN_D」指向另一株，新旧日志就对不上号了。
        for s, letter in zip(tv_strains, _LEGACY_STRAIN_ALIAS_ORDER):
            cur["strains"][s] = f"STRAIN_{letter}"
    new_s = sorted(s for s in strains if s not in cur["strains"])
    used = set(cur["strains"].values())
    nxt = 0
    for s in new_s:
        while f"STRAIN_{chr(ord('A') + nxt)}" in used:
            nxt += 1
        cur["strains"][s] = f"STRAIN_{chr(ord('A') + nxt)}"
        used.add(cur["strains"][s])

    n = len(cur["compounds"])
    for c in sorted(c for c in comps if c not in cur["compounds"]):
        n += 1
        cur["compounds"][c] = f"COMPOUND_{n:02d}"

    with open(ALIAS_FILE, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(cur, fh, ensure_ascii=False, indent=2, sort_keys=True)
    return cur


def _stem_names(alias: dict) -> dict:
    """化合物的**盐/水合物词干**也要换，并且要顶得住被截断的写法。

    2026-08-24 实测两处漏网：报告里的近邻表把化合物名截到 25/29 字，
    于是登记的全名逐字匹配不上；而泄漏扫描器做词干归一化，照样抓得到。
    这里把词干（去掉 hydrochloride / monohydrate 之类后缀的母体名）也登记成模式，
    截断后的写法只要还留着母体名就会被换掉。

    词干短于 6 个字符的不登记——那种会撞进普通英文单词。
    """
    from data.pkg_leak_scan import _salt_stem

    out = {}
    for real, ph in alias["compounds"].items():
        stem = _salt_stem(real)
        if stem and stem != real.lower() and len(stem) >= 6:
            out[stem] = ph
    return out


def _extra_names(alias: dict) -> dict:
    """别名表里登记的**同义写法**也要换。

    2026-08-24 实测：`compound_aliases.json` 把某些化合物的元数据名映射到了正式名，
    正文与解析结果用的是正式名；只逐字匹配元数据名就漏了，而泄漏扫描器做词干与
    大小写归一化照样抓得到。扫描器比脱敏器狠，脱敏器就必须补上这一层，
    否则每次发布都要在闸门 2 被拦一次。
    """
    out = {}
    p = os.path.join(paths.DATA_EXTERNAL, "compound_aliases.json")
    if os.path.exists(p):
        try:
            for real, formal in json.load(open(p, encoding="utf-8")).items():
                ph = alias["compounds"].get(real)
                if ph and formal:
                    out[str(formal)] = ph
        except (json.JSONDecodeError, OSError):
            pass
    return out


def _patterns(alias: dict):
    """长名优先，避免短名先替换把长名切碎（如 `DMSO` 与 `DMSO-d6`）。

    大小写不敏感：日志与解析结果里常见母体名的小写写法。
    """
    pairs = (list(alias["strains"].items()) + list(alias["compounds"].items())
             + list(_extra_names(alias).items()) + list(_stem_names(alias).items()))
    pairs.sort(key=lambda kv: -len(kv[0]))
    out = []
    for real, ph in pairs:
        # 实体名里可能有空格、括号、连字符，不能一律用 \b
        left = r"(?<![0-9A-Za-z_])" if re.match(r"[0-9A-Za-z_]", real) else ""
        right = r"(?![0-9A-Za-z_])" if re.search(r"[0-9A-Za-z_]$", real) else ""
        out.append((re.compile(left + re.escape(real) + right, re.IGNORECASE), ph, real))
    return out


def scrub_text(text: str, pats) -> tuple[str, dict]:
    hits = {}
    for rx, ph, real in pats:
        text, n = rx.subn(ph, text)
        if n:
            hits[real] = hits.get(real, 0) + n
    return text, hits


def scrub_dir(root: str, alias: dict, dry_run: bool = False,
              only: list | None = None) -> dict:
    """就地脱敏。`only` 给定时，**只处理本次要发布的那些文件**。

    为什么要有 `only`：公开仓库里还躺着上一轮发布的文稿，那份已经核验过与提交定稿
    逐字节一致（SHA-256 对得上）。本轮再拿脱敏器扫一遍会把它也改掉，
    那个「与提交稿同版」的性质就没了。本轮该动的只有本轮写进去的文件。
    """
    pats = _patterns(alias)
    allow = None if only is None else {str(x).replace("\\", "/") for x in only}
    total, per_file = {}, {}
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in {".git", "__pycache__"}]
        for f in files:
            if os.path.splitext(f)[1].lower() not in TEXT_EXT:
                continue
            p = os.path.join(dirpath, f)
            if allow is not None and \
                    os.path.relpath(p, root).replace("\\", "/") not in allow:
                continue
            try:
                src = open(p, encoding="utf-8").read()
            except (UnicodeDecodeError, OSError):
                continue
            out, hits = scrub_text(src, pats)
            if not hits:
                continue
            rel = os.path.relpath(p, root).replace("\\", "/")
            per_file[rel] = hits
            for k, v in hits.items():
                total[k] = total.get(k, 0) + v
            if not dry_run:
                with open(p, "w", encoding="utf-8", newline="\n") as fh:
                    fh.write(out)
    return {"per_file": per_file, "total": total}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", nargs="?", help="要脱敏的目录")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--show-map", action="store_true")
    ap.add_argument("--rebuild-map", action="store_true")
    args = ap.parse_args()

    alias = build_alias(force=args.rebuild_map)
    if args.show_map:
        print(json.dumps(alias, ensure_ascii=False, indent=2))
        print(f"\n[映射文件] {ALIAS_FILE}（已 gitignore）")
        return
    if not args.target:
        raise SystemExit("要么给目录，要么用 --show-map")
    if not os.path.isdir(args.target):
        raise SystemExit(f"不是目录：{args.target}")

    r = scrub_dir(args.target, alias, dry_run=args.dry_run)
    print(f"[脱敏]{'（演练）' if args.dry_run else ''} {args.target}")
    print(f"  菌株 {len(alias['strains'])} 个 / 化合物 {len(alias['compounds'])} 个 已登记占位符")
    print(f"  命中 {len(r['per_file'])} 个文件，{sum(r['total'].values())} 处")
    for rel, hits in sorted(r["per_file"].items())[:20]:
        print(f"    {rel}: {sum(hits.values())} 处")
    if len(r["per_file"]) > 20:
        print(f"    …另有 {len(r['per_file']) - 20} 个文件")


if __name__ == "__main__":
    main()
