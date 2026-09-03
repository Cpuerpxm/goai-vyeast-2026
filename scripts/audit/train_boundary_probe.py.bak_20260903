"""训练边界的**经验**验证：破坏全部非 train 行，看模型是否逐位不变。

为什么要有这个脚本（2026-08-24 复赛整改 L1-1 的关闭证据）：

手册第 17 页要求「验证集与测试集均不得参与训练，也不得用于估计任何统计量」。
静态地读代码只能说「我看着像是没用」，那正是上一轮翻车的地方——旧 `design()`
的拟合行确实只有 train，泄漏藏在词表与标准化参数里，读代码时看不出来。

所以这里做一个**可判定的**检验，判据是复现核验的人能自己跑一遍的：

    把 split_final != 'train' 的每一行都破坏掉
      —— 蛋白丰度换成随机数、缺失位置重画、类别字段换成假水平、时间乘上乱数
    然后从头再拟合一次。
    若模型真的只由 train 折决定，冻结的 spec、mu / U / W、保留蛋白列表、
    以及最终写给组委会的 4,454 x 5,243 预测矩阵，**都必须逐位相同**。

任何一处不同，就说明有 val/test 的信息进了模型，退出码非 0。

另外附一遍静态扫描，专挡已经踩过的三类写法：
  1. 在整张 meta 上 pd.factorize 建词表
  2. 在整张表上算 mean/std 做标准化
  3. 允许用 train 以外的行拟合的开关（如已删除的 --fit-rows all）

运行：
    python train_boundary_probe.py              # 全量（约 2-4 分钟）
    python train_boundary_probe.py --static-only
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import paths, provenance
from data import split_guard as sg
from scorer import evaluate as ev

OUT_DIR = os.path.join(paths.RESULTS, "step0_compliance")

CAT_FIELDS = ["Strains", "Medium", "Temperature", "data_source", "instrument",
              "perturbation_no_concentration", "Yeast_cell_plate"]

# ---- 静态扫描规则：模型/评分路径里不许出现的写法 ----
SCAN_DIRS = ["models", "scorer", "data"]
SCAN_SKIP = {"models/design.py", "audit/train_boundary_probe.py"}
# token 串以空格分隔，正则一律用 \s* 允许空白
PATTERNS = [
    ("在整张 meta 上 pd.factorize 建词表",
     re.compile(r"pd\s*\.\s*factorize\s*\(\s*meta\s*\[")),
    ("在整张表上算 log-time 标准化参数",
     re.compile(r"t\s*-\s*t\s*\.\s*mean\s*\(\s*\)\s*\)\s*/\s*t\s*\.\s*std\s*\(\s*\)")),
]
#: 第 3 类违规（拟合行开关允许取 train 以外的值）藏在**字符串字面量**里，
#: 而上面的扫描恰恰把字符串抹掉了，正则看不见它。所以这一类单独走 AST。
SWITCH_NAME = "fit_rows"



def _code_lines(path: str) -> dict:
    """把文件里的注释与字符串字面量全部抹掉，只留可执行代码，按行号返回。

    ❗必须这么做，不能按行 split("#")。第一版就是这么写的，结果两条命中全是
    误报：一条是 predict_test.py 文档字符串里「旧版有 --fit-rows」这句说明，
    一条是权威表作废数字表里的一行 Markdown 字面量。扫描器把讲规则的文字
    当成了违规代码本身。
    """
    import io as _io
    import tokenize

    out = {}
    with open(path, "rb") as fh:
        try:
            toks = list(tokenize.tokenize(fh.readline))
        except (tokenize.TokenError, IndentationError, SyntaxError):
            return out
    for t in toks:
        if t.type in (tokenize.COMMENT, tokenize.STRING, tokenize.NL,
                      tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT):
            continue
        ln = t.start[0]
        out[ln] = out.get(ln, "") + t.string + " "
    return out


SWITCH_WHAT = "拟合行开关允许取 train 以外的值"


def _switch_scan(src: str) -> list:
    """AST 找「让拟合行可选 train 以外的值」的开关。

    两种形态：
      1) argparse 里出现名字含 fit-rows / fit_rows 的选项，且候选值不止 train
      2) 代码里比较 `... fit_rows ... == "all"`
    """
    import ast

    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    out = []

    def _strs(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return [node.value]
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            return [e.value for e in node.elts
                    if isinstance(e, ast.Constant) and isinstance(e.value, str)]
        return []

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)                 and node.func.attr == "add_argument":
            names = [v for arg in node.args for v in _strs(arg)]
            if any(SWITCH_NAME.replace("_", "-") in n or SWITCH_NAME in n
                   for n in names):
                vals = [v for kw in node.keywords if kw.arg in ("choices", "default")
                        for v in _strs(kw.value)]
                if any(v != "train" for v in vals) or not vals:
                    out.append((node.lineno, f"argparse 选项 {names}，候选/默认 {vals}"))
        if isinstance(node, ast.Compare):
            left = ast.dump(node.left)
            if SWITCH_NAME in left:
                for c in node.comparators:
                    for v in _strs(c):
                        if v != "train":
                            out.append((node.lineno, f"{SWITCH_NAME} == {v!r}"))
    return out


def static_scan() -> list:
    hits = []
    for rel in provenance.source_files():
        if rel in SCAN_SKIP or rel.split("/")[0] not in SCAN_DIRS:
            continue
        p = os.path.join(paths.SCRIPTS_ROOT, rel)
        src = open(p, encoding="utf-8").read()
        raw = src.splitlines()

        def _txt(ln):
            return raw[ln - 1].strip() if 1 <= ln <= len(raw) else ""

        for ln, code in _code_lines(p).items():
            for what, rx in PATTERNS:
                if rx.search(code):
                    hits.append({"file": rel, "line": ln, "what": what, "text": _txt(ln)})
        for ln, why in _switch_scan(src):
            hits.append({"file": rel, "line": ln, "what": SWITCH_WHAT,
                         "text": f"{_txt(ln)}   <- {why}"})
    return sorted(hits, key=lambda h: (h["file"], h["line"]))


FIXTURE_BAD = '''
import pandas as pd, numpy as np, argparse

def design(meta, cols):
    for c in cols:
        codes, uniq = pd.factorize(meta[c].astype(str))
    t = np.log1p(meta["pert_time"].to_numpy())
    t = (t - t.mean()) / t.std()
    return t

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-rows", choices=["train", "all"], default="all")
    args = ap.parse_args()
    fit_rows = args.fit_rows
    if fit_rows == "all":
        pass
'''

FIXTURE_OK = '''
"""说明：旧版有 --fit-rows all，且用 pd.factorize(meta[c]) 建词表，已删除。"""
NOTE = "| 提交模型用全部 train_val 拟合 | predict_test.py --fit-rows all | 已删除 |"

def design(meta, fit_rows, cols):
    spec = freeze(meta, fit_rows, cols)
    return encode(meta, spec)
'''


def scanner_selftest() -> int:
    """扫描器自己先过一遍阳性/阴性对照。

    阳性：一个把三类违规都写全的假文件，必须三类全中。
    阴性：一个只是在**文档与字符串里谈论**这些违规的干净文件，必须零命中——
    第一版就栽在这里，两条命中全是讲规则的文字。
    """
    import tempfile

    ok = 0
    with tempfile.TemporaryDirectory() as d:
        pb = os.path.join(d, "bad.py")
        po = os.path.join(d, "ok.py")
        for path, src in ((pb, FIXTURE_BAD), (po, FIXTURE_OK)):
            with open(path, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(src)

        bad_lines = _code_lines(pb)
        got = set()
        for ln, code in bad_lines.items():
            for what, rx in PATTERNS:
                if rx.search(code):
                    got.add(what)
        for _, _why in _switch_scan(FIXTURE_BAD):
            got.add(SWITCH_WHAT)
        want = {w for w, _ in PATTERNS} | {SWITCH_WHAT}
        missed = want - got
        print(f"  [阳性对照] 三类违规命中 {len(got)}/{len(want)}"
              + (f"  漏掉 {missed}" if missed else ""))
        assert not missed, f"扫描器抓不到已知违规：{missed}"
        ok += 1

        got2 = set()
        for ln, code in _code_lines(po).items():
            for what, rx in PATTERNS:
                if rx.search(code):
                    got2.add((what, ln))
        for ln, _why in _switch_scan(FIXTURE_OK):
            got2.add((SWITCH_WHAT, ln))
        print(f"  [阴性对照] 只谈论违规的干净文件命中 {len(got2)} 处（应为 0）")
        assert not got2, f"误报：{got2}"
        ok += 1
    return ok


def corrupt_non_train(ctx: ev.EvalContext, seed: int = 20260824) -> ev.EvalContext:
    """把每一个非 train 行破坏到面目全非，train 行一个字节都不动。"""
    rng = np.random.default_rng(seed)
    tr = sg.train_rows(ctx.meta)
    bad = ~tr
    n_bad = int(bad.sum())

    X = ctx.X.copy()
    # 丰度：换成与真实量级完全不同的随机数，并重画缺失掩码
    X[bad] = rng.normal(100.0, 30.0, size=(n_bad, X.shape[1])).astype(np.float32)
    drop = rng.random((n_bad, X.shape[1])) < 0.5
    Xb = X[bad]
    Xb[drop] = np.nan
    X[bad] = Xb

    C = ctx.C.copy()
    C[bad] = rng.normal(-50.0, 10.0, size=(n_bad, C.shape[1])).astype(np.float32)
    D = X - C

    meta = ctx.meta.copy()
    idx = meta.index[bad]
    for c in CAT_FIELDS:
        if c in meta.columns:
            # 先转成字符串列再写假水平。`Temperature` 是整数列，直接塞字符串会被
            # pandas 以 LossySetitemError 拒绝（2026-08-24 实测）。编码路径本来
            # 就走 astype(str)，转换对 train 行的取值没有任何影响。
            meta[c] = meta[c].astype(str)
            meta.loc[idx, c] = [f"__CORRUPT_{c}_{k}__" for k in range(n_bad)]
    meta.loc[idx, "pert_time"] = rng.integers(10_000, 99_999, size=n_bad)

    # split_final / sample_ID 保持不变：它们定义了「谁是 train」与行序，
    # 破坏它们等于换了一道题，证明不了任何事。
    return dataclasses.replace(ctx, meta=meta, X=X, C=C, D=D)


def _model_signature(spec: dict, model: dict, y_te: np.ndarray) -> dict:
    import hashlib

    def h(arr) -> str:
        a = np.ascontiguousarray(arr)
        return hashlib.sha256(a.tobytes()).hexdigest()[:16]

    return {
        "design_spec": json.dumps(spec, ensure_ascii=False, sort_keys=True),
        "mu": h(model["mu"]),
        "U": h(model["U"]),
        "W": h(model["W"]),
        "dead_cols": h(model["dead_cols"]),
        "n_dead": int(model["dead_cols"].sum()),
        "fallback": float(model["fallback"]),
        "prediction_test": h(y_te),
        "prediction_shape": list(y_te.shape),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k0", type=int, default=16)
    ap.add_argument("--lam", type=float, default=30.0)
    ap.add_argument("--static-only", action="store_true")
    ap.add_argument("--selftest", action="store_true", help="只跑扫描器的阳性/阴性对照")
    args = ap.parse_args()

    if args.selftest:
        print("[探针] 扫描器自检")
        n = scanner_selftest()
        print(f"[探针] 扫描器自检 {n} 项全部通过")
        return

    paths.ensure_dir(OUT_DIR)
    L: list[str] = []
    a = L.append
    a("=" * 92)
    a("训练边界合规探针（复赛整改 L1-1 的关闭证据）")
    a("=" * 92)
    a("判据：破坏全部非 train 行后重新拟合，模型与提交预测必须逐位不变。")
    a("")

    # ---------------------------------------------------------- 静态扫描
    a("-" * 92)
    a("一 · 静态扫描（挡已经踩过的三类写法）")
    a("-" * 92)
    print("[探针] 扫描器阳性/阴性对照 …")
    scanner_selftest()
    a("  扫描器已先过阳性对照（三类违规全中）与阴性对照（只谈论违规的文字零命中）")
    hits = static_scan()
    for what in [w for w, _ in PATTERNS] + [SWITCH_WHAT]:
        n = sum(1 for h in hits if h["what"] == what)
        a(f"  [{'FAIL' if n else 'PASS'}] {what}：命中 {n} 处")
    for h in hits:
        a(f"      {h['file']}:{h['line']}  {h['text'][:80]}")
    a("")
    static_ok = not hits

    if args.static_only:
        a("（--static-only：跳过经验探针）")
        txt = "\n".join(L)
        print(txt)
        sys.exit(0 if static_ok else 1)

    # ---------------------------------------------------------- 经验探针
    a("-" * 92)
    a("二 · 经验探针：破坏非 train 行后重跑，逐位比对")
    a("-" * 92)
    from models.predict_test import build_submission

    print("[探针] 载入原始数据 …")
    ctx = ev.build_context(verbose=False)
    tr = sg.train_rows(ctx.meta)
    a(f"  train 行 {int(tr.sum())} / 全表 {ctx.n}；被破坏的非 train 行 {int((~tr).sum())}")

    print("[探针] 第 1 遍：原始数据拟合 …")
    y_a, meta_te, spec_a, model_a, val_a, _ = build_submission(ctx, args.k0, args.lam)
    sig_a = _model_signature(spec_a, model_a, y_a)

    print("[探针] 破坏全部非 train 行 …")
    ctx_bad = corrupt_non_train(ctx)
    # 破坏是否真的发生了——否则这个检验会因为「什么都没改」而假通过
    changed = int((~np.isclose(np.nan_to_num(ctx.X[~tr], nan=-1e9),
                               np.nan_to_num(ctx_bad.X[~tr], nan=-1e9))).sum())
    same_train = np.array_equal(np.nan_to_num(ctx.X[tr], nan=-1e9),
                                np.nan_to_num(ctx_bad.X[tr], nan=-1e9))
    a(f"  破坏生效：非 train 行有 {changed:,} 个单元格被改动")
    a(f"  train 行未被触碰：{same_train}")
    if changed == 0 or not same_train:
        a("  ❗探针自身失效（没改到东西，或误伤了 train 行），结论无效")
        print("\n".join(L))
        sys.exit(2)

    print("[探针] 第 2 遍：破坏后的数据拟合 …")
    y_b, _, spec_b, model_b, val_b, _ = build_submission(ctx_bad, args.k0, args.lam)
    sig_b = _model_signature(spec_b, model_b, y_b)

    a("")
    a(f"  {'比对项':<26}{'原始':>20}{'破坏后':>20}{'':>8}")
    ok = True
    for k in ["design_spec", "mu", "U", "W", "dead_cols", "n_dead", "fallback",
              "prediction_test", "prediction_shape"]:
        same = sig_a[k] == sig_b[k]
        ok &= same
        va, vb = str(sig_a[k]), str(sig_b[k])
        if k == "design_spec":
            va, vb = f"(spec {len(va)} 字节)", f"(spec {len(vb)} 字节)"
        a(f"  {k:<26}{va[:20]:>20}{vb[:20]:>20}{'  一致' if same else '  ❌不同':>8}")
    a("")
    a(f"  提交预测矩阵逐位相同：{'是' if sig_a['prediction_test'] == sig_b['prediction_test'] else '否'}"
      f"（{sig_a['prediction_shape']}，SHA-256 前 16 位 {sig_a['prediction_test']}）")
    a("")
    a("  对照读数（不作判据，只说明破坏确实改变了非 train 行的一切）：")
    a(f"    原始数据下官方四类 val 总分 {val_a['total']:.4f}；"
      f"破坏后同一评估口径给出 {val_b['total']:.4f}")
    a("    —— 分数变了而模型没变，正是「val 只用于评估、不进拟合」的样子。")
    a("")

    # ---------------------------------------------------------- 三 · test 元数据
    # ❗2026-08-25（GPT Pro R6 · L2-01）：上一段只破坏了 metadata_train_val 里
    # split_final != 'train' 的行，**没有碰独立的 metadata_test.csv**。
    # 于是它能证明「输出不依赖 train_val 的非 train 行」，却证明不了
    # 「模型参数没有从 test 元数据估计过任何统计量」——手册第 17 页两样都禁。
    a("-" * 92)
    a("三 · 测试集元数据探针：破坏 metadata_test 后，模型参数必须原样不动")
    a("-" * 92)
    a("  与上一段的差别：这次改的是 test 侧。预测**理应**跟着变（输入变了），")
    a("  但冻结的设计 spec 与 mu / U / W **必须逐位不变**——")
    a("  它们只许由 train 折估计，test 元数据在模型里连一个统计量都不该出现。")

    from data import loader as _loader
    import models.predict_test as _pt

    meta_te_real = _loader.load_metadata("test")
    rng2 = np.random.default_rng(20260825)
    meta_te_bad = meta_te_real.copy()
    for c in CAT_FIELDS:
        if c in meta_te_bad.columns:
            meta_te_bad[c] = [f"__CORRUPT_TEST_{c}_{k}__" for k in range(len(meta_te_bad))]
    meta_te_bad["pert_time"] = rng2.integers(10_000, 99_999, size=len(meta_te_bad))

    _orig = _loader.load_metadata

    def _patched(which="train_val", *a_, **k_):
        return meta_te_bad.copy() if which == "test" else _orig(which, *a_, **k_)

    try:
        _pt.loader.load_metadata = _patched
        y_c, _, spec_c, model_c, _, _ = build_submission(ctx, args.k0, args.lam)
    finally:
        _pt.loader.load_metadata = _orig

    sig_c = _model_signature(spec_c, model_c, y_c)
    model_same = all(sig_a[k] == sig_c[k] for k in
                     ("design_spec", "mu", "U", "W", "dead_cols", "n_dead", "fallback"))
    pred_changed = sig_a["prediction_test"] != sig_c["prediction_test"]
    a("")
    a(f"  破坏 test 元数据 {len(meta_te_bad)} 行（类别字段换假水平、时间换乱数）")
    a(f"  设计 spec / mu / U / W / 保留蛋白列表：{'逐位相同' if model_same else '**有变化**'}")
    a(f"  test 预测：{'变了（本该如此，输入变了）' if pred_changed else '**没变，反常**'}")
    ok_test = model_same and pred_changed
    a(f"  -> {'PASS' if ok_test else 'FAIL'}：模型参数与 test 元数据无关，"
      "而预测确实依赖它——")
    a("     这正是「只用 train 折估计、对 test 只做编码」应有的样子。")
    a("")

    a("-" * 92)
    a(f"结论：静态扫描 {'PASS' if static_ok else 'FAIL'} · "
      f"train_val 探针 {'PASS' if ok else 'FAIL'} · "
      f"test 元数据探针 {'PASS' if ok_test else 'FAIL'}")
    a("-" * 92)
    ok = ok and ok_test

    txt = "\n".join(L)
    print(txt)
    p = os.path.join(OUT_DIR, "train_boundary_probe.txt")
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(txt)
    payload = {
        "static_scan_hits": hits,
        "static_ok": static_ok,
        "probe_ok": bool(ok),
        "n_train_rows": int(tr.sum()),
        "n_corrupted_rows": int((~tr).sum()),
        "n_corrupted_cells": changed,
        "signature_original": sig_a,
        "signature_corrupted": sig_b,
        "signature_test_metadata_corrupted": sig_c,
        "test_metadata_probe_ok": bool(ok_test),
        "test_metadata_probe_note": (
            "破坏 metadata_test 后：模型参数逐位不变（不从 test 估计任何统计量），"
            "预测随输入改变（对 test 只做编码）。补 GPT Pro R6 · L2-01 指出的盲区。"),
        "val_total_original": val_a["total"],
        "val_total_after_corruption": val_b["total"],
        "_provenance": provenance.stamp(),
    }
    pj = os.path.join(OUT_DIR, "train_boundary_probe.json")
    with open(pj, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)
    print(f"\n[写出] {p}\n[写出] {pj}")
    sys.exit(0 if (static_ok and ok) else 1)


if __name__ == "__main__":
    main()
