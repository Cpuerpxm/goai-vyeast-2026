"""A/B：共享 Gram 的旧 masked_ridge vs 逐蛋白 Gram 的 masked_ridge_exact。

GPT Pro 会诊 2026-09-02 指出旧实现把缺失行也算进了 Gram 矩阵。
本脚本在 4,422 契约空间内，对满秩 ridge 与低秩骨架各跑一遍两种实现，
在官方四类 val 上比总分与六项，判断这个修正值不值得进最终候选。

用法：
    python scripts/models/ridge_exact_ab.py --lam 30 --ks 16 32
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import paths, split_guard as sg          # noqa: E402
from models.baseline_cfree import masked_ridge, masked_ridge_exact  # noqa: E402
from models.design import freeze, encode, FEATURE_SETS  # noqa: E402
from models.select_k0 import fit_lowrank_pipeline  # noqa: E402
from scorer import evaluate as ev                  # noqa: E402
from scorer.config import ScorerConfig             # noqa: E402

COLS = ["total", "abs_pcc", "abs_r2", "fc_pcc", "ctx_resid", "drug_resid"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, default=30.0)
    ap.add_argument("--ks", type=int, nargs="*", default=[16, 32])
    args = ap.parse_args()

    cfg = ScorerConfig()
    ctx = ev.build_context()
    val = ctx.rows(ev.VAL_SPLITS)
    train_all = sg.train_rows(ctx.meta)
    cols = FEATURE_SETS["bio_tech"]
    spec = freeze(ctx.meta, train_all, cols, with_drug=False)
    Z = encode(ctx.meta, spec)

    print(f"契约空间 {ctx.X.shape[1]} 维 · 训练行 {int(train_all.sum())} · 设计 {Z.shape[1]} 维")
    print(f"{'模型':<28}" + "".join(f"{c:>11}" for c in COLS) + f"{'耗时s':>8}")
    print("-" * (28 + 11 * len(COLS) + 8))

    rows = {}

    # 满秩：两种实现
    for name, fn in (("满秩 · 旧共享Gram", masked_ridge),
                     ("满秩 · 逐蛋白Gram", masked_ridge_exact)):
        t0 = time.time()
        mu, W = fn(Z, ctx.X, train_all, args.lam, ctx.meta)
        y = (mu[None, :] + Z.astype(np.float64) @ W).astype(np.float32)
        y = np.nan_to_num(y, nan=float(np.nanmedian(mu)))
        f = ev.flatten(ev.evaluate(y, ctx, val, cfg))
        rows[name] = f
        print(f"{name:<28}" + "".join(f"{f[c]:>11.4f}" for c in COLS) + f"{time.time()-t0:>8.0f}")

    d = rows["满秩 · 逐蛋白Gram"]["total"] - rows["满秩 · 旧共享Gram"]["total"]
    print(f"\n满秩上，逐蛋白 Gram 相对旧实现 {d:+.4f}")

    # 低秩：现有管线用的是旧 masked_ridge，这里只报旧实现的低秩作参照
    for k in args.ks:
        t0 = time.time()
        y = fit_lowrank_pipeline(ctx, Z, train_all, k, args.lam)
        y = np.nan_to_num(y, nan=0.0)
        f = ev.flatten(ev.evaluate(y, ctx, val, cfg))
        name = f"低秩 K0={k} · 现有管线"
        rows[name] = f
        print(f"{name:<28}" + "".join(f"{f[c]:>11.4f}" for c in COLS) + f"{time.time()-t0:>8.0f}")

    best = max(rows.items(), key=lambda kv: kv[1]["total"])
    print(f"\n本轮最高：{best[0]}  total {best[1]['total']:.4f}")

    out = os.path.join(paths.RESULTS, "step12_ridge_ab")
    paths.ensure_dir(out)
    with open(os.path.join(out, "report.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write(f"契约空间 {ctx.X.shape[1]} 维 · lam={args.lam}\n\n")
        f.write(f"{'模型':<28}" + "".join(f"{c:>11}" for c in COLS) + "\n")
        for k, v in rows.items():
            f.write(f"{k:<28}" + "".join(f"{v[c]:>11.4f}" for c in COLS) + "\n")
        f.write(f"\n满秩上逐蛋白 Gram 相对旧实现 {d:+.4f}\n")
    print(f"写出 {out}/report.txt")


if __name__ == "__main__":
    main()
