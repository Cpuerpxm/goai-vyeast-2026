"""满秩 + 逐蛋白 Gram 的 λ 扫描（GPT Pro L1-03）。

ridge_exact_ab 已证实：把共享 Gram 换成逐蛋白 Gram，满秩 total 由 0.4858 → 0.5060。
λ 是 Pro 指出的「尚未搜索的维度」，这里在契约空间内扫一遍。
"""
from __future__ import annotations
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import paths, split_guard as sg
from models.baseline_cfree import masked_ridge_exact
from models.design import freeze, encode, FEATURE_SETS
from models.select_k0 import fit_lowrank_pipeline
from scorer import evaluate as ev
from scorer.config import ScorerConfig

COLS = ["total", "abs_pcc", "abs_r2", "fc_pcc", "ctx_resid", "drug_resid"]
LAMS = [3.0, 10.0, 30.0, 100.0, 300.0]

def main():
    cfg = ScorerConfig()
    ctx = ev.build_context(verbose=False)
    val = ctx.rows(ev.VAL_SPLITS)
    tr = sg.train_rows(ctx.meta)
    spec = freeze(ctx.meta, tr, FEATURE_SETS["bio_tech"], with_drug=False)
    Z = encode(ctx.meta, spec)
    print(f"契约 {ctx.X.shape[1]} 维 · 设计 {Z.shape[1]} 维")
    print(f"{'模型':<26}" + "".join(f"{c:>11}" for c in COLS))
    print("-" * (26 + 11 * len(COLS)))
    rows = {}
    for lam in LAMS:
        t0 = time.time()
        mu, W = masked_ridge_exact(Z, ctx.X, tr, lam, ctx.meta)
        y = (mu[None, :] + Z.astype(np.float64) @ W).astype(np.float32)
        y = np.nan_to_num(y, nan=float(np.nanmedian(mu)))
        f = ev.flatten(ev.evaluate(y, ctx, val, cfg))
        rows[f"满秩exact λ={lam:g}"] = f
        print(f"{'满秩exact λ='+format(lam,'g'):<26}" + "".join(f"{f[c]:>11.4f}" for c in COLS))
    best = max(rows.items(), key=lambda kv: kv[1]["total"])
    print(f"\n最优：{best[0]}  total {best[1]['total']:.4f}")
    out = os.path.join(paths.RESULTS, "step13_lam_sweep"); paths.ensure_dir(out)
    with open(os.path.join(out, "report.txt"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"{'模型':<26}" + "".join(f"{c:>11}" for c in COLS) + "\n")
        for k, v in rows.items():
            fh.write(f"{k:<26}" + "".join(f"{v[c]:>11.4f}" for c in COLS) + "\n")
        fh.write(f"\n最优 {best[0]} total {best[1]['total']:.4f}\n")
    print(f"写出 {out}/report.txt")

if __name__ == "__main__":
    main()
