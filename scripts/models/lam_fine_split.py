"""λ 细扫 + 按 val split 分解（Pro 保留门槛：不得只由单个 split 拉升）。"""
from __future__ import annotations
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import paths, split_guard as sg
from models.baseline_cfree import masked_ridge, masked_ridge_exact
from models.design import freeze, encode, FEATURE_SETS
from models.select_k0 import fit_lowrank_pipeline
from scorer import evaluate as ev
from scorer.config import ScorerConfig

def fit_full(ctx, Z, tr, lam, exact=True):
    fn = masked_ridge_exact if exact else masked_ridge
    mu, W = fn(Z, ctx.X, tr, lam, ctx.meta)
    y = (mu[None, :] + Z.astype(np.float64) @ W).astype(np.float32)
    return np.nan_to_num(y, nan=float(np.nanmedian(mu)))

def main():
    cfg = ScorerConfig()
    ctx = ev.build_context(verbose=False)
    val = ctx.rows(ev.VAL_SPLITS)
    tr = sg.train_rows(ctx.meta)
    spec = freeze(ctx.meta, tr, FEATURE_SETS["bio_tech"], with_drug=False)
    Z = encode(ctx.meta, spec)

    print("=== λ 细扫（满秩 exact）===")
    best = (None, -9)
    for lam in [0.3, 1.0, 3.0, 5.0, 10.0]:
        y = fit_full(ctx, Z, tr, lam)
        t = ev.flatten(ev.evaluate(y, ctx, val, cfg))["total"]
        print(f"  λ={lam:<6g} total {t:.4f}")
        if t > best[1]:
            best = (lam, t)
    print(f"  最优 λ={best[0]:g}  total {best[1]:.4f}\n")

    print("=== 按 val split 分解：旧共享Gram vs 逐蛋白Gram（λ=最优）===")
    for tag, exact in (("旧共享Gram", False), ("逐蛋白Gram", True)):
        y = fit_full(ctx, Z, tr, best[0], exact=exact)
        df = ev.evaluate_by_split(y, ctx, cfg)
        print(f"\n[{tag}]")
        print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    out = os.path.join(paths.RESULTS, "step14_lam_fine"); paths.ensure_dir(out)
    with open(os.path.join(out, "best_lam.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write(f"best_lam={best[0]:g}\ntotal={best[1]:.6f}\n")
    print(f"\n写出 {out}/best_lam.txt")

if __name__ == "__main__":
    main()
