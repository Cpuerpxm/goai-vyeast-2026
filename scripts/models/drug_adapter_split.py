"""按四类 val 分别看药物列的收益，决定 λ_drug 取 100 还是 300。

GPT Pro 的门槛（drug_resid 变化 ≥ -0.003）在 λ_drug=100 上不通过。
本脚本给出逐 split 的证据，判断这个不通过是真的风险还是聚合口径的假象。
"""
from __future__ import annotations
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import paths, split_guard as sg
from models.baseline_cfree import masked_ridge_exact
from models.design import freeze, encode, FEATURE_SETS
from scorer import evaluate as ev
from scorer.config import ScorerConfig

BASE_LAM = 3.0

def predict(ctx, Z, tr, lam_vec):
    mu, W = masked_ridge_exact(Z, ctx.X, tr, lam_vec, ctx.meta)
    y = (mu[None, :] + Z.astype(np.float64) @ W).astype(np.float32)
    return np.nan_to_num(y, nan=float(np.nanmedian(mu)))

def main():
    cfg = ScorerConfig()
    ctx = ev.build_context(verbose=False)
    tr = sg.train_rows(ctx.meta)
    cols = FEATURE_SETS["bio_tech"]

    spec0 = freeze(ctx.meta, tr, cols, with_drug=False)
    Z0 = encode(ctx.meta, spec0)
    spec1 = freeze(ctx.meta, tr, cols, with_drug=True)
    Z1 = encode(ctx.meta, spec1)
    n_ctx, n_drug = Z0.shape[1], Z1.shape[1] - Z0.shape[1]

    runs = {"无药物列": predict(ctx, Z0, tr, BASE_LAM)}
    for ld in (100.0, 300.0):
        lam = np.concatenate([np.full(n_ctx, BASE_LAM), np.full(n_drug, ld)])
        runs[f"药物 λ={ld:g}"] = predict(ctx, Z1, tr, lam)

    for name, y in runs.items():
        df = ev.evaluate_by_split(y, ctx, cfg)
        print(f"\n===== {name} =====")
        print(df.to_string(index=False))

if __name__ == "__main__":
    main()
