"""小比例集成（GPT Pro L1-03 的剩余部分）。

λ 扫描已经做完，满秩也已经反超低秩，所以 L1-03 里「K16/K32 + 尾部秩连续收缩」
那半条失效了。剩下能试的是把两条不同形状的预测按小比例混合：

    y = (1-β)·满秩精确岭  +  β·低秩

两者的误差结构不同（一个满秩逐蛋白解，一个走 K0 维瓶颈），
小比例混合通常能吃到一点方差下降。β 只走粗网格，避免在单个 val 上过拟合。
"""
from __future__ import annotations
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import paths, split_guard as sg
from models.baseline_cfree import masked_ridge_exact
from models.design import freeze, encode, FEATURE_SETS
from models.select_k0 import fit_lowrank_pipeline
from scorer import evaluate as ev
from scorer.config import ScorerConfig

COLS = ["total", "abs_pcc", "abs_r2", "fc_pcc", "ctx_resid", "drug_resid"]


def main():
    cfg = ScorerConfig()
    ctx = ev.build_context(verbose=False)
    val = ctx.rows(ev.VAL_SPLITS)
    tr = sg.train_rows(ctx.meta)
    cols = FEATURE_SETS["bio_tech"]

    s0 = freeze(ctx.meta, tr, cols, with_drug=False)
    Z0 = encode(ctx.meta, s0); n_ctx = Z0.shape[1]
    s1 = freeze(ctx.meta, tr, cols, with_drug=True)
    Z1 = encode(ctx.meta, s1); n_drug = Z1.shape[1] - n_ctx
    lam = np.concatenate([np.full(n_ctx, 3.0), np.full(n_drug, 300.0)])

    mu, W = masked_ridge_exact(Z1, ctx.X, tr, lam, ctx.meta)
    y_full = np.nan_to_num((mu[None, :] + Z1.astype(np.float64) @ W).astype(np.float32),
                           nan=float(np.nanmedian(mu)))
    y_low = np.nan_to_num(fit_lowrank_pipeline(ctx, Z0, tr, 32, 3.0), nan=0.0)

    print(f"{'配置':<30}" + "".join(f"{c:>11}" for c in COLS))
    print("-" * (30 + 11 * len(COLS)))
    f_full = ev.flatten(ev.evaluate(y_full, ctx, val, cfg))
    print(f"{'满秩 β=0（当前）':<30}" + "".join(f"{f_full[c]:>11.4f}" for c in COLS))
    base = f_full["total"]
    best = ("满秩 β=0", base, f_full)

    for beta in (0.05, 0.10, 0.20, 0.35, 0.50):
        y = ((1 - beta) * y_full.astype(np.float64) + beta * y_low.astype(np.float64)).astype(np.float32)
        f = ev.flatten(ev.evaluate(y, ctx, val, cfg))
        tag = f"混合 β={beta:g}"
        mark = ""
        if f["total"] > best[1]:
            best = (tag, f["total"], f); mark = "  ←"
        print(f"{tag:<30}" + "".join(f"{f[c]:>11.4f}" for c in COLS) + mark)

    f_low = ev.flatten(ev.evaluate(y_low, ctx, val, cfg))
    print(f"{'低秩 β=1（参照）':<30}" + "".join(f"{f_low[c]:>11.4f}" for c in COLS))

    d = best[1] - base
    print(f"\n最优：{best[0]}  total {best[1]:.4f}  相对当前 {d:+.4f}")
    print(f"止损线 +0.0007 → {'通过' if d >= 0.0007 else '未通过，不采用'}")
    if d >= 0.0007:
        for k, thr in (("abs_r2", -0.0015), ("ctx_resid", -0.003), ("drug_resid", -0.003)):
            dv = best[2][k] - f_full[k]
            print(f"  {k} 变化 {dv:+.4f}（门槛 ≥ {thr}） {'通过' if dv >= thr else '不通过'}")

    out = os.path.join(paths.RESULTS, "step19_blend"); paths.ensure_dir(out)
    with open(os.path.join(out, "report.txt"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"满秩 total {base:.6f}\n{best[0]} total {best[1]:.6f}  {d:+.6f}\n")
    print(f"写出 {out}/report.txt")


if __name__ == "__main__":
    main()
