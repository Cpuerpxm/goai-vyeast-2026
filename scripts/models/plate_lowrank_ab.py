"""板号系数矩阵的低秩截断，以及按观测数自适应的 λ。

动机：板号块是 144 × 4,422 个系数，每块板平均只有 41 行训练数据，
逐蛋白单独估这 144 个偏移，噪声不小。批次效应通常活在低维子空间里，
把板号系数矩阵做 SVD 截到秩 r，能在几乎不损失结构的前提下降方差。

做法：先按现配置解出 W，取出板号那 144 行，SVD 截断到 r，写回去再预测。
这是后验截断，不是联合拟合，但足够判断「板号块该不该降秩」。

第二件：λ 现在对所有蛋白一样。缺失多的蛋白有效观测少，本该收得更紧。
试 λ_j = λ0 · (n_ref / n_obs_j)^p，p ∈ {0（现状）, 0.5, 1}。
"""
from __future__ import annotations
import os, sys, gc
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import paths, split_guard as sg
from models.baseline_cfree import ChunkedGram
from models.design import freeze, encode, FEATURE_SETS
from scorer import evaluate as ev
from scorer.config import ScorerConfig

COLS = ["total", "abs_pcc", "abs_r2", "fc_pcc", "ctx_resid", "drug_resid"]
LAM_CTX, LAM_PLATE, LAM_DRUG = 3.0, 10.0, 300.0
PLATE = "Yeast_cell_plate"


def score_W(ctx, Z, mu, W, val, cfg):
    y = np.nan_to_num((mu[None, :] + Z.astype(np.float64) @ W).astype(np.float32),
                      nan=float(np.nanmedian(mu)))
    return ev.flatten(ev.evaluate(y, ctx, val, cfg))


def main():
    cfg = ScorerConfig()
    ctx = ev.build_context(verbose=False)
    val = ctx.rows(ev.VAL_SPLITS)
    tr = sg.train_rows(ctx.meta)
    base_cols = FEATURE_SETS["bio_tech"]

    s0 = freeze(ctx.meta, tr, base_cols, with_drug=False)
    n_ctx = encode(ctx.meta, s0).shape[1]
    sp = freeze(ctx.meta, tr, list(base_cols) + [PLATE], with_drug=True)
    Z = encode(ctx.meta, sp)
    n_plate = len(sp["levels"][PLATE])
    n_drug = Z.shape[1] - n_ctx - n_plate
    lam = np.concatenate([np.full(n_ctx, LAM_CTX), np.full(n_plate, LAM_PLATE),
                          np.full(n_drug, LAM_DRUG)])

    g = ChunkedGram(Z, ctx.X, tr, ctx.meta)
    mu, W = g.solve(lam)
    print(f"{'配置':<32}" + "".join(f"{c:>11}" for c in COLS), flush=True)
    print("-" * (32 + 11 * len(COLS)))
    f0 = score_W(ctx, Z, mu, W, val, cfg)
    print(f"{'当前 板号满秩 144':<32}" + "".join(f"{f0[c]:>11.4f}" for c in COLS), flush=True)
    base = f0["total"]
    rows = {"当前 板号满秩": f0}

    # ---- 板号块 SVD 截断 ----
    sl = slice(n_ctx, n_ctx + n_plate)
    Wp = W[sl].copy()
    U, S, Vt = np.linalg.svd(Wp, full_matrices=False)
    ratio = np.cumsum(S**2) / np.sum(S**2)
    print(f"\n板号系数矩阵奇异值：前 5 解释 {ratio[4]:.1%}，前 20 解释 {ratio[19]:.1%}，"
          f"前 50 解释 {ratio[min(49,len(ratio)-1)]:.1%}\n", flush=True)
    for r in (5, 10, 20, 40, 80):
        W2 = W.copy()
        W2[sl] = (U[:, :r] * S[:r]) @ Vt[:r]
        f = score_W(ctx, Z, mu, W2, val, cfg)
        rows[f"板号降到秩 {r}"] = f
        print(f"{'板号降到秩 '+str(r):<32}" + "".join(f"{f[c]:>11.4f}" for c in COLS), flush=True)
        del W2; gc.collect()

    # ---- 按观测数自适应的 λ ----
    n_obs = np.isfinite(ctx.X[tr]).sum(axis=0).astype(np.float64)
    n_ref = float(np.median(n_obs))
    print(f"\n每蛋白训练观测数：中位 {n_ref:.0f}  最少 {n_obs.min():.0f}  最多 {n_obs.max():.0f}\n", flush=True)
    for p in (0.5, 1.0):
        scale = (n_ref / np.maximum(n_obs, 1.0)) ** p
        mu2, W3 = g.solve_perprotein(lam, scale)
        f = score_W(ctx, Z, mu2, W3, val, cfg)
        rows[f"λ 按观测数缩放 p={p:g}"] = f
        print(f"{'λ 按观测数缩放 p='+f'{p:g}':<32}" + "".join(f"{f[c]:>11.4f}" for c in COLS), flush=True)
        del W3; gc.collect()

    name, f = max(rows.items(), key=lambda kv: kv[1]["total"])
    d = f["total"] - base
    print(f"\n最优：{name}  total {f['total']:.4f}  相对当前 {d:+.4f}")
    print(f"止损线 +0.0007 → {'通过' if d >= 0.0007 else '未通过'}")

    out = os.path.join(paths.RESULTS, "step23_plate_lowrank"); paths.ensure_dir(out)
    with open(os.path.join(out, "report.txt"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"{'配置':<32}" + "".join(f"{c:>11}" for c in COLS) + "\n")
        for k, v in rows.items():
            fh.write(f"{k:<32}" + "".join(f"{v[c]:>11.4f}" for c in COLS) + "\n")
        fh.write(f"\n最优 {name} {f['total']:.6f}  {d:+.6f}\n")
    print(f"写出 {out}/report.txt")


if __name__ == "__main__":
    main()
