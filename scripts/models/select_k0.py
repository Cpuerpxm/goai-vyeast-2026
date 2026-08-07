"""Day 3 · 用**端到端预测分数**选 K₀，不再用半观测重建。

Pro R2 L1-06：此前用「观察留出样本一半蛋白、重建另一半」选 K，那是**插补**口径——
真实预测时看不到留出样本的任何蛋白值。两个口径会选出完全不同的 K。

本脚本的口径与部署一致：

    metadata --ridge--> z₀ (K₀ 维) --U₀--> 完整 5,243 维预测

留出样本的**任何蛋白值都不参与**其自身预测；U₀ 与 ridge 权重都只由训练行拟合。
选择依据是官方评分器的 total，不是 RMSE。

对照两条曲线：
  - 端到端预测（本脚本）
  - 逐蛋白直接 ridge（不走低秩，等价于 K₀ = 全维），看低秩到底有没有必要

运行：python select_k0.py
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import paths
from models.baseline_cfree import FEATURE_SETS, design, masked_ridge
from models.lowrank import masked_pca
from scorer import evaluate as ev
from scorer.config import ScorerConfig

OUT_DIR = os.path.join(paths.RESULTS, "step8_select_k0")


def fit_lowrank_pipeline(ctx, Z, train_rows, k, lam):
    """训练侧：掩码 PCA 取 U₀ → metadata 岭回归预测 z₀ → 解码回全维。

    留出行只提供 metadata，其蛋白值不参与任何一步。
    """
    Xtr = ctx.X[train_rows]
    mu, U, Ztr = masked_pca(Xtr, k, np.isfinite(Xtr), n_iter=12, center=True, seed=0)
    # 目标是训练行的低秩系数；用 metadata 回归它
    Zd = Z[train_rows].astype(np.float64)
    G = Zd.T @ Zd + lam * np.eye(Z.shape[1])
    W = np.linalg.solve(G, Zd.T @ Ztr.astype(np.float64))
    z_hat = Z.astype(np.float64) @ W
    return (mu[None, :] + z_hat @ U.T.astype(np.float64)).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", choices=list(FEATURE_SETS), default="bio_tech")
    ap.add_argument("--lam", type=float, default=30.0)
    args = ap.parse_args()

    paths.ensure_dir(OUT_DIR)
    cfg = ScorerConfig()
    ctx = ev.build_context()
    val = ctx.rows(ev.VAL_SPLITS)
    train_all = (ctx.meta["split_final"] == "train").to_numpy()
    cols = FEATURE_SETS[args.features]
    Z = design(ctx.meta, cols, with_drug=False)

    L: list[str] = []
    a = L.append
    a("=" * 96)
    a("Day 3 · 端到端选 K₀（口径与部署一致，留出样本的蛋白值全程不参与）")
    a("=" * 96)
    a(f"特征集 {args.features}：{' / '.join(cols)} + log-time 三次多项式（{Z.shape[1]} 维）")
    a(f"训练行 {int(train_all.sum())}；评估在官方四类 val 上，依据是评分器 total")
    a("")
    a("对比的是两个**不同的问题**：")
    a("  半观测重建 = 给定该样本一半蛋白，能否补出另一半（插补）→ 此前用它选出 K₀=96")
    a("  端到端预测 = 只给 metadata，能否预测完整蛋白谱（部署）→ 本表")
    a("")

    cols_out = ["total", "abs_pcc", "abs_r2", "fc_pcc", "ctx_resid", "drug_resid"]
    a(f"  {'K₀':>5}" + "".join(f"{c:>11}" for c in cols_out))
    a("  " + "-" * (5 + 11 * len(cols_out)))
    rows = []
    for k in [2, 4, 8, 16, 24, 32, 48, 64, 96, 128]:
        y = fit_lowrank_pipeline(ctx, Z, train_all, k, args.lam)
        y = np.nan_to_num(y, nan=0.0)
        f = ev.flatten(ev.evaluate(y, ctx, val, cfg))
        rows.append((k, f))
        a(f"  {k:>5d}" + "".join(f"{f[c]:>11.4f}" for c in cols_out))

    # 不走低秩：逐蛋白直接 ridge（等价于满秩）
    mu, W = masked_ridge(Z, ctx.X, train_all, args.lam)
    y_full = (mu[None, :] + Z.astype(np.float64) @ W).astype(np.float32)
    y_full = np.nan_to_num(y_full, nan=float(np.nanmedian(mu)))
    f_full = ev.flatten(ev.evaluate(y_full, ctx, val, cfg))
    a(f"  {'满秩':>5}" + "".join(f"{f_full[c]:>11.4f}" for c in cols_out)
      + "   ← 逐蛋白直接 ridge，不走低秩")
    a("")

    best_k, best_f = max(rows, key=lambda t: t[1]["total"])
    a(f"端到端最优 K₀ = {best_k}（total {best_f['total']:.4f}）")
    a(f"满秩 ridge total {f_full['total']:.4f}，差 {best_f['total'] - f_full['total']:+.4f}")
    a("")
    if f_full["total"] >= best_f["total"] - 1e-4:
        a("→ **低秩没有带来增益**：metadata 的自由度（%d 维）本来就远小于 K₀，"
          % Z.shape[1])
        a("  再压一层低秩只会丢信息。低秩的价值在响应模块 Δ，不在绝对基线 b̂。")
    else:
        a(f"→ 低秩有增益，保留 K₀ = {best_k}。")
    a("")
    a("对照此前用半观测重建选出的 K₀ = 96：")
    k96 = [f for k, f in rows if k == 96]
    if k96:
        a(f"  端到端 total {k96[0]['total']:.4f}，与端到端最优（K₀={best_k}）差 "
          f"{k96[0]['total'] - best_f['total']:+.4f}")
    a("  两个口径选出的 K 不同就说明：重建 RMSE 不能用来定部署用的秩。")

    txt = "\n".join(L)
    print(txt)
    p = os.path.join(OUT_DIR, "report.txt")
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(txt)
    print(f"\n[写出] {p}")


if __name__ == "__main__":
    main()
