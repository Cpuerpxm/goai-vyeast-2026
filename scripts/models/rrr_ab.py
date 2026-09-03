"""监督式降秩回归（GPT Pro 会诊 2026-09-02 的 L1-02，最后一项没做的）。

想法：现在每个蛋白各解一次 247 维的岭回归，等于 4,422 个互不相干的问题。
但这些蛋白的响应显然共享结构，把系数矩阵 W(247 × 4,422) 限制在低秩子空间里，
能用蛋白之间的相关性给每一个蛋白降方差。

与 `select_k0.py` 那条低秩管线的区别：那条对**原始 X** 做无监督 PCA，
先决定基再拟合；RRR 是对**元数据能解释的那部分拟合信号** Ŷ = ZW 做 SVD，
基由监督信号决定。Pro 的原话是「先做真正的逐蛋白掩码 ridge，再对元数据可预测的
拟合信号做 SVD，而不是先对原始 X 做无监督 PCA」。

怎么算得快（否则 Ŷ 是 5,920 × 4,422，SVD 很贵）：

    W = U Σ V'                      （247 × 4,422 的 SVD，便宜）
    Ŷ'Ŷ = V Σ (U' Z'Z U) Σ V' = V K V'
    K = Σ (U' G_z U) Σ              （247 × 247）
    K = Q Λ Q'  →  Ŷ 的右奇异向量就是 V Q

所以秩 r 的投影是  W_r = U Σ (Q_r Q_r') V'，全程只碰 247 维的东西。

除了硬截断，也试连续收缩：把 Q Q' 换成 Q diag(λ/(λ+γ)) Q'，比一刀切平滑。
"""
from __future__ import annotations
import os, sys, gc
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import paths, split_guard as sg
from models.baseline_cfree import ChunkedGram
from models.final_grid import build
from scorer import evaluate as ev
from scorer.config import ScorerConfig

COLS = ["total", "abs_pcc", "abs_r2", "fc_pcc", "ctx_resid", "drug_resid"]
LAM = (3.0, 10.0, 100.0, 30.0)      # step24 网格最优


def main():
    cfg = ScorerConfig()
    ctx = ev.build_context(verbose=False)
    val = ctx.rows(ev.VAL_SPLITS)
    tr = sg.train_rows(ctx.meta)
    Z, n_ctx, n_plate, n_drug, _ = build(ctx, tr)
    lc, lp, ld, li = LAM
    lam = np.concatenate([np.full(n_ctx, lc), np.full(n_plate, lp),
                          np.full(n_drug, ld), np.full(n_drug, li)])

    g = ChunkedGram(Z, ctx.X, tr, ctx.meta)
    mu, W = g.solve(lam)
    d, p = W.shape

    def sc(Wx, tag, store):
        y = np.nan_to_num((mu[None, :] + Z.astype(np.float64) @ Wx).astype(np.float32),
                          nan=float(np.nanmedian(mu)))
        f = ev.flatten(ev.evaluate(y, ctx, val, cfg))
        store[tag] = f
        print(f"{tag:<30}" + "".join(f"{f[c]:>11.4f}" for c in COLS), flush=True)
        del y; gc.collect()
        return f

    print(f"{'配置':<30}" + "".join(f"{c:>11}" for c in COLS), flush=True)
    print("-" * (30 + 11 * len(COLS)))
    rows = {}
    f0 = sc(W, "满秩 W（当前）", rows)
    base = f0["total"]

    # W 的 SVD 与 Ŷ 的右奇异子空间
    U, S, Vt = np.linalg.svd(W, full_matrices=False)      # U:(d,k) S:(k,) Vt:(k,p)
    Zt = Z[tr].astype(np.float64)
    G_z = Zt.T @ Zt
    K = (S[:, None] * (U.T @ G_z @ U)) * S[None, :]
    K = (K + K.T) / 2
    lam_k, Q = np.linalg.eigh(K)
    order = np.argsort(-lam_k)
    lam_k, Q = lam_k[order], Q[:, order]
    ratio = np.cumsum(np.maximum(lam_k, 0)) / np.sum(np.maximum(lam_k, 0))
    k = len(lam_k)
    print(f"\n拟合信号 Ŷ 的谱：前 10 解释 {ratio[9]:.1%}，前 30 解释 {ratio[29]:.1%}，"
          f"前 100 解释 {ratio[min(99,k-1)]:.1%}，满秩 {k}\n", flush=True)

    US = U * S[None, :]                                    # (d, k)
    for r in (10, 30, 60, 100, 150, 200):
        if r >= k:
            continue
        Qr = Q[:, :r]
        Wr = US @ (Qr @ (Qr.T @ Vt))
        sc(Wr, f"RRR 硬截断 秩 {r}", rows)
        del Wr; gc.collect()

    # 连续收缩：λ/(λ+γ)，γ 取谱的若干分位
    pos = lam_k[lam_k > 0]
    for q in (0.5, 0.75, 0.9):
        gamma = float(np.quantile(pos, 1 - q))
        w = lam_k / (lam_k + gamma)
        w = np.clip(w, 0.0, 1.0)
        Wг = US @ ((Q * w[None, :]) @ (Q.T @ Vt))
        sc(Wг, f"RRR 连续收缩 γ=谱{q:.0%}分位", rows)
        del Wг; gc.collect()

    name, f = max(rows.items(), key=lambda kv: kv[1]["total"])
    dlt = f["total"] - base
    print(f"\n最优：{name}  total {f['total']:.4f}  相对满秩 {dlt:+.4f}")
    print(f"止损线 +0.0007 → {'通过' if dlt >= 0.0007 else '未通过，RRR 不采用'}")

    out = os.path.join(paths.RESULTS, "step27_rrr"); paths.ensure_dir(out)
    with open(os.path.join(out, "report.txt"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"{'配置':<30}" + "".join(f"{c:>11}" for c in COLS) + "\n")
        for kk, v in rows.items():
            fh.write(f"{kk:<30}" + "".join(f"{v[c]:>11.4f}" for c in COLS) + "\n")
        fh.write(f"\n最优 {name} {f['total']:.6f}  {dlt:+.6f}\n")
    print(f"写出 {out}/report.txt")


if __name__ == "__main__":
    main()
