"""第 5 步 · 掩码低秩分解（基线谱 K₀ 与响应谱 K_Δ）。

docs/05 §5.2：
    y₀ ≈ μ + U₀ z₀      K₀ ∈ [32, 64]
    Δ  ≈ U_Δ z_Δ        K_Δ ≤ 42        ← 硬上限（43 个训练化合物中心化后秩 ≤ 42）

**只在观测位置计算重建损失**（27% 缺失不填补）。做法是加权 EM-PCA：
把缺失位置用当前重建值补上再做一次 SVD，反复迭代——等价于对观测位置的
加权最小二乘，缺失位置不贡献梯度。

三件事：
  1. 定 K₀：留出一部分**观测**单元格当验证，看重建误差随 K₀ 怎么走
  2. 定 K_Δ：同上，并核对 42 这个硬上限是不是真的在数据里成立
  3. 把两组基存盘给第 6 步用

❗K_Δ > 42 的任何配置直接拒绝（CLAUDE.md R5）。

运行：python lowrank.py
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import paths
from scorer import evaluate as ev

K_DELTA_HARD_CAP = 42          # CLAUDE.md R5，不可放宽
OUT_DIR = os.path.join(paths.RESULTS, "step5_lowrank")


def masked_pca(
    M: np.ndarray, k: int, mask: np.ndarray, n_iter: int = 12, tol: float = 1e-5,
    center: bool = True, seed: int = 0,
):
    """加权 EM-PCA：只在 mask 为真的位置拟合。

    返回 (mu, U (p,k), Z (n,k))，重建 = mu + Z @ U.T

    用截断随机 SVD 而不是完整 SVD：5,078 × 5,243 的完整 SVD 每次约 1 分钟，
    乘上 EM 迭代和 K 网格要跑几小时；只要前 k 个奇异向量，randomized_svd 快两个量级。
    """
    from sklearn.utils.extmath import randomized_svd

    Mf = M.astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu = (np.nanmean(np.where(mask, Mf, np.nan), axis=0)
              if center else np.zeros(Mf.shape[1], dtype=np.float32))
    mu = np.where(np.isfinite(mu), mu, 0.0).astype(np.float32)
    T = np.where(mask, Mf - mu, 0.0).astype(np.float32)     # 观测位置的目标
    F = T.copy()                                            # 缺失位置初值 0
    prev, U, Z = np.inf, None, None
    if k == 0:
        return mu, np.zeros((Mf.shape[1], 0), np.float32), np.zeros((Mf.shape[0], 0), np.float32)
    for _ in range(n_iter):
        Uf, S, Vt = randomized_svd(F, n_components=k, n_iter=2, random_state=seed)
        Z = (Uf * S).astype(np.float32)
        U = Vt.T.astype(np.float32)
        R = Z @ U.T
        F = np.where(mask, T, R).astype(np.float32)
        err = float(np.sqrt(np.mean((T[mask] - R[mask]) ** 2)))
        if abs(prev - err) < tol:
            break
        prev = err
    return mu, U, Z


def holdout_rmse(M: np.ndarray, k: int, frac: float = 0.08, seed: int = 0,
                 n_iter: int = 12, center: bool = True) -> tuple[float, float]:
    """把一部分**观测**单元格挖掉当验证集，报训练/验证 RMSE。"""
    rng = np.random.default_rng(seed)
    obs = np.isfinite(M)
    hold = obs & (rng.random(M.shape) < frac)
    fit = obs & ~hold
    mu, U, Z = masked_pca(M, k, fit, n_iter=n_iter, center=center, seed=seed)
    R = (np.broadcast_to(mu, M.shape) if k == 0 else mu[None, :] + Z @ U.T)
    tr = float(np.sqrt(np.mean((M[fit] - R[fit]) ** 2)))
    te = float(np.sqrt(np.mean((M[hold] - R[hold]) ** 2)))
    return tr, te


def encode_masked(M: np.ndarray, mu: np.ndarray, U: np.ndarray, enc: np.ndarray,
                  n_iter: int = 8, lam: float = 1e-2) -> np.ndarray:
    """同 project_masked，但返回低秩系数 Z 而不是重建矩阵。"""
    k = U.shape[1]
    if k == 0:
        return np.zeros((M.shape[0], 0), dtype=np.float32)
    A = np.linalg.inv(U.T @ U + lam * np.trace(U.T @ U) / k * np.eye(k)).astype(np.float32)
    T = np.where(enc, np.nan_to_num(M, nan=0.0) - mu, 0.0).astype(np.float32)
    F = T.copy()
    Z = None
    for _ in range(n_iter):
        Z = (F @ U) @ A
        F = np.where(enc, T, Z @ U.T).astype(np.float32)
    return Z.astype(np.float32)


def project_masked(M: np.ndarray, mu: np.ndarray, U: np.ndarray, enc: np.ndarray,
                   n_iter: int = 8, lam: float = 1e-2) -> np.ndarray:
    """在 enc 掩码下把每行投到基 U 上（掩码最小二乘），返回重建矩阵。

    用 EM 迭代（缺失位置填当前重建值）而不是逐行解正规方程：
    逐行建 UᵀU 是 O(n·p·k²)，在 k=128、n≈1700 时要几小时；EM 只有矩阵乘，
    O(n·p·k) 每轮，收敛到同一个解。
    """
    k = U.shape[1]
    if k == 0:
        return np.broadcast_to(mu, M.shape).astype(np.float32)
    A = np.linalg.inv(U.T @ U + lam * np.trace(U.T @ U) / k * np.eye(k)).astype(np.float32)
    T = np.where(enc, np.nan_to_num(M, nan=0.0) - mu, 0.0).astype(np.float32)
    F = T.copy()
    R = None
    for _ in range(n_iter):
        Z = (F @ U) @ A
        R = Z @ U.T
        F = np.where(enc, T, R).astype(np.float32)
    return (mu[None, :] + R).astype(np.float32)


def grouped_holdout_rmse(
    M: np.ndarray, k: int, group: np.ndarray, n_fold: int = 3, seed: int = 0,
    center: bool = True, n_iter: int = 10, encode_frac: float = 0.5,
) -> tuple[float, float]:
    """按**整组**留出（整样本 / 整化合物），再在留出行内部分编码/评分单元格。

    随机挖单元格的做法会让 K 越大越好——因为基可以顺着同一行其它单元格
    把该行自己的噪声也装进去。要问的是「这组基能不能装下**没见过的**行/化合物」，
    就必须按组留出。
    返回 (基线RMSE, 秩K的RMSE)，两者都只在评分单元格上算。
    """
    rng = np.random.default_rng(seed)
    obs = np.isfinite(M)
    ug = np.unique(group)
    fold = rng.permutation(len(ug)) % n_fold
    num_k = den = num_b = 0.0
    for f in range(n_fold):
        te = np.isin(group, ug[fold == f])
        tr = ~te
        if tr.sum() < 10 or te.sum() < 2:
            continue
        mu, U, _ = masked_pca(M[tr], k, obs[tr], n_iter=n_iter, center=center, seed=seed)
        sub = M[te]
        o = obs[te]
        enc = o & (rng.random(sub.shape) < encode_frac)
        sc = o & ~enc
        R = project_masked(sub, mu, U, enc, n_iter=8)
        num_k += float(np.sum((sub[sc] - R[sc]) ** 2))
        num_b += float(np.sum((sub[sc] - np.broadcast_to(mu, sub.shape)[sc]) ** 2))
        den += float(sc.sum())
    if den == 0:
        return np.nan, np.nan
    return float(np.sqrt(num_b / den)), float(np.sqrt(num_k / den))


def compound_response_matrix(ctx: ev.EvalContext) -> tuple[np.ndarray, list]:
    """43 × 5243 的训练化合物平均响应矩阵，用来核对秩上限。"""
    pert = ctx.meta["perturbation_no_concentration"].astype(str).to_numpy()
    names, rows = [], []
    for d in sorted(np.unique(pert[ctx.train_mask])):
        sel = ctx.train_mask & (pert == d)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            rows.append(np.nanmean(ctx.D[sel], axis=0, dtype=np.float64))
        names.append(d)
    return np.vstack(rows), names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=15)
    args = ap.parse_args()
    paths.ensure_dir(OUT_DIR)
    ctx = ev.build_context()
    tr = ctx.train_mask

    L: list[str] = []
    a = L.append
    a("=" * 92)
    a("第 5 步 · 掩码低秩分解")
    a("=" * 92)
    a(f"只用训练处理样本拟合（{int(tr.sum())} 行），缺失位置不进损失。")
    a("留出 8% 的**已观测**单元格作验证，报重建 RMSE。")
    a("")

    # ---------------------------------------------------------- K₀
    a("-" * 92)
    a("1. 基线谱 y₀ 的秩 K₀（docs/05 建议区间 32–64）")
    a("-" * 92)
    X = ctx.X[tr].astype(np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        _sd = float(np.nanstd(X - np.nanmean(X, axis=0)))
    a(f"  观测单元格 {int(np.isfinite(X).sum()):,}   逐蛋白中心化后的总标准差 {_sd:.4f}")
    a("")
    a("  两种留出协议对照：")
    a("    随机单元格  = 在同一行里挖掉部分观测值。K 越大越好，因为基可以顺着")
    a("                  该行其它单元格把这一行自己的噪声也装进去 → **选 K 会选爆**")
    a("    整样本留出  = 整行不参与拟合；留出行内一半单元格编码、另一半评分")
    a("                  → 问的是「基能不能装下没见过的样本」，这才是要的口径")
    a("")
    samp_group = np.arange(int(tr.sum()))          # 每行一组 = 整样本留出
    a(f"  {'K₀':>5}{'随机格·验证':>14}{'整样本·验证':>14}{'整样本/基线':>14}")
    _, base_cell = holdout_rmse(X, 0, seed=1, n_iter=1)
    base_grp, _ = grouped_holdout_rmse(X, 1, samp_group, seed=1, n_iter=args.iters)
    best_k0, best_e = 1, np.inf
    for k in [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128]:
        _, e_cell = holdout_rmse(X, k, seed=1, n_iter=args.iters)
        _, e_grp = grouped_holdout_rmse(X, k, samp_group, seed=1, n_iter=args.iters)
        if e_grp < best_e:
            best_e, best_k0 = e_grp, k
        a(f"  {k:>5d}{e_cell:>14.4f}{e_grp:>14.4f}{e_grp/base_grp:>14.3f}")
    a(f"  （只用逐蛋白均值：随机格 {base_cell:.4f}，整样本留出 {base_grp:.4f}）")
    a(f"  → 按**整样本留出**选出的 K₀ = {best_k0}")
    a("")

    # ---------------------------------------------------------- 秩上限核对
    a("-" * 92)
    a("2. 响应矩阵的秩上限核对（CLAUDE.md R5：K_Δ ≤ 42）")
    a("-" * 92)
    Dm, names = compound_response_matrix(ctx)
    a(f"  ❗区分两个口径：train_val 全集有 43 个非对照化合物（秩上限 42，即 R5 的来源）；")
    a(f"    但官方 split_final=='train' 只有 {len(names)} 个——另外几个整个被划进了 val。")
    a(f"    模型只能用 train 拟合，所以**实际**可用秩上限是 {len(names)-1}，比 42 还紧。")
    a(f"  训练化合物数 {len(names)}   矩阵 {Dm.shape}")
    Dc = Dm - np.nanmean(Dm, axis=0)
    Dc = np.nan_to_num(Dc, nan=0.0)
    sv = np.linalg.svd(Dc, compute_uv=False)
    rank = int((sv > sv[0] * 1e-10).sum())
    a(f"  中心化后数值秩 {rank}   理论上限 {len(names)-1}")
    a(f"  奇异值前 10：" + "  ".join(f"{v:.2f}" for v in sv[:10]))
    ev_ratio = np.cumsum(sv ** 2) / np.sum(sv ** 2)
    for q in [0.5, 0.8, 0.9, 0.95]:
        a(f"  解释 {q:.0%} 方差需要 {int(np.searchsorted(ev_ratio, q)) + 1} 个方向")
    assert rank <= K_DELTA_HARD_CAP, f"秩 {rank} 超过硬上限 {K_DELTA_HARD_CAP}"
    a(f"  ✅ 秩 {rank} ≤ 硬上限 {K_DELTA_HARD_CAP}，R5 成立")
    a("")

    # ---------------------------------------------------------- K_Δ
    a("-" * 92)
    a("3. 响应谱 Δ 的秩 K_Δ（硬上限 42）")
    a("-" * 92)
    Dt = ctx.D[tr].astype(np.float64)
    a(f"  观测单元格 {int(np.isfinite(Dt).sum()):,}")
    a("  留出协议：**整化合物留出**（3 折）。基只由其余化合物拟合，")
    a("  留出化合物的样本一半单元格编码、一半评分——这才是「新化合物的响应能不能")
    a("  被这组基表示」，与 S1 的任务形态一致。")
    a("")
    drug_group = ctx.meta["perturbation_no_concentration"].astype(str).to_numpy()[tr]
    a(f"  {'K_Δ':>5}{'随机格·验证':>14}{'整化合物·验证':>16}{'整化合物/基线':>16}")
    _, based_cell = holdout_rmse(Dt, 0, seed=2, n_iter=1, center=True)
    based_grp, _ = grouped_holdout_rmse(Dt, 1, drug_group, seed=2, n_iter=args.iters)
    best_kd, best_ed = 1, np.inf
    for k in [1, 2, 4, 8, 12, 16, 24, 32, 36, 42]:
        if k > K_DELTA_HARD_CAP:
            a(f"  {k:>5d}   拒绝：超过秩上限 {K_DELTA_HARD_CAP}（R5）")
            continue
        _, e_cell = holdout_rmse(Dt, k, seed=2, n_iter=args.iters)
        _, e_grp = grouped_holdout_rmse(Dt, k, drug_group, seed=2, n_iter=args.iters)
        if e_grp < best_ed:
            best_ed, best_kd = e_grp, k
        a(f"  {k:>5d}{e_cell:>14.4f}{e_grp:>16.4f}{e_grp/based_grp:>16.3f}")
    a(f"  （只用逐蛋白平均 Δ：随机格 {based_cell:.4f}，整化合物留出 {based_grp:.4f}）")
    a(f"  → 按**整化合物留出**选出的 K_Δ = {best_kd}")
    a("")
    a("  注意区分两件事：这里的 RMSE 衡量『低秩基能不能装下 Δ』，")
    a("  不等于『从药物+上下文能不能预测出 z_Δ』。后者是第 6 步的事，")
    a("  而单样本 Δ 的可靠性只有 0.115（见 noise_ceiling.txt），")
    a("  所以 K_Δ 取大反而容易把噪声一起装进来。")
    a("")

    # ---------------------------------------------------------- 存盘
    mu0, U0, _ = masked_pca(X, best_k0, np.isfinite(X), n_iter=args.iters)
    mud, Ud, _ = masked_pca(Dt, best_kd, np.isfinite(Dt), n_iter=args.iters)
    fp = os.path.join(OUT_DIR, "bases.npz")
    np.savez(fp, mu0=mu0.astype(np.float32), U0=U0.astype(np.float32), K0=best_k0,
             mu_delta=mud.astype(np.float32), U_delta=Ud.astype(np.float32),
             K_delta=best_kd, proteins=ctx.proteins,
             compound_svals=sv.astype(np.float32),
             compounds=np.asarray([str(x) for x in names], dtype="<U64"))
    a(f"产出：{fp}")
    a(f"  mu0/U0 (K₀={best_k0})、mu_delta/U_delta (K_Δ={best_kd})，均仅由训练行拟合。")

    txt = "\n".join(L)
    print(txt)
    p = os.path.join(OUT_DIR, "report.txt")
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(txt)
    print(f"\n[写出] {p}")


if __name__ == "__main__":
    main()
