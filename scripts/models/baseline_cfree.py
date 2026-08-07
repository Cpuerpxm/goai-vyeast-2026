"""C-free 可部署骨架：预测函数**不接收**真实对照。

为什么必须有这个（2026-08-05 Pro R2 L1-04）：
A 分支隔离 `proteome_raw_test.csv`，而测试集的 Water/DMSO 对照就是该文件里的行。
所以推断时拿不到 C。此前 B1/B2/B3/B4/低秩响应/α 收缩**全部**是 `ŷ = C + Δ̂` 形态，
它们是 **oracle 诊断模型**，不是可提交模型。

    ŷ = b̂(metadata)  +  γ · Δ̂(compound, context)

关键性质：C-free 模型的 `Δ_pred = b̂ − C` **完整包含 −C**，
所以共享参照那份红利自动拿满（B0 拿到 fc_pcc 0.1872 正是这个原因）。
放弃读真实对照**不损失** FC 那份分，反而把胜负手推回 b̂ 的准确度——
即指标 1 的 R²，也是唯一没被参照污染的地方。

**硬约束**：本模块所有预测函数的签名里都不得出现 C / 对照 / 测试蛋白质组。
`assert_c_free()` 在返回前检查预测未被对照信息污染。

运行：
    python baseline_cfree.py
    python baseline_cfree.py --features bio_tech   # 含仪器/来源
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import paths
from scorer import evaluate as ev
from scorer.config import ScorerConfig

PERT_COL = "perturbation_no_concentration"
OUT_DIR = os.path.join(paths.RESULTS, "step7_cfree")

FEATURE_SETS = {
    # 只有生物条件。板号与培养基/温度/时间的 Cramér's V = 0.992，
    # 加板号等于把这三个生物变量的身份直接背下来，对新板不可泛化。
    "bio": ["Strains", "Medium", "Temperature"],
    # 加测量上下文（手册明说测量上下文用于吸收测量变异）
    "bio_tech": ["Strains", "Medium", "Temperature", "data_source", "instrument"],
}


def design(meta: pd.DataFrame, cols, with_drug: bool, drugs=None) -> np.ndarray:
    """one-hot + log-time 三次多项式。未见水平整行为 0 → 岭回归自动回退到总体先验。"""
    blocks = [np.ones((len(meta), 1), dtype=np.float32)]
    for c in cols:
        codes, uniq = pd.factorize(meta[c].astype(str))
        M = np.zeros((len(meta), len(uniq)), dtype=np.float32)
        M[np.arange(len(meta)), codes] = 1.0
        blocks.append(M)
    t = np.log1p(meta["pert_time"].to_numpy(dtype=np.float64))
    t = (t - t.mean()) / t.std()
    blocks.append(np.stack([t, t ** 2, t ** 3], axis=1).astype(np.float32))
    if with_drug:
        idx = {d: i for i, d in enumerate(drugs)}
        M = np.zeros((len(meta), len(drugs)), dtype=np.float32)
        for i, d in enumerate(meta[PERT_COL].astype(str)):
            j = idx.get(d)
            if j is not None:
                M[i, j] = 1.0       # 未见化合物整行为 0 → 响应项归零
        blocks.append(M)
    return np.hstack(blocks)


def masked_ridge(Z: np.ndarray, Y: np.ndarray, fit_rows: np.ndarray, lam: float):
    """逐蛋白掩码岭回归。

    缺失位置不进 Z'y 的求和，再按该蛋白的观测数重标定。
    ❗Y 必须先逐蛋白中心化：绝对丰度 ~20，直接把缺失当 0 会把预测拉向 0。
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu = np.nanmean(Y[fit_rows], axis=0, dtype=np.float64)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    Yc = Y - mu
    M = np.isfinite(Yc)
    Yf = np.where(M, Yc, 0.0).astype(np.float64)

    Zt = Z[fit_rows].astype(np.float64)
    G = Zt.T @ Zt + lam * np.eye(Z.shape[1])
    B = Zt.T @ Yf[fit_rows]
    n_all = float(fit_rows.sum())
    n_obs = M[fit_rows].sum(axis=0).astype(np.float64)
    scale = np.where(n_obs > 0, n_all / np.maximum(n_obs, 1.0), 0.0)
    W = np.linalg.solve(G, B * scale[None, :])
    return mu.astype(np.float32), W


def assert_c_free(y_pred: np.ndarray, ctx: ev.EvalContext, cond_cols,
                  tol: float = 0.01) -> None:
    """守卫：预测不得携带样本特异的对照信息。

    判据：预测与「该样本自己的对照谱」的相关，不应高于它与
    「**在模型所用条件变量上完全相同**的另一个样本的对照谱」的相关。

    ❗置换集必须与模型的条件集一致，否则全是误报：
      - 做全局置换 → b̂ 与 C_own 共享菌株/培养基/温度/时间，本来就该更像（误报 +0.06）
      - 只固定生物条件 → 还漏了 data_source / instrument（误报 +0.04）
    两者都是**合法的条件相关**，不是泄漏。只有在模型声明的全部条件变量上都相同、
    却仍然更像自己的对照，才说明预测里带了样本特异信息。
    """
    from scorer.metrics import pcc_axis

    cfg = ScorerConfig(min_valid_points=30)
    ok = ctx.treated & np.isfinite(ctx.C).any(axis=1)
    key = ctx.meta[list(cond_cols) + ["pert_time"]] \
        .astype(str).agg("\x1f".join, axis=1).to_numpy()
    rng = np.random.default_rng(0)
    rows, perm = [], []
    for k in np.unique(key[ok]):
        g = np.nonzero(ok & (key == k))[0]
        if g.size < 2:
            continue
        sh = rng.permutation(g)
        keep = sh != g                       # 只保留真的换掉了的
        rows.append(g[keep])
        perm.append(sh[keep])
    rows, perm = np.concatenate(rows), np.concatenate(perm)
    own = np.nanmean(pcc_axis(ctx.C[rows], y_pred[rows], cfg, axis=1))
    other = np.nanmean(pcc_axis(ctx.C[perm], y_pred[rows], cfg, axis=1))
    gap = own - other
    print(f"[C-free 守卫] 与自身对照相关 {own:.4f} / 与他人对照相关 {other:.4f}"
          f"  差 {gap:+.4f}")
    if gap > tol:
        raise AssertionError(
            f"预测疑似携带样本特异对照信息（差 {gap:+.4f} > {tol}）。"
            "C-free 模型不得在推断时读取真实对照。")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", choices=list(FEATURE_SETS), default="bio_tech")
    ap.add_argument("--lam", type=float, default=30.0)
    args = ap.parse_args()

    paths.ensure_dir(OUT_DIR)
    cfg = ScorerConfig()
    ctx = ev.build_context()
    val = ctx.rows(ev.VAL_SPLITS)
    # 绝对基线用**全部** train 行（含对照）——对照也是合法的绝对丰度监督
    train_all = (ctx.meta["split_final"] == "train").to_numpy()
    drugs = sorted(np.unique(ctx.meta[PERT_COL].astype(str).to_numpy()[train_all]))
    cols = FEATURE_SETS[args.features]

    L: list[str] = []
    a = L.append
    a("=" * 100)
    a("第 7 步 · C-free 可部署骨架（预测函数不接收真实对照）")
    a("=" * 100)
    a(f"特征集 {args.features}：{' / '.join(cols)} + log-time 三次多项式")
    a(f"拟合行：split_final=='train' 共 {int(train_all.sum())} 行（含对照）")
    a(f"岭参数 λ={args.lam}；未见水平整行为 0 → 自动回退总体先验")
    a("")
    a("❗与此前基线的根本区别：此前 B1–B4、低秩响应、α 收缩都是 ŷ = C + Δ̂，")
    a("  推断时需要该样本的真实对照。A 分支下测试集对照在被隔离的文件里，拿不到，")
    a("  所以那些数字是 oracle 诊断，不是可提交模型。本表才是可交卷的形态。")
    a("")

    cols_out = ["total", "abs_pcc", "abs_r2", "fc_pcc", "ctx_resid", "drug_resid", "both_time"]
    a(f"  {'模型':<34}" + "".join(f"{c:>11}" for c in cols_out))
    a("  " + "-" * (34 + 11 * len(cols_out)))

    results = {}
    for label, with_drug in [("C-free 仅上下文 b̂", False),
                             ("C-free b̂ + 药物项 Δ̂", True)]:
        Z = design(ctx.meta, cols, with_drug, drugs)
        mu, W = masked_ridge(Z, ctx.X, train_all, args.lam)
        y = (mu[None, :] + Z.astype(np.float64) @ W).astype(np.float32)
        y = np.nan_to_num(y, nan=float(np.nanmedian(mu)))
        f = ev.flatten(ev.evaluate(y, ctx, val, cfg))
        results[label] = (f, y)
        a(f"  {label:<34}" + "".join(f"{f[c]:>11.4f}" for c in cols_out))

    # 对照参考：全局均值谱（最朴素的 C-free 模型）
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        g = np.nanmean(ctx.X[train_all], axis=0, dtype=np.float64)
    g = np.where(np.isfinite(g), g, np.nanmedian(g)).astype(np.float32)
    y0 = np.tile(g, (ctx.n, 1))
    f0 = ev.flatten(ev.evaluate(y0, ctx, val, cfg))
    a(f"  {'（参照）B0 全局均值谱 · 也是 C-free':<34}"
      + "".join(f"{f0[c]:>11.4f}" for c in cols_out))
    a("")

    best_lab = max(results, key=lambda k: results[k][0]["total"])
    a(f"最好的 C-free 模型：{best_lab}  total = {results[best_lab][0]['total']:.4f}")
    a(f"  相对最朴素的全局均值谱（{f0['total']:.4f}）提升 "
      f"{results[best_lab][0]['total'] - f0['total']:+.4f}")
    a("")
    a("注意 fc_pcc 这一列：C-free 模型的 Δ_pred = b̂ − C 完整包含 −C，")
    a("共享参照的红利照样拿满，并不因为放弃读对照而损失。")
    a("")

    a("-" * 100)
    a("C-free 守卫检查（同上下文内置换对照）")
    a("-" * 100)
    a("守卫本身先做阳性对照：拿一个**已知偷用了 C** 的模型（ŷ = C）去测，")
    a("它必须被拦下。守卫抓不到已知泄漏，就没有资格说别的模型干净。")
    leaky = np.nan_to_num(ctx.C, nan=float(np.nanmedian(g)))
    a("  [阳性对照] ŷ = C（B1，定义上就是读了对照）")
    try:
        assert_c_free(leaky, ctx, cols)
        a("    ❌ 守卫失效：已知泄漏的模型竟然通过了，守卫不可信")
    except AssertionError as e:
        a(f"    ✅ 已知泄漏被拦下：{str(e).split('（')[1][:40]}…")
    for lab, (_, y) in results.items():
        a(f"  {lab}")
        try:
            assert_c_free(y, ctx, cols)
            a("    ✅ 通过")
        except AssertionError as e:
            a(f"    ❌ {e}")
    a("")
    a("-" * 100)
    a("逐划分（最好的 C-free 模型）")
    a("-" * 100)
    sp = ev.evaluate_by_split(results[best_lab][1], ctx, cfg)
    a("    " + "  ".join(f"{c:>14}" for c in ["split", "n", "abs_pcc", "abs_r2", "fc_pcc"]))
    for _, r in sp.iterrows():
        a("    " + "  ".join([f"{r['split']:>14}", f"{int(r['n']):>14d}",
                              f"{r.get('abs_pcc', np.nan):>14.4f}",
                              f"{r.get('abs_r2', np.nan):>14.4f}",
                              f"{r.get('fc_pcc', np.nan):>14.4f}"]))

    txt = "\n".join(L)
    print(txt)
    p = os.path.join(OUT_DIR, f"report_{args.features}.txt")
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(txt)
    print(f"\n[写出] {p}")


if __name__ == "__main__":
    main()
