"""第 3 步之一 · 复制样本可靠性 → 六项指标的经验上限。

问题：模型能拿多少分，上限不是 1.0，而是由测量噪声决定的。

做法：找「同化合物 + 同生物上下文（菌株/培养基/温度/时间）但不同技术批次」的
复制样本对，计算两条 Δ_true 向量的相关。

若每次观测 = 真实信号 + 独立噪声，则

    corr(rep1, rep2) = Var(signal) / (Var(signal) + Var(noise)) = ρ   （可靠性）
    完美模型对单次观测的相关上限 = sqrt(ρ)

分别对四个空间做：
    绝对 log2 丰度      → 指标 1 (20%)
    Δ_true             → 指标 2 (25%)
    Δ − μ_ctx          → 指标 3 (20%)
    Δ − μ_drug         → 指标 4 (20%)

μ 一律只用 split_final == 'train' 计算（与官方「参照统计须仅用训练数据冻结」一致）。

运行：python noise_ceiling.py
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import loader, paths
from data.control_match import QC_LABEL

BIO_CTX = ["Strains", "Medium", "Temperature", "pert_time"]
PERT_COL = "perturbation_no_concentration"
MAX_PAIRS = 40000
MIN_VALID = 30
RNG = np.random.default_rng(20260805)

OUT_DIR = os.path.join(paths.RESULTS, "step3_diagnostics")


# --------------------------------------------------------------- 相关工具


def pcc_pairwise(a: np.ndarray, b: np.ndarray, min_valid: int = MIN_VALID) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < min_valid:
        return np.nan
    x, y = a[m].astype(np.float64), b[m].astype(np.float64)
    sx, sy = x.std(), y.std()
    if sx < 1e-12 or sy < 1e-12:
        return np.nan
    return float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy))


def replicate_pairs(meta: pd.DataFrame, keep: np.ndarray) -> list:
    """同扰动 + 同生物上下文、但不同板/来源的样本对。"""
    idx = np.nonzero(keep)[0]
    sub = meta.iloc[idx]
    key = sub[[PERT_COL] + BIO_CTX].astype(str).agg("\x1f".join, axis=1).to_numpy()
    pairs = []
    for k in np.unique(key):
        rows = idx[key == k]
        if rows.size < 2:
            continue
        plate = meta["Yeast_cell_plate"].to_numpy()[rows]
        src = meta["data_source"].to_numpy()[rows]
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                # 要求技术批次不同，否则测的是同板重复而非可泛化信号
                if plate[i] != plate[j] or src[i] != src[j]:
                    pairs.append((rows[i], rows[j]))
    if len(pairs) > MAX_PAIRS:
        sel = RNG.choice(len(pairs), MAX_PAIRS, replace=False)
        pairs = [pairs[i] for i in sel]
    return pairs


def reliability(M: np.ndarray, pairs: list) -> dict:
    """样本轴：逐对相关；蛋白轴：把配对拆成两个矩阵后逐蛋白相关。"""
    rs = np.asarray([pcc_pairwise(M[i], M[j]) for i, j in pairs], dtype=float)
    rs = rs[np.isfinite(rs)]
    A = M[[i for i, _ in pairs]]
    B = M[[j for _, j in pairs]]
    rp = np.asarray([pcc_pairwise(A[:, k], B[:, k]) for k in range(M.shape[1])], dtype=float)
    rp = rp[np.isfinite(rp)]

    def _c(r):
        return float(np.sqrt(max(r, 0.0)))

    return {
        "n_pairs": len(pairs),
        "rho_sample": float(np.mean(rs)) if rs.size else np.nan,
        "rho_sample_median": float(np.median(rs)) if rs.size else np.nan,
        "ceil_sample": _c(np.mean(rs)) if rs.size else np.nan,
        "n_proteins": int(rp.size),
        "rho_protein": float(np.mean(rp)) if rp.size else np.nan,
        "ceil_protein": _c(np.mean(rp)) if rp.size else np.nan,
    }


# --------------------------------------------------------------- 参照均值


def group_mean(M: np.ndarray, keys: np.ndarray, fit_rows: np.ndarray) -> np.ndarray:
    """按 keys 分组，仅用 fit_rows 的样本算逐蛋白均值，广播回每一行。"""
    out = np.full(M.shape, np.nan, dtype=np.float32)
    fit_mask = np.zeros(M.shape[0], dtype=bool)
    fit_mask[fit_rows] = True
    for k in np.unique(keys):
        rows_fit = np.nonzero((keys == k) & fit_mask)[0]
        if rows_fit.size == 0:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN 列
            mu = np.nanmean(M[rows_fit], axis=0, dtype=np.float64).astype(np.float32)
        out[keys == k] = mu
    return out


# --------------------------------------------------------------- 主流程


def main() -> None:
    paths.ensure_dir(OUT_DIR)
    meta = loader.load_metadata("train_val")
    sid, proteins, Xr = loader.load_proteome_log2(verbose=False)
    X = loader.align_proteome_to_metadata(meta, sid, Xr)
    del Xr

    z = np.load(os.path.join(paths.RESULTS, "step2_control_match",
                             "delta_true_train_val_median.npz"), allow_pickle=False)
    D = z["delta"]
    is_ctrl, is_qc = z["is_control"], z["is_qc"]
    assert list(z["sample_ids"]) == list(meta["sample_ID"].astype(str)), "样本顺序不一致"

    treat = (~is_ctrl) & (~is_qc) & np.isfinite(D).any(axis=1)
    pairs = replicate_pairs(meta, treat)

    pert = meta[PERT_COL].astype(str).to_numpy()
    ctx_key = meta[BIO_CTX].astype(str).agg("\x1f".join, axis=1).to_numpy()
    train_rows = np.nonzero((meta["split_final"] == "train").to_numpy() & treat)[0]

    mu_ctx = group_mean(D, ctx_key, train_rows)
    mu_drug = group_mean(D, pert, train_rows)

    spaces = [
        ("指标1 绝对 log2 丰度", X, 0.20),
        ("指标2 Δ_true (匹配FC)", D, 0.25),
        ("指标3 Δ − μ_ctx", D - mu_ctx, 0.20),
        ("指标4 Δ − μ_drug", D - mu_drug, 0.20),
    ]

    L = []
    a = L.append
    a("=" * 86)
    a("第 3 步之一 · 复制样本可靠性 → 各指标的经验上限")
    a("=" * 86)
    a(f"复制对定义：同化合物 + 同 {'/'.join(BIO_CTX)}，且板号或数据来源不同")
    a(f"可用复制对 {len(pairs)}（上限抽样 {MAX_PAIRS}，种子 20260805）")
    a(f"μ_ctx / μ_drug 仅由 split_final=='train' 的 {len(train_rows)} 个处理样本计算")
    a("")
    a(f"{'空间':<24}{'权重':>6}{'对数':>8}{'ρ_样本':>9}{'上限_样本':>11}{'ρ_蛋白':>9}{'上限_蛋白':>11}")
    a("-" * 86)
    rows = []
    for name, M, w in spaces:
        r = reliability(M, pairs)
        rows.append((name, w, r))
        a(f"{name:<24}{w:>6.0%}{r['n_pairs']:>8d}{r['rho_sample']:>9.3f}"
          f"{r['ceil_sample']:>11.3f}{r['rho_protein']:>9.3f}{r['ceil_protein']:>11.3f}")
    a("")
    a("ρ = 两次独立观测的相关（可靠性）；上限 = sqrt(ρ) = 完美模型对单次观测的相关上限。")
    a("")

    a("-" * 86)
    a("加权总分的粗略天花板（假设各项都打满各自上限）")
    a("-" * 86)
    tot_w, tot = 0.0, 0.0
    for name, w, r in rows:
        c = np.nanmean([r["ceil_sample"], r["ceil_protein"]])
        tot += w * c
        tot_w += w
        a(f"  {name:<24} 上限≈{c:.3f} × {w:.0%}")
    a(f"  这四项合计权重 {tot_w:.0%}，其加权上限 ≈ {tot/tot_w:.3f}")
    a("  （余下 10% 双重未知/时间外推 + 5% DEP 未计入本表）")
    a("")

    a("-" * 86)
    a("复制对的分层：同来源换板 vs 跨数据来源（判断是否存在批次×药物交互）")
    a("-" * 86)
    src = meta["data_source"].to_numpy()
    same_src = [(i, j) for i, j in pairs if src[i] == src[j]]
    diff_src = [(i, j) for i, j in pairs if src[i] != src[j]]
    for lab, pp in [("同来源·不同板", same_src), ("跨数据来源", diff_src)]:
        if len(pp) < 50:
            a(f"  {lab:<14} 对数 {len(pp)}（不足 50，跳过）")
            continue
        r = reliability(D, pp)
        a(f"  {lab:<14} 对数 {r['n_pairs']:5d}   ρ_样本 {r['rho_sample']:.3f}"
          f"   上限 {r['ceil_sample']:.3f}")
    a("  两者接近 → 噪声以板内随机测量误差为主；同来源明显更高 → 存在来源级系统差异，")
    a("  对未见批次的泛化会比表中的上限更差。")
    a("")

    a("-" * 86)
    a("另一口径：全矩阵池化的信号方差占比（与上表不是同一个量，勿直接对比）")
    a("-" * 86)
    ctrl_ok = is_ctrl & np.isfinite(D).any(axis=1)
    dt = D[treat]
    dc = D[ctrl_ok]
    vt = float(np.nanvar(dt[np.isfinite(dt)]))
    vc = float(np.nanvar(dc[np.isfinite(dc)]))
    frac = max(vt - vc, 0.0) / vt
    a(f"  处理 Δ 方差 {vt:.4f}   对照-对照噪声方差 {vc:.4f}")
    a(f"  池化信号方差占比 ≈ {frac:.1%}  → 全矩阵单一 PCC 的上限 ≈ {np.sqrt(frac):.3f}")
    a("  差异来源：本口径把『药物之间的差异』也算进信号，而上表的 ρ_样本 只问")
    a("  『同一药物同一上下文重测一次，单条 5243 维向量能重现多少』。官方按样本轴/")
    a("  蛋白轴聚合，故上表才是对齐评分口径的那个数；本行仅作量级旁证，偏乐观。")
    a("  另注：对照走留一法、可聚合的对照数比处理侧少 1，噪声方差略被高估。")
    a("")

    a("-" * 86)
    a("对下游的直接含义")
    a("-" * 86)
    a("1. 排行榜分数不会接近 1.0；判断改动是否有效，要对着上限而不是对着 1.0 看。")
    a("2. 若某指标上限本身很低，那 20-25% 的权重上大家差距天然被压缩，")
    a("   把力气放在上限高的指标上边际收益更大。")
    a("3. 单样本 Δ 噪声大 → 训练目标应尽量用『同条件多样本聚合后的 Δ』做监督，")
    a("   或在损失里按每个样本的对照数/检出深度加权。")

    txt = "\n".join(L)
    print(txt)
    p = os.path.join(OUT_DIR, "noise_ceiling.txt")
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(txt)
    print(f"\n[写出] {p}")


if __name__ == "__main__":
    main()
