"""第 3 步之三 · 批次与生物信号的混杂诊断（决定能不能做批次校正）。

docs/05 §2.4 的过度校正防线：**不能无条件删除所有与批次相关的方向**。
先量化混杂程度，再决定校正强度。

本脚本做五项：
  1. 生物变量 × 技术变量 的 Cramér's V（关联强度矩阵）
  2. 设计矩阵秩（生物 one-hot + 技术 one-hot 是否发生别名/共线）
  3. 仅在对照样本上的技术方差解释率（生物基本恒定，剩下的就是技术）
  4. 匹配 FC 之后，残差 Δ 是否仍依赖 instrument / plate / well
  5. LOSO / LOCO 与官方划分的可行性核查（每折是否还有足够对照与化合物）

运行：python diagnose_batch.py
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import loader, paths

BIO = ["Strains", "Medium", "Temperature", "pert_time", "perturbation_no_concentration"]
TECH = ["data_source", "instrument", "Yeast_cell_plate", "protein_well"]
OUT_DIR = os.path.join(paths.RESULTS, "step3_diagnostics")
N_PC = 10


def cramers_v(a: np.ndarray, b: np.ndarray) -> float:
    """列联表关联强度，0=独立，1=完全决定。含 bias 校正。"""
    tab = pd.crosstab(pd.Series(a), pd.Series(b)).to_numpy(dtype=float)
    n = tab.sum()
    if n == 0:
        return np.nan
    exp = np.outer(tab.sum(1), tab.sum(0)) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum(np.where(exp > 0, (tab - exp) ** 2 / exp, 0.0))
    r, k = tab.shape
    phi2 = chi2 / n
    phi2c = max(0.0, phi2 - (r - 1) * (k - 1) / (n - 1))
    rc = r - (r - 1) ** 2 / (n - 1)
    kc = k - (k - 1) ** 2 / (n - 1)
    d = min(rc - 1, kc - 1)
    return float(np.sqrt(phi2c / d)) if d > 0 else np.nan


def onehot(meta: pd.DataFrame, cols) -> np.ndarray:
    blocks = []
    for c in cols:
        codes, uniq = pd.factorize(meta[c].astype(str))
        M = np.zeros((len(meta), len(uniq)), dtype=np.float32)
        M[np.arange(len(meta)), codes] = 1.0
        blocks.append(M[:, 1:])          # 丢参照水平
    return np.hstack([np.ones((len(meta), 1), dtype=np.float32)] + blocks)


def eta_squared(v: np.ndarray, g: np.ndarray) -> float:
    ok = np.isfinite(v)
    v, g = v[ok], g[ok]
    if v.size < 2:
        return np.nan
    ss_tot = float(((v - v.mean()) ** 2).sum())
    if ss_tot < 1e-12:
        return np.nan
    grand = v.mean()
    ss_b = sum(int((g == k).sum()) * (v[g == k].mean() - grand) ** 2 for k in np.unique(g))
    return float(ss_b / ss_tot)


def top_pcs(M: np.ndarray, k: int = N_PC) -> np.ndarray:
    """对含 NaN 的矩阵取主成分：缺失按列均值居中后置 0（仅用于诊断，不进模型）。"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu = np.nanmean(M, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    A = np.where(np.isfinite(M), M - mu, 0.0).astype(np.float32)
    keep = A.std(axis=0) > 1e-6
    A = A[:, keep]
    U, S, _ = np.linalg.svd(A, full_matrices=False)
    return (U[:, :k] * S[:k]), (S ** 2 / (S ** 2).sum())[:k]


def main() -> None:
    paths.ensure_dir(OUT_DIR)
    meta = loader.load_metadata("train_val")
    z = np.load(os.path.join(paths.RESULTS, "step2_control_match",
                             "delta_true_train_val_median.npz"), allow_pickle=False)
    D, is_ctrl, is_qc = z["delta"], z["is_control"], z["is_qc"]
    sid, proteins, Xr = loader.load_proteome_log2(verbose=False)
    X = loader.align_proteome_to_metadata(meta, sid, Xr)
    del Xr

    L: list[str] = []
    a = L.append
    a("=" * 88)
    a("第 3 步之三 · 批次与生物信号混杂诊断")
    a("=" * 88)
    a("")

    # ---------------------------------------------------------------- 1
    a("-" * 88)
    a("1. Cramér's V：生物变量（行）× 技术变量（列）")
    a("-" * 88)
    a(f"  {'':<32}" + "".join(f"{c[:14]:>16}" for c in TECH))
    worst = []
    for b in BIO:
        line = f"  {b:<32}"
        for t in TECH:
            v = cramers_v(meta[b].astype(str).to_numpy(), meta[t].astype(str).to_numpy())
            line += f"{v:>16.3f}"
            worst.append((v, b, t))
        a(line)
    a("")
    worst.sort(reverse=True)
    a("  混杂最强的三对：")
    for v, b, t in worst[:3]:
        a(f"    {b} × {t}  V = {v:.3f}")
    a("  判读：V > 0.7 表示该生物变量几乎能被技术变量决定，此时任何按技术变量的")
    a("        强校正都会连生物信号一起删掉（docs/05 §2.4 过度校正防线）。")
    a("")

    # ---------------------------------------------------------------- 2
    a("-" * 88)
    a("2. 设计矩阵秩（生物 + 技术 one-hot 是否别名）")
    a("-" * 88)
    for label, cols in [("仅生物", BIO), ("仅技术", TECH), ("生物+技术", BIO + TECH)]:
        Zm = onehot(meta, cols)
        r = int(np.linalg.matrix_rank(Zm))
        a(f"  {label:<12} 列数 {Zm.shape[1]:5d}   秩 {r:5d}   亏秩 {Zm.shape[1]-r:4d}")
    a("  判读：亏秩 = 0 说明生物效应与技术效应在设计上可分离（能同时进同一个线性模型）；")
    a("        亏秩 > 0 说明存在完全别名，必须靠收缩/先验而非最小二乘来定")
    a("")

    # ---------------------------------------------------------------- 3
    a("-" * 88)
    a("3. 仅对照样本上的技术方差解释率（生物近乎恒定，剩下的就是技术）")
    a("-" * 88)
    ctrl = is_ctrl
    Pc, varc = top_pcs(X[ctrl], N_PC)
    a(f"  对照样本 {int(ctrl.sum())}   前 5 PC 解释方差 "
      + "  ".join(f"{v:.1%}" for v in varc[:5]))
    a("")
    a(f"  {'变量':<32}" + "".join(f"{'PC'+str(i+1):>8}" for i in range(5)))
    for c in TECH + ["Strains", "Medium", "Temperature", "pert_time"]:
        g = meta[c].astype(str).to_numpy()[ctrl]
        tag = " [技术]" if c in TECH else " [生物]"
        a(f"  {c + tag:<32}" + "".join(f"{eta_squared(Pc[:, i], g):>8.3f}" for i in range(5)))
    a("")
    a("  这一节回答：在没有药物的情况下，蛋白质组的主要变异到底来自哪里。")
    a("")

    # ---------------------------------------------------------------- 4
    a("-" * 88)
    a("4. 匹配 FC 之后，残差 Δ 是否仍依赖技术变量")
    a("-" * 88)
    treat = (~is_ctrl) & (~is_qc) & np.isfinite(D).any(axis=1)
    Pd, vard = top_pcs(D[treat], N_PC)
    a(f"  处理样本 {int(treat.sum())}   Δ 空间前 5 PC 解释方差 "
      + "  ".join(f"{v:.1%}" for v in vard[:5]))
    a("")
    a(f"  {'变量':<32}" + "".join(f"{'PC'+str(i+1):>8}" for i in range(5)))
    for c in TECH + BIO:
        g = meta[c].astype(str).to_numpy()[treat]
        tag = " [技术]" if c in TECH else " [生物]"
        a(f"  {c + tag:<32}" + "".join(f"{eta_squared(Pd[:, i], g):>8.3f}" for i in range(5)))
    a("")
    a("  判读：匹配对照已消去在处理与对照间共享的加性技术偏移。若这里技术变量的 η²")
    a("        仍然很高，说明存在蛋白特异/非线性的批次效应，匹配 FC 消不掉；")
    a("        若已经很低，则不必再做额外的批次校正（做了只会伤生物信号）。")
    a("")
    a("  同表中生物变量的 η² 就是『Δ 里到底有多少可被条件解释的结构』，")
    a("  它给基线阶梯 B2（上下文均值 Δ）能达到的水平定了个预期。")
    a("")

    # ---------------------------------------------------------------- 5
    a("-" * 88)
    a("5. 划分可行性核查（官方四类 val + LOSO + LOCO）")
    a("-" * 88)
    pert = meta["perturbation_no_concentration"].astype(str).to_numpy()
    a("  官方划分：")
    for k, g in meta.groupby("split_final"):
        nt = int((~np.isin(g["perturbation_no_concentration"], ["Water", "DMSO", "Quality Control"])).sum())
        a(f"    {k:<18} 样本 {len(g):5d}  处理 {nt:5d}  菌株 {g['Strains'].nunique()}"
          f"  化合物 {g['perturbation_no_concentration'].nunique()}")
    a("")
    a("  LOSO（留一菌株）：每折训练侧剩余化合物数与对照数")
    for s in sorted(meta["Strains"].unique()):
        tr = meta["Strains"] != s
        te = ~tr
        a(f"    留出 {s:<8} 训练 {int(tr.sum()):5d} 样本 / {meta.loc[tr,'perturbation_no_concentration'].nunique()} 化合物"
          f"    留出侧 {int(te.sum()):5d} 样本 / 对照 "
          f"{int(meta.loc[te,'perturbation_no_concentration'].isin(['Water','DMSO']).sum())}")
    a("")
    a("  LOCO（留一化合物）：可用化合物数与每化合物样本数")
    cmp_counts = pd.Series(pert).value_counts()
    cmp_counts = cmp_counts[~cmp_counts.index.isin(["Water", "DMSO", "Quality Control"])]
    a(f"    可留出化合物 {len(cmp_counts)}   样本数 中位 {cmp_counts.median():.0f}"
      f"  最少 {cmp_counts.min()} ({cmp_counts.idxmin()})  最多 {cmp_counts.max()} ({cmp_counts.idxmax()})")
    a(f"    样本数 < 60 的化合物 {int((cmp_counts < 60).sum())} 个："
      f" {', '.join(cmp_counts[cmp_counts < 60].index[:8])}")
    a("")
    a("  ❗ 绝不使用随机行划分：同一化合物在多个批次有重复样本，随机划分会让")
    a("     同化合物同上下文的样本同时出现在训练与验证侧，分数会虚高。")

    txt = "\n".join(L)
    print(txt)
    fp = os.path.join(OUT_DIR, "batch_confounding.txt")
    with open(fp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(txt)
    print(f"\n[写出] {fp}")


if __name__ == "__main__":
    main()
