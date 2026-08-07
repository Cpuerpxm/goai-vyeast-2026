"""第 3 步之二 · 缺失机制诊断（27% 缺失是 MNAR 还是 MAR？）。

docs/05 §2.3 列了六项验证，本脚本做其中可做的五项：
  1. 每蛋白缺失率 vs 其观测中位 log2 丰度        → 丰度依赖的左删失（MNAR）
  2. 每个仪器/来源/板/孔的检出深度                → 技术依赖（MAR）
  3. 缺失指示回归 P(missing) ~ 丰度代理 + 技术 + 生物
  4. 同条件重复样本的缺失不一致率                 → 随机成分占比
  5. 对缺失矩阵本身做 PCA，看是否按技术变量聚集

第 6 项（比较 train / test 的每蛋白缺失率与共缺失结构）**被 CLAUDE.md R2 阻断**：
它需要读 proteome_raw_test.csv 做缺失统计，而该文件在分支 A 下物理隔离。
本脚本明确记录这一点，不偷偷绕过。

运行：python diagnose_missing.py
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import loader, paths

BIO = ["Strains", "Medium", "Temperature", "pert_time", "perturbation_no_concentration"]
TECH = ["data_source", "instrument", "Yeast_cell_plate", "protein_well"]
OUT_DIR = os.path.join(paths.RESULTS, "step3_diagnostics")
RNG = np.random.default_rng(20260805)


def eta_squared(values: np.ndarray, groups: np.ndarray) -> float:
    """单因素方差解释率 η²（组间平方和 / 总平方和）。"""
    ok = np.isfinite(values)
    v, g = values[ok], groups[ok]
    if v.size < 2:
        return np.nan
    grand = v.mean()
    ss_tot = float(((v - grand) ** 2).sum())
    if ss_tot < 1e-12:
        return np.nan
    ss_b = 0.0
    for k in np.unique(g):
        m = g == k
        ss_b += m.sum() * (v[m].mean() - grand) ** 2
    return float(ss_b / ss_tot)


def main() -> None:
    paths.ensure_dir(OUT_DIR)
    meta = loader.load_metadata("train_val")
    sid, proteins, Xr = loader.load_proteome_log2(verbose=False)
    X = loader.align_proteome_to_metadata(meta, sid, Xr)
    del Xr
    miss = np.isnan(X)
    n, p = X.shape

    L: list[str] = []
    a = L.append
    a("=" * 84)
    a("第 3 步之二 · 缺失机制诊断")
    a("=" * 84)
    a(f"矩阵 {n} 样本 × {p} 蛋白   总缺失率 {miss.mean():.4%}")
    a("")

    # ---------------------------------------------------------------- 1
    a("-" * 84)
    a("1. 丰度依赖（MNAR 的判据）：每蛋白缺失率 vs 其观测中位 log2 丰度")
    a("-" * 84)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        prot_med = np.nanmedian(X, axis=0)
    prot_miss = miss.mean(axis=0)
    ok = np.isfinite(prot_med)
    rho, pv = stats.spearmanr(prot_med[ok], prot_miss[ok])
    a(f"有观测的蛋白 {int(ok.sum())} / {p}（{p - int(ok.sum())} 个全缺失）")
    a(f"Spearman(中位丰度, 缺失率) = {rho:+.3f}   p = {pv:.2e}")
    a("")
    a("按中位丰度十分位分箱：")
    a(f"  {'十分位':<8}{'蛋白数':>7}{'中位log2丰度':>14}{'平均缺失率':>12}")
    dec = pd.qcut(prot_med[ok], 10, labels=False, duplicates="drop")
    for d in np.unique(dec):
        m = dec == d
        a(f"  {int(d)+1:<8}{int(m.sum()):>7d}{np.median(prot_med[ok][m]):>14.2f}"
          f"{prot_miss[ok][m].mean():>12.2%}")
    a("")
    a("判读：负相关且低丰度箱缺失率显著更高 → 存在检出限左删失（MNAR）。")
    a("若各箱缺失率接近 → 缺失与丰度无关，更可能是技术/随机（MAR/MCAR）。")
    a("")

    # ---------------------------------------------------------------- 2
    a("-" * 84)
    a("2. 技术依赖：每样本检出深度（非缺失蛋白数）按技术/生物变量的方差解释率 η²")
    a("-" * 84)
    depth = (~miss).sum(axis=1).astype(float)
    a(f"检出深度  min {depth.min():.0f}  p25 {np.percentile(depth,25):.0f}  "
      f"中位 {np.median(depth):.0f}  p75 {np.percentile(depth,75):.0f}  max {depth.max():.0f}")
    a("")
    a(f"  {'变量':<32}{'水平数':>7}{'η² (检出深度)':>16}")
    eta_rows = []
    for c in TECH + BIO:
        g = meta[c].astype(str).to_numpy()
        e = eta_squared(depth, g)
        eta_rows.append((c, len(np.unique(g)), e))
        tag = "技术" if c in TECH else "生物"
        a(f"  {c + ' [' + tag + ']':<32}{len(np.unique(g)):>7d}{e:>16.3f}")
    a("")
    a("判读：技术变量 η² 远高于生物变量 → 缺失主要由测量过程决定，")
    a("      模型必须显式吸收测量上下文，否则会把技术缺失当成生物信号。")
    a("")

    # ---------------------------------------------------------------- 3
    a("-" * 84)
    a("3. 缺失指示回归 P(missing) ~ 蛋白丰度代理 + 仪器 + 来源 + 温度 + 菌株")
    a("-" * 84)
    n_cell = 400_000
    ri = RNG.integers(0, n, n_cell)
    rj = RNG.integers(0, p, n_cell)
    y = miss[ri, rj].astype(int)
    prot_med_f = np.where(np.isfinite(prot_med), prot_med, np.nanmin(prot_med))
    feat = [((prot_med_f[rj] - np.nanmean(prot_med_f)) / np.nanstd(prot_med_f)).reshape(-1, 1)]
    names = ["蛋白中位丰度(标准化)"]
    for c in ["instrument", "data_source", "Temperature", "Strains"]:
        codes = pd.factorize(meta[c].astype(str))[0][ri]
        oh = np.zeros((n_cell, codes.max() + 1), dtype=np.float32)
        oh[np.arange(n_cell), codes] = 1.0
        feat.append(oh[:, 1:])          # 丢一列做参照
        names += [f"{c}={v}" for v in pd.factorize(meta[c].astype(str))[1][1:]]
    Z = np.hstack(feat).astype(np.float32)

    def _auc_dev(cols):
        lr = LogisticRegression(max_iter=400, solver="lbfgs", C=1.0)
        lr.fit(Z[:, cols], y)
        pr = lr.predict_proba(Z[:, cols])[:, 1]
        pr = np.clip(pr, 1e-9, 1 - 1e-9)
        ll = float(np.mean(y * np.log(pr) + (1 - y) * np.log(1 - pr)))
        return ll, lr

    base = y.mean()
    ll0 = float(base * np.log(base) + (1 - base) * np.log(1 - base))
    ll_ab, lr_ab = _auc_dev([0])
    ll_tech, _ = _auc_dev(list(range(1, Z.shape[1])))
    ll_all, lr_all = _auc_dev(list(range(Z.shape[1])))
    a(f"抽样单元格 {n_cell:,}   基础缺失率 {base:.4f}")
    a(f"  仅截距          伪R² = 0.000")
    a(f"  仅丰度代理      伪R² = {1 - ll_ab/ll0:.4f}")
    a(f"  仅技术+生物因子 伪R² = {1 - ll_tech/ll0:.4f}")
    a(f"  全部            伪R² = {1 - ll_all/ll0:.4f}")
    a(f"  丰度系数 β = {lr_ab.coef_[0][0]:+.3f}  "
      f"（负 = 丰度越高越不易缺失，支持左删失）")
    a("")
    top = np.argsort(-np.abs(lr_all.coef_[0]))[:8]
    a("  全模型中绝对值最大的 8 个系数：")
    for t in top:
        a(f"    {names[t]:<34} β = {lr_all.coef_[0][t]:+.3f}")
    a("")

    # ---------------------------------------------------------------- 4
    a("-" * 84)
    a("4. 同条件重复样本的缺失不一致率（随机成分有多大）")
    a("-" * 84)
    key = meta[BIO].astype(str).agg("\x1f".join, axis=1).to_numpy()
    incons, jac, npair = [], [], 0
    for k in np.unique(key):
        rows = np.nonzero(key == k)[0]
        if rows.size < 2:
            continue
        for i in range(min(rows.size, 4)):
            for j in range(i + 1, min(rows.size, 4)):
                A, B = miss[rows[i]], miss[rows[j]]
                incons.append(float((A ^ B).mean()))
                u = float((A | B).sum())
                jac.append(float((A & B).sum() / u) if u else np.nan)
                npair += 1
    a(f"同生物条件的样本对 {npair}")
    a(f"  缺失状态不一致率  中位 {np.median(incons):.2%}   均值 {np.mean(incons):.2%}")
    a(f"  共缺失 Jaccard    中位 {np.nanmedian(jac):.3f}")
    a("  判读：不一致率 ≈ 0 → 缺失几乎完全由蛋白本身决定（结构性，可当固定掩码）；")
    a("        不一致率高 → 每次测量随机丢，掩码必须逐样本处理，且填补风险大。")
    a("")

    # ---------------------------------------------------------------- 5
    a("-" * 84)
    a("5. 缺失矩阵 PCA：主成分是否对齐技术变量")
    a("-" * 84)
    M = miss.astype(np.float32)
    M = M - M.mean(axis=0, keepdims=True)
    keep = M.std(axis=0) > 1e-6
    M = M[:, keep]
    U, S, _ = np.linalg.svd(M, full_matrices=False)
    var = (S ** 2) / float((S ** 2).sum())
    a(f"参与列 {int(keep.sum())}（去掉零方差列）")
    a("  前 5 个主成分解释方差：" + "  ".join(f"PC{i+1} {var[i]:.1%}" for i in range(5)))
    a("")
    a(f"  {'变量':<32}" + "".join(f"{'PC'+str(i+1):>9}" for i in range(5)) + "   (η²)")
    for c in TECH + BIO:
        g = meta[c].astype(str).to_numpy()
        line = f"  {c + (' [技术]' if c in TECH else ' [生物]'):<32}"
        for i in range(5):
            line += f"{eta_squared(U[:, i] * S[i], g):>9.3f}"
        a(line)
    a("")
    a("  判读：若前几个 PC 的 η² 在技术变量上接近 1、在生物变量上接近 0，")
    a("        则缺失结构基本是批次指纹，与生物学无关。")
    a("")

    # ---------------------------------------------------------------- 6
    a("-" * 84)
    a("6. train vs test 缺失结构比较 —— 【未执行 · 被规则阻断】")
    a("-" * 84)
    a("docs/05 §2.3 第 6 项要求比较 train / test 的每蛋白缺失率与共缺失结构。")
    a("该操作需读 proteome_raw_test.csv 做缺失统计，而 CLAUDE.md R2 明令该文件")
    a("在组委会书面答复到达前不得进入包括『缺失统计』在内的任何环节。")
    a("→ 本项标记为阻塞，不绕过。若日后解除隔离，在此补做。")
    a("")
    a("影响：无法预先知道 test 的缺失模式是否与 train 同分布。缓解办法是让模型对")
    a("      掩码模式不敏感（掩码损失 + 不把缺失当 0），而不是去拟合 train 的掩码。")

    txt = "\n".join(L)
    print(txt)
    fp = os.path.join(OUT_DIR, "missing_mechanism.txt")
    with open(fp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(txt)
    print(f"\n[写出] {fp}")


if __name__ == "__main__":
    main()
