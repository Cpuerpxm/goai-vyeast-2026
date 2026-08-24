"""严格 nested LOCO：化合物表示到底有没有可验证的预测增益。

架构是 C-free 的：

    y = b(metadata)  +  Delta(compound, context)

Delta 拟合在训练残差 (X - b) 上，推断时不接触任何对照。

方法学关卡（Pro R2 E5 点名的五条，全部保留）：

1. **外层按整化合物留出**（不是随机行、也不是官方 split）
2. **化合物等权**：样本最多的化合物有 699 行、最少的 30 行，按行拟合会让前者
   的权重是后者的 23 倍。这里每个化合物总权重相等
3. **预处理只在外层训练折内拟合**：指纹 bit 过滤、描述符标准化都不许看留出化合物
4. **超参在内层选**：lambda 由外层训练折内部的二次留出决定
5. **shuffled-label 对照**：把化合物标签打乱后重跑，真增益必须显著超过它

❗2026-08-24 复赛整改，这个脚本被改了两处，旧结果全部作废：

- **L1-2（科学证据违规）**：旧版 `compounds` 取自**全 train_val** 的非对照化合物，
  外层只按 `tr = ~held` 划分。于是只要某个 val 化合物当折没被留出，它就进了训练。
  现在整个 LOCO 宇宙收进 `split_final == 'train'`，合法训练化合物只有 **37 个**。
  因此旧的三个数不能再对外使用：Morgan 增益 -0.0004 / 低于全部 5 次随机标签对照 /
  神谕近邻 +0.0237。
- **L1-1（训练边界违规）**：旧版 `Z = design(ctx.meta, ...)` 在全表建 one-hot 词表、
  在全表算 log-time 均值方差。现在每折的设计矩阵**只由该折的训练行冻结**
  （`models.design.freeze`），未见水平整块归零。

外层给的是 train 折内部的交叉验证证据；最后再在官方 `val_chem_only` 上**一次性**
评一次作确认——那 6 个化合物只看一眼，不参与任何选参。

运行：
    python loco_response.py                       # 默认 8 折、含 shuffled 对照
    python loco_response.py --folds 4 --n-perm 0  # 快速版
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import paths, provenance
from data import split_guard as sg
from models.design import FEATURE_SETS, PERT_COL, encode, freeze
from models.lowrank import masked_pca
from scorer import evaluate as ev
from scorer.config import ScorerConfig
from scorer.metrics import pcc_axis, r2_axis

NON_COMPOUND = ["Water", "DMSO", "Quality Control"]
OUT_DIR = os.path.join(paths.RESULTS, "step9_loco")
SMILES_CSV = os.path.join(paths.DATA_EXTERNAL, "compound_smiles.csv")
CAT_COLS = FEATURE_SETS["bio_tech"]


# ------------------------------------------------------------------ 化学表示


def ecfp_table(compounds, train_compounds, seed=0, shuffle=False):
    """Morgan 指纹 + RDKit 描述符。

    ❗bit 过滤与描述符标准化**只用 train_compounds 拟合**（Pro E5 第 3 条）。
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")

    tab = pd.read_csv(SMILES_CSV)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    dfn = [Descriptors.MolWt, Descriptors.MolLogP, Descriptors.TPSA,
           Descriptors.NumHDonors, Descriptors.NumHAcceptors,
           Descriptors.NumRotatableBonds, Descriptors.RingCount,
           Descriptors.FractionCSP3]
    fp, ds = {}, {}
    for _, r in tab.iterrows():
        s = str(r["smiles"]).strip()
        if not s or s.lower() == "nan":
            continue
        m = Chem.MolFromSmiles(s)
        if m is None or m.GetNumAtoms() == 0:
            continue
        fp[str(r["compound"])] = np.asarray(gen.GetFingerprint(m), dtype=np.float32)
        ds[str(r["compound"])] = np.asarray([f(m) for f in dfn], dtype=np.float32)

    have = [c for c in compounds if c in fp]
    tr = [c for c in train_compounds if c in fp]
    if len(tr) < 5:
        return None
    B = np.vstack([fp[c] for c in tr])
    keep = (B.sum(0) >= 2) & (B.sum(0) <= len(tr) - 2)     # 只看训练折
    D = np.vstack([ds[c] for c in tr])
    mu, sd = D.mean(0), D.std(0) + 1e-8                    # 只看训练折
    out = {c: np.concatenate([fp[c][keep], (ds[c] - mu) / sd]) for c in have}
    if shuffle:
        rng = np.random.default_rng(seed)
        vals = [out[c] for c in have]
        perm = rng.permutation(len(have))
        out = {c: vals[perm[i]] for i, c in enumerate(have)}
    return out


def feature_matrix(pert: np.ndarray, tab: dict, n: int) -> np.ndarray:
    dim = len(next(iter(tab.values())))
    P = np.zeros((n, dim), dtype=np.float32)
    for i, c in enumerate(pert):
        if c in tab:
            P[i] = tab[c]
    return np.hstack([np.ones((n, 1), np.float32), P]).astype(np.float32)


# ------------------------------------------------------------------ 拟合


def fit_baseline(ctx, meta_all, fit_rows, k0, lam):
    """C-free 绝对骨架 b：设计矩阵按 fit_rows 冻结，掩码 PCA + metadata 岭回归。

    返回对**全部行**的 b 预测（留出行只用到自己的 metadata，蛋白值不参与）。
    """
    fit_rows = sg.assert_train_only(meta_all, fit_rows, what="LOCO 外层折的 b 拟合行")
    spec = freeze(meta_all, fit_rows, CAT_COLS, with_drug=False)
    Z = encode(meta_all, spec)
    Xtr = ctx.X[fit_rows]
    mu, U, Ztr = masked_pca(Xtr, k0, np.isfinite(Xtr), n_iter=12, center=True, seed=0)
    Zd = Z[fit_rows].astype(np.float64)
    W = np.linalg.solve(Zd.T @ Zd + lam * np.eye(Z.shape[1]), Zd.T @ Ztr.astype(np.float64))
    B = (mu[None, :] + (Z.astype(np.float64) @ W) @ U.T.astype(np.float64)).astype(np.float32)
    return B, spec


def fit_response(F, R, fit_rows, w, lam):
    """加权掩码岭回归：残差 R 上学 Delta。w 是逐样本权重（化合物等权）。"""
    M = np.isfinite(R)
    Y = np.where(M, R, 0.0).astype(np.float64)
    Fd = F[fit_rows].astype(np.float64)
    wv = w[fit_rows][:, None]
    G = Fd.T @ (Fd * wv) + lam * np.eye(F.shape[1])
    B = Fd.T @ (Y[fit_rows] * wv)
    n_eff = float(w[fit_rows].sum())
    n_obs = (M[fit_rows] * wv).sum(axis=0)
    scale = np.where(n_obs > 1e-9, n_eff / np.maximum(n_obs, 1e-9), 0.0)
    W = np.linalg.solve(G, B * scale[None, :])
    return (F.astype(np.float64) @ W).astype(np.float32)


def compound_weights(pert, rows) -> np.ndarray:
    """每个化合物总权重相等（Pro E5 第 4 条）。"""
    w = np.zeros(len(pert), dtype=np.float64)
    for c in np.unique(pert[rows]):
        sel = rows & (pert == c)
        n = int(sel.sum())
        if n:
            w[sel] = 1.0 / n
    return w


# ------------------------------------------------------------------ 主流程


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=8)
    ap.add_argument("--k0", type=int, default=16)
    ap.add_argument("--lam-b", type=float, default=30.0)
    ap.add_argument("--n-perm", type=int, default=5, help="shuffled-label 重复次数")
    ap.add_argument("--skip-final-val", action="store_true",
                    help="跳过官方 val_chem_only 的一次性确认（调试用）")
    args = ap.parse_args()

    paths.ensure_dir(OUT_DIR)
    cfg = ScorerConfig()
    ctx = ev.build_context()
    meta = ctx.meta
    pert = meta[PERT_COL].astype(str).to_numpy()
    is_cmpd = ~np.isin(pert, NON_COMPOUND)

    # ---- L1-2：整个 LOCO 宇宙收进 train 折 ----
    train_rows = sg.train_rows(meta)
    train_cmpd_rows = train_rows & is_cmpd
    compounds = sorted(np.unique(pert[train_cmpd_rows]))

    rng = np.random.default_rng(20260805)
    fold_of = {c: i for i, c in zip(rng.permutation(len(compounds)) % args.folds, compounds)}

    L: list[str] = []
    a = L.append
    a("=" * 96)
    a("严格 nested LOCO：化合物表示有没有可验证的预测增益（2026-08-24 合规重跑）")
    a("=" * 96)
    a(f"外层：{args.folds} 折整化合物留出，宇宙 = split_final=='train'，"
      f"合法训练化合物 {len(compounds)} 个")
    a(f"训练行 {int(train_rows.sum())}（含对照）/ 其中处理行 {int(train_cmpd_rows.sum())}")
    a(f"架构：C-free  y = b(metadata, K0={args.k0}) + Delta(compound, context)")
    a("化合物等权：每个化合物总权重相同（最多 699 行 vs 最少 30 行，按行拟合差 23 倍）")
    a("指纹 bit 过滤与描述符标准化只在外层训练折内拟合")
    a("每折的设计矩阵词表与 log-time 标准化参数**只由该折的训练行冻结**")
    a("")
    a("❗与 2026-08-07 那版的区别（旧数字全部作废）：")
    a("  旧版 compounds 取自全 train_val（43 个），外层 tr = ~held，")
    a("  于是当折未被留出的 val 化合物照样进了训练——违反手册第 17 页。")
    a("  旧的 Morgan 增益 -0.0004 / 神谕近邻 +0.0237 / 神谕自身 +0.0617 不得再引用。")
    a("")

    # 每折的 b 只拟合一次，所有 phi 配置复用
    folds = []
    for f in range(args.folds):
        held_c = [c for c in compounds if fold_of[c] == f]
        held = np.isin(pert, held_c) & train_cmpd_rows
        if held.sum() < 20:
            continue
        tr = train_rows & ~held
        B, spec = fit_baseline(ctx, meta, tr, args.k0, args.lam_b)
        folds.append({"f": f, "te": held, "tr": tr, "B": B, "spec": spec,
                      "held_c": held_c,
                      "tr_c": [c for c in compounds if fold_of[c] != f]})
        print(f"[LOCO] 折 {f}: 留出 {len(held_c)} 个化合物 / {int(held.sum())} 行；"
              f"训练 {int(tr.sum())} 行；设计矩阵 {spec['n_cols']} 列")
    if not folds:
        raise SystemExit("没有任何一折留出行数达标")
    a(f"实际成折 {len(folds)} / {args.folds}；"
      f"每折留出化合物数 {[len(fd['held_c']) for fd in folds]}")
    a("")

    def _score(y, r):
        """两轴都算，与官方 _both_axes 口径一致，便于和 val 表对比。"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            dp = y[r].astype(np.float64) - ctx.C[r].astype(np.float64)
            return {
                "abs_pcc": np.nanmean([np.nanmean(pcc_axis(ctx.X[r], y[r], cfg, ax))
                                       for ax in (1, 0)]),
                "abs_r2": np.nanmean([np.nanmean(r2_axis(ctx.X[r], y[r], cfg, ax))
                                      for ax in (1, 0)]),
                "fc_pcc": np.nanmean([np.nanmean(pcc_axis(ctx.D[r], dp, cfg, ax))
                                      for ax in (1, 0)]),
            }

    def oracle_profile_pred(fd):
        """阳性对照：用留出化合物**自己的真值**去训练化合物里挑最像的，照搬其残差。

        这是作弊的，但必须做：若连它都拿不到增益，说明「b + 化合物残差」这个
        架构本身或评分口径不给分，那 ECFP 的失败就不能归因于表示不好。
        测试本身有没有效力，要先用已知有信号的输入验一次。
        """
        te, tr, B = fd["te"], fd["tr"], fd["B"]
        R = ctx.X - B
        prof = {}
        for c in np.unique(pert[tr & train_cmpd_rows]):
            sel = (tr & train_cmpd_rows) & (pert == c)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                prof[c] = np.nanmean(R[sel], axis=0, dtype=np.float64)
        names = sorted(prof)
        T = np.vstack([prof[k] for k in names])
        y = B.copy()
        for c in np.unique(pert[te]):
            sel = te & (pert == c)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                own = np.nanmean(R[sel], axis=0, dtype=np.float64)
            m = np.isfinite(own)[None, :] & np.isfinite(T)
            A = np.where(m, own[None, :], 0.0)
            Bz = np.where(m, T, 0.0)
            n = m.sum(1).astype(float)
            with np.errstate(invalid="ignore", divide="ignore"):
                cov = (A * Bz).sum(1) / n - A.sum(1) * Bz.sum(1) / n ** 2
                va = (A * A).sum(1) / n - (A.sum(1) / n) ** 2
                vb = (Bz * Bz).sum(1) / n - (Bz.sum(1) / n) ** 2
                r = cov / np.sqrt(np.maximum(va, 1e-12) * np.maximum(vb, 1e-12))
            best = T[int(np.nanargmax(r))]
            y[sel] = B[sel] + np.where(np.isfinite(best), best, 0.0).astype(np.float32)
        return y

    def oracle_feature_table(fd, dim: int = 8) -> dict:
        """阳性对照之二：给响应模块喂一个**已知有信息**的化合物特征。

        为什么必须有它。上面那个「神谕残差照搬」绕过了整个岭回归模块，
        它只能证明「架构与评分口径给分」，证明不了「模块本身还有没有容量」。
        万一内层把 lambda 一路选到最大、把 Delta 压成 0，ECFP 拿不到增益就成了
        必然结果，检验对「结构表示行不行」这个问题其实没有效力——而报告里
        却会写成「结构信息无用」。这是评审最容易问倒人的一处。

        做法：把每个化合物自己的平均残差压到 dim 维当特征，走**同一条**
        fit_response 路径、同一套内层 lambda 选择。留出化合物的特征用到了它自己的
        残差，所以这是神谕、不可实现；但它与 ECFP 唯一的差别就是特征内容。
        它拿得到增益而 ECFP 拿不到 => 失败在表示层面。
        """
        te, tr, B = fd["te"], fd["tr"], fd["B"]
        R = ctx.X - B
        prof = {}
        for c in np.unique(pert[train_cmpd_rows]):
            sel = train_cmpd_rows & (pert == c)
            if not sel.any():
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                prof[c] = np.nan_to_num(
                    np.nanmean(R[sel], axis=0, dtype=np.float64), nan=0.0)
        names = sorted(prof)
        M = np.vstack([prof[c] for c in names])
        # 基只由**外层训练折**的化合物拟合，与 ECFP 的 bit 过滤同一条纪律
        tr_idx = [i for i, c in enumerate(names) if c in set(fd["tr_c"])]
        mu = M[tr_idx].mean(axis=0)
        U, S, Vt = np.linalg.svd(M[tr_idx] - mu, full_matrices=False)
        V = Vt[:min(dim, Vt.shape[0])].T
        Z = (M - mu) @ V
        sd = Z[tr_idx].std(axis=0) + 1e-8
        Z = Z / sd
        return {c: Z[i].astype(np.float32) for i, c in enumerate(names)}

    def run(phi_mode: str, seed: int = 0, lam_r: float = None) -> dict:
        """phi_mode in {none, ecfp, ecfp_shuffled, oracle_feature,
        oracle_profile, oracle_self}。"""
        acc, lams = {"abs_pcc": [], "abs_r2": [], "fc_pcc": [], "n": []}, []
        for fd in folds:
            te, tr, B = fd["te"], fd["tr"], fd["B"]
            if phi_mode == "none":
                y = B
            elif phi_mode == "oracle_profile":
                y = oracle_profile_pred(fd)
            elif phi_mode == "oracle_self":
                # 上限：直接用留出化合物**自己**的平均残差当 Delta。
                # 任何「预测每个化合物一条响应谱」的模型都不可能超过它。
                R2 = ctx.X - B
                y = B.copy()
                for c in np.unique(pert[te]):
                    sel = te & (pert == c)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        own = np.nanmean(R2[sel], axis=0, dtype=np.float64)
                    y[sel] = B[sel] + np.where(np.isfinite(own), own, 0.0).astype(np.float32)
            else:
                if phi_mode == "oracle_feature":
                    tab = oracle_feature_table(fd)
                else:
                    tab = ecfp_table(compounds, fd["tr_c"], seed=seed,
                                     shuffle=(phi_mode == "ecfp_shuffled"))
                if tab is None:
                    return {}
                F = feature_matrix(pert, tab, ctx.n)
                R = ctx.X - B
                fit_rows = tr & train_cmpd_rows
                w = compound_weights(pert, fit_rows)
                if lam_r is None:
                    # 内层：在外层训练折内部再留出一批化合物来选 lambda（Pro E5 第 5 条）
                    inner_c = fd["tr_c"][::3]           # 取 1/3 化合物作内层验证
                    inner_te = np.isin(pert, inner_c) & fit_rows
                    inner_tr = fit_rows & ~inner_te
                    w_in = compound_weights(pert, inner_tr)
                    best, best_l = -np.inf, 200.0
                    ri = np.nonzero(inner_te)[0]
                    for l in (20.0, 60.0, 200.0, 600.0, 2000.0):
                        yi = B + fit_response(F, R, inner_tr, w_in, lam=l)
                        s = _score(yi, ri)["fc_pcc"]
                        if np.isfinite(s) and s > best:
                            best, best_l = s, l
                    lams.append(best_l)
                    l_use = best_l
                else:
                    l_use = lam_r
                y = B + fit_response(F, R, fit_rows, w, lam=l_use)
            acc_i = _score(y, np.nonzero(te)[0])
            for k, v in acc_i.items():
                acc[k].append(v)
            acc["n"].append(int(te.sum()))
        out = {k: float(np.nanmean(v)) for k, v in acc.items() if k != "n"}
        out["n_total"] = int(np.sum(acc["n"]))
        if lams:
            out["lam"] = f"内层选出 {sorted(set(lams))}"
            out["lam_list"] = [float(x) for x in lams]
        return out

    a("-" * 96)
    a("外层 LOCO 结果（在留出化合物的样本上，逐样本相关后按折平均）")
    a("-" * 96)
    a(f"  {'配置':<28}{'abs_pcc':>10}{'abs_r2':>10}{'fc_pcc':>10}{'留出样本':>10}")
    base = run("none")
    a(f"  {'仅上下文 b（无化合物信息）':<28}{base['abs_pcc']:>10.4f}"
      f"{base['abs_r2']:>10.4f}{base['fc_pcc']:>10.4f}{base['n_total']:>10d}")
    ecfp = run("ecfp")
    a(f"  {'+ ECFP 结构表示':<28}{ecfp['abs_pcc']:>10.4f}"
      f"{ecfp['abs_r2']:>10.4f}{ecfp['fc_pcc']:>10.4f}{ecfp['n_total']:>10d}")
    ofe = run("oracle_feature")
    a(f"  {'[阳性对照2] 神谕特征走同一模块':<28}{ofe['abs_pcc']:>10.4f}"
      f"{ofe['abs_r2']:>10.4f}{ofe['fc_pcc']:>10.4f}{ofe['n_total']:>10d}")
    orc = run("oracle_profile")
    a(f"  {'[阳性对照] 神谕残差照搬':<28}{orc['abs_pcc']:>10.4f}"
      f"{orc['abs_r2']:>10.4f}{orc['fc_pcc']:>10.4f}{orc['n_total']:>10d}")
    slf = run("oracle_self")
    a(f"  {'[上限] 神谕用自身平均残差':<28}{slf['abs_pcc']:>10.4f}"
      f"{slf['abs_r2']:>10.4f}{slf['fc_pcc']:>10.4f}{slf['n_total']:>10d}")
    a("")
    gain = ecfp["fc_pcc"] - base["fc_pcc"]
    gain_o = orc["fc_pcc"] - base["fc_pcc"]
    gain_s = slf["fc_pcc"] - base["fc_pcc"]
    a(f"  【整条化合物特异路线的上限】用留出化合物自己的平均残差：{gain_s:+.4f}")
    a(f"    按指标 2 的 25% 权重折成总分约 {0.25 * gain_s:+.4f}。")
    a("    ⚠ 该折算假设其余五项不受影响，未经检验，不可与总分口径的数直接比大小。")
    a("    任何「给每个化合物预测一条响应谱」的模型都不可能超过 fc_pcc 上的这个数。")
    a("")
    gain_f = ofe["fc_pcc"] - base["fc_pcc"]
    a(f"  ECFP 相对仅上下文的 fc_pcc 增益：{gain:+.4f}")
    a(f"  神谕残差照搬的增益（绕过岭回归模块，验证架构与评分口径给分）：{gain_o:+.4f}")
    a(f"  神谕特征走同一模块的增益（**不绕过**模块，验证模块本身还有容量）：{gain_f:+.4f}")
    a("")
    a("  这两个阳性对照分工不同，缺了第二个就答不上评审最可能问的那一句：")
    a("  「你的响应模块是不是被内层 lambda 压成 0 了？那 ECFP 拿不到增益是必然的，")
    a("   你的检验对『结构表示行不行』根本没有效力。」")
    if gain_f < 0.005:
        a(f"  → 实测 {gain_f:+.4f}：**模块确实吃不下任何化合物特征**。")
        a("    此时不能把 ECFP 的失败归因于结构表示，得先修模块或换 Delta 的形态。")
    else:
        a(f"  → 实测 {gain_f:+.4f}：同一条 fit_response 路径、同一套内层 lambda，")
        a("    换成有信息的特征就拿得到增益。模块没被压死，检验有效力；")
        a("    ECFP 拿不到，就确实是**表示层面**的问题。")
    a("")
    if gain_o < 0.005:
        a("  ⚠ **连神谕都拿不到增益** → 问题不在化合物表示，而在"
          "「b + 化合物残差」这个架构或评分口径本身。")
        a("    此时不能把 ECFP 的失败归因于结构信息无用——本检验对这个问题没有效力。")
    else:
        a("  → 神谕能拿到增益，说明架构可用、检验有效力；")
        a("    ECFP 拿不到就确实是表示层面的问题。")
    a("")
    if lam_used := ecfp.get("lam"):
        a(f"  响应模块的正则强度：{lam_used}")
        a("  （若内层一律选到最大 lambda，说明模块被压成 0，检验会失去效力——需核对）")
    a("")

    perms = np.asarray([])
    if args.n_perm > 0:
        a("-" * 96)
        a(f"shuffled-label 对照（{args.n_perm} 次，把结构表示随机指派给别的化合物）")
        a("-" * 96)
        pl = []
        for s in range(args.n_perm):
            r = run("ecfp_shuffled", seed=1000 + s)
            pl.append(r["fc_pcc"] - base["fc_pcc"])
            a(f"    第 {s+1} 次：fc_pcc 增益 {pl[-1]:+.4f}")
        perms = np.asarray(pl)
        a("")
        a(f"  打乱标签的增益：均值 {perms.mean():+.4f}  最大 {perms.max():+.4f}")
        a(f"  真实结构表示的增益：{gain:+.4f}")
        if gain > perms.max():
            a("  → 真增益超过全部打乱对照，结构信息**通过**本关")
        else:
            a("  → 真增益**未**超过打乱对照的最大值 → 观察到的提升与随机指派无法区分。")
            a("    按 Pro E5 的预注册标准，此时应把结构项权重设为 0；")
            a("    但只能说『当前样本下无可验证的预测增益』，不能说结构在生物学上无关。")
    a("")

    # ------------------------------------------------ 官方 val_chem_only 一次性确认
    final = {}
    if not args.skip_final_val:
        a("-" * 96)
        a("官方 val_chem_only 上的一次性确认（只看一眼，不参与任何选参）")
        a("-" * 96)
        val_c = ctx.rows(["val_chem_only"])
        B_all, spec_all = fit_baseline(ctx, meta, train_rows, args.k0, args.lam_b)
        f_base = ev.flatten(ev.evaluate(B_all, ctx, val_c, cfg))
        val_compounds = sorted(np.unique(pert[val_c & ctx.treated]))
        tab = ecfp_table(compounds + val_compounds, compounds, seed=0)
        if tab is None:
            a("  ⚠ SMILES 不足，跳过")
        else:
            F = feature_matrix(pert, tab, ctx.n)
            R = ctx.X - B_all
            fit_rows = train_cmpd_rows
            w = compound_weights(pert, fit_rows)
            # lambda 用外层内层选出的众数，不在 val 上再选
            lam_pick = (float(pd.Series(ecfp.get("lam_list", [200.0])).mode().iloc[0])
                        if ecfp.get("lam_list") else 200.0)
            y_ecfp = B_all + fit_response(F, R, fit_rows, w, lam=lam_pick)
            f_ecfp = ev.flatten(ev.evaluate(y_ecfp, ctx, val_c, cfg))
            n_cov = int(sum(1 for c in val_compounds if c in tab))
            a(f"  val_chem_only 化合物 {len(val_compounds)} 个，其中有 SMILES 的 {n_cov} 个")
            a(f"  响应模块 lambda = {lam_pick:g}（取外层内层选出的众数，未在 val 上调）")
            a("")
            a(f"  {'配置':<22}{'total':>10}{'fc_pcc':>10}{'ctx_resid':>11}{'abs_r2':>10}")
            for lab, f_ in [("仅上下文 b", f_base), ("+ ECFP", f_ecfp)]:
                a(f"  {lab:<22}{f_['total']:>10.4f}{f_['fc_pcc']:>10.4f}"
                  f"{f_['ctx_resid']:>11.4f}{f_['abs_r2']:>10.4f}")
            a(f"  差：total {f_ecfp['total']-f_base['total']:+.4f}   "
              f"fc_pcc {f_ecfp['fc_pcc']-f_base['fc_pcc']:+.4f}   "
              f"ctx_resid {f_ecfp['ctx_resid']-f_base['ctx_resid']:+.4f}")
            a("  ⚠ 只有 6 个留出化合物，这一格的不确定性很大，只作与外层 LOCO 的方向一致性核对，")
            a("    不单独构成结论。")
            final = {
                "n_val_compounds": len(val_compounds),
                "n_with_smiles": n_cov,
                "lam": lam_pick,
                "context_only": f_base,
                "ecfp": f_ecfp,
                "delta_total": f_ecfp["total"] - f_base["total"],
                "delta_fc_pcc": f_ecfp["fc_pcc"] - f_base["fc_pcc"],
                "delta_ctx_resid": f_ecfp["ctx_resid"] - f_base["ctx_resid"],
            }
    a("")
    a("-" * 96)
    a("与官方 val 口径的关系")
    a("-" * 96)
    a(f"外层 LOCO 在 train 折内部把 {len(compounds)} 个合法训练化合物全部轮流留出，")
    a("比只有 6 个化合物的官方 val_chem_only 更稳定；后者作一次性方向确认。")
    a("两者都不许用于选参——选参在外层训练折的内层留出里完成。")

    txt = "\n".join(L)
    print(txt)
    p = os.path.join(OUT_DIR, "report.txt")
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(txt)
    print(f"\n[写出] {p}")

    # 机器可读副本：权威表要收录 LOCO 的数，不能靠人从 report.txt 里抄。
    # 2026-08-07 外审发现文中的三个增益都不在注册范围内，数字核对脚本对它们完全无感。
    # 2026-08-24 起再加一层：把源码摘要写进来，权威表收录时要核对它是不是当前这份代码。
    payload = {
        "_schema": "loco/2.1-train-only",
        "universe": "split_final=='train'",
        "n_compounds": len(compounds),
        "compounds": compounds,
        "n_folds_requested": args.folds,
        "n_folds_effective": len(folds),
        "n_train_rows": int(train_rows.sum()),
        "n_train_compound_rows": int(train_cmpd_rows.sum()),
        "k0": args.k0,
        "lam_b": args.lam_b,
        "context_only_fc_pcc": base["fc_pcc"],
        "ecfp_fc_pcc": ecfp["fc_pcc"],
        "gain_ecfp": gain,
        "gain_oracle_feature_same_module": gain_f,
        "gain_oracle_neighbor": gain_o,
        "gain_oracle_self": gain_s,
        "gain_oracle_self_scaled_to_total": 0.25 * gain_s,
        "outer_rows": {"context_only": base, "ecfp": ecfp,
                       "oracle_feature": ofe,
                       "oracle_neighbor": orc, "oracle_self": slf},
        "n_perm": args.n_perm,
        "shuffled_gains": [float(x) for x in perms],
        "shuffled_mean": float(perms.mean()) if perms.size else None,
        "shuffled_max": float(perms.max()) if perms.size else None,
        "passes_shuffled_control": bool(perms.size and gain > perms.max()),
        "final_val_chem_only": final,
        "_note": "增益均为 fc_pcc（指标 2）口径，不是六项总分。折成总分的换算未经检验。",
        "_provenance": provenance.stamp(),
    }
    pj = os.path.join(OUT_DIR, "loco.json")
    with open(pj, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)
    print(f"[写出] {pj}")


if __name__ == "__main__":
    main()
