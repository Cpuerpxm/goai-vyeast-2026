"""Day 4-6 · 严格 nested LOCO：化合物表示到底有没有可验证的预测增益。

Pro R2 E5 的关卡。此前 B3 与 φ=morgan 的失败都可能是实现问题而非"信息不存在"，
所以这里把 Pro 点名的五个方法学漏洞一次补齐：

1. **外层按整化合物留出**（不是随机行、也不是官方 split）
2. **化合物等权**：样本最多的化合物有 699 行、最少的 30 行，按行拟合会让前者
   的权重是后者的 23 倍。这里每个化合物总权重相等
3. **预处理只在外层训练折内拟合**：指纹 bit 过滤、描述符标准化都不许看留出化合物
4. **超参在内层选**：λ 与 K_Δ 由外层训练折内部的二次留出决定
5. **shuffled-label 对照**：把化合物标签打乱后重跑，真增益必须显著超过它

架构是 C-free 的（Pro L1-04）：

    ŷ = b̂(metadata)  +  Δ̂(compound, context)

Δ̂ 拟合在训练残差 (X − b̂) 上，推断时不接触任何对照。

运行：
    python loco_response.py                  # 默认 8 折、含 shuffled 对照
    python loco_response.py --folds 4 --n-perm 0    # 快速版
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
from models.baseline_cfree import FEATURE_SETS, design
from models.lowrank import masked_pca
from scorer import evaluate as ev
from scorer.config import ScorerConfig
from scorer.metrics import pcc_axis, r2_axis

PERT_COL = "perturbation_no_concentration"
NON_COMPOUND = ["Water", "DMSO", "Quality Control"]
OUT_DIR = os.path.join(paths.RESULTS, "step9_loco")
SMILES_CSV = os.path.join(paths.DATA_EXTERNAL, "compound_smiles.csv")


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


# ------------------------------------------------------------------ 拟合


def fit_baseline(ctx, Z, train_rows, k0, lam):
    """C-free 绝对骨架 b̂：掩码 PCA + metadata 岭回归。"""
    Xtr = ctx.X[train_rows]
    mu, U, Ztr = masked_pca(Xtr, k0, np.isfinite(Xtr), n_iter=12, center=True, seed=0)
    Zd = Z[train_rows].astype(np.float64)
    W = np.linalg.solve(Zd.T @ Zd + lam * np.eye(Z.shape[1]), Zd.T @ Ztr.astype(np.float64))
    return (mu[None, :] + (Z.astype(np.float64) @ W) @ U.T.astype(np.float64)).astype(np.float32)


def fit_response(F, R, fit_rows, w, lam):
    """加权掩码岭回归：残差 R 上学 Δ̂。w 是逐样本权重（化合物等权）。"""
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
    args = ap.parse_args()

    paths.ensure_dir(OUT_DIR)
    cfg = ScorerConfig()
    ctx = ev.build_context()
    pert = ctx.meta[PERT_COL].astype(str).to_numpy()
    is_cmpd = ~np.isin(pert, NON_COMPOUND)
    compounds = sorted(np.unique(pert[is_cmpd]))
    Z = design(ctx.meta, FEATURE_SETS["bio_tech"], with_drug=False)

    rng = np.random.default_rng(20260805)
    fold_of = {c: i for i, c in zip(rng.permutation(len(compounds)) % args.folds, compounds)}

    L: list[str] = []
    a = L.append
    a("=" * 92)
    a("Day 4-6 · 严格 nested LOCO：化合物表示有没有可验证的预测增益")
    a("=" * 92)
    a(f"外层：{args.folds} 折整化合物留出，共 {len(compounds)} 个化合物")
    a(f"架构：C-free  ŷ = b̂(metadata, K₀={args.k0}) + Δ̂(compound, context)")
    a("化合物等权：每个化合物总权重相同（最多 699 行 vs 最少 30 行，按行拟合差 23 倍）")
    a("指纹 bit 过滤与描述符标准化只在外层训练折内拟合")
    a("")

    # 每折的 b̂ 只拟合一次，所有 φ 配置复用（否则 7 个配置 × 8 折 = 56 次掩码 PCA）
    folds = []
    for f in range(args.folds):
        held = np.asarray([fold_of.get(c, -1) == f for c in pert], dtype=bool) & is_cmpd
        if held.sum() < 20:
            continue
        tr = ~held
        folds.append({"f": f, "te": held, "tr": tr,
                      "B": fit_baseline(ctx, Z, tr, args.k0, args.lam_b),
                      "tr_c": [c for c in compounds if fold_of[c] != f]})
    print(f"[LOCO] {len(folds)} 折的绝对骨架已拟合并缓存")

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

        这是作弊的，但必须做：若连它都拿不到增益，说明「b̂ + 化合物残差」这个
        架构本身或评分口径不给分，那 ECFP 的失败就不能归因于表示不好。
        测试本身有没有效力，要先用已知有信号的输入验一次。
        """
        te, tr, B = fd["te"], fd["tr"], fd["B"]
        R = ctx.X - B
        prof = {}
        for c in np.unique(pert[tr & is_cmpd]):
            sel = (tr & is_cmpd) & (pert == c)
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

    def run(phi_mode: str, seed: int = 0, lam_r: float = None) -> dict:
        """phi_mode ∈ {none, ecfp, ecfp_shuffled, oracle_profile}。lam_r=None 时内层选。"""
        acc, lams = {"abs_pcc": [], "abs_r2": [], "fc_pcc": [], "n": []}, []
        for fd in folds:
            te, tr, B = fd["te"], fd["tr"], fd["B"]
            if phi_mode == "none":
                y = B
            elif phi_mode == "oracle_profile":
                y = oracle_profile_pred(fd)
            elif phi_mode == "oracle_self":
                # 上限：直接用留出化合物**自己**的平均残差当 Δ̂。
                # 任何「预测每个化合物一条响应谱」的模型都不可能超过它。
                te2, B2 = fd["te"], fd["B"]
                R2 = ctx.X - B2
                y = B2.copy()
                for c in np.unique(pert[te2]):
                    sel = te2 & (pert == c)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        own = np.nanmean(R2[sel], axis=0, dtype=np.float64)
                    y[sel] = B2[sel] + np.where(np.isfinite(own), own, 0.0).astype(np.float32)
            else:
                tab = ecfp_table(compounds, fd["tr_c"], seed=seed,
                                 shuffle=(phi_mode == "ecfp_shuffled"))
                if tab is None:
                    return {}
                dim = len(next(iter(tab.values())))
                P = np.zeros((ctx.n, dim), dtype=np.float32)
                for i, c in enumerate(pert):
                    if c in tab:
                        P[i] = tab[c]
                F = np.hstack([np.ones((ctx.n, 1), np.float32), P]).astype(np.float32)
                R = ctx.X - B
                fit_rows = tr & is_cmpd
                w = compound_weights(pert, fit_rows)
                if lam_r is None:
                    # 内层：在外层训练折内部再留出一批化合物来选 λ（Pro E5 第 5 条）
                    inner_c = fd["tr_c"][::3]           # 取 1/3 化合物作内层验证
                    inner_te = np.isin(pert, inner_c) & is_cmpd
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
        return out

    a("-" * 92)
    a("外层 LOCO 结果（在留出化合物的样本上，逐样本相关后按折平均）")
    a("-" * 92)
    a(f"  {'配置':<26}{'abs_pcc':>10}{'abs_r2':>10}{'fc_pcc':>10}{'留出样本':>10}")
    base = run("none")
    a(f"  {'仅上下文 b̂（无化合物信息）':<26}{base['abs_pcc']:>10.4f}"
      f"{base['abs_r2']:>10.4f}{base['fc_pcc']:>10.4f}{base['n_total']:>10d}")
    ecfp = run("ecfp")
    a(f"  {'+ ECFP 结构表示':<26}{ecfp['abs_pcc']:>10.4f}"
      f"{ecfp['abs_r2']:>10.4f}{ecfp['fc_pcc']:>10.4f}{ecfp['n_total']:>10d}")
    orc = run("oracle_profile")
    a(f"  {'[阳性对照] 神谕残差照搬 ⚠':<26}{orc['abs_pcc']:>10.4f}"
      f"{orc['abs_r2']:>10.4f}{orc['fc_pcc']:>10.4f}{orc['n_total']:>10d}")
    slf = run("oracle_self")
    a(f"  {'[上限] 神谕用自身平均残差 ⚠':<26}{slf['abs_pcc']:>10.4f}"
      f"{slf['abs_r2']:>10.4f}{slf['fc_pcc']:>10.4f}{slf['n_total']:>10d}")
    a("")
    gain = ecfp["fc_pcc"] - base["fc_pcc"]
    gain_o = orc["fc_pcc"] - base["fc_pcc"]
    gain_s = slf["fc_pcc"] - base["fc_pcc"]
    a(f"  【整条化合物特异路线的上限】用留出化合物自己的平均残差：{gain_s:+.4f}")
    a(f"    按指标 2 的 25% 权重折成总分约 {0.25 * gain_s:+.4f}。")
    a("    ⚠ 该折算假设其余五项不受影响，未经检验；它与总分口径的 +0.0287 不可直接比大小。")
    a("    任何「给每个化合物预测一条响应谱」的模型都不可能超过 fc_pcc 上的这个数。")
    a("")
    a(f"  ECFP 相对仅上下文的 fc_pcc 增益：{gain:+.4f}")
    a(f"  神谕残差照搬的增益（不可实现，用于验证检验有功效）：{gain_o:+.4f}")
    if gain_o < 0.005:
        a("  ⚠ **连神谕都拿不到增益** → 问题不在化合物表示，而在"
          "「b̂ + 化合物残差」这个架构或评分口径本身。")
        a("    此时不能把 ECFP 的失败归因于结构信息无用——本检验对这个问题没有效力。")
    else:
        a("  → 神谕能拿到增益，说明架构可用、检验有效力；")
        a("    ECFP 拿不到就确实是表示层面的问题。")
    a("")
    if lam_used := ecfp.get("lam"):
        a(f"  响应模块的正则强度：{lam_used}")
        a("  （若内层一律选到最大 λ，说明模块被压成 0，检验会失去效力——需核对）")
    a("")

    if args.n_perm > 0:
        a("-" * 92)
        a(f"shuffled-label 对照（{args.n_perm} 次，把结构表示随机指派给别的化合物）")
        a("-" * 92)
        perms = []
        for s in range(args.n_perm):
            r = run("ecfp_shuffled", seed=1000 + s)
            perms.append(r["fc_pcc"] - base["fc_pcc"])
            a(f"    第 {s+1} 次：fc_pcc 增益 {perms[-1]:+.4f}")
        perms = np.asarray(perms)
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
    a("-" * 92)
    a("与官方 val 口径的关系")
    a("-" * 92)
    a("本表是**外层 LOCO**，比官方 val_chem_only 更严格也更稳定（43 个化合物全部轮流留出，")
    a("而 val_chem_only 只有 6 个）。最终提交前仍需在官方划分上冻结一次评估。")

    txt = "\n".join(L)
    print(txt)
    p = os.path.join(OUT_DIR, "report.txt")
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(txt)
    print(f"\n[写出] {p}")

    # 机器可读副本：权威表要收录 LOCO 的数，不能靠人从 report.txt 里抄。
    # 2026-08-07 外审发现文中的 +0.0617 / +0.0237 / −0.0004 都不在注册范围内，
    # 于是数字核对脚本对它们完全无感——这就是那次漏检的根因。
    import json as _json
    payload = {
        "context_only_fc_pcc": base["fc_pcc"],
        "ecfp_fc_pcc": ecfp["fc_pcc"],
        "gain_ecfp": gain,
        "gain_oracle_neighbor": gain_o,
        "gain_oracle_self": gain_s,
        "gain_oracle_self_scaled_to_total": 0.25 * gain_s,
        "n_folds": args.folds,
        "n_compounds": len(compounds),
        "n_perm": args.n_perm,
        "shuffled_gains": [float(x) for x in perms] if args.n_perm > 0 else [],
        "shuffled_mean": float(perms.mean()) if args.n_perm > 0 else None,
        "shuffled_max": float(perms.max()) if args.n_perm > 0 else None,
        "_note": "增益均为 fc_pcc（指标 2）口径，不是六项总分。折成总分的换算未经检验。",
    }
    pj = os.path.join(OUT_DIR, "loco.json")
    with open(pj, "w", encoding="utf-8", newline="\n") as fh:
        _json.dump(payload, fh, ensure_ascii=False, indent=1)
    print(f"[写出] {pj}")


if __name__ == "__main__":
    main()
