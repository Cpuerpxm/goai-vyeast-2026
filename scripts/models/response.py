"""第 6 步 · 响应模型 + 低秩交互。

docs/05 §5.4：**纯加性拿不到指标 3/4 那 40%**，所以必须带交互项。

    z_Δ = A·φ(d) + B·ψ(u) + W·[(P·φ(d)) ⊙ (Q·ψ(u))]

    φ(d) 化合物表示  = [Morgan/ECFP 指纹, RDKit 物化描述符, 粗粒度 MoA]
    ψ(u) 上下文表示  = [strain, medium, temperature, time_spline]
    交互维度 r ∈ [8, 16]（不展开完整 Kronecker 积）

预测目标是低秩系数 z_Δ（第 5 步的 U_Δ 基），不是 5,243 维原始向量。
解码 Δ̂ = U_Δ z_Δ + μ_Δ，再 ŷ = y_control + Δ̂。

**φ(d) 是可插拔的**：
  - `onehot`：药物 one-hot。对已见化合物（S2/val_strain_only）有效，
              对未见化合物（S1）整列为 0 → 自动退化成上下文模型。
  - `morgan`：Morgan 指纹 + RDKit 描述符。需要 data/external/compound_smiles.csv。
              这是唯一能让 S1 有戏的一路，缺 SMILES 时跳过并明确报出来。

分阶段训练（docs/05 §7.1）：本步只学响应 Δ̂，基线 ŷ₀ 不动，防止批次效应
和共享应激被吸进药物特异模块。

运行：
    python response.py                    # φ=onehot
    python response.py --phi morgan       # 需 SMILES
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
from models.baselines import SMILES_CSV, _fill, b0_global_mean
from scorer import evaluate as ev
from scorer.config import ScorerConfig

PERT_COL = "perturbation_no_concentration"
OUT_DIR = os.path.join(paths.RESULTS, "step6_response")
BASES = os.path.join(paths.RESULTS, "step5_lowrank", "bases.npz")


# --------------------------------------------------------------- 表示


def psi_context(meta: pd.DataFrame) -> np.ndarray:
    """ψ(u)：菌株 / 培养基 / 温度 one-hot + log-time 样条。"""
    blocks = []
    for c in ["Strains", "Medium", "Temperature"]:
        codes, uniq = pd.factorize(meta[c].astype(str))
        M = np.zeros((len(meta), len(uniq)), dtype=np.float32)
        M[np.arange(len(meta)), codes] = 1.0
        blocks.append(M)
    t = np.log1p(meta["pert_time"].to_numpy(dtype=np.float64))
    t = (t - t.mean()) / t.std()
    # 自然样条的简易替代：三次多项式基（6 个时间点撑不起更复杂的基）
    blocks.append(np.stack([t, t ** 2, t ** 3], axis=1).astype(np.float32))
    return np.hstack(blocks)


def phi_onehot(meta: pd.DataFrame, drugs: list) -> np.ndarray:
    pert = meta[PERT_COL].astype(str).to_numpy()
    M = np.zeros((len(meta), len(drugs)), dtype=np.float32)
    idx = {d: i for i, d in enumerate(drugs)}
    for i, d in enumerate(pert):
        j = idx.get(d)
        if j is not None:
            M[i, j] = 1.0
    return M


def phi_morgan(meta: pd.DataFrame, smiles_csv: str):
    """Morgan 指纹（半径 2，2048 位，先按训练集出现频率过滤）+ RDKit 描述符。"""
    if not os.path.exists(smiles_csv):
        return None, f"缺 {smiles_csv}"
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")

    tab = pd.read_csv(smiles_csv)
    if not {"compound", "smiles"}.issubset(tab.columns):
        return None, f"{smiles_csv} 需要列 compound / smiles"
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    desc_fn = [("MolWt", Descriptors.MolWt), ("LogP", Descriptors.MolLogP),
               ("TPSA", Descriptors.TPSA), ("HBD", Descriptors.NumHDonors),
               ("HBA", Descriptors.NumHAcceptors),
               ("RotB", Descriptors.NumRotatableBonds),
               ("Rings", Descriptors.RingCount),
               ("FracCSP3", Descriptors.FractionCSP3)]
    fp, desc = {}, {}
    for _, r in tab.iterrows():
        s = str(r["smiles"]).strip()
        # 同 baselines.py：MolFromSmiles("") 返回空分子而非 None，必须先挡
        if not s or s.lower() == "nan":
            continue
        mol = Chem.MolFromSmiles(s)
        if mol is None or mol.GetNumAtoms() == 0:
            continue
        fp[str(r["compound"])] = np.asarray(gen.GetFingerprint(mol), dtype=np.float32)
        desc[str(r["compound"])] = np.asarray([f(mol) for _, f in desc_fn], dtype=np.float32)

    pert = meta[PERT_COL].astype(str).to_numpy()
    have = [d for d in np.unique(pert) if d in fp]
    if len(have) < 10:
        return None, f"只有 {len(have)} 个化合物解析出结构，不足以建化学表示"
    B = np.vstack([fp[d] for d in have])
    keep = (B.sum(0) >= 3) & (B.sum(0) <= len(have) - 3)     # 去掉全 0/全 1 的位
    Dsc = np.vstack([desc[d] for d in have])
    Dsc = (Dsc - Dsc.mean(0)) / (Dsc.std(0) + 1e-8)
    table = {d: np.concatenate([fp[d][keep], Dsc[i]]) for i, d in enumerate(have)}
    dim = len(next(iter(table.values())))
    M = np.zeros((len(meta), dim), dtype=np.float32)
    for i, d in enumerate(pert):
        if d in table:
            M[i] = table[d]
    return M, f"{len(have)} 个化合物有结构，特征维 {dim}（指纹 {int(keep.sum())} + 描述符 8）"


# --------------------------------------------------------------- 模型


def fit_response(
    phi: np.ndarray, psi: np.ndarray, Zt: np.ndarray, train: np.ndarray,
    r_inter: int = 12, lam: float = 50.0, seed: int = 0,
):
    """z_Δ = A·φ + B·ψ + W·[(P·φ) ⊙ (Q·ψ)]，P/Q 随机投影后整体走岭回归。

    P、Q 用固定随机投影而不是学出来：43 个独立化合物撑不起可辨识的双线性
    参数，随机投影 + 岭回归是同等表达力下方差小得多的做法。
    """
    rng = np.random.default_rng(seed)
    P = rng.normal(0, 1 / np.sqrt(max(phi.shape[1], 1)), (phi.shape[1], r_inter)).astype(np.float32)
    Q = rng.normal(0, 1 / np.sqrt(psi.shape[1]), (psi.shape[1], r_inter)).astype(np.float32)
    inter = (phi @ P) * (psi @ Q)
    F = np.hstack([np.ones((len(phi), 1), np.float32), phi, psi, inter]).astype(np.float64)

    Ft = F[train]
    G = Ft.T @ Ft + lam * np.eye(F.shape[1])
    W = np.linalg.solve(G, Ft.T @ Zt[train])
    return F @ W, {"P": P, "Q": Q, "W": W, "n_feat": F.shape[1], "r_inter": r_inter}


def build_targets(ctx: ev.EvalContext, U: np.ndarray, mu_d: np.ndarray) -> np.ndarray:
    """把 Δ 投到低秩系数 z_Δ。缺失位置不参与投影（掩码最小二乘）。

    用 EM 迭代而不是逐行解正规方程：逐行建 UᵀU 是 O(n·p·k²)，
    8,958 行 × 5,243 蛋白 × k=42 要十几分钟；EM 只有矩阵乘，几十秒。
    """
    from models.lowrank import encode_masked

    return encode_masked(ctx.D, mu_d, U, np.isfinite(ctx.D), n_iter=10)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phi", choices=["onehot", "morgan"], default="onehot")
    ap.add_argument("--smiles", default=SMILES_CSV)
    ap.add_argument("--r-inter", type=int, default=12)
    ap.add_argument("--lam", type=float, default=50.0)
    args = ap.parse_args()

    if not os.path.exists(BASES):
        print(f"❗先跑第 5 步生成 {BASES}")
        sys.exit(2)
    paths.ensure_dir(OUT_DIR)
    cfg = ScorerConfig()
    ctx = ev.build_context()
    val = ctx.rows(ev.VAL_SPLITS)
    fb = b0_global_mean(ctx)[0]

    z = np.load(BASES, allow_pickle=False)
    U, mu_d, K = z["U_delta"], z["mu_delta"], int(z["K_delta"])

    L: list[str] = []
    a = L.append
    a("=" * 100)
    a("第 6 步 · 响应模型 + 低秩交互")
    a("=" * 100)
    a(f"低秩基来自第 5 步：K_Δ = {K}（硬上限 42）")
    a(f"目标 z_Δ 由逐样本掩码最小二乘投影得到（缺失位置不参与）")
    a("")

    psi = psi_context(ctx.meta)
    drugs = sorted(np.unique(ctx.meta[PERT_COL].astype(str).to_numpy()[ctx.train_mask]))
    if args.phi == "onehot":
        phi, note = phi_onehot(ctx.meta, drugs), f"药物 one-hot，{len(drugs)} 个训练化合物"
    else:
        phi, note = phi_morgan(ctx.meta, args.smiles)
        if phi is None:
            a(f"❗φ=morgan 无法构建：{note}")
            a("   → 化学表示这一路需要 data/external/compound_smiles.csv")
            a("     （列：compound, smiles）。没有它，S1（未见化合物）在结构上")
            a("     就没有任何可外推的信息，只能退化成上下文模型。")
            print("\n".join(L))
            sys.exit(3)
    a(f"φ(d) = {args.phi}：{note}")
    a(f"ψ(u) = 菌株/培养基/温度 one-hot + log-time 三次多项式，维 {psi.shape[1]}")
    a(f"交互秩 r = {args.r_inter}（P、Q 为固定随机投影，整体岭回归 λ={args.lam}）")
    a("")

    Zt = build_targets(ctx, U, mu_d)
    a(f"z_Δ 目标：{Zt.shape}，训练行 {int(ctx.train_mask.sum())}")
    a("")

    a("-" * 100)
    a("消融：加性 vs 加性+交互")
    a("-" * 100)
    cols = ["total", "abs_pcc", "abs_r2", "fc_pcc", "ctx_resid", "drug_resid", "both_time"]
    a(f"  {'配置':<28}" + "".join(f"{c:>11}" for c in cols))
    rows = {}
    for label, use_phi, use_inter in [
        ("仅上下文 ψ（无药物项）", False, False),
        ("加性 φ + ψ", True, False),
        (f"加性 + 交互 r={args.r_inter}", True, True),
    ]:
        p_ = phi if use_phi else np.zeros((len(phi), 1), np.float32)
        r_ = args.r_inter if use_inter else 0
        if r_ == 0:
            F = np.hstack([np.ones((len(psi), 1), np.float32), p_, psi]).astype(np.float64)
            Ft = F[ctx.train_mask]
            W = np.linalg.solve(Ft.T @ Ft + args.lam * np.eye(F.shape[1]),
                                Ft.T @ Zt[ctx.train_mask])
            Zp = F @ W
        else:
            Zp, _ = fit_response(p_, psi, Zt, ctx.train_mask, r_inter=r_, lam=args.lam)
        delta = mu_d[None, :] + Zp.astype(np.float32) @ U.T
        y = _fill(ctx.C + delta, fb)
        f = ev.flatten(ev.evaluate(y, ctx, val, cfg))
        rows[label] = f
        a(f"  {label:<28}" + "".join(f"{f[c]:>11.4f}" for c in cols))
    a("")
    add = rows["加性 φ + ψ"]["total"]
    itr = rows[f"加性 + 交互 r={args.r_inter}"]["total"]
    a(f"  交互项带来的总分变化 {itr - add:+.4f}")
    a("  docs/05 §5.4 的论断是『纯加性拿不到指标 3/4 那 40%』；上表的 ctx_resid /")
    a("  drug_resid 两列就是这个论断的直接检验。")
    a("")

    a("-" * 100)
    a("逐划分（加性+交互）—— S1 看 val_chem_only，S2 看 val_strain_only")
    a("-" * 100)
    if args.phi == "onehot":
        a("⚠ 药物 one-hot 对**未见化合物**整列为 0，模型在 S1 上已退化为上下文模型，")
        a("  故 val_chem_only / val_both 只是下限，不代表化学外推能力。")
    else:
        a("φ=morgan 对未见化合物仍有非零特征，val_chem_only 上的数字才真正回答")
        a("「分子结构能不能外推到没见过的化合物」。与 φ=onehot 的同格数字对比即为增益。")
    a("")
    y_full = _fill(ctx.C + mu_d[None, :] + (fit_response(
        phi, psi, Zt, ctx.train_mask, r_inter=args.r_inter, lam=args.lam)[0]
        ).astype(np.float32) @ U.T, fb)
    sp = ev.evaluate_by_split(y_full, ctx, cfg)
    a("    " + "  ".join(f"{c:>14}" for c in ["split", "n", "abs_pcc", "abs_r2", "fc_pcc"]))
    for _, r in sp.iterrows():
        a("    " + "  ".join([f"{r['split']:>14}", f"{int(r['n']):>14d}",
                              f"{r.get('abs_pcc', np.nan):>14.4f}",
                              f"{r.get('abs_r2', np.nan):>14.4f}",
                              f"{r.get('fc_pcc', np.nan):>14.4f}"]))
    a("")
    # 只在 val_chem_only 上单独打指标 3，这是 S1 的核心问题
    chem = ctx.rows(["val_chem_only"])
    f_chem = ev.flatten(ev.evaluate(y_full, ctx, chem, cfg))
    a(f"  仅 val_chem_only（S1，{int(chem.sum())} 样本）：")
    a(f"    fc_pcc {f_chem['fc_pcc']:.4f}   ctx_resid {f_chem['ctx_resid']:.4f}"
      f"   abs_r2 {f_chem['abs_r2']:.4f}")
    a("")

    txt = "\n".join(L)
    print(txt)
    p = os.path.join(OUT_DIR, f"report_{args.phi}.txt")
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(txt)
    print(f"\n[写出] {p}")


if __name__ == "__main__":
    main()
