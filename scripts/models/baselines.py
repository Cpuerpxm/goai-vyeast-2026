"""第 4 步 · 基线阶梯 B0–B4。

docs/05 §四：**B2 与 B3 的差距是本项目最关键的一个数字**。
若化学近邻显著超过上下文均值，化合物特异信息可提取；不超则指标 3 那 20%
大家都拿不到。

| 编号  | 做法                              | 回答什么 |
|-------|-----------------------------------|----------|
| B0    | 训练集全局均值谱                   | 指标 1 靠蛋白丰度差异能虚高到多少 |
| B1    | ŷ = 匹配对照值（Δ ≡ 0）            | 指标 2 的绝对下限 |
| B2g   | ŷ = 对照 + 训练集总体平均 Δ        | 只学「所有药物的共同应激」能拿多少 |
| B2    | ŷ = 对照 + 同上下文训练药物平均 Δ  | 加上上下文调制能拿多少；指标 3 是否归零 |
| B3o   | ŷ = 对照 + **神谕**最近训练化合物 Δ | 任何「照搬某个训练化合物」策略的**上限** |
| B3    | ŷ = 对照 + Tanimoto 最近训练化合物 Δ | 化学结构信息是否真的可提取（需 SMILES） |
| B4    | ŷ = 对照 + ridge(上下文 + 药物 one-hot) Δ | 常规线性方法水位 |

B3o 是神谕基线：它用留出化合物**自己的真值**去挑最相似的训练化合物，
因此不是合法提交，只用来给 B3 这一族方法定天花板。
若 B3o 相比 B2 增益很小，说明「照搬近邻」这条路本身就没多少可挖，
此时再去找 SMILES 做真 B3 的边际价值也有限。

运行：
    python baselines.py                 # 全部基线 + 官方四类 val
    python baselines.py --bootstrap 100 # 附按药物分组的 bootstrap 区间
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
OUT_DIR = os.path.join(paths.RESULTS, "step4_baselines")
SMILES_CSV = os.path.join(paths.DATA_EXTERNAL, "compound_smiles.csv")


def _nanmean0(M: np.ndarray, rows: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        v = np.nanmean(M[rows], axis=0, dtype=np.float64).astype(np.float32)
    return v


def _fill(y: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """预测里的 NaN 用兜底谱补上——提交必须是完整向量，不能交 NaN。"""
    out = np.where(np.isfinite(y), y, fallback[None, :])
    return np.nan_to_num(out, nan=0.0).astype(np.float32)


# --------------------------------------------------------------- 基线


def b0_global_mean(ctx: ev.EvalContext) -> np.ndarray:
    mu = _nanmean0(ctx.X, ctx.train_mask)
    mu = np.where(np.isfinite(mu), mu, np.float32(np.nanmedian(mu)))
    return np.tile(mu, (ctx.n, 1)).astype(np.float32)


def b1_control(ctx: ev.EvalContext, fb: np.ndarray) -> np.ndarray:
    return _fill(ctx.C, fb)


def b2g_global_delta(ctx: ev.EvalContext, fb: np.ndarray) -> np.ndarray:
    d = _nanmean0(ctx.D, ctx.train_mask)
    d = np.where(np.isfinite(d), d, 0.0)
    return _fill(ctx.C + d[None, :], fb)


def b2_ctx_mean_delta(ctx: ev.EvalContext, fb: np.ndarray) -> np.ndarray:
    return _fill(ctx.C + np.where(np.isfinite(ctx.mu_ctx), ctx.mu_ctx, 0.0), fb)


def _drug_mean_delta(ctx: ev.EvalContext, rows: np.ndarray) -> dict:
    """每个化合物在给定行集合上的平均 Δ。"""
    pert = ctx.meta[PERT_COL].astype(str).to_numpy()
    out = {}
    for d in np.unique(pert[rows]):
        sel = rows & (pert == d)
        if sel.sum() == 0:
            continue
        out[d] = _nanmean0(ctx.D, sel)
    return out


def b3o_oracle_neighbor(ctx: ev.EvalContext, fb: np.ndarray, eval_mask: np.ndarray):
    """神谕近邻：用留出化合物自己的平均 Δ 去训练化合物里挑最像的那个。

    **不是合法提交**，只用于给「照搬训练化合物响应」这一族方法定上限。
    """
    train_prof = _drug_mean_delta(ctx, ctx.train_mask)
    eval_prof = _drug_mean_delta(ctx, eval_mask & ctx.treated)
    names = sorted(train_prof)
    T = np.vstack([train_prof[k] for k in names])

    def corr(a, B):
        m = np.isfinite(a)[None, :] & np.isfinite(B)
        A = np.where(m, a[None, :], 0.0)
        Bz = np.where(m, B, 0.0)
        n = m.sum(1).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            sa, sb = A.sum(1), Bz.sum(1)
            cov = (A * Bz).sum(1) / n - sa * sb / n ** 2
            va = (A * A).sum(1) / n - (sa / n) ** 2
            vb = (Bz * Bz).sum(1) / n - (sb / n) ** 2
            return cov / np.sqrt(np.maximum(va, 1e-12) * np.maximum(vb, 1e-12))

    pert = ctx.meta[PERT_COL].astype(str).to_numpy()
    delta = np.zeros((ctx.n, ctx.X.shape[1]), dtype=np.float32)
    picks = {}
    for d, prof in eval_prof.items():
        cand = [i for i, k in enumerate(names) if k != d]   # 不许挑自己
        r = corr(prof, T[cand])
        best = cand[int(np.nanargmax(r))]
        picks[d] = (names[best], float(np.nanmax(r)))
        delta[pert == d] = np.where(np.isfinite(T[best]), T[best], 0.0)
    return _fill(ctx.C + delta, fb), picks


def b3_chem_neighbor(ctx: ev.EvalContext, fb: np.ndarray, smiles_csv: str):
    """Tanimoto 最近训练化合物的 Δ。需要 data/external/compound_smiles.csv。"""
    if not os.path.exists(smiles_csv):
        return None, f"缺 {smiles_csv}"
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.DataStructs import BulkTanimotoSimilarity
    RDLogger.DisableLog("rdApp.*")

    tab = pd.read_csv(smiles_csv)
    need = {"compound", "smiles"}
    if not need.issubset(tab.columns):
        return None, f"{smiles_csv} 需要列 {need}"
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fp = {}
    for _, r in tab.iterrows():
        s = str(r["smiles"]).strip()
        # ❗空串必须先挡掉：MolFromSmiles("") 返回的是**空分子**而不是 None，
        # 会生成全 0 指纹，与所有化合物的 Tanimoto 都是 0，静默污染近邻。
        if not s or s.lower() == "nan":
            continue
        mol = Chem.MolFromSmiles(s)
        if mol is not None and mol.GetNumAtoms() > 0:
            fp[str(r["compound"])] = gen.GetFingerprint(mol)

    train_prof = _drug_mean_delta(ctx, ctx.train_mask)
    train_names = [k for k in sorted(train_prof) if k in fp]
    if len(train_names) < 5:
        return None, f"只有 {len(train_names)} 个训练化合物有 SMILES，不足以做近邻"
    train_fps = [fp[k] for k in train_names]

    pert = ctx.meta[PERT_COL].astype(str).to_numpy()
    delta = np.zeros((ctx.n, ctx.X.shape[1]), dtype=np.float32)
    picks = {}
    for d in np.unique(pert[ctx.treated]):
        if d not in fp:
            continue
        cand = [i for i, k in enumerate(train_names) if k != d]
        sims = BulkTanimotoSimilarity(fp[d], [train_fps[i] for i in cand])
        best = cand[int(np.argmax(sims))]
        picks[d] = (train_names[best], float(np.max(sims)))
        prof = train_prof[train_names[best]]
        delta[pert == d] = np.where(np.isfinite(prof), prof, 0.0)
    return _fill(ctx.C + delta, fb), picks


def _design(meta: pd.DataFrame) -> np.ndarray:
    """上下文 one-hot + log-time 多项式 + 药物 one-hot。"""
    blocks = [np.ones((len(meta), 1), dtype=np.float32)]
    for c in ["Strains", "Medium", "Temperature", PERT_COL]:
        codes, uniq = pd.factorize(meta[c].astype(str))
        M = np.zeros((len(meta), len(uniq)), dtype=np.float32)
        M[np.arange(len(meta)), codes] = 1.0
        blocks.append(M)
    t = np.log1p(meta["pert_time"].to_numpy(dtype=np.float64))
    t = (t - t.mean()) / t.std()
    blocks.append(np.stack([t, t ** 2, t ** 3], axis=1).astype(np.float32))
    return np.hstack(blocks)


def b4_ridge(ctx: ev.EvalContext, fb: np.ndarray, lam: float = 30.0) -> np.ndarray:
    """岭回归预测 Δ。

    缺失处理：Δ 的缺失位置不进 Z'δ 的求和，再按该蛋白的观测数重标定
    （近似：Gram 矩阵用全部训练行，未按蛋白逐个重算）。Δ 均值 ≈ 0，
    该近似只带来一个温和的均匀收缩，对基线足够。
    """
    Z = _design(ctx.meta)
    tr = np.nonzero(ctx.train_mask)[0]
    Zt = Z[tr].astype(np.float64)
    Dt = ctx.D[tr]
    M = np.isfinite(Dt)
    Y = np.where(M, Dt, 0.0).astype(np.float64)

    G = Zt.T @ Zt + lam * np.eye(Zt.shape[1])
    B = Zt.T @ Y                                    # (k,p)
    n_all = float(len(tr))
    n_obs = M.sum(axis=0).astype(np.float64)
    scale = np.where(n_obs > 0, n_all / np.maximum(n_obs, 1.0), 0.0)
    W = np.linalg.solve(G, B * scale[None, :])
    delta = (Z.astype(np.float64) @ W).astype(np.float32)
    return _fill(ctx.C + delta, fb)


# --------------------------------------------------------------- 主流程


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", type=int, default=0,
                    help="按药物分组 bootstrap 次数（0 = 不做）")
    ap.add_argument("--smiles", default=SMILES_CSV)
    args = ap.parse_args()

    paths.ensure_dir(OUT_DIR)
    cfg = ScorerConfig()
    ctx = ev.build_context()
    val = ctx.rows(ev.VAL_SPLITS)
    fb = b0_global_mean(ctx)[0]

    preds = {}
    preds["B0 全局均值谱"] = b0_global_mean(ctx)
    preds["B1 Δ≡0（=对照）"] = b1_control(ctx, fb)
    preds["B2g 总体平均 Δ"] = b2g_global_delta(ctx, fb)
    preds["B2 上下文均值 Δ"] = b2_ctx_mean_delta(ctx, fb)
    p3o, picks_o = b3o_oracle_neighbor(ctx, fb, val)
    preds["B3o 神谕近邻 Δ ⚠"] = p3o
    p3, info3 = b3_chem_neighbor(ctx, fb, args.smiles)
    if p3 is not None:
        preds["B3 化学近邻 Δ"] = p3
    preds["B4 ridge Δ"] = b4_ridge(ctx, fb)
    preds["【真值】上限参照 ⚠"] = np.nan_to_num(ctx.X, nan=0.0)

    rows = []
    for name, y in preds.items():
        f = ev.flatten(ev.evaluate(y, ctx, val, cfg))
        f["模型"] = name
        rows.append(f)
    df = pd.DataFrame(rows).set_index("模型")

    L: list[str] = []
    a = L.append
    a("=" * 104)
    a("第 4 步 · 基线阶梯（官方四类 val 划分：chem_only / strain_only / both / time）")
    a("=" * 104)
    a("指标分配：绝对=全部 val 行；FC/DEP=全部 val 处理行；")
    a("          上下文残差=val_chem_only；药物残差=val_strain_only；双重/时间=val_both+val_time")
    a("")
    cols = ["total", "abs_pcc", "abs_r2", "fc_pcc", "ctx_resid", "drug_resid",
            "both_time", "dep_dir", "dep_f1"]
    hdr = f"{'模型':<22}" + "".join(f"{c:>11}" for c in cols)
    a(hdr)
    a("-" * len(hdr))
    for name in df.index:
        a(f"{name:<22}" + "".join(f"{df.loc[name, c]:>11.4f}" for c in cols))
    a("")

    # 关键对比
    a("-" * 104)
    a("关键读数")
    a("-" * 104)
    g = lambda n, c: float(df.loc[n, c])
    a(f"1) 指标 1 的虚高地板：B0 只报一条全局均值谱，abs_pcc = {g('B0 全局均值谱','abs_pcc'):.4f}")
    a(f"   → 20% 权重里绝大部分是「蛋白之间丰度差异」白送的，不代表任何扰动理解。")
    a(f"   B0 的 abs_r2 = {g('B0 全局均值谱','abs_r2'):.4f}，R² 才是真正区分模型的那一项。")
    a("")
    a(f"2) 指标 2 的下限：B1（Δ≡0）fc_pcc = {g('B1 Δ≡0（=对照）','fc_pcc'):.4f}")
    a(f"   B2g（只加总体平均 Δ）= {g('B2g 总体平均 Δ','fc_pcc'):.4f}"
      f"   B2（上下文均值 Δ）= {g('B2 上下文均值 Δ','fc_pcc'):.4f}")
    a("")
    b2, b3o = g("B2 上下文均值 Δ", "total"), g("B3o 神谕近邻 Δ ⚠", "total")
    a(f"3) **B2 vs B3 的差距**（本项目最关键的单一数字）")
    a(f"   B2  总分 {b2:.4f}    B3o(神谕上限) 总分 {b3o:.4f}    差 {b3o-b2:+.4f}")
    a(f"   B2  fc  {g('B2 上下文均值 Δ','fc_pcc'):.4f}    B3o fc {g('B3o 神谕近邻 Δ ⚠','fc_pcc'):.4f}"
      f"    差 {g('B3o 神谕近邻 Δ ⚠','fc_pcc')-g('B2 上下文均值 Δ','fc_pcc'):+.4f}")
    a(f"   B2  ctx_resid {g('B2 上下文均值 Δ','ctx_resid'):+.4f}"
      f"    B3o ctx_resid {g('B3o 神谕近邻 Δ ⚠','ctx_resid'):+.4f}")
    a("   B3o 用留出化合物自己的真值挑近邻，是**任何**化学近邻方法的天花板。")
    a("   若这个天花板本身就贴着 B2，说明「照搬某个训练化合物的响应」这条路走不通，")
    a("   化学表示要发挥作用只能靠加权组合/回归，而不是最近邻照搬。")
    if p3 is None:
        a(f"   ❗真 B3（Tanimoto 近邻）未运行：{info3}")
    a("")
    a(f"4) B4 ridge 总分 {g('B4 ridge Δ','total'):.4f}"
      f"（上下文 one-hot + 药物 one-hot；新化合物处药物项归零 → 退化成上下文模型）")
    a("")
    a(f"5) 真值参照（把真值当预测）总分 {g('【真值】上限参照 ⚠','total'):.4f}，")
    a("   它 ≠ 可达上限：真实可达上限受测量噪声限制，见 step3_diagnostics/noise_ceiling.txt。")
    a("")

    a("-" * 104)
    a("6) ⚠ 共享参照污染的实测（docs/05 §8 第一条陷阱）")
    a("-" * 104)
    a(f"   B0 对药物一无所知，却拿到 fc_pcc = {g('B0 全局均值谱','fc_pcc'):.4f}，")
    a(f"   比认真学了上下文的 B2（{g('B2 上下文均值 Δ','fc_pcc'):.4f}）和 B4"
      f"（{g('B4 ridge Δ','fc_pcc'):.4f}）还高。")
    a("   原因：Δ_pred = ŷ − y_ctrl 与 Δ_true = y − y_ctrl 共享同一个真实对照向量。")
    a("   ŷ 越不去追踪该样本自己的对照噪声，Δ_pred 里残留的 −y_ctrl 成分越大，")
    a("   与 Δ_true 里的同一成分对上，白拿相关。**指标 2 不可单独作模型选择依据。**")
    a("")
    a("   把对照向全局均值谱收缩：ŷ = y_ctrl + α·(μ_global − y_ctrl)")
    a("   α=0 就是 B1（完全信对照），α=1 就是 B0（完全不信对照）。")
    a("")
    a(f"   {'α':>6}{'total':>10}{'abs_pcc':>10}{'abs_r2':>10}{'fc_pcc':>10}"
      f"{'ctx_resid':>11}{'drug_resid':>12}")
    mu_g = b0_global_mean(ctx)[0]
    for alpha in [0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.0]:
        y = _fill(ctx.C + alpha * (mu_g[None, :] - ctx.C), fb)
        f = ev.flatten(ev.evaluate(y, ctx, val, cfg))
        a(f"   {alpha:>6.2f}{f['total']:>10.4f}{f['abs_pcc']:>10.4f}{f['abs_r2']:>10.4f}"
          f"{f['fc_pcc']:>10.4f}{f['ctx_resid']:>11.4f}{f['drug_resid']:>12.4f}")
    a("")
    a("   这条曲线是纯粹的『信不信对照』取舍，完全不含任何药物知识。")
    a("   最终模型的融合权重必须**超过**这条曲线才算真的学到了东西。")
    a("")

    a("-" * 104)
    a("神谕近邻挑中了谁（留出化合物 → 最相似的训练化合物，及其 Δ 谱相关）")
    a("-" * 104)
    for d in sorted(picks_o, key=lambda k: -picks_o[k][1])[:20]:
        nb, r = picks_o[d]
        a(f"  {d:<34} → {nb:<34} r = {r:+.3f}")
    a("")

    if p3 is not None:
        a("-" * 104)
        a("7) Tanimoto 挑的近邻 vs 神谕挑的近邻 —— B3 为什么输")
        a("-" * 104)
        a("  神谕用留出化合物自己的 Δ 真值挑；Tanimoto 只看分子结构。")
        a("  两者挑中同一个化合物的比例，就是「结构相似 ⇒ 响应相似」这个假设的成立率。")
        a("")
        a(f"  {'留出化合物':<30}{'神谕挑的':<26}{'Δ相关':>8}   {'Tanimoto挑的':<26}{'Tc':>6}{'该选择的Δ相关':>12}")
        train_prof = _drug_mean_delta(ctx, ctx.train_mask)
        eval_prof = _drug_mean_delta(ctx, val & ctx.treated)

        def _corr1(a_, b_):
            m = np.isfinite(a_) & np.isfinite(b_)
            if m.sum() < 50:
                return np.nan
            x, y = a_[m], b_[m]
            sx, sy = x.std(), y.std()
            if sx < 1e-12 or sy < 1e-12:
                return np.nan
            return float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy))

        same = tot = 0
        for d in sorted(picks_o):
            if d not in info3:
                continue
            onb, orr = picks_o[d]
            cnb, tc = info3[d]
            rc = _corr1(eval_prof[d], train_prof[cnb]) if cnb in train_prof else np.nan
            tot += 1
            same += int(onb == cnb)
            flag = " ✓" if onb == cnb else ""
            a(f"  {d[:29]:<30}{onb[:25]:<26}{orr:>+8.3f}   {cnb[:25]:<26}{tc:>6.2f}{rc:>+12.3f}{flag}")
        a("")
        a(f"  结构近邻与神谕近邻一致的：{same}/{tot}")
        a("  → 一致率低说明在这 43 个化合物的稀疏化学空间里，Tanimoto 相似度**不是**")
        a("    响应相似度的好代理。结构表示要发挥作用，得靠对全部训练化合物加权回归，")
        a("    而不是最近邻照搬；或者改用机制标注（MoA）做类别级收缩。")
        a("")

    a("-" * 104)
    a("逐划分细分（指标 1 与指标 2）")
    a("-" * 104)
    for name, y in preds.items():
        sp = ev.evaluate_by_split(y, ctx, cfg)
        a(f"  {name}")
        a("    " + "  ".join(f"{c:>14}" for c in ["split", "n", "abs_pcc", "abs_r2", "fc_pcc"]))
        for _, r in sp.iterrows():
            a("    " + "  ".join([f"{r['split']:>14}", f"{int(r['n']):>14d}",
                                  f"{r.get('abs_pcc', np.nan):>14.4f}",
                                  f"{r.get('abs_r2', np.nan):>14.4f}",
                                  f"{r.get('fc_pcc', np.nan):>14.4f}"]))
        a("")

    if args.bootstrap > 0:
        a("-" * 104)
        a(f"按药物分组 bootstrap（{args.bootstrap} 次，2.5–97.5% 区间）")
        a("-" * 104)
        a("❗按行 bootstrap 会严重低估不确定性：同一化合物有上百个相关样本。")
        a("")
        a(f"  {'模型':<22}{'total 区间':>28}{'fc_pcc 区间':>28}")
        for name, y in preds.items():
            bs = ev.grouped_bootstrap(y, ctx, val, group_by=PERT_COL,
                                      n_boot=args.bootstrap, cfg=cfg)
            a(f"  {name:<22}[{bs['total'][0]:.4f}, {bs['total'][1]:.4f}]".ljust(52)
              + f"[{bs['fc_pcc'][0]:.4f}, {bs['fc_pcc'][1]:.4f}]")
            print(f"    bootstrap 完成 {name}")
        a("")

    txt = "\n".join(L)
    print(txt)
    df.to_csv(os.path.join(OUT_DIR, "baseline_scores.csv"), encoding="utf-8-sig")
    fp = os.path.join(OUT_DIR, "report.txt")
    with open(fp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(txt)
    print(f"\n[写出] {fp}")
    print(f"[写出] {os.path.join(OUT_DIR, 'baseline_scores.csv')}")


if __name__ == "__main__":
    main()
