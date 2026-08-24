"""未见菌株的效应搬运：把训练菌株的效应按 SNP 距离加权，搬到没见过的菌株上。

问题在哪。C-free 骨架把菌株编成 one-hot；碰到训练里没有的菌株（S2），这一整块
是 0，预测只剩截距。而训练菌株拿的是「截距 + 该株系数」。岭回归下这些系数的
平均并不为 0，所以未见菌株被系统性地推离了「平均菌株」。test 里有一株训练时
从未出现过的菌株，占 50.1%，它就吃这个亏。

做法只改**编码**，不改模型：把未见菌株写成训练菌株的一个凸组合

    未见菌株  ->  0.87 x 最近的训练菌株 + 0.12 x 次近 + 0.01 x 最远
                （权重来自 SNP 距离核；具体代号运行时从 metadata 读，源码不写）

容量锁死在两个标量（手册没给容量限制，是数据给的——合法训练菌株只有 4 株）：

    w(s*, s) = exp(-(d(s*,s)/h)^2)，归一化后整体乘收缩系数 gamma

`gamma = 0` 就是现状（整块 0），`gamma = 1` 就是完全按核权重顶上去。

**三个必须同时跑的对照**，缺一个结论就不成立：

| 方案 | 含义 | 它排除什么 |
|---|---|---|
| `zero` | 未见菌株整块 0 | 现状，作基准 |
| `uniform` | 有坐标的训练菌株**等权** | 排除「把 0 换成平均值」本身带来的收益——这部分与基因组无关 |
| `snp` | SNP 距离核加权 | 真正要检验的东西 |
| `null` | 把目标菌株换成面板里**随机一株**的距离行 | 排除「任何一株的坐标都行」 |

选参只用 train 折内的留一菌株（LOSO）；整株落在 val 的那一株只在最后一次性验证。

⚠ 一个不能回避的不对称（见 `data/strain_genome.py`，数字由脚本现算）：
未见菌株到最近供体 0.398，落在面板距离分布的 14.6% 分位，三个供体的距离跨度 1.70
——加权在这一格差别极大；而唯一能验证的那一株（整株在 val）到最近供体 1.362，
分位 73.8%，跨度只有 0.41，加权几乎不起作用。所以 val 那一格的结果只能证明
「搬运机制有没有用」，**不能替未见菌株那一格的基因组加权背书**。材料里必须这么写。

运行：
    python strain_transport.py                 # LOSO 选参 + val 菌株一次性验证
    python strain_transport.py --skip-final     # 只跑 LOSO
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

from data import paths, provenance, strain_genome
from data import split_guard as sg
from models.design import FEATURE_SETS, PERT_COL, encode, freeze
from models.lowrank import masked_pca
from scorer import evaluate as ev
from scorer.config import ScorerConfig
from scorer.metrics import metric_absolute, metric_dep, metric_fc, metric_residual

OUT_DIR = os.path.join(paths.RESULTS, "step11_strain_transport")
CAT_COLS = FEATURE_SETS["bio_tech"]
STRAIN_COL = "Strains"

BANDWIDTHS = [0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0]
GAMMAS = [0.0, 0.25, 0.5, 0.75, 1.0]


# ------------------------------------------------------------------ 拟合与预测


def fit_once(ctx, fit_rows, k0: int, lam: float) -> dict:
    """在 fit_rows 上冻结设计矩阵并拟合低秩 + 岭回归。

    ❗关键性质（也是这个脚本能在几分钟内跑完网格的原因）：
    留出菌株的行不在 fit_rows 里，所以**软权重完全不影响拟合**，
    只影响预测时怎么给留出行编码。因此每折只拟合一次，网格在预测端展开。
    """
    fit_rows = sg.assert_train_only(ctx.meta, fit_rows, what="菌株搬运实验拟合行")
    spec = freeze(ctx.meta, fit_rows, CAT_COLS, with_drug=False)
    Z = encode(ctx.meta, spec)                       # 拟合行不含未见菌株
    Xtr = ctx.X[fit_rows]
    mu, U, Ztr = masked_pca(Xtr, k0, np.isfinite(Xtr), n_iter=12, center=True, seed=0)
    Zd = Z[fit_rows].astype(np.float64)
    W = np.linalg.solve(Zd.T @ Zd + lam * np.eye(Z.shape[1]), Zd.T @ Ztr.astype(np.float64))
    n_obs = np.isfinite(Xtr).sum(axis=0)
    dead = n_obs == 0
    fallback = float(np.median(mu[~dead])) if (~dead).any() else 0.0
    return {"spec": spec, "mu": mu, "U": U, "W": W, "dead": dead, "fallback": fallback,
            "fit_rows": fit_rows}


def predict_with_soft(model: dict, meta: pd.DataFrame, soft: dict | None) -> np.ndarray:
    Z = encode(meta, model["spec"], soft_levels=soft)
    mu, U, W = model["mu"], model["U"], model["W"]
    y = (np.nan_to_num(mu, nan=model["fallback"])[None, :]
         + (Z.astype(np.float64) @ W) @ U.T.astype(np.float64)).astype(np.float32)
    y[:, model["dead"]] = model["fallback"]
    return y


def soft_for(target: str, donors, Dm, scheme: str, h: float, gamma: float,
             rng: np.random.Generator | None = None, panel=None) -> dict:
    """给一个未见菌株算软权重。返回 {} 表示整块 0（即现状）。"""
    if gamma <= 0 or scheme == "zero":
        return {}
    if scheme == "uniform":
        w = strain_genome.uniform_weights(list(donors))
    elif scheme == "snp":
        w = strain_genome.kernel_weights(target, list(donors), Dm, h)
    elif scheme == "null":
        # 零假设：目标菌株的基因组身份换成面板里随机一株，看权重还灵不灵
        pick = str(rng.choice(panel))
        w = strain_genome.kernel_weights(pick, list(donors), Dm, h)
    else:
        raise ValueError(scheme)
    return {k: gamma * v for k, v in w.items()} if w else {}


# ------------------------------------------------------------------ 打分


def score_rows(ctx, y, rows, mu_drug, cfg) -> dict:
    """留出菌株那些行上的分项。drug_resid 就是官方指标 4（S2 那 20%）。"""
    r = np.nonzero(rows)[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        dp = y[r].astype(np.float64) - ctx.C[r].astype(np.float64)
        ab = metric_absolute(ctx.X[r], y[r], cfg)
        out = {
            "abs_pcc": ab["pcc"], "abs_r2": ab["r2"],
            "fc_pcc": metric_fc(ctx.D[r], dp, cfg)["pcc"],
            "drug_resid": metric_residual(ctx.D[r], dp, mu_drug[r], cfg)["pcc"],
            "dep_dir": metric_dep(ctx.D[r], dp, cfg).get("direction_acc", np.nan),
            "n": int(rows.sum()),
        }
    return {k: (float(v) if k != "n" else v) for k, v in out.items()}


def frozen_mu_drug(ctx, fit_treated) -> np.ndarray:
    """按折重新冻结 mu_drug：留出菌株的行不许参与自己的参照均值。"""
    key = ctx.meta[PERT_COL].astype(str).to_numpy()
    mu, _ = ev._group_mean_frozen(ctx.D, key, fit_treated)
    return mu


# ------------------------------------------------------------------ 主流程


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k0", type=int, default=16)
    ap.add_argument("--lam", type=float, default=30.0)
    ap.add_argument("--n-null", type=int, default=20, help="随机坐标对照次数")
    ap.add_argument("--skip-final", action="store_true")
    args = ap.parse_args()

    paths.ensure_dir(OUT_DIR)
    cfg = ScorerConfig()
    ctx = ev.build_context(verbose=False)
    meta = ctx.meta
    strains = meta[STRAIN_COL].astype(str).to_numpy()
    train_rows = sg.train_rows(meta)
    train_strains = sorted(set(strains[train_rows]))

    Dm, ginfo = strain_genome.contest_submatrix()
    panel = strain_genome.load_matrix().index.tolist()
    coord_train = [s for s in train_strains if s in Dm.index]
    roles = ginfo["roles"]
    val_strain = roles["val_only"][0]          # 整株落在 val 的那一株
    unseen_strain = roles["test_only"][0]      # 只出现在 test 的那一株

    L: list[str] = []
    a = L.append
    a("=" * 100)
    a("未见菌株的效应搬运：SNP 距离核加权（2026-08-24）")
    a("=" * 100)
    a(f"训练菌株 {train_strains}；其中有 1011 面板坐标的 {coord_train}")
    a(f"未命中面板：{[s for s in train_strains if s not in Dm.index]}"
      "（实验室菌株命名，不属野生分离株面板；面板里也找不到可辨认的参考株条目）")
    a(f"外部资源：{ginfo['citation']}")
    a(f"  文件 {ginfo['matrix_file']}  SHA-256 {ginfo['matrix_sha256'][:16]}…"
      f"  面板 {ginfo['panel_size']} 株")
    a("")
    a("赛题菌株两两 SNP 距离：")
    for line in Dm.round(4).to_string().splitlines():
        a("  " + line)
    pct = ginfo["panel_distance_percentiles"]
    d_unseen = float(Dm.loc[unseen_strain, coord_train].min())
    d_val = float(Dm.loc[val_strain, coord_train].min())
    q_unseen = strain_genome.panel_quantile(d_unseen)
    q_val = strain_genome.panel_quantile(d_val)
    sp_unseen = float(Dm.loc[unseen_strain, coord_train].max() - d_unseen)
    sp_val = float(Dm.loc[val_strain, coord_train].max() - d_val)
    a("")
    a(f"  面板全体距离分位：p5 {pct['p5']:.3f} · 中位 {pct['p50']:.3f} · p95 {pct['p95']:.3f}")
    a(f"  未见菌株（占 test 50.1%）到最近供体 {d_unseen:.3f}，分位 {q_unseen:.1%}；"
      f"三供体距离跨度 {sp_unseen:.3f} → 加权在这一格差别极大")
    a(f"  val 菌株到最近供体 {d_val:.3f}，分位 {q_val:.1%}；"
      f"跨度 {sp_val:.3f} → 加权在这一格几乎不起作用")
    a("")
    a("⚠ 因此：加权最可能起作用的那一格（未见菌株）无法验证；")
    a("  唯一能验证的那一格对加权本身不敏感。这个不对称是数据给的，不是选择。")
    a("")

    # ---------------------------------------------------------- LOSO 选参
    a("-" * 100)
    a("一 · train 折内留一菌株（LOSO）选 (h, gamma)")
    a("-" * 100)
    a("只有有坐标的训练菌株才能当留出目标；每折的供体池 = 该折其余**有坐标**的训练菌株。")
    a("三个方案共用同一个供体池，所以差异只来自权重本身，不来自池子大小。")
    a("")

    folds = []
    for s in coord_train:
        held = train_rows & (strains == s)
        fit = train_rows & (strains != s)
        donors = [d for d in coord_train if d != s]
        if held.sum() < 50 or not donors:
            a(f"  跳过 {s}：留出 {int(held.sum())} 行 / 供体 {donors}")
            continue
        print(f"[搬运] 折 {s}: 留出 {int(held.sum())} 行，拟合 {int(fit.sum())} 行，供体 {donors}")
        model = fit_once(ctx, fit, args.k0, args.lam)
        mu_drug = frozen_mu_drug(ctx, fit & ctx.treated)
        folds.append({"s": s, "held": held, "fit": fit, "donors": donors,
                      "model": model, "mu_drug": mu_drug})
    if not folds:
        raise SystemExit("没有可用的 LOSO 折")

    # 留出菌株在该折里是「未见水平」，freeze 时不在词表内 -> 软编码合法
    grid = []
    for fd in folds:
        base = score_rows(ctx, predict_with_soft(fd["model"], meta, None),
                          fd["held"], fd["mu_drug"], cfg)
        grid.append({"strain": fd["s"], "scheme": "zero", "h": None, "gamma": 0.0, **base})
    for scheme in ("uniform", "snp"):
        for g in GAMMAS:
            if g == 0:
                continue
            hs = [None] if scheme == "uniform" else BANDWIDTHS
            for h in hs:
                for fd in folds:
                    soft = {STRAIN_COL: {fd["s"]: soft_for(
                        fd["s"], fd["donors"], Dm, scheme, h or 1.0, g)}}
                    if not soft[STRAIN_COL][fd["s"]]:
                        continue
                    y = predict_with_soft(fd["model"], meta, soft)
                    grid.append({"strain": fd["s"], "scheme": scheme, "h": h, "gamma": g,
                                 **score_rows(ctx, y, fd["held"], fd["mu_drug"], cfg)})
    G = pd.DataFrame(grid)
    agg = (G.groupby(["scheme", "h", "gamma"], dropna=False)
             [["abs_r2", "fc_pcc", "drug_resid", "abs_pcc"]].mean().reset_index())
    zero = agg[agg.scheme == "zero"].iloc[0]

    a(f"  基准（zero，现状）：drug_resid {zero['drug_resid']:+.4f}   "
      f"abs_r2 {zero['abs_r2']:.4f}   fc_pcc {zero['fc_pcc']:.4f}")
    a("")
    a(f"  {'方案':<10}{'h':>6}{'gamma':>7}{'drug_resid':>12}{'Δ vs zero':>11}"
      f"{'abs_r2':>10}{'Δ abs_r2':>10}")
    for _, r in agg[agg.scheme != "zero"].sort_values(
            "drug_resid", ascending=False).head(18).iterrows():
        a(f"  {r['scheme']:<10}{('-' if pd.isna(r['h']) else f'{r.h:g}'):>6}"
          f"{r['gamma']:>7.2f}{r['drug_resid']:>12.4f}"
          f"{r['drug_resid'] - zero['drug_resid']:>+11.4f}"
          f"{r['abs_r2']:>10.4f}{r['abs_r2'] - zero['abs_r2']:>+10.4f}")
    a("")

    # 选参：drug_resid 最高，且 abs_r2 不得比 zero 掉超过 0.005（CURRENT.md 硬判据）
    cand = agg[(agg.scheme != "zero") & (agg.abs_r2 >= zero["abs_r2"] - 0.005)]
    if cand.empty:
        a("  ⚠ 没有任何配置能在不掉 abs_r2 的前提下提高 drug_resid")
        best_snp = best_uni = None
    else:
        best_uni = (cand[cand.scheme == "uniform"]
                    .sort_values("drug_resid", ascending=False).head(1))
        best_snp = (cand[cand.scheme == "snp"]
                    .sort_values("drug_resid", ascending=False).head(1))
        best_uni = None if best_uni.empty else best_uni.iloc[0]
        best_snp = None if best_snp.empty else best_snp.iloc[0]
        a("  LOSO 选出（判据：drug_resid 最高，且 abs_r2 相对 zero 不掉超过 0.005）")
        for lab, b in [("等权对照", best_uni), ("SNP 核", best_snp)]:
            if b is None:
                a(f"    {lab}：无合格配置")
            else:
                a(f"    {lab}：h={'-' if pd.isna(b['h']) else f'{b.h:g}'} "
                  f"gamma={b['gamma']:.2f}  drug_resid {b['drug_resid']:+.4f}"
                  f"（相对 zero {b['drug_resid'] - zero['drug_resid']:+.4f}）")
        if best_snp is not None and best_uni is not None:
            a(f"    SNP 相对等权的增量：{best_snp['drug_resid'] - best_uni['drug_resid']:+.4f}"
              "  ← 这一项才是「基因组信息本身」的贡献")
    a("")

    # 逐折明细：3 折里哪几折的权重是真有对比度
    a("  逐折明细（选中的 SNP 配置）")
    if best_snp is not None:
        a(f"  {'留出菌株':<10}{'供体与权重':<46}{'drug_resid':>12}{'Δ vs zero':>11}")
        for fd in folds:
            w = strain_genome.kernel_weights(fd["s"], fd["donors"], Dm,
                                             float(best_snp["h"]))
            z = G[(G.strain == fd["s"]) & (G.scheme == "zero")].iloc[0]
            row = G[(G.strain == fd["s"]) & (G.scheme == "snp")
                    & (G.h == best_snp["h"]) & (G.gamma == best_snp["gamma"])]
            v = float(row.iloc[0]["drug_resid"]) if len(row) else np.nan
            ws = " / ".join(f"{k} {x:.2f}" for k, x in sorted(w.items(), key=lambda t: -t[1]))
            a(f"  {fd['s']:<10}{ws:<46}{v:>12.4f}{v - z['drug_resid']:>+11.4f}")
    a("")

    # ------------------------------------------------ val 菌株一次性验证
    final = {}
    if not args.skip_final and best_snp is not None:
        a("-" * 100)
        a("二 · 官方 val_strain_only 一次性验证 —— 只看一眼，不回头调参")
        a("-" * 100)
        model = fit_once(ctx, train_rows, args.k0, args.lam)
        donors = coord_train
        val_s = ctx.rows(["val_strain_only"])
        val_c = ctx.rows(["val_chem_only"])
        rng = np.random.default_rng(20260824)
        pool = [p for p in panel if p not in donors]

        def _eval(soft, s2_only: bool = False):
            """s2_only：随机坐标对照只看 val_strain_only 的 drug_resid，
            没必要把六项和 S1 都算一遍（20 次重复，省下来的是分钟级）。"""
            y = predict_with_soft(model, meta, soft)
            f_s2 = ev.flatten(ev.evaluate(y, ctx, val_s, cfg))
            if s2_only:
                return None, f_s2, None
            f_all = ev.flatten(ev.evaluate(y, ctx, ctx.rows(ev.VAL_SPLITS), cfg))
            f_s1 = ev.flatten(ev.evaluate(y, ctx, val_c, cfg))
            return f_all, f_s2, f_s1

        variants = {"zero": {}}
        if best_uni is not None:
            variants["uniform"] = {STRAIN_COL: {val_strain: soft_for(
                val_strain, donors, Dm, "uniform", 1.0, float(best_uni["gamma"]))}}
        variants["snp"] = {STRAIN_COL: {val_strain: soft_for(
            val_strain, donors, Dm, "snp", float(best_snp["h"]), float(best_snp["gamma"]))}}

        a(f"  {'方案':<10}{'六项总分':>10}{'drug_resid':>12}{'abs_r2':>10}"
          f"{'S1 abs_r2':>11}{'权重':<34}")
        base_all = base_s2 = base_s1 = None
        for name, soft in variants.items():
            f_all, f_s2, f_s1 = _eval(soft or None)
            if name == "zero":
                base_all, base_s2, base_s1 = f_all, f_s2, f_s1
            w = soft.get(STRAIN_COL, {}).get(val_strain, {})
            ws = " / ".join(f"{k} {v:.2f}" for k, v in sorted(w.items(), key=lambda t: -t[1])) or "-"
            a(f"  {name:<10}{f_all['total']:>10.4f}{f_s2['drug_resid']:>12.4f}"
              f"{f_all['abs_r2']:>10.4f}{f_s1['abs_r2']:>11.4f}  {ws:<34}")
            final[name] = {"total": f_all["total"], "drug_resid": f_s2["drug_resid"],
                           "abs_r2": f_all["abs_r2"], "s1_abs_r2": f_s1["abs_r2"],
                           "weights": w}

        # 随机坐标对照
        nulls = []
        for i in range(args.n_null):
            soft = {STRAIN_COL: {val_strain: soft_for(
                val_strain, donors, Dm, "null", float(best_snp["h"]),
                float(best_snp["gamma"]), rng=rng, panel=pool)}}
            if not soft[STRAIN_COL][val_strain]:
                continue
            _, f_s2, _ = _eval(soft, s2_only=True)
            nulls.append(f_s2["drug_resid"])
        if nulls:
            nz = np.asarray(nulls)
            snp_v = final["snp"]["drug_resid"]
            p = float((nz >= snp_v).mean())
            a("")
            a(f"  随机坐标对照 {len(nulls)} 次（把 val 菌株换成面板里随机一株的距离行）：")
            a(f"    drug_resid 均值 {nz.mean():+.4f}  最大 {nz.max():+.4f}  "
              f"真实坐标 {snp_v:+.4f}  经验 p = {p:.3f}")
            final["null"] = {"n": len(nulls), "mean": float(nz.mean()),
                             "max": float(nz.max()), "p_value": p,
                             "values": [float(x) for x in nz]}

        # Day 5 硬判据
        d_drug = final["snp"]["drug_resid"] - base_s2["drug_resid"]
        d_tot = final["snp"]["total"] - base_all["total"]
        d_abs = final["snp"]["abs_r2"] - base_all["abs_r2"]
        d_s1 = final["snp"]["s1_abs_r2"] - base_s1["abs_r2"]
        passed = ((d_drug >= 0.015 or d_tot >= 0.003)
                  and d_abs >= -0.005 and d_s1 >= -0.005)
        a("")
        a("  Day 5 硬判据（_handoff/CURRENT.md 预先写死，不是事后定的）：")
        a(f"    drug_resid 增量 {d_drug:+.4f}（要求 ≥ +0.015）")
        a(f"    六项总分增量   {d_tot:+.4f}（要求 ≥ +0.003，与上一条满足其一即可）")
        a(f"    abs_r2 变化    {d_abs:+.4f}（要求 ≥ -0.005）")
        a(f"    S1 abs_r2 变化 {d_s1:+.4f}（要求 ≥ -0.005）")
        a(f"    => {'保留' if passed else '不达标，按预案永久停，不再调参'}")
        final["verdict"] = {"passed": bool(passed), "delta_drug_resid": d_drug,
                            "delta_total": d_tot, "delta_abs_r2": d_abs,
                            "delta_s1_abs_r2": d_s1}
        a("")
        a("  ⚠ 无论上面是什么结论，都不能外推到未见菌株：val 菌株到三株供体近乎等距，")
        a("    这一格检验的是「搬运机制」，不是「基因组加权」。未见菌株那种近缘情形")
        a("    在 train_val 里根本不存在，无从验证。")

    txt = "\n".join(L)
    print(txt)
    p = os.path.join(OUT_DIR, "report.txt")
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(txt)
    G.to_csv(os.path.join(OUT_DIR, "loso_grid.csv"), index=False, encoding="utf-8-sig")
    payload = {
        "_schema": "strain_transport/1.0",
        "external_resource": ginfo,
        "train_strains": train_strains,
        "donors_with_coordinates": coord_train,
        "distances": Dm.round(6).to_dict(),
        "unseen_nearest_donor_distance": d_unseen,
        "unseen_panel_quantile": q_unseen,
        "unseen_donor_distance_spread": sp_unseen,
        "val_nearest_donor_distance": d_val,
        "val_panel_quantile": q_val,
        "val_donor_distance_spread": sp_val,
        "loso_zero_baseline": {k: float(zero[k]) for k in
                               ("abs_r2", "fc_pcc", "drug_resid", "abs_pcc")},
        "loso_best_uniform": (None if best_uni is None else
                              {k: (None if pd.isna(best_uni[k]) else float(best_uni[k]))
                               if k != "scheme" else best_uni[k]
                               for k in ("scheme", "h", "gamma", "drug_resid", "abs_r2")}),
        "loso_best_snp": (None if best_snp is None else
                          {k: (None if pd.isna(best_snp[k]) else float(best_snp[k]))
                           if k != "scheme" else best_snp[k]
                           for k in ("scheme", "h", "gamma", "drug_resid", "abs_r2")}),
        "final_val_strain_only": final,
        "_provenance": provenance.stamp(),
    }
    pj = os.path.join(OUT_DIR, "strain_transport.json")
    with open(pj, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)
    print(f"\n[写出] {p}\n[写出] {pj}")


if __name__ == "__main__":
    main()
