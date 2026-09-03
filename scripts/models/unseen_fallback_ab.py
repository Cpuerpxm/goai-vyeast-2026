"""未见水平的回退方式：整块归零 vs 按训练频率软编码。

问题是纯代数的，`strain_transport.py` 开头就点出来了：
one-hot 加岭回归，碰到训练里没有的水平，整块是 0，预测只剩截距。
而训练水平拿的是「截距 + 该水平系数」，这些系数在岭回归下平均并不为 0。
于是未见水平被系统性地推离了「平均水平」，推的方向和幅度都不受控。

正确的回退应该是「训练总体的平均」，也就是把该块写成训练频率向量 p：

    未见水平  ->  Σ_k p_k · (水平 k 的 one-hot)

这样预测就是 b0 + Σ_k p_k b_k，正好是训练总体均值。
`design.encode` 的 soft_levels 接口本来就支持，只是一直没这么用过。

与 2026-08-24 那次菌株搬运的区别：那次比的是 SNP 距离核 vs 等权，
等权还只在**有面板坐标的 3 株**上取，不含 STRAIN_D；结论是搬运不保留。
本轮不用基因组，权重就是训练频率，覆盖全部训练水平，且同时试药物列。

test 里未见水平的规模：菌株 2,663 行 / 4,454，药物 2,769 行 / 4,454。
"""
from __future__ import annotations
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import paths, split_guard as sg
from models.baseline_cfree import masked_ridge_exact
from models.design import freeze, encode, FEATURE_SETS, PERT_COL
from scorer import evaluate as ev
from scorer.config import ScorerConfig

COLS = ["total", "abs_pcc", "abs_r2", "fc_pcc", "ctx_resid", "drug_resid"]
LAM_CTX, LAM_DRUG = 3.0, 300.0


def freq_soft(meta, fit_mask, spec, cols, gamma=1.0):
    """给 cols 里每个未见水平配一份训练频率权重，整体乘收缩系数 gamma。"""
    out = {}
    sub = meta.loc[fit_mask]
    for c in cols:
        seen = spec["levels"][c]
        vc = sub[c].astype(str).value_counts()
        p = np.array([vc.get(lv, 0) for lv in seen], dtype=np.float64)
        if p.sum() <= 0:
            continue
        p = p / p.sum() * float(gamma)
        w = {lv: float(v) for lv, v in zip(seen, p) if v > 0}
        unseen = sorted(set(meta[c].astype(str)) - set(seen))
        if unseen:
            out[c] = {u: dict(w) for u in unseen}
    return out


def run(ctx, Z, tr, n_ctx, n_drug, val, cfg):
    lam = np.concatenate([np.full(n_ctx, LAM_CTX), np.full(n_drug, LAM_DRUG)])
    mu, W = masked_ridge_exact(Z, ctx.X, tr, lam, ctx.meta)
    y = (mu[None, :] + Z.astype(np.float64) @ W).astype(np.float32)
    y = np.nan_to_num(y, nan=float(np.nanmedian(mu)))
    return ev.flatten(ev.evaluate(y, ctx, val, cfg)), y


def main():
    cfg = ScorerConfig()
    ctx = ev.build_context(verbose=False)
    val = ctx.rows(ev.VAL_SPLITS)
    tr = sg.train_rows(ctx.meta)
    cols = FEATURE_SETS["bio_tech"]

    spec_ctx = freeze(ctx.meta, tr, cols, with_drug=False)
    n_ctx = encode(ctx.meta, spec_ctx).shape[1]
    spec = freeze(ctx.meta, tr, cols, with_drug=True)
    n_drug = spec["n_cols"] - n_ctx

    print(f"设计 {spec['n_cols']} 列 = 上下文 {n_ctx} + 药物 {n_drug}")
    for c in list(cols):
        unseen = sorted(set(ctx.meta[c].astype(str)) - set(spec["levels"][c]))
        if unseen:
            n = int(ctx.meta[c].astype(str).isin(unseen).sum())
            print(f"  train_val 里 {c} 未见水平 {unseen} 覆盖 {n} 行")
    print()

    print(f"{'回退方式':<40}" + "".join(f"{c:>11}" for c in COLS))
    print("-" * (40 + 11 * len(COLS)))

    Z0 = encode(ctx.meta, spec)
    f0, _ = run(ctx, Z0, tr, n_ctx, n_drug, val, cfg)
    print(f"{'整块归零（当前）':<40}" + "".join(f"{f0[c]:>11.4f}" for c in COLS))
    base = f0["total"]

    best = ("整块归零（当前）", base, f0)

    # 药物块不在 spec["cat_cols"] 里，encode 的 soft_levels 接口够不着它，
    # 所以直接改设计矩阵：末尾 n_drug 列就是药物 one-hot（n_cols = 1 + Σ|levels| + 3 + n_drug）。
    dseen = spec["drugs"]
    dcol = ctx.meta[PERT_COL].astype(str).to_numpy()
    unseen_rows = ~np.isin(dcol, dseen)
    cnt = ctx.meta.loc[tr, PERT_COL].astype(str).value_counts()
    p_drug = np.array([cnt.get(d, 0) for d in dseen], dtype=np.float64)
    p_drug = p_drug / p_drug.sum()
    print(f"train_val 里未见药物覆盖 {int(unseen_rows.sum())} 行")
    print()

    for gamma in (0.25, 0.5, 1.0):
        Z = Z0.copy()
        Z[unseen_rows, -n_drug:] = (p_drug * gamma).astype(Z.dtype)
        f, _ = run(ctx, Z, tr, n_ctx, n_drug, val, cfg)
        name = f"未见药物按训练频率回退 γ={gamma:g}"
        mark = ""
        if f["total"] > best[1]:
            best = (name, f["total"], f); mark = "  ←"
        print(f"{name:<40}" + "".join(f"{f[c]:>11.4f}" for c in COLS) + mark)

    # 菌株那一格上面已经试过，负的；这里再试菌株与药物同时回退
    sl = freq_soft(ctx.meta, tr, spec, ["Strains"], 0.5)
    if sl:
        Zs = encode(ctx.meta, spec, soft_levels=sl)
        Zs[unseen_rows, -n_drug:] = (p_drug * 0.5).astype(Zs.dtype)
        f, _ = run(ctx, Zs, tr, n_ctx, n_drug, val, cfg)
        name = "菌株 γ=0.5 + 药物 γ=0.5 同时回退"
        mark = ""
        if f["total"] > best[1]:
            best = (name, f["total"], f); mark = "  ←"
        print(f"{name:<40}" + "".join(f"{f[c]:>11.4f}" for c in COLS) + mark)

    d = best[1] - base
    print(f"\n最优：{best[0]}  total {best[1]:.4f}  相对当前 {d:+.4f}")
    print(f"止损线 +0.0007 → {'通过' if d >= 0.0007 else '未通过，不采用'}")
    if d >= 0.0007:
        for k, thr in (("abs_r2", -0.0015), ("ctx_resid", -0.003), ("drug_resid", -0.003)):
            dv = best[2][k] - f0[k]
            print(f"  {k} 变化 {dv:+.4f}（门槛 ≥ {thr}） {'通过' if dv >= thr else '不通过'}")

    out = os.path.join(paths.RESULTS, "step18_unseen_fallback"); paths.ensure_dir(out)
    with open(os.path.join(out, "report.txt"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"整块归零 total {base:.6f}\n{best[0]} total {best[1]:.6f}  {d:+.6f}\n")
    print(f"写出 {out}/report.txt")


if __name__ == "__main__":
    main()
