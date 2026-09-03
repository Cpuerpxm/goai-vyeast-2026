"""L1-01：处理角色 + 强收缩药物残差（GPT Pro 会诊 2026-09-02 第一优先提分项）。

依据：同 tag 探索日志 step7.log 记录满秩 C-free 加药物项后 0.4817 → 0.4841。
本轮在契约空间 + 逐蛋白 Gram 的新基线上重做，并按 Pro 的建议给药物列单独的强正则。

未见药物：one-hot 全 0，自动退回上下文预测，不冒充任何已见药物。
"""
from __future__ import annotations
import os, sys, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import paths, split_guard as sg
from models.baseline_cfree import masked_ridge_exact
from models.design import freeze, encode, FEATURE_SETS
from scorer import evaluate as ev
from scorer.config import ScorerConfig

COLS = ["total", "abs_pcc", "abs_r2", "fc_pcc", "ctx_resid", "drug_resid"]
BASE_LAM = 3.0          # step14 细扫结果

def run(ctx, Z, tr, lam_vec, val, cfg):
    mu, W = masked_ridge_exact(Z, ctx.X, tr, lam_vec, ctx.meta)
    y = (mu[None, :] + Z.astype(np.float64) @ W).astype(np.float32)
    y = np.nan_to_num(y, nan=float(np.nanmedian(mu)))
    return ev.flatten(ev.evaluate(y, ctx, val, cfg))

def main():
    cfg = ScorerConfig()
    ctx = ev.build_context(verbose=False)
    val = ctx.rows(ev.VAL_SPLITS)
    tr = sg.train_rows(ctx.meta)
    cols = FEATURE_SETS["bio_tech"]

    print(f"{'配置':<34}" + "".join(f"{c:>11}" for c in COLS))
    print("-" * (34 + 11 * len(COLS)))

    # 基线：无药物列
    spec0 = freeze(ctx.meta, tr, cols, with_drug=False)
    Z0 = encode(ctx.meta, spec0)
    f0 = run(ctx, Z0, tr, BASE_LAM, val, cfg)
    print(f"{'基线 无药物列 λ=3':<34}" + "".join(f"{f0[c]:>11.4f}" for c in COLS))
    base = f0["total"]

    # 带药物列
    spec1 = freeze(ctx.meta, tr, cols, with_drug=True)
    Z1 = encode(ctx.meta, spec1)
    n_ctx = Z0.shape[1]
    n_drug = Z1.shape[1] - n_ctx
    print(f"\n设计矩阵：上下文 {n_ctx} 列 + 药物 {n_drug} 列 = {Z1.shape[1]} 列")
    print(f"训练药物 {len(spec1['drugs'])} 个\n")

    best = ("基线", base, f0)
    for lam_drug in [3.0, 30.0, 100.0, 300.0, 1000.0, 3000.0]:
        lam_vec = np.concatenate([np.full(n_ctx, BASE_LAM), np.full(n_drug, lam_drug)])
        f = run(ctx, Z1, tr, lam_vec, val, cfg)
        tag = f"药物列 λ_drug={lam_drug:g}"
        mark = ""
        if f["total"] > best[1]:
            best = (tag, f["total"], f); mark = "  ←"
        print(f"{tag:<34}" + "".join(f"{f[c]:>11.4f}" for c in COLS) + mark)

    d = best[1] - base
    print(f"\n最优：{best[0]}  total {best[1]:.4f}  相对基线 {d:+.4f}")
    print(f"止损线 +0.0007 → {'通过' if d >= 0.0007 else '未通过，药物项不采用'}")
    if d >= 0.0007:
        bf = best[2]
        print(f"  abs_r2 变化 {bf['abs_r2']-f0['abs_r2']:+.4f}（门槛 ≥ -0.0015）")
        print(f"  ctx_resid 变化 {bf['ctx_resid']-f0['ctx_resid']:+.4f}（门槛 ≥ -0.003）")
        print(f"  drug_resid 变化 {bf['drug_resid']-f0['drug_resid']:+.4f}（门槛 ≥ -0.003）")

    out = os.path.join(paths.RESULTS, "step15_drug_adapter"); paths.ensure_dir(out)
    with open(os.path.join(out, "report.txt"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"基线(无药物) total {base:.6f}\n最优 {best[0]} total {best[1]:.6f}  {d:+.6f}\n")
    print(f"写出 {out}/report.txt")

if __name__ == "__main__":
    main()
