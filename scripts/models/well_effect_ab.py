"""把 protein_well（制备孔位）接进设计矩阵，作强收缩随机效应。

来源：`_handoff/CURRENT.md` 第 7 项自列的廉价消融，复赛报告 §已知限制 也点了名
（`docs/20` 第 509 行）；2026-09-03 GrokBuild 独立复核也把它挑成剩下最值钱的合法特征。

为什么它可能有用：手册的对照匹配键是 7 项，**不含 protein_well**
（`docs/06` 第 43 行）。所以处理样本与它的匹配对照来自同一块板但可能是不同孔位，
孔位效应在 Δ = 处理 − 对照 里消不掉。响应空间残留 η² = 0.272（`docs/06` 第 139 行），
是唯一还没吃掉的强技术变量。

设计维从 63 涨到 ~151，逐蛋白 Gram 的外积一次性展开会吃 1 GB 以上内存，
所以这里自带一份**按行分块**的实现，验证通过再决定要不要并回 baseline_cfree。
"""
from __future__ import annotations
import os, sys, warnings
import numpy as np
from threadpoolctl import threadpool_limits
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import paths, split_guard as sg
from models.baseline_cfree import ChunkedGram
from models.design import freeze, encode, FEATURE_SETS
from scorer import evaluate as ev
from scorer.config import ScorerConfig

COLS = ["total", "abs_pcc", "abs_r2", "fc_pcc", "ctx_resid", "drug_resid"]


def score(ctx, Z, mu, W, val, cfg):
    y = (mu[None, :] + Z.astype(np.float64) @ W).astype(np.float32)
    y = np.nan_to_num(y, nan=float(np.nanmedian(mu)))
    return ev.flatten(ev.evaluate(y, ctx, val, cfg)), y


def main():
    cfg = ScorerConfig()
    ctx = ev.build_context(verbose=False)
    val = ctx.rows(ev.VAL_SPLITS)
    tr = sg.train_rows(ctx.meta)
    base_cols = FEATURE_SETS["bio_tech"]
    well_cols = list(base_cols) + ["protein_well"]

    print(f"{'配置':<36}" + "".join(f"{c:>11}" for c in COLS))
    print("-" * (36 + 11 * len(COLS)))

    # 当前最优：上下文 + 药物，无孔位
    s0 = freeze(ctx.meta, tr, base_cols, with_drug=False)
    n_ctx0 = encode(ctx.meta, s0).shape[1]
    s1 = freeze(ctx.meta, tr, base_cols, with_drug=True)
    Z1 = encode(ctx.meta, s1)
    n_drug = Z1.shape[1] - n_ctx0
    lam1 = np.concatenate([np.full(n_ctx0, 3.0), np.full(n_drug, 300.0)])
    g1 = ChunkedGram(Z1, ctx.X, tr, ctx.meta)
    f0, _ = score(ctx, Z1, *g1.solve(lam1), val, cfg)
    del g1
    print(f"{'当前最优 上下文+药物 无孔位':<36}" + "".join(f"{f0[c]:>11.4f}" for c in COLS))
    base = f0["total"]

    # 加孔位
    sw0 = freeze(ctx.meta, tr, well_cols, with_drug=False)
    n_ctxw = encode(ctx.meta, sw0).shape[1]
    sw = freeze(ctx.meta, tr, well_cols, with_drug=True)
    Zw = encode(ctx.meta, sw)
    n_drugw = Zw.shape[1] - n_ctxw
    n_well = len(sw["levels"]["protein_well"])
    print(f"\n设计矩阵：{Zw.shape[1]} 列（原上下文 {n_ctx0} + 孔位 {n_well} + 药物 {n_drugw}）")
    te_well = set(__import__("data.loader", fromlist=["loader"])
                  .load_metadata("test")["protein_well"].astype(str))
    unseen = sorted(te_well - set(sw["levels"]["protein_well"]))
    print(f"train 孔位 {n_well} 个；test 里未见孔位 {len(unseen)} 个（整块归零，退回上下文）\n")

    best = ("当前最优", base, f0, None)
    gw = ChunkedGram(Zw, ctx.X, tr, ctx.meta)     # Gram 只算一次，λ 只改对角
    for lam_well in [10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0]:
        lam = np.concatenate([np.full(n_ctx0, 3.0),
                              np.full(n_well, lam_well),
                              np.full(n_drugw, 300.0)])
        assert lam.size == Zw.shape[1], f"{lam.size} != {Zw.shape[1]}"
        f, y = score(ctx, Zw, *gw.solve(lam), val, cfg)
        tag = f"加孔位 λ_well={lam_well:g}"
        mark = ""
        if f["total"] > best[1]:
            best = (tag, f["total"], f, y); mark = "  ←"
        print(f"{tag:<36}" + "".join(f"{f[c]:>11.4f}" for c in COLS) + mark)

    d = best[1] - base
    print(f"\n最优：{best[0]}  total {best[1]:.4f}  相对当前最优 {d:+.4f}")
    print(f"止损线 +0.0007 → {'通过' if d >= 0.0007 else '未通过，孔位不采用'}")
    if d >= 0.0007:
        bf = best[2]
        for k, thr in (("abs_r2", -0.0015), ("ctx_resid", -0.003), ("drug_resid", -0.003)):
            dv = bf[k] - f0[k]
            print(f"  {k} 变化 {dv:+.4f}（门槛 ≥ {thr}） {'通过' if dv >= thr else '不通过'}")
        print("\n逐 split：")
        print(ev.evaluate_by_split(best[3], ctx, cfg).to_string(index=False))

    out = os.path.join(paths.RESULTS, "step16_well_effect"); paths.ensure_dir(out)
    with open(os.path.join(out, "report.txt"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"当前最优(无孔位) total {base:.6f}\n{best[0]} total {best[1]:.6f}  {d:+.6f}\n")
    print(f"写出 {out}/report.txt")


if __name__ == "__main__":
    main()
