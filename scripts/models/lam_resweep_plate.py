"""板号进来之后重扫 λ_ctx 与 λ_drug。

上午定的 λ_ctx=3 / λ_drug=300 是在**没有板号**的设计上选的。
板号吃掉了一大块此前落在上下文列和药物列上的方差，两个 λ 的最优点大概率移动了。
粗网格，不细扫。
"""
from __future__ import annotations
import os, sys, gc
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import paths, split_guard as sg
from models.baseline_cfree import ChunkedGram
from models.design import freeze, encode, FEATURE_SETS
from scorer import evaluate as ev
from scorer.config import ScorerConfig

COLS = ["total", "abs_pcc", "abs_r2", "fc_pcc", "ctx_resid", "drug_resid"]
PLATE = "Yeast_cell_plate"


def main():
    cfg = ScorerConfig()
    ctx = ev.build_context(verbose=False)
    val = ctx.rows(ev.VAL_SPLITS)
    tr = sg.train_rows(ctx.meta)
    base_cols = FEATURE_SETS["bio_tech"]

    s0 = freeze(ctx.meta, tr, base_cols, with_drug=False)
    n_ctx = encode(ctx.meta, s0).shape[1]
    sp = freeze(ctx.meta, tr, list(base_cols) + [PLATE], with_drug=True)
    Z = encode(ctx.meta, sp)
    n_plate = len(sp["levels"][PLATE])
    n_drug = Z.shape[1] - n_ctx - n_plate

    g = ChunkedGram(Z, ctx.X, tr, ctx.meta)     # Gram 只算一次
    print(f"{'λ_ctx / λ_plate / λ_drug':<34}" + "".join(f"{c:>11}" for c in COLS), flush=True)
    print("-" * (34 + 11 * len(COLS)))

    rows = {}
    grid = [(lc, lp, ld)
            for lc in (0.3, 1.0, 3.0, 10.0)
            for lp in (3.0, 10.0, 30.0)
            for ld in (30.0, 100.0, 300.0, 1000.0)]
    for lc, lp, ld in grid:
        lam = np.concatenate([np.full(n_ctx, lc), np.full(n_plate, lp), np.full(n_drug, ld)])
        mu, W = g.solve(lam)
        y = np.nan_to_num((mu[None, :] + Z.astype(np.float64) @ W).astype(np.float32),
                          nan=float(np.nanmedian(mu)))
        f = ev.flatten(ev.evaluate(y, ctx, val, cfg))
        tag = f"{lc:g} / {lp:g} / {ld:g}"
        rows[tag] = f
        print(f"{tag:<34}" + "".join(f"{f[c]:>11.4f}" for c in COLS), flush=True)
        del y; gc.collect()

    cur = rows.get("3 / 10 / 300")
    name, f = max(rows.items(), key=lambda kv: kv[1]["total"])
    print(f"\n当前配置 3 / 10 / 300  total {cur['total']:.4f}")
    print(f"网格最优 {name}  total {f['total']:.4f}  相对当前 {f['total']-cur['total']:+.4f}")
    if f["total"] - cur["total"] >= 0.0007:
        for k, thr in (("abs_r2", -0.0015), ("ctx_resid", -0.003), ("drug_resid", -0.003)):
            dv = f[k] - cur[k]
            print(f"  {k} 变化 {dv:+.4f}（门槛 ≥ {thr}） {'通过' if dv >= thr else '不通过'}")

    out = os.path.join(paths.RESULTS, "step22_lam_resweep"); paths.ensure_dir(out)
    with open(os.path.join(out, "report.txt"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"{'λ_ctx / λ_plate / λ_drug':<34}" + "".join(f"{c:>11}" for c in COLS) + "\n")
        for k, v in rows.items():
            fh.write(f"{k:<34}" + "".join(f"{v[c]:>11.4f}" for c in COLS) + "\n")
        fh.write(f"\n最优 {name} {f['total']:.6f}\n")
    print(f"写出 {out}/report.txt")


if __name__ == "__main__":
    main()
