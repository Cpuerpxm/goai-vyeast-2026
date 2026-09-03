"""网格边界探针：λ_ctx=1 落在 step24 网格的下边界上，往下再看两档。

只动 λ_ctx，其余固定在网格最优 (λ_plate=10, λ_drug=100, λ_int=30)。
"""
from __future__ import annotations
import os, sys, gc
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import paths, split_guard as sg
from models.baseline_cfree import ChunkedGram
from models.final_grid import build
from scorer import evaluate as ev
from scorer.config import ScorerConfig

COLS = ["total", "abs_pcc", "abs_r2", "fc_pcc", "ctx_resid", "drug_resid"]

def main():
    cfg = ScorerConfig(); ctx = ev.build_context(verbose=False)
    val = ctx.rows(ev.VAL_SPLITS); tr = sg.train_rows(ctx.meta)
    Z, n_ctx, n_plate, n_drug, _ = build(ctx, tr)
    g = ChunkedGram(Z, ctx.X, tr, ctx.meta)
    print(f"{'λ_ctx':<12}" + "".join(f"{c:>11}" for c in COLS), flush=True)
    print("-" * (12 + 11 * len(COLS)))
    rows = {}
    for lc in (0.03, 0.1, 0.3, 1.0, 3.0):
        lam = np.concatenate([np.full(n_ctx, lc), np.full(n_plate, 10.0),
                              np.full(n_drug, 100.0), np.full(n_drug, 30.0)])
        mu, W = g.solve(lam)
        y = np.nan_to_num((mu[None, :] + Z.astype(np.float64) @ W).astype(np.float32),
                          nan=float(np.nanmedian(mu)))
        f = ev.flatten(ev.evaluate(y, ctx, val, cfg)); rows[lc] = f
        print(f"{lc:<12g}" + "".join(f"{f[c]:>11.4f}" for c in COLS), flush=True)
        del W, y; gc.collect()
    best = max(rows.items(), key=lambda kv: kv[1]["total"])
    print(f"\n最优 λ_ctx={best[0]:g}  total {best[1]['total']:.4f}"
          f"  相对 λ_ctx=1 {best[1]['total']-rows[1.0]['total']:+.4f}")
    out = os.path.join(paths.RESULTS, "step26_edge"); paths.ensure_dir(out)
    with open(os.path.join(out, "report.txt"), "w", encoding="utf-8", newline="\n") as fh:
        for k, v in rows.items():
            fh.write(f"{k:<12g}" + "".join(f"{v[c]:>11.4f}" for c in COLS) + "\n")
    print(f"写出 {out}/report.txt")

if __name__ == "__main__":
    main()
