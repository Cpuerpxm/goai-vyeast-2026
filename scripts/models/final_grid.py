"""当前完整设计的四块 λ 粗网格。

设计 = 截距与上下文 23 + 板号 144 + 药物 40 + 药物×log时间 40 = 247 列。
Gram 只算一次，网格只改对角，所以 81 个组合的成本主要在评分而不是解方程。

粗网格，每个 λ 只取 3 档。不细扫，避免在单个 val 上把 λ 调过头。
"""
from __future__ import annotations
import os, sys, gc
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import paths, split_guard as sg
from models.baseline_cfree import ChunkedGram
from models.design import freeze, encode, FEATURE_SETS, TIME_COL
from scorer import evaluate as ev
from scorer.config import ScorerConfig

COLS = ["total", "abs_pcc", "abs_r2", "fc_pcc", "ctx_resid", "drug_resid"]
PLATE = "Yeast_cell_plate"
CUR = (3.0, 10.0, 300.0, 100.0)     # 当前配置


def build(ctx, tr):
    base_cols = FEATURE_SETS["bio_tech"]
    s0 = freeze(ctx.meta, tr, base_cols, with_drug=False)
    n_ctx = encode(ctx.meta, s0).shape[1]
    sp = freeze(ctx.meta, tr, list(base_cols) + [PLATE], with_drug=True)
    Zp = encode(ctx.meta, sp)
    n_plate = len(sp["levels"][PLATE])
    n_drug = Zp.shape[1] - n_ctx - n_plate
    t = np.log1p(ctx.meta[TIME_COL].to_numpy(dtype=np.float64))
    tc = ((t - t[tr].mean()) / t[tr].std()).astype(np.float32)
    Z = np.hstack([Zp, Zp[:, -n_drug:] * tc[:, None]])
    return Z, n_ctx, n_plate, n_drug, sp


def main():
    cfg = ScorerConfig()
    ctx = ev.build_context(verbose=False)
    val = ctx.rows(ev.VAL_SPLITS)
    tr = sg.train_rows(ctx.meta)
    Z, n_ctx, n_plate, n_drug, spec = build(ctx, tr)
    print(f"设计 {Z.shape[1]} 列 = 上下文 {n_ctx} + 板号 {n_plate} + 药物 {n_drug} + 交互 {n_drug}",
          flush=True)

    g = ChunkedGram(Z, ctx.X, tr, ctx.meta, row_chunk=512)
    print(f"{'λ_ctx/plate/drug/交互':<28}" + "".join(f"{c:>11}" for c in COLS), flush=True)
    print("-" * (28 + 11 * len(COLS)))

    rows = {}
    for lc in (1.0, 3.0, 10.0):
        for lp in (3.0, 10.0, 30.0):
            for ld in (100.0, 300.0, 1000.0):
                for li in (30.0, 100.0, 300.0):
                    lam = np.concatenate([np.full(n_ctx, lc), np.full(n_plate, lp),
                                          np.full(n_drug, ld), np.full(n_drug, li)])
                    mu, W = g.solve(lam, prot_chunk=384)
                    y = np.nan_to_num((mu[None, :] + Z.astype(np.float64) @ W).astype(np.float32),
                                      nan=float(np.nanmedian(mu)))
                    f = ev.flatten(ev.evaluate(y, ctx, val, cfg))
                    tag = f"{lc:g}/{lp:g}/{ld:g}/{li:g}"
                    rows[tag] = f
                    mark = "  ← 当前" if (lc, lp, ld, li) == CUR else ""
                    print(f"{tag:<28}" + "".join(f"{f[c]:>11.4f}" for c in COLS) + mark, flush=True)
                    del y, W; gc.collect()

    cur_tag = "/".join(f"{v:g}" for v in CUR)
    cur = rows[cur_tag]
    name, f = max(rows.items(), key=lambda kv: kv[1]["total"])
    d = f["total"] - cur["total"]
    print(f"\n当前 {cur_tag}  total {cur['total']:.4f}")
    print(f"网格最优 {name}  total {f['total']:.4f}  相对当前 {d:+.4f}")
    print(f"止损线 +0.0007 → {'通过' if d >= 0.0007 else '未通过，保持当前配置'}")

    top = sorted(rows.items(), key=lambda kv: -kv[1]["total"])[:8]
    print("\n前 8 名：")
    for k, v in top:
        print(f"  {k:<22} {v['total']:.4f}")

    out = os.path.join(paths.RESULTS, "step24_final_grid"); paths.ensure_dir(out)
    with open(os.path.join(out, "report.txt"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"{'λ_ctx/plate/drug/交互':<28}" + "".join(f"{c:>11}" for c in COLS) + "\n")
        for k, v in sorted(rows.items(), key=lambda kv: -kv[1]["total"]):
            fh.write(f"{k:<28}" + "".join(f"{v[c]:>11.4f}" for c in COLS) + "\n")
        fh.write(f"\n当前 {cur_tag} {cur['total']:.6f}\n最优 {name} {f['total']:.6f}  {d:+.6f}\n")
    print(f"写出 {out}/report.txt")


if __name__ == "__main__":
    main()
