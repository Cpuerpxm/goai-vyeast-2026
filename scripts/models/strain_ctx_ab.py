"""菌株 × 上下文交互往深了走一格。

step29 的单项扫描结果（都相对 0.5483）：

    菌株×温度            0.5481   （负）
    菌株×时间            0.5490
    菌株×培养基          0.5509
    菌株×仪器            0.5540
    菌株×来源            0.5564
    菌株×来源×培养基     0.5589   ← 越细越好，所以再往下试一格

理论终点是「菌株 × 板号」，也就是手册匹配键的完整 7 元组（383 组），
但那样设计维到 630，逐蛋白 Gram 要 14 GB，本机放不下。
所以只走到菌株 × 三四个上下文变量。
"""
from __future__ import annotations
import os, sys, gc
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import paths, split_guard as sg
from models.baseline_cfree import ChunkedGram
from models.design import TIME_COL
from models.final_grid import build
from models.interaction_ab import onehot
from scorer import evaluate as ev
from scorer.config import ScorerConfig

COLS = ["total", "abs_pcc", "abs_r2", "fc_pcc", "ctx_resid", "drug_resid"]

CANDS = [
    ("菌株×来源×培养基", ["Strains", "data_source", "Medium"]),
    ("菌株×来源×培养基×温度", ["Strains", "data_source", "Medium", "Temperature"]),
    ("菌株×来源×培养基×仪器", ["Strains", "data_source", "Medium", "instrument"]),
    ("菌株×来源×培养基×时间", ["Strains", "data_source", "Medium", TIME_COL]),
    ("菌株×来源×仪器", ["Strains", "data_source", "instrument"]),
]


def main():
    cfg = ScorerConfig()
    ctx = ev.build_context(verbose=False)
    val = ctx.rows(ev.VAL_SPLITS)
    tr = sg.train_rows(ctx.meta)
    Z0, n_ctx, n_plate, n_drug, _ = build(ctx, tr)
    lam0 = np.concatenate([np.full(n_ctx, 3.0), np.full(n_plate, 10.0),
                           np.full(n_drug, 100.0), np.full(n_drug, 30.0)])

    print(f"{'配置':<40}" + "".join(f"{c:>11}" for c in COLS), flush=True)
    print("-" * (40 + 11 * len(COLS)))
    rows = {}
    for name, cols in CANDS:
        M, lv = onehot(ctx.meta, cols, tr)
        hit = float(M[val].sum(axis=1).mean())
        Z = np.hstack([Z0, M])
        g = ChunkedGram(Z, ctx.X, tr, ctx.meta, row_chunk=256)
        for lm in (30.0, 100.0, 300.0):
            lam = np.concatenate([lam0, np.full(len(lv), lm)])
            mu, W = g.solve(lam, prot_chunk=128)
            y = np.nan_to_num((mu[None, :] + Z.astype(np.float64) @ W).astype(np.float32),
                              nan=float(np.nanmedian(mu)))
            f = ev.flatten(ev.evaluate(y, ctx, val, cfg))
            tag = f"{name}({len(lv)}) λ={lm:g}"
            rows[tag] = f
            print(f"{tag:<40}" + "".join(f"{f[c]:>11.4f}" for c in COLS)
                  + (f"   val命中{hit:.0%}" if lm == 30.0 else ""), flush=True)
            del W, y; gc.collect()
        del g, Z, M; gc.collect()

    name, f = max(rows.items(), key=lambda kv: kv[1]["total"])
    print(f"\n最优：{name}  total {f['total']:.4f}")
    out = os.path.join(paths.RESULTS, "step30_strain_ctx"); paths.ensure_dir(out)
    with open(os.path.join(out, "report.txt"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"{'配置':<40}" + "".join(f"{c:>11}" for c in COLS) + "\n")
        for k, v in sorted(rows.items(), key=lambda kv: -kv[1]["total"]):
            fh.write(f"{k:<40}" + "".join(f"{v[c]:>11.4f}" for c in COLS) + "\n")
    print(f"写出 {out}/report.txt")


if __name__ == "__main__":
    main()
