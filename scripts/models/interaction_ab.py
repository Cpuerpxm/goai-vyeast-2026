"""匹配键变量之间的小交互项。

板号已经决定了匹配键 7 项里的 6 项（板号 × 时间的水平数与板号相同，
说明每块板只对应一个时间点；来源、培养基、温度、仪器同理）。
唯一不被板号决定的是菌株，所以完整的匹配组 = 板号 × 菌株，383 个水平。

那个完整交互不做，两个原因：
  · 设计维会到 590，逐蛋白 Gram 要 12 GB，本机放不下；
  · test 里 60% 的行菌株未见，这一块对它们全是 0，能覆盖的只有 40%。

改成试它的几个粗化版本，都很便宜：菌株 × 时间、菌株 × 培养基、菌株 × 来源。
"""
from __future__ import annotations
import os, sys, gc
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import paths, split_guard as sg
from models.baseline_cfree import ChunkedGram
from models.design import freeze, encode, FEATURE_SETS, TIME_COL
from models.final_grid import build
from scorer import evaluate as ev
from scorer.config import ScorerConfig

COLS = ["total", "abs_pcc", "abs_r2", "fc_pcc", "ctx_resid", "drug_resid"]


def onehot(meta, cols, tr):
    k = meta[cols].astype(str).agg("\x1f".join, axis=1).to_numpy()
    levels = sorted(set(k[tr]))
    idx = {v: i for i, v in enumerate(levels)}
    M = np.zeros((len(k), len(levels)), dtype=np.float32)
    for i, v in enumerate(k):
        j = idx.get(v)
        if j is not None:
            M[i, j] = 1.0
    return M, levels


def main():
    cfg = ScorerConfig()
    ctx = ev.build_context(verbose=False)
    val = ctx.rows(ev.VAL_SPLITS)
    tr = sg.train_rows(ctx.meta)
    Z0, n_ctx, n_plate, n_drug, _ = build(ctx, tr)
    # step24 网格最优
    lam0 = np.concatenate([np.full(n_ctx, 3.0), np.full(n_plate, 10.0),
                           np.full(n_drug, 100.0), np.full(n_drug, 30.0)])

    print(f"{'配置':<34}" + "".join(f"{c:>11}" for c in COLS), flush=True)
    print("-" * (34 + 11 * len(COLS)))
    g0 = ChunkedGram(Z0, ctx.X, tr, ctx.meta)
    mu, W = g0.solve(lam0)
    y = np.nan_to_num((mu[None, :] + Z0.astype(np.float64) @ W).astype(np.float32),
                      nan=float(np.nanmedian(mu)))
    f0 = ev.flatten(ev.evaluate(y, ctx, val, cfg))
    print(f"{'当前 板号+药物×时间':<34}" + "".join(f"{f0[c]:>11.4f}" for c in COLS), flush=True)
    base = f0["total"]
    del g0, W, y; gc.collect()
    rows = {"当前": f0}

    # 药物 × log时间² ：给已见药物一条二次的时间轨迹，未见药物仍全 0
    t = np.log1p(ctx.meta[TIME_COL].to_numpy(dtype=np.float64))
    tc = ((t - t[tr].mean()) / t[tr].std())
    Zq = (Z0[:, -n_drug:] * ((tc ** 2 - (tc[tr] ** 2).mean())[:, None])).astype(np.float32)
    Z = np.hstack([Z0, Zq])
    print()
    print(f"药物 × log时间²：{n_drug} 列", flush=True)
    gq = ChunkedGram(Z, ctx.X, tr, ctx.meta, row_chunk=384)
    for li in (100.0, 300.0, 1000.0):
        lam = np.concatenate([lam0, np.full(n_drug, li)])
        mu, W = gq.solve(lam, prot_chunk=256)
        y = np.nan_to_num((mu[None, :] + Z.astype(np.float64) @ W).astype(np.float32),
                          nan=float(np.nanmedian(mu)))
        f = ev.flatten(ev.evaluate(y, ctx, val, cfg))
        tag = f"+药物×时间² λ={li:g}"
        rows[tag] = f
        print(f"{tag:<34}" + "".join(f"{f[c]:>11.4f}" for c in COLS), flush=True)
        del W, y; gc.collect()
    del gq, Z, Zq; gc.collect()

    for cols in (["Strains", TIME_COL], ["Strains", "Medium"], ["Strains", "data_source"]):
        M, levels = onehot(ctx.meta, cols, tr)
        hit = float(M[val].sum(axis=1).mean())
        Z = np.hstack([Z0, M])
        print(f"\n{'×'.join(cols)}：{len(levels)} 个水平，val 命中 {hit:.1%}", flush=True)
        g = ChunkedGram(Z, ctx.X, tr, ctx.meta, row_chunk=384)
        for li in (30.0, 100.0, 300.0):
            lam = np.concatenate([lam0, np.full(len(levels), li)])
            mu, W = g.solve(lam, prot_chunk=256)
            y = np.nan_to_num((mu[None, :] + Z.astype(np.float64) @ W).astype(np.float32),
                              nan=float(np.nanmedian(mu)))
            f = ev.flatten(ev.evaluate(y, ctx, val, cfg))
            tag = f"+{'×'.join(cols)} λ={li:g}"
            rows[tag] = f
            print(f"{tag:<34}" + "".join(f"{f[c]:>11.4f}" for c in COLS), flush=True)
            del W, y; gc.collect()
        del g, Z, M; gc.collect()

    name, f = max(rows.items(), key=lambda kv: kv[1]["total"])
    d = f["total"] - base
    print(f"\n最优：{name}  total {f['total']:.4f}  相对当前 {d:+.4f}")
    print(f"止损线 +0.0007 → {'通过' if d >= 0.0007 else '未通过'}")

    out = os.path.join(paths.RESULTS, "step25_interaction"); paths.ensure_dir(out)
    with open(os.path.join(out, "report.txt"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"{'配置':<34}" + "".join(f"{c:>11}" for c in COLS) + "\n")
        for k, v in rows.items():
            fh.write(f"{k:<34}" + "".join(f"{v[c]:>11.4f}" for c in COLS) + "\n")
        fh.write(f"\n最优 {name} {f['total']:.6f}  {d:+.6f}\n")
    print(f"写出 {out}/report.txt")


if __name__ == "__main__":
    main()
