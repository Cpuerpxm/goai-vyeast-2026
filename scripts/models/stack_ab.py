"""把 step25 挑出来的交互项和 step28 的结构迁移叠在一起，看是否可加。

单项收益（都相对 0.5483 的基线）：
  · 菌株 × 培养基（8 水平）          +0.0026
  · 菌株 × 时间（24 水平）           +0.0007
  · 未见药物结构迁移                 +0.0011

前两项只作用于菌株已见的行（val 命中 40.2%），第三项只作用于药物未见的行，
支撑集不同，理论上应该基本可加。这里实测。
"""
from __future__ import annotations
import os, sys, gc, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import paths, split_guard as sg
from models.baseline_cfree import ChunkedGram
from models.design import freeze, encode, FEATURE_SETS, TIME_COL
from models.final_grid import build
from models.interaction_ab import onehot
from models.predict_test import chem_transfer
from scorer import evaluate as ev
from scorer.config import ScorerConfig

COLS = ["total", "abs_pcc", "abs_r2", "fc_pcc", "ctx_resid", "drug_resid"]
PLATE = "Yeast_cell_plate"


def main():
    cfg = ScorerConfig()
    ctx = ev.build_context(verbose=False)
    val = ctx.rows(ev.VAL_SPLITS)
    tr = sg.train_rows(ctx.meta)
    Z0, n_ctx, n_plate, n_drug, _ = build(ctx, tr)
    lam0 = np.concatenate([np.full(n_ctx, 3.0), np.full(n_plate, 10.0),
                           np.full(n_drug, 100.0), np.full(n_drug, 30.0)])
    spec = freeze(ctx.meta, tr, list(FEATURE_SETS["bio_tech"]) + [PLATE], with_drug=True)
    drugs = spec["drugs"]

    # 先把「菌株 × 各个匹配键变量」都扫一遍，再叠赢的
    cands = {
        "菌株×培养基": (["Strains", "Medium"], 30.0),
        "菌株×温度": (["Strains", "Temperature"], 100.0),
        "菌株×来源": (["Strains", "data_source"], 100.0),
        "菌株×仪器": (["Strains", "instrument"], 100.0),
        "菌株×时间": (["Strains", TIME_COL], 100.0),
        "菌株×来源×培养基": (["Strains", "data_source", "Medium"], 100.0),
    }
    blocks = {}
    for name, (cols, lm) in cands.items():
        M, lv = onehot(ctx.meta, cols, tr)
        blocks[name] = (M, np.full(len(lv), lm), len(lv))

    print(f"{'配置':<40}" + "".join(f"{c:>11}" for c in COLS), flush=True)
    print("-" * (40 + 11 * len(COLS)))
    rows = {}

    def run(names, use_chem):
        Zs, lams = [Z0], [lam0]
        for nm in names:
            M, lm, _ = blocks[nm]
            Zs.append(M); lams.append(lm)
        Z = np.hstack(Zs) if len(Zs) > 1 else Z0
        lam = np.concatenate(lams)
        g = ChunkedGram(Z, ctx.X, tr, ctx.meta, row_chunk=384)
        mu, W = g.solve(lam, prot_chunk=256)
        y = np.nan_to_num((mu[None, :] + Z.astype(np.float64) @ W).astype(np.float32),
                          nan=float(np.nanmedian(mu)))
        tag_parts = list(names)
        if use_chem:
            E = W[n_ctx + n_plate: n_ctx + n_plate + n_drug]
            add, hit, _ = chem_transfer(ctx.meta, drugs, E, 10.0, 1.0)
            if add is not None and hit.any():
                y = (y.astype(np.float64) + add).astype(np.float32)
            tag_parts.append("结构迁移")
        f = ev.flatten(ev.evaluate(y, ctx, val, cfg))
        tag = " + ".join(tag_parts) if tag_parts else "基线（都不加）"
        rows[tag] = f
        print(f"{tag:<40}" + "".join(f"{f[c]:>11.4f}" for c in COLS), flush=True)
        del g, W, y, Z; gc.collect()
        return f

    run([], False)
    for nm in cands:
        run([nm], False)

    top = sorted((k for k in rows if k != "基线（都不加）"),
                 key=lambda k: -rows[k]["total"])[:2]
    print()
    print(f"单项前二：{top}", flush=True)
    print()
    run(top, False)
    run(top[:1], True)
    run(top, True)

    base = rows["基线（都不加）"]["total"]
    name, f = max(rows.items(), key=lambda kv: kv[1]["total"])
    d = f["total"] - base
    print(f"\n最优：{name}  total {f['total']:.4f}  相对基线 {d:+.4f}")
    for k, thr in (("abs_r2", -0.0015), ("ctx_resid", -0.003), ("drug_resid", -0.003)):
        dv = f[k] - rows["基线（都不加）"][k]
        print(f"  {k} 变化 {dv:+.4f}（门槛 ≥ {thr}） {'通过' if dv >= thr else '不通过'}")

    out = os.path.join(paths.RESULTS, "step29_stack"); paths.ensure_dir(out)
    with open(os.path.join(out, "report.txt"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"{'配置':<40}" + "".join(f"{c:>11}" for c in COLS) + "\n")
        for k, v in rows.items():
            fh.write(f"{k:<40}" + "".join(f"{v[c]:>11.4f}" for c in COLS) + "\n")
        fh.write(f"\n最优 {name} {f['total']:.6f}  {d:+.6f}\n")
    print(f"写出 {out}/report.txt")


if __name__ == "__main__":
    main()
