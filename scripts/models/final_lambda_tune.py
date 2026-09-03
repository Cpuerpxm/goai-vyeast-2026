"""定稿前把交互块的 λ 调到四道门槛全过。

step29 的最优组合（菌株×来源×培养基 + 菌株×来源 + 结构迁移）总分 0.5616，
但 drug_resid 变化 −0.0032，刚好越过预注册的 −0.003 止损线。

止损线是实验前定死的，不能因为总分好看就放宽。所以这里扫两个交互块的 λ，
找「总分最高且四道门槛全过」的那一组。宁可少 0.001 分，也不破自己定的规则
——「复现与合规」在手册里是晋级门槛项，不是加分项。
"""
from __future__ import annotations
import os, sys, gc, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import paths, split_guard as sg
from models.baseline_cfree import ChunkedGram
from models.design import freeze, FEATURE_SETS
from models.final_grid import build
from models.interaction_ab import onehot
from models.predict_test import chem_transfer
from scorer import evaluate as ev
from scorer.config import ScorerConfig

COLS = ["total", "abs_pcc", "abs_r2", "fc_pcc", "ctx_resid", "drug_resid"]
PLATE = "Yeast_cell_plate"
GATES = {"abs_r2": -0.0015, "ctx_resid": -0.003, "drug_resid": -0.003}


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

    M3, lv3 = onehot(ctx.meta, ["Strains", "data_source", "Medium"], tr)
    M2, lv2 = onehot(ctx.meta, ["Strains", "data_source"], tr)
    Z = np.hstack([Z0, M3, M2])
    print(f"设计 {Z.shape[1]} 列 = 基线 {Z0.shape[1]} + 菌株×来源×培养基 {len(lv3)}"
          f" + 菌株×来源 {len(lv2)}", flush=True)

    g = ChunkedGram(Z, ctx.X, tr, ctx.meta, row_chunk=384)

    # 门槛的参照是「不加任何交互、不加结构迁移」的基线
    mu, W = g.solve(np.concatenate([lam0, np.full(len(lv3), 1e9), np.full(len(lv2), 1e9)]),
                    prot_chunk=256)
    y = np.nan_to_num((mu[None, :] + Z.astype(np.float64) @ W).astype(np.float32),
                      nan=float(np.nanmedian(mu)))
    f0 = ev.flatten(ev.evaluate(y, ctx, val, cfg))
    print(f"{'λ3 / λ2 / 结构迁移':<26}" + "".join(f"{c:>11}" for c in COLS) + "   门槛", flush=True)
    print("-" * (26 + 11 * len(COLS) + 8))
    print(f"{'基线（交互块压死）':<26}" + "".join(f"{f0[c]:>11.4f}" for c in COLS), flush=True)
    del W, y; gc.collect()

    rows = {}
    for l3, l2, chem in itertools.product((30.0, 100.0, 300.0, 1000.0),
                                          (100.0, 300.0, 1000.0), (True,)):
        mu, W = g.solve(np.concatenate([lam0, np.full(len(lv3), l3), np.full(len(lv2), l2)]),
                        prot_chunk=256)
        y = np.nan_to_num((mu[None, :] + Z.astype(np.float64) @ W).astype(np.float32),
                          nan=float(np.nanmedian(mu)))
        if chem:
            E = W[n_ctx + n_plate: n_ctx + n_plate + n_drug]
            add, hit, _ = chem_transfer(ctx.meta, drugs, E, 10.0, 1.0)
            if add is not None and hit.any():
                y = (y.astype(np.float64) + add).astype(np.float32)
        f = ev.flatten(ev.evaluate(y, ctx, val, cfg))
        ok = all(f[k] - f0[k] >= thr for k, thr in GATES.items())
        tag = f"{l3:g} / {l2:g} / {'开' if chem else '关'}"
        rows[tag] = (f, ok)
        bad = [k for k, thr in GATES.items() if f[k] - f0[k] < thr]
        print(f"{tag:<26}" + "".join(f"{f[c]:>11.4f}" for c in COLS)
              + ("   全过" if ok else f"   {'/'.join(bad)} 不过"), flush=True)
        del W, y; gc.collect()

    passing = {k: v for k, v in rows.items() if v[1]}
    if passing:
        name, (f, _) = max(passing.items(), key=lambda kv: kv[1][0]["total"])
        print(f"\n门槛全过里的最高分：{name}  total {f['total']:.4f}")
    else:
        print("\n没有一组四道门槛全过")
    name_all, (f_all, ok_all) = max(rows.items(), key=lambda kv: kv[1][0]["total"])
    print(f"不看门槛的最高分：{name_all}  total {f_all['total']:.4f}  {'全过' if ok_all else '有不过'}")

    out = os.path.join(paths.RESULTS, "step31_final_tune"); paths.ensure_dir(out)
    with open(os.path.join(out, "report.txt"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"{'λ3 / λ2 / 结构迁移':<26}" + "".join(f"{c:>11}" for c in COLS) + "  门槛\n")
        for k, (v, o) in sorted(rows.items(), key=lambda kv: -kv[1][0]["total"]):
            fh.write(f"{k:<26}" + "".join(f"{v[c]:>11.4f}" for c in COLS)
                     + ("  全过\n" if o else "  有不过\n"))
    print(f"写出 {out}/report.txt")


if __name__ == "__main__":
    main()
