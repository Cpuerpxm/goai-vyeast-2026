"""把 Yeast_cell_plate（培养板号）接进设计矩阵。

为什么它跟孔位不一样，值得单独试：

手册的对照匹配键**含板号**（`docs/06` 第 43 行），所以处理样本和它的匹配对照
来自同一块板。板效应在 Δ_true = y_treat − y_ctrl 里正好抵消。
但我们的 ŷ 里没有板项，于是

    Δ_pred = ŷ − y_ctrl = base − (base_ctrl + plate + noise)

**凭空多出一个 −plate**。Δ_true 里那一项是抵消掉的，Δ_pred 里却带着，
这是一个我们一直在犯的系统性偏差，不是缺一点信息。

孔位不在匹配键里，情况相反：Δ_true 里带着 (well_treat − well_ctrl)，
所以孔位只能加信息、不能消偏差。实测孔位没用（step16），板号未必同理。

覆盖率：train 折有全部 144 块板，val 3038 行与 test 4454 行**全部**命中已见板。
所以推断时这一块是满的，没有回退问题。

板号与培养基/温度/时间的 Cramér's V = 0.992（`docs/06` 第 323 行），高度共线，
所以必须用强收缩，让它只吃残差里那一层。
"""
from __future__ import annotations
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import paths, split_guard as sg
from models.design import freeze, encode, FEATURE_SETS
from models.well_effect_ab import ChunkedGram, score
from scorer import evaluate as ev
from scorer.config import ScorerConfig

COLS = ["total", "abs_pcc", "abs_r2", "fc_pcc", "ctx_resid", "drug_resid"]


def main():
    cfg = ScorerConfig()
    ctx = ev.build_context(verbose=False)
    val = ctx.rows(ev.VAL_SPLITS)
    tr = sg.train_rows(ctx.meta)
    base_cols = FEATURE_SETS["bio_tech"]
    plate_cols = list(base_cols) + ["Yeast_cell_plate"]

    print(f"{'配置':<36}" + "".join(f"{c:>11}" for c in COLS))
    print("-" * (36 + 11 * len(COLS)))

    s0 = freeze(ctx.meta, tr, base_cols, with_drug=False)
    n_ctx0 = encode(ctx.meta, s0).shape[1]
    s1 = freeze(ctx.meta, tr, base_cols, with_drug=True)
    Z1 = encode(ctx.meta, s1)
    n_drug = Z1.shape[1] - n_ctx0
    g1 = ChunkedGram(Z1, ctx.X, tr, ctx.meta)
    f0, _ = score(ctx, Z1, *g1.solve(np.concatenate(
        [np.full(n_ctx0, 3.0), np.full(n_drug, 300.0)])), val, cfg)
    del g1
    print(f"{'当前最优 无板号':<36}" + "".join(f"{f0[c]:>11.4f}" for c in COLS))
    base = f0["total"]

    sp0 = freeze(ctx.meta, tr, plate_cols, with_drug=False)
    n_ctxp = encode(ctx.meta, sp0).shape[1]
    sp = freeze(ctx.meta, tr, plate_cols, with_drug=True)
    Zp = encode(ctx.meta, sp)
    n_drugp = Zp.shape[1] - n_ctxp
    n_plate = len(sp["levels"]["Yeast_cell_plate"])
    print(f"\n设计 {Zp.shape[1]} 列（原上下文 {n_ctx0} + 板号 {n_plate} + 药物 {n_drugp}）\n")

    best = ("当前最优", base, f0, None)
    gp = ChunkedGram(Zp, ctx.X, tr, ctx.meta)
    for lam_plate in (3.0, 10.0, 30.0, 100.0, 300.0, 1000.0):
        lam = np.concatenate([np.full(n_ctx0, 3.0),
                              np.full(n_plate, lam_plate),
                              np.full(n_drugp, 300.0)])
        assert lam.size == Zp.shape[1]
        f, y = score(ctx, Zp, *gp.solve(lam), val, cfg)
        tag = f"加板号 λ_plate={lam_plate:g}"
        mark = ""
        if f["total"] > best[1]:
            best = (tag, f["total"], f, y); mark = "  ←"
        print(f"{tag:<36}" + "".join(f"{f[c]:>11.4f}" for c in COLS) + mark)

    d = best[1] - base
    print(f"\n最优：{best[0]}  total {best[1]:.4f}  相对当前 {d:+.4f}")
    print(f"止损线 +0.0007 → {'通过' if d >= 0.0007 else '未通过，板号不采用'}")
    if d >= 0.0007:
        for k, thr in (("abs_r2", -0.0015), ("ctx_resid", -0.003), ("drug_resid", -0.003)):
            dv = best[2][k] - f0[k]
            print(f"  {k} 变化 {dv:+.4f}（门槛 ≥ {thr}） {'通过' if dv >= thr else '不通过'}")
        print("\n逐 split：")
        print(ev.evaluate_by_split(best[3], ctx, cfg).to_string(index=False))

    out = os.path.join(paths.RESULTS, "step20_plate_effect"); paths.ensure_dir(out)
    with open(os.path.join(out, "report.txt"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"无板号 total {base:.6f}\n{best[0]} total {best[1]:.6f}  {d:+.6f}\n")
    print(f"写出 {out}/report.txt")


if __name__ == "__main__":
    main()
