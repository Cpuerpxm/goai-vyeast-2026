"""板号进模型之后，接着挖三样东西。

板号那一笔（step20，+0.0347）说明一件事：Δ 类指标其实奖励的是**把处理样本的
绝对谱预测准**，因为 Δ_pred = ŷ − C 而 Δ_true = y_treat − C，两者之差就是 ŷ − y_treat。
所以凡是能让 ŷ 更贴近 y_treat 的合法结构，都会同时抬高六项。

本轮三个候选：

1. **板号 + 孔位**。step16 测孔位时基线里还没有板号，那次是负结果。板号和孔位是
   同一块板上的两个不同维度，加了板号之后孔位的残差结构会变，值得在新基线上重测。

2. **药物 × 时间**。药物现在是主效应，等于假设同一个药在 15 分钟和 240 分钟
   引起的响应方向一样。用「药物 one-hot × 中心化 log 时间」加 40 列，
   给每个已见药物一个随时间变化的斜率，未见药物照样全 0。

3. **匹配组 one-hot**（= 板号 × 菌株，383 个水平）。手册的对照匹配是 7 元组，
   实测这个 7 元组恰好由「板号 × 菌株」决定。给每个匹配组一个系数，等于把
   「对照那一侧带的所有共享结构」显式建出来。
   ⚠ 风险：组平均只有 15.5 行，若某个对照本身在训练折里，它对组系数的贡献约 6%，
   有把对照读进预测的成分。所以这一项必须过 C-free 守卫才能采用。
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
LAM_CTX, LAM_PLATE, LAM_DRUG = 3.0, 10.0, 300.0
PLATE = "Yeast_cell_plate"


def sc(ctx, Z, tr, lam, val, cfg, row_chunk=512, prot_chunk=256):
    g = ChunkedGram(Z, ctx.X, tr, ctx.meta, row_chunk=row_chunk)
    mu, W = g.solve(lam, prot_chunk=prot_chunk)
    del g; gc.collect()
    y = (mu[None, :] + Z.astype(np.float64) @ W).astype(np.float32)
    y = np.nan_to_num(y, nan=float(np.nanmedian(mu)))
    return ev.flatten(ev.evaluate(y, ctx, val, cfg)), y


def main():
    cfg = ScorerConfig()
    ctx = ev.build_context(verbose=False)
    val = ctx.rows(ev.VAL_SPLITS)
    tr = sg.train_rows(ctx.meta)
    base_cols = FEATURE_SETS["bio_tech"]

    s0 = freeze(ctx.meta, tr, base_cols, with_drug=False)
    n_ctx = encode(ctx.meta, s0).shape[1]
    sp = freeze(ctx.meta, tr, list(base_cols) + [PLATE], with_drug=True)
    Zp = encode(ctx.meta, sp)
    n_plate = len(sp["levels"][PLATE])
    n_drug = Zp.shape[1] - n_ctx - n_plate
    lam_p = np.concatenate([np.full(n_ctx, LAM_CTX), np.full(n_plate, LAM_PLATE),
                            np.full(n_drug, LAM_DRUG)])

    print(f"{'配置':<34}" + "".join(f"{c:>11}" for c in COLS), flush=True)
    print("-" * (34 + 11 * len(COLS)))
    f0, _ = sc(ctx, Zp, tr, lam_p, val, cfg)
    print(f"{'当前最优 板号 λ=10':<34}" + "".join(f"{f0[c]:>11.4f}" for c in COLS), flush=True)
    base = f0["total"]
    rows = {"当前最优": f0}

    # ---- 1. 板号 + 孔位 ----
    sw = freeze(ctx.meta, tr, list(base_cols) + [PLATE, "protein_well"], with_drug=True)
    Zw = encode(ctx.meta, sw)
    n_well = len(sw["levels"]["protein_well"])
    print(f"\n[1] 板号 + 孔位  设计 {Zw.shape[1]} 列（孔位 {n_well}）", flush=True)
    for lw in (30.0, 100.0, 300.0):
        lam = np.concatenate([np.full(n_ctx, LAM_CTX), np.full(n_plate, LAM_PLATE),
                              np.full(n_well, lw), np.full(n_drug, LAM_DRUG)])
        assert lam.size == Zw.shape[1], f"{lam.size} != {Zw.shape[1]}"
        f, _ = sc(ctx, Zw, tr, lam, val, cfg, row_chunk=256, prot_chunk=128)
        rows[f"板号+孔位 λ_well={lw:g}"] = f
        print(f"{'板号+孔位 λ_well='+f'{lw:g}':<34}" + "".join(f"{f[c]:>11.4f}" for c in COLS), flush=True)
    del Zw; gc.collect()

    # ---- 2. 药物 × 中心化 log 时间 ----
    t = np.log1p(ctx.meta[TIME_COL].to_numpy(dtype=np.float64))
    tc = ((t - t[tr].mean()) / t[tr].std()).astype(np.float32)
    Zd = Zp[:, -n_drug:]                      # 药物 one-hot 块
    Zi = np.hstack([Zp, Zd * tc[:, None]])
    print(f"\n[2] 药物 × log 时间  设计 {Zi.shape[1]} 列（交互 {n_drug}）", flush=True)
    for li in (100.0, 300.0, 1000.0):
        lam = np.concatenate([lam_p, np.full(n_drug, li)])
        f, _ = sc(ctx, Zi, tr, lam, val, cfg, row_chunk=384, prot_chunk=192)
        rows[f"药物×时间 λ={li:g}"] = f
        print(f"{'药物×时间 λ='+f'{li:g}':<34}" + "".join(f"{f[c]:>11.4f}" for c in COLS), flush=True)
    del Zi, Zd; gc.collect()

    # ---- 3. 匹配组 one-hot ----
    # 手动建 one-hot：这块不走 design.freeze/encode，因为它不是一个真的类别特征，
    # 只是「板号 × 菌株」的乘积，冻结词表仍然只用 train 折。
    grp = ctx.meta[[PLATE, "Strains"]].astype(str).agg("\x1f".join, axis=1).to_numpy()
    levels = sorted(set(grp[tr]))
    idx = {v: i for i, v in enumerate(levels)}
    n_grp = len(levels)
    Zg = np.zeros((len(grp), n_grp), dtype=np.float32)
    for i, v in enumerate(grp):
        j = idx.get(v)
        if j is not None:
            Zg[i, j] = 1.0          # 未见组整行 0，退回板号 + 菌株的主效应
    hit_val = float(Zg[ctx.rows(ev.VAL_SPLITS)].sum(axis=1).mean())
    Zm = np.hstack([Zp, Zg])
    print()
    print(f"[3] 匹配组 one-hot  设计 {Zm.shape[1]} 列（组 {n_grp}，val 命中率 {hit_val:.1%}）", flush=True)
    for lg in (30.0, 100.0, 300.0):
        lam = np.concatenate([lam_p, np.full(n_grp, lg)])
        f, y = sc(ctx, Zm, tr, lam, val, cfg, row_chunk=256, prot_chunk=128)
        rows[f"匹配组 λ={lg:g}"] = f
        print(f"{'匹配组 λ='+f'{lg:g}':<34}" + "".join(f"{f[c]:>11.4f}" for c in COLS), flush=True)

    name, f = max(rows.items(), key=lambda kv: kv[1]["total"])
    d = f["total"] - base
    print(f"\n最优：{name}  total {f['total']:.4f}  相对当前 {d:+.4f}")
    print(f"止损线 +0.0007 → {'通过' if d >= 0.0007 else '未通过'}")
    if d >= 0.0007:
        for k, thr in (("abs_r2", -0.0015), ("ctx_resid", -0.003), ("drug_resid", -0.003)):
            dv = f[k] - f0[k]
            print(f"  {k} 变化 {dv:+.4f}（门槛 ≥ {thr}） {'通过' if dv >= thr else '不通过'}")

    out = os.path.join(paths.RESULTS, "step21_plate_extras"); paths.ensure_dir(out)
    with open(os.path.join(out, "report.txt"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"{'配置':<34}" + "".join(f"{c:>11}" for c in COLS) + "\n")
        for k, v in rows.items():
            fh.write(f"{k:<34}" + "".join(f"{v[c]:>11.4f}" for c in COLS) + "\n")
        fh.write(f"\n最优 {name} {f['total']:.6f}  {d:+.6f}\n")
    print(f"写出 {out}/report.txt")


if __name__ == "__main__":
    main()
