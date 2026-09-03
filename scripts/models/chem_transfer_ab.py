"""用分子结构给**未见药物**补一个响应预测。

这是眼下唯一还能动的那块空白：test 有 2,769 行的药物在训练折里没出现过，
药物 one-hot 对它们整块是 0，模型等于预测「这个药什么也没干」。
而这 2,769 行**全部**有 SMILES（val 那边 1,334 行也全部有），所以结构信息是齐的。

为什么不直接把描述符当特征塞进设计矩阵：训练折只有 37 个有 SMILES 的药物，
而药物 one-hot 正好有 37 列，one-hot 会把药物那一层方差吃干净，
描述符块的系数会被压到 0，学不到东西。

所以走两步：

  1. 照常拟合（含药物 one-hot），拿到已见药物的效应矩阵 E(37 × 4,422)，
     它就是 W 里药物那几行。
  2. 在这 37 个点上做岭回归 E ~ D（D 是标准化后的描述符矩阵），
     得到「结构 → 响应」的映射 C(n_desc × 4,422)。
  3. 未见药物用它自己的描述符 d 算 α·(d·C) 补进预测；α 是全局收缩系数。

和初赛 B3 那个化学近邻的区别：B3 是挑一个最相似的药把它的 Δ 整条搬过来，
这里是把 37 个药一起回归出一张结构到响应的映射，再对新分子求值。
B3 输给上下文均值基线（0.3208 vs 0.3393），但那不代表回归也不行。

外部资源：`data/external/compound_smiles.csv`（PubChem 离线映射，已在依赖披露里）。
描述符由本机 rdkit 2024.09.5 现算，不联网。
"""
from __future__ import annotations
import os, sys, gc
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import paths, split_guard as sg
from models.baseline_cfree import ChunkedGram
from models.design import freeze, encode, FEATURE_SETS
from models.final_grid import build
from scorer import evaluate as ev
from scorer.config import ScorerConfig

COLS = ["total", "abs_pcc", "abs_r2", "fc_pcc", "ctx_resid", "drug_resid"]
LAM = (3.0, 10.0, 100.0, 30.0)
PERT = "perturbation_no_concentration"

DESC = ["MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors",
        "NumRotatableBonds", "RingCount", "NumAromaticRings", "FractionCSP3",
        "HeavyAtomCount", "NumHeteroatoms", "MolMR", "BertzCT", "Chi0n",
        "Kappa2", "NumSaturatedRings", "NHOHCount", "NOCount", "LabuteASA", "qed"]


def descriptors(smiles_map):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, QED
    RDLogger.DisableLog("rdApp.*")
    fn = {n: f for n, f in Descriptors.descList}
    fn["qed"] = lambda m: QED.qed(m)
    out = {}
    for name, smi in smiles_map.items():
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        try:
            out[name] = np.array([float(fn[d](m)) for d in DESC], dtype=np.float64)
        except Exception:
            continue
    return out


def main():
    cfg = ScorerConfig()
    ctx = ev.build_context(verbose=False)
    val = ctx.rows(ev.VAL_SPLITS)
    tr = sg.train_rows(ctx.meta)
    Z, n_ctx, n_plate, n_drug, _ = build(ctx, tr)
    lc, lp, ld, li = LAM
    lam = np.concatenate([np.full(n_ctx, lc), np.full(n_plate, lp),
                          np.full(n_drug, ld), np.full(n_drug, li)])
    g = ChunkedGram(Z, ctx.X, tr, ctx.meta)
    mu, W = g.solve(lam)

    def sc(y, tag, store):
        f = ev.flatten(ev.evaluate(y, ctx, val, cfg))
        store[tag] = f
        print(f"{tag:<34}" + "".join(f"{f[c]:>11.4f}" for c in COLS), flush=True)
        return f

    y0 = np.nan_to_num((mu[None, :] + Z.astype(np.float64) @ W).astype(np.float32),
                       nan=float(np.nanmedian(mu)))
    print(f"{'配置':<34}" + "".join(f"{c:>11}" for c in COLS), flush=True)
    print("-" * (34 + 11 * len(COLS)))
    rows = {}
    f0 = sc(y0, "当前（未见药物整块 0）", rows)
    base = f0["total"]

    # 训练折冻结的药物词表 = W 里药物块的行序
    spec = freeze(ctx.meta, tr, list(FEATURE_SETS["bio_tech"]) + ["Yeast_cell_plate"],
                  with_drug=True)
    drugs = spec["drugs"]
    assert len(drugs) == n_drug

    sm = pd.read_csv(os.path.join(paths.DATA_EXTERNAL, "compound_smiles.csv"))
    smap = dict(zip(sm["compound"].astype(str), sm["smiles"].astype(str)))
    dmap = descriptors(smap)
    print(f"\nrdkit 算出描述符的化合物 {len(dmap)} 个，维度 {len(DESC)}")

    fit_drugs = [d for d in drugs if d in dmap]
    print(f"训练折药物里有描述符的 {len(fit_drugs)} / {len(drugs)}")
    D_fit = np.vstack([dmap[d] for d in fit_drugs])
    dmu, dsd = D_fit.mean(axis=0), D_fit.std(axis=0)
    dsd[dsd < 1e-9] = 1.0
    D_fit = (D_fit - dmu) / dsd
    E = W[n_ctx + n_plate: n_ctx + n_plate + n_drug]          # (n_drug, p)
    E_fit = E[[drugs.index(d) for d in fit_drugs]]            # (n_fit, p)

    # 未见药物的行
    pert = ctx.meta[PERT].astype(str).to_numpy()
    unseen = np.array([(p not in drugs) and (p in dmap) for p in pert])
    print(f"train_val 里未见且有描述符的行 {int(unseen.sum())}")
    D_un = np.zeros((len(pert), len(DESC)), dtype=np.float64)
    for i in np.nonzero(unseen)[0]:
        D_un[i] = (dmap[pert[i]] - dmu) / dsd

    print()
    G_d = D_fit.T @ D_fit
    best_y = [None, -1.0]
    for lam_c in (3.0, 10.0, 30.0):
        C = np.linalg.solve(G_d + lam_c * np.eye(len(DESC)), D_fit.T @ E_fit)   # (n_desc, p)
        add_full = D_un @ C                                                     # (n, p)
        for alpha in (0.5, 1.0, 1.5, 2.0):
            y = y0.astype(np.float64).copy()
            y[unseen] += alpha * add_full[unseen]
            yf = y.astype(np.float32)
            f_ = sc(yf, f"结构迁移 λ_C={lam_c:g} α={alpha:g}", rows)
            if f_["total"] > best_y[1]:
                best_y = [yf, f_["total"]]
            del y; gc.collect()
        del C, add_full; gc.collect()

    name, f = max(rows.items(), key=lambda kv: kv[1]["total"])
    d = f["total"] - base
    print(f"\n最优：{name}  total {f['total']:.4f}  相对当前 {d:+.4f}")
    print(f"止损线 +0.0007 → {'通过' if d >= 0.0007 else '未通过，结构迁移不采用'}")
    if d >= 0.0007:
        for k, thr in (("abs_r2", -0.0015), ("ctx_resid", -0.003), ("drug_resid", -0.003)):
            dv = f[k] - f0[k]
            print(f"  {k} 变化 {dv:+.4f}（门槛 ≥ {thr}） {'通过' if dv >= thr else '不通过'}")

    if best_y[0] is not None:
        print()
        print("逐 split（最优结构迁移）：")
        print(ev.evaluate_by_split(best_y[0], ctx, cfg).to_string(index=False))
        print()
        print("逐 split（不加结构迁移）：")
        print(ev.evaluate_by_split(y0, ctx, cfg).to_string(index=False))

    out = os.path.join(paths.RESULTS, "step28_chem_transfer"); paths.ensure_dir(out)
    with open(os.path.join(out, "report.txt"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"{'配置':<34}" + "".join(f"{c:>11}" for c in COLS) + "\n")
        for k, v in rows.items():
            fh.write(f"{k:<34}" + "".join(f"{v[c]:>11.4f}" for c in COLS) + "\n")
        fh.write(f"\n最优 {name} {f['total']:.6f}  {d:+.6f}\n")
    print(f"写出 {out}/report.txt")


if __name__ == "__main__":
    main()
