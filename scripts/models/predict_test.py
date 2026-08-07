"""产出提交文件 `prediction.csv`（4,454 × 5,243）。

形态与 4.2.3 报告的可提交模型一致：

    metadata --ridge--> z₀ (K₀=16) --U₀--> 完整 5,243 维 log2 丰度预测

**推断路径不接触任何对照，也不读 `proteome_raw_test.csv`**——
后者由 `paths.assert_readable` 硬性拦截，测试侧只读 `metadata_test.csv`。

❗此处不能复用 baseline_cfree.design()。那个函数用 pd.factorize 就地建词表，
列的顺序与个数取决于传进去的是哪张表；用 train_val 拟合、拿 test 去预测时
列会错位，而且**不会报错**——它只是安静地把"菌株 A 的权重"用到了菌株 B 上。
本脚本改为先从训练行冻结词表，再按同一词表给两边编码；未见水平整行为 0，
岭回归自然回退到总体先验。

用法：
    python predict_test.py                    # 用 split_final=='train' 拟合（与文档报告口径一致）
    python predict_test.py --fit-rows all     # 用全部 train_val 拟合（最终提交可选，未经 val 验证）
    python predict_test.py --self-check       # 只跑自检，不写文件
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import loader, paths
from models.baseline_cfree import FEATURE_SETS
from models.lowrank import masked_pca
from scorer import evaluate as ev
from scorer.config import ScorerConfig

OUT_DIR = os.path.join(paths.RESULTS, "step10_submission")
CAT_COLS = FEATURE_SETS["bio_tech"]          # 菌株/培养基/温度/来源/仪器


def freeze_vocab(meta: pd.DataFrame, rows: np.ndarray) -> dict:
    """只用拟合行建词表，并把 log-time 的标准化参数一并冻结。

    标准化参数若在 test 上重算，等于让测试集的分布参与了自己的预测。
    """
    sub = meta.loc[rows]
    vocab = {c: sorted(sub[c].astype(str).unique().tolist()) for c in CAT_COLS}
    t = np.log1p(sub["pert_time"].to_numpy(dtype=np.float64))
    return {"levels": vocab, "t_mean": float(t.mean()), "t_std": float(t.std())}


def encode(meta: pd.DataFrame, voc: dict) -> np.ndarray:
    """按冻结词表编码。未见水平 → 该块整行 0 → 岭回归回退到总体先验。"""
    blocks = [np.ones((len(meta), 1), dtype=np.float32)]
    for c in CAT_COLS:
        levels = voc["levels"][c]
        idx = {v: i for i, v in enumerate(levels)}
        M = np.zeros((len(meta), len(levels)), dtype=np.float32)
        for i, v in enumerate(meta[c].astype(str)):
            j = idx.get(v)
            if j is not None:
                M[i, j] = 1.0
        blocks.append(M)
    t = np.log1p(meta["pert_time"].to_numpy(dtype=np.float64))
    t = (t - voc["t_mean"]) / voc["t_std"]
    blocks.append(np.stack([t, t ** 2, t ** 3], axis=1).astype(np.float32))
    return np.hstack(blocks)


def fit(X: np.ndarray, Z_fit: np.ndarray, k: int, lam: float, seed: int = 0) -> dict:
    """掩码 PCA 取基，再用 metadata 岭回归预测低秩系数。全部只用拟合行。"""
    obs = np.isfinite(X)
    mu, U, Zc = masked_pca(X, k, obs, n_iter=12, center=True, seed=seed)
    Zd = Z_fit.astype(np.float64)
    G = Zd.T @ Zd + lam * np.eye(Z_fit.shape[1])
    W = np.linalg.solve(G, Zd.T @ Zc.astype(np.float64))

    # ❗全缺失列必须按**观测计数**判定，不能看 mu 是不是 NaN。
    # masked_pca 内部把无观测列的 mu 置成了 0（lowrank.py 第 56 行），
    # 于是 isfinite(mu) 对它们恒为真，回退分支永远不触发，
    # 这些列会带着 mu=0 走完低秩解码，落盘变成 log2 丰度 ≈ 0
    # ——相当于宣称"该蛋白丰度为 1"，比缺失还糟。2026-08-07 自检抓到。
    n_obs = obs.sum(axis=0)
    dead = n_obs == 0
    fallback = float(np.median(mu[~dead])) if (~dead).any() else 0.0
    return {"mu": mu, "U": U, "W": W, "fallback": fallback,
            "dead_cols": dead, "n_obs": n_obs}


def predict(model: dict, Z: np.ndarray) -> np.ndarray:
    mu, U, W = model["mu"], model["U"], model["W"]
    z_hat = Z.astype(np.float64) @ W
    y = np.nan_to_num(mu, nan=model["fallback"])[None, :] + z_hat @ U.T.astype(np.float64)
    y = y.astype(np.float32)
    y[:, model["dead_cols"]] = model["fallback"]     # 无信号列：确定性回退
    return y


def report_unseen(meta_te: pd.DataFrame, voc: dict) -> None:
    """未见水平的覆盖情况——评审必问 S2 未见菌株怎么处理。"""
    for c in CAT_COLS:
        seen = set(voc["levels"][c])
        vals = meta_te[c].astype(str)
        bad = ~vals.isin(seen)
        if bad.any():
            print(f"  {c}: {int(bad.sum())} 行为未见水平 {sorted(set(vals[bad]))}"
                  f" → 该块整行 0，回退总体先验")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k0", type=int, default=16)
    ap.add_argument("--lam", type=float, default=30.0)
    ap.add_argument("--fit-rows", choices=["train", "all"], default="all")
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()

    paths.ensure_dir(OUT_DIR)
    cfg = ScorerConfig()
    print("[提交] 载入 train_val …")
    ctx = ev.build_context(verbose=False)

    # 第一步永远先跑 train-only 拟合，用来在官方四类 val 上给出**样本外**分数。
    # 这是文档中报告的那个数；提交模型若改用全部 train_val 拟合，
    # 它在 val 上就是样本内，不能再拿来当成绩。两者必须分开。
    tr = (ctx.meta["split_final"] == "train").to_numpy()
    voc_tr = freeze_vocab(ctx.meta, tr)
    Z_tr_all = encode(ctx.meta, voc_tr)
    m_tr = fit(ctx.X[tr], Z_tr_all[tr], args.k0, args.lam)
    y_tv = predict(m_tr, Z_tr_all)
    f = ev.flatten(ev.evaluate(y_tv, ctx, ctx.rows(ev.VAL_SPLITS), cfg))
    print(f"\n[样本外自检] train 折拟合（{int(tr.sum())}行），官方四类 val 总分 = {f['total']:.4f}")
    by = ev.evaluate_by_split(y_tv, ctx, cfg)
    print(by.to_string(index=False))

    # 第二步：提交模型。默认用全部 train_val 拟合——
    # val 行的标签是组委会给的，最终提交没有理由丢掉；
    # 尤其有一株菌株只出现在 val_strain_only，只用 train 折会把它整株丢掉，
    # 到 test 上反而变成"未见菌株"，白白损失一株的信息。
    fit_mask = (np.ones(ctx.n, dtype=bool) if args.fit_rows == "all" else tr)
    print(f"\n[提交模型] 拟合行 {int(fit_mask.sum())} / {ctx.n}"
          f"（--fit-rows {args.fit_rows}）  K₀={args.k0}  λ={args.lam}")
    voc = freeze_vocab(ctx.meta, fit_mask)
    Z_tv = encode(ctx.meta, voc)
    model = fit(ctx.X[fit_mask], Z_tv[fit_mask], args.k0, args.lam)
    thin = int(((model["n_obs"] > 0) & (model["n_obs"] < 10)).sum())
    print(f"  设计矩阵 {Z_tv.shape[1]} 列；无任何观测的蛋白列 {int(model['dead_cols'].sum())}"
          f" → 统一回退为 {model['fallback']:.4f}；观测数 <10 的列另有 {thin} 个（仍走低秩）")

    # ---- 测试侧：只读 metadata，绝不碰 proteome_raw_test.csv ----
    print("\n[提交] 载入 metadata_test.csv（测试侧只读元数据）…")
    meta_te = loader.load_metadata("test")
    for c in CAT_COLS + ["pert_time", "sample_ID"]:
        if c not in meta_te.columns:
            raise KeyError(f"metadata_test 缺列 {c}")
    Z_te = encode(meta_te, voc)
    assert Z_te.shape[1] == Z_tv.shape[1], (
        f"设计矩阵列数不一致 train={Z_tv.shape[1]} test={Z_te.shape[1]}——词表没冻住")
    y_te = predict(model, Z_te)
    report_unseen(meta_te, voc)

    # ---- 落盘前的硬校验 ----
    proteins = ctx.proteins
    ok = True
    checks = [
        ("行数 = 4,454", y_te.shape[0] == len(meta_te) == 4454),
        ("列数 = 5,243", y_te.shape[1] == len(proteins) == 5243),
        ("无 NaN", not np.isnan(y_te).any()),
        ("无 Inf", not np.isinf(y_te).any()),
        # 手册："不得修改样本 ID"。拿 loader 处理过的 meta 跟自己比是同义反复，
        # 必须回到磁盘上那份原始 CSV 去比顺序，才能证明没被重排过。
        ("样本 ID 顺序与 metadata_test.csv 原始行序一致",
         list(meta_te["sample_ID"].astype(str))
         == list(pd.read_csv(paths.assert_readable(paths.META_TEST),
                             usecols=["sample_ID"])["sample_ID"].astype(str))),
        ("样本 ID 无重复", meta_te["sample_ID"].astype(str).is_unique),
        ("蛋白名无重复", len(set(proteins.tolist())) == len(proteins)),
        ("数值落在训练观测范围的合理邻域",
         float(np.nanmin(y_te)) > float(np.nanmin(ctx.X)) - 5
         and float(np.nanmax(y_te)) < float(np.nanmax(ctx.X)) + 5),
    ]
    print("\n[校验]")
    for name, cond in checks:
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok &= bool(cond)
    print(f"  预测值范围 [{y_te.min():.3f}, {y_te.max():.3f}]  "
          f"训练观测范围 [{np.nanmin(ctx.X):.3f}, {np.nanmax(ctx.X):.3f}]")
    if not ok:
        print("\n❗校验未通过，不写出提交文件")
        sys.exit(1)

    if args.self_check:
        print("\n[自检模式] 校验通过，按要求不写文件")
        return

    out = os.path.join(OUT_DIR, "prediction.csv")
    df = pd.DataFrame(y_te, columns=proteins.tolist())
    df.insert(0, "sample_ID", meta_te["sample_ID"].astype(str).to_numpy())
    df.to_csv(out, index=False, float_format="%.6f", encoding="utf-8", lineterminator="\n")
    size = os.path.getsize(out) / 1e6
    print(f"\n[写出] {out}   {size:.1f} MB")

    # 回读验证：写出去的东西必须能原样读回来
    back = pd.read_csv(out, nrows=5)
    assert list(back.columns[:1]) == ["sample_ID"], "首列不是 sample_ID"
    assert len(back.columns) == 5244, f"回读列数 {len(back.columns)} ≠ 5244"
    n_rows = sum(1 for _ in open(out, encoding="utf-8")) - 1
    assert n_rows == 4454, f"回读行数 {n_rows} ≠ 4454"
    print(f"[回读] {n_rows} 行 × {len(back.columns)} 列（首列 sample_ID + 5,243 蛋白）✅")


if __name__ == "__main__":
    main()
