"""产出提交文件 `prediction.csv`（4,454 × 5,243）。

形态与报告的可提交模型一致：

    metadata --ridge--> z0 (K0=16) --U0--> 完整 5,243 维 log2 丰度预测

**推断路径不接触任何对照，也不读 `proteome_raw_test.csv`**——
后者由 `paths.assert_readable` 硬性拦截，测试侧只读 `metadata_test.csv`。

拟合行**只有 `split_final == 'train'`**。手册（2026-08 修订版）第 17 页：
「训练仅可使用 train 划分的蛋白质组标签……验证集与测试集均不得参与训练，
也不得用于估计任何统计量（含保留蛋白列表与归一化参数）」。

❗2026-08-24（复赛整改 L1-1）删掉的东西，记在这里免得有人再加回来：

- 旧版有 `--fit-rows {train,all}` 且**默认 all**，即提交模型用全部 train_val 拟合。
  当时的理由是「val 标签是组委会给的，最终提交没理由丢掉」。这个理由在
  2026-08 修订版下不成立，且违规后果是「取消成绩与参赛资格」。
  删的是**选项本身**，不是改默认值——留着开关就等于留着违规路径。
- 因此 dead_cols（无观测的蛋白列，即「保留蛋白列表」）与 log-time 标准化参数
  现在也只由 train 折估计，这正是手册点名的两类统计量。

代价要说清楚：只用 train 折就丢掉了整整一株菌（它整株落在 `val_strain_only`）。
它在 test 里其实是出现过的，真正未见的是另一株；丢掉的是 val 折那 3,038 行样本量。

用法：
    python predict_test.py                  # 拟合 train 折，写出 prediction.csv
    python predict_test.py --self-check      # 只跑校验，不写文件
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import loader, paths
from data import split_guard as sg
from models.design import FEATURE_SETS, encode, freeze, unseen_report
from models.lowrank import masked_pca
from scorer import evaluate as ev
from scorer.config import ScorerConfig

OUT_DIR = os.path.join(paths.RESULTS, "step10_submission")
CAT_COLS = FEATURE_SETS["bio_tech"]          # 菌株/培养基/温度/来源/仪器


def fit(X: np.ndarray, Z_fit: np.ndarray, k: int, lam: float, seed: int = 0) -> dict:
    """掩码 PCA 取基，再用 metadata 岭回归预测低秩系数。全部只用拟合行。

    X 与 Z_fit 必须**已经**是拟合行的子集；训练边界守卫在调用方执行
    （那里才拿得到 meta），见 build_submission。
    """
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


def build_submission(ctx, k0: int, lam: float):
    """从 ctx 拟合提交模型并预测测试集。

    抽成函数是为了让 `scripts/audit/train_boundary_probe.py` 能拿同一条路径，
    在「非 train 行被破坏」的上下文里再跑一次，逐位比对。
    返回 (y_te, meta_te, spec, model, val_scores, y_train_val)。
    """
    cfg = ScorerConfig()
    tr = sg.assert_train_only(ctx.meta, sg.train_rows(ctx.meta), what="提交模型拟合行")
    spec = freeze(ctx.meta, tr, CAT_COLS, with_drug=False)
    Z_tv = encode(ctx.meta, spec)
    model = fit(ctx.X[tr], Z_tv[tr], k0, lam)

    y_tv = predict(model, Z_tv)
    val_scores = ev.flatten(ev.evaluate(y_tv, ctx, ctx.rows(ev.VAL_SPLITS), cfg))

    meta_te = loader.load_metadata("test")
    for c in list(CAT_COLS) + ["pert_time", "sample_ID"]:
        if c not in meta_te.columns:
            raise KeyError(f"metadata_test 缺列 {c}")
    Z_te = encode(meta_te, spec)
    if Z_te.shape[1] != Z_tv.shape[1]:
        raise AssertionError(
            f"设计矩阵列数不一致 train={Z_tv.shape[1]} test={Z_te.shape[1]}——词表没冻住")
    y_te = predict(model, Z_te)
    return y_te, meta_te, spec, model, val_scores, y_tv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k0", type=int, default=16)
    ap.add_argument("--lam", type=float, default=30.0)
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()

    paths.ensure_dir(OUT_DIR)
    cfg = ScorerConfig()
    print("[提交] 载入 train_val …")
    ctx = ev.build_context(verbose=False)

    y_te, meta_te, spec, model, f, y_tv = build_submission(ctx, args.k0, args.lam)
    n_fit = int(sg.train_rows(ctx.meta).sum())

    print(f"\n[提交模型] 拟合行 {n_fit} / {ctx.n}（只有 split_final=='train'）"
          f"  K0={args.k0}  lam={args.lam}")
    print(f"  设计矩阵 {spec['n_cols']} 列；拟合行指纹 {spec['fit_rows_digest']}")
    thin = int(((model["n_obs"] > 0) & (model["n_obs"] < 10)).sum())
    print(f"  无任何观测的蛋白列 {int(model['dead_cols'].sum())}"
          f" -> 统一回退为 {model['fallback']:.4f}；观测数 <10 的列另有 {thin} 个（仍走低秩）")

    # val 分数是**样本外**的：val 行完全没参与拟合，也没参与冻结任何统计量。
    print(f"\n[样本外] 官方四类 val 总分 = {f['total']:.4f}")
    print(ev.evaluate_by_split(y_tv, ctx, cfg).to_string(index=False))

    print("\n[提交] metadata_test.csv 的未见水平（-> 该块整行 0，回退总体先验）：")
    un = unseen_report(meta_te, spec)
    if un:
        for c, d in un.items():
            print(f"  {c}: {d['n_rows']} 行为未见水平 {d['levels']}")
    else:
        print("  无")

    # ---- 落盘前的硬校验 ----
    proteins = ctx.proteins
    ok = True
    orig_ids = list(pd.read_csv(paths.assert_readable(paths.META_TEST),
                                usecols=["sample_ID"])["sample_ID"].astype(str))
    fit_splits_clean = not (ctx.meta.loc[sg.train_rows(ctx.meta), "split_final"]
                            .astype(str) != "train").any()
    checks = [
        ("行数 = 4,454", y_te.shape[0] == len(meta_te) == 4454),
        ("列数 = 5,243", y_te.shape[1] == len(proteins) == 5243),
        ("无 NaN", not np.isnan(y_te).any()),
        ("无 Inf", not np.isinf(y_te).any()),
        # 手册："不得修改样本 ID"。拿 loader 处理过的 meta 跟自己比是同义反复，
        # 必须回到磁盘上那份原始 CSV 去比顺序，才能证明没被重排过。
        ("样本 ID 顺序与 metadata_test.csv 原始行序一致",
         list(meta_te["sample_ID"].astype(str)) == orig_ids),
        ("样本 ID 无重复", meta_te["sample_ID"].astype(str).is_unique),
        ("蛋白名无重复", len(set(proteins.tolist())) == len(proteins)),
        ("数值落在训练观测范围的合理邻域",
         float(np.nanmin(y_te)) > float(np.nanmin(ctx.X)) - 5
         and float(np.nanmax(y_te)) < float(np.nanmax(ctx.X)) + 5),
        ("拟合行不含任何 val 样本", fit_splits_clean),
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
    assert len(back.columns) == 5244, f"回读列数 {len(back.columns)} != 5244"
    n_rows = sum(1 for _ in open(out, encoding="utf-8")) - 1
    assert n_rows == 4454, f"回读行数 {n_rows} != 4454"
    print(f"[回读] {n_rows} 行 x {len(back.columns)} 列（首列 sample_ID + 5,243 蛋白）OK")

    # 提交清单：预测文件的内容指纹 + 冻结 spec + 配置，供复现核验逐项对上
    sha = hashlib.sha256(open(out, "rb").read()).hexdigest()
    manifest = {
        "prediction_sha256": sha,
        "prediction_rows": n_rows,
        "prediction_cols": len(back.columns),
        "k0": args.k0,
        "lam": args.lam,
        "fit_split": "train",
        "n_fit_rows": n_fit,
        "design_spec": spec,
        "n_dead_cols": int(model["dead_cols"].sum()),
        "fallback": model["fallback"],
        "val_scores_out_of_sample": f,
        "unseen_in_test": un,
    }
    mp = os.path.join(OUT_DIR, "submission_manifest.json")
    with open(mp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    print(f"[写出] {mp}   prediction SHA-256 {sha[:16]}…")


if __name__ == "__main__":
    main()
