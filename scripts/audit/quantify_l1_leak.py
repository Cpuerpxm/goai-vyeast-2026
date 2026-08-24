"""把 L1-1 那两处违规**值多少分**量出来，而不是含糊地说「数字会变」。

复现核验要的是「代码合规」，但材料里还得回答评审的一个自然问题：
既然违规了，之前报的成绩虚高了多少？

这里把旧写法原样复刻一遍（只在本文件里，且本文件不在训练路径上，
`audit/` 也不在合规扫描的目录清单内），与合规写法在同一批数据上对打：

    A 旧 · 全表建词表 + 全表算 log-time 标准化参数 → 只用 train 行拟合
    B 新 · 词表与标准化参数都只由 train 行冻结     → 只用 train 行拟合

    C 旧 · 提交模型用全部 train_val 拟合（--fit-rows all，已删除）
    D 新 · 提交模型只用 train 折拟合

A vs B 量的是「统计量泄漏」这一处；C vs D 量的是「拟合行泄漏」那一处。
两处的量级差很远，材料里不能混着讲。

运行：python quantify_l1_leak.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import loader, paths, provenance
from data import split_guard as sg
from models.baseline_cfree import masked_ridge
from models.design import FEATURE_SETS, encode, freeze
from models.lowrank import masked_pca
from models.predict_test import fit as fit_lowrank
from models.predict_test import predict as predict_lowrank
from scorer import evaluate as ev
from scorer.config import ScorerConfig

OUT_DIR = os.path.join(paths.RESULTS, "step0_compliance")
CAT_COLS = FEATURE_SETS["bio_tech"]


def legacy_design(meta: pd.DataFrame, cols) -> np.ndarray:
    """2026-08-07 那版 `baseline_cfree.design()` 的逐行复刻（with_drug=False 分支）。

    保留在这里只为量化，不给任何训练路径调用。
    """
    blocks = [np.ones((len(meta), 1), dtype=np.float32)]
    for c in cols:
        codes, uniq = pd.factorize(meta[c].astype(str))       # ← 全表建词表
        M = np.zeros((len(meta), len(uniq)), dtype=np.float32)
        M[np.arange(len(meta)), codes] = 1.0
        blocks.append(M)
    t = np.log1p(meta["pert_time"].to_numpy(dtype=np.float64))
    t = (t - t.mean()) / t.std()                              # ← 全表算标准化参数
    blocks.append(np.stack([t, t ** 2, t ** 3], axis=1).astype(np.float32))
    return np.hstack(blocks)


def lowrank_from_Z(ctx, Z, fit_rows, k, lam):
    Xtr = ctx.X[fit_rows]
    mu, U, Ztr = masked_pca(Xtr, k, np.isfinite(Xtr), n_iter=12, center=True, seed=0)
    Zd = Z[fit_rows].astype(np.float64)
    W = np.linalg.solve(Zd.T @ Zd + lam * np.eye(Z.shape[1]), Zd.T @ Ztr.astype(np.float64))
    y = (mu[None, :] + (Z.astype(np.float64) @ W) @ U.T.astype(np.float64)).astype(np.float32)
    return np.nan_to_num(y, nan=0.0)


def main() -> None:
    cfg = ScorerConfig()
    ctx = ev.build_context(verbose=False)
    train = sg.train_rows(ctx.meta)
    val = ctx.rows(ev.VAL_SPLITS)
    k0, lam = 16, 30.0

    L: list[str] = []
    a = L.append
    a("=" * 96)
    a("L1-1 两处违规的量化（旧写法 vs 合规写法，同一批数据）")
    a("=" * 96)
    a("")

    # ---------------------------------------------------------- A vs B
    a("-" * 96)
    a("一 · 统计量泄漏：设计矩阵的词表与 log-time 标准化参数在哪估")
    a("-" * 96)
    Z_old = legacy_design(ctx.meta, CAT_COLS)
    spec = freeze(ctx.meta, train, CAT_COLS, with_drug=False)
    Z_new = encode(ctx.meta, spec)
    a(f"  旧设计矩阵 {Z_old.shape[1]} 列（含 val 独有的那一个菌株水平）")
    a(f"  新设计矩阵 {Z_new.shape[1]} 列（词表只含 train 折出现过的水平）")
    a("")

    rows = []
    for lab, Z in [("A 旧 · 全表估统计量", Z_old), ("B 新 · 只由 train 折冻结", Z_new)]:
        y_lr = lowrank_from_Z(ctx, Z, train, k0, lam)
        f_lr = ev.flatten(ev.evaluate(y_lr, ctx, val, cfg))
        mu, W = masked_ridge(Z, ctx.X, train, lam, ctx.meta)
        y_fr = np.nan_to_num((mu[None, :] + Z.astype(np.float64) @ W).astype(np.float32),
                             nan=float(np.nanmedian(mu)))
        f_fr = ev.flatten(ev.evaluate(y_fr, ctx, val, cfg))
        rows.append((lab, f_lr, f_fr, y_lr))
    a(f"  {'配置':<26}{'低秩K0=16 total':>18}{'满秩ridge total':>18}"
      f"{'低秩 abs_r2':>14}{'低秩 fc_pcc':>13}")
    for lab, f_lr, f_fr, _ in rows:
        a(f"  {lab:<26}{f_lr['total']:>18.6f}{f_fr['total']:>18.6f}"
          f"{f_lr['abs_r2']:>14.6f}{f_lr['fc_pcc']:>13.6f}")
    d_lr = rows[1][1]["total"] - rows[0][1]["total"]
    d_fr = rows[1][2]["total"] - rows[0][2]["total"]
    a("")
    a(f"  合规写法 − 旧写法：低秩 {d_lr:+.6f}   满秩 {d_fr:+.6f}")
    dy = np.abs(rows[1][3] - rows[0][3])
    a(f"  两种写法的预测矩阵逐元素差：中位 {np.median(dy):.3e}  最大 {dy.max():.3e}")
    a("")
    a("  为什么差这么小（必须解释清楚，否则读者会以为在敷衍）：")
    a("   1) val 独有的那个菌株水平在 train 行上整列为 0，岭回归给它的系数**恰好是 0**，")
    a("      于是它既不影响别的系数，也不改变任何预测——多出来的那一列是空转的。")
    a("   2) log-time 三次多项式 {u, u², u³} 在有截距的情况下，与 {t, t², t³} 张成同一个")
    a("      列空间；换标准化参数只是同一空间内的重参数化，仅通过岭惩罚的几何影响解，")
    a("      量级远小于第 4 位小数。")
    a("  → 结论：这一处违规是**程序性**的，成绩没有被它显著抬高。但它必须修，")
    a("    因为复现核验查的是代码里有没有这条路径，不是它这次值多少分；")
    a("    而且换一套数据（比如 val 独有水平在 train 里也出现过部分行）它就会真的影响结果。")
    a("")

    # ---------------------------------------------------------- C vs D
    a("-" * 96)
    a("二 · 拟合行泄漏：提交模型拿哪些行拟合（旧默认 --fit-rows all）")
    a("-" * 96)
    meta_te = loader.load_metadata("test")
    out = {}
    for lab, mask in [("C 旧 · 全部 train_val 拟合", np.ones(ctx.n, dtype=bool)),
                      ("D 新 · 只用 train 折拟合", train)]:
        sub = ctx.meta.loc[mask]
        levels = {c: sorted(sub[c].astype(str).unique().tolist()) for c in CAT_COLS}
        t = np.log1p(sub["pert_time"].to_numpy(dtype=np.float64))
        sp = {"_version": spec["_version"], "cat_cols": CAT_COLS, "levels": levels,
              "t_mean": float(t.mean()), "t_std": float(t.std()), "t_degenerate": False,
              "with_drug": False, "drug_col": spec["drug_col"], "drugs": [],
              "fit_rows_digest": "(量化用)"}
        sp["n_cols"] = 1 + sum(len(v) for v in levels.values()) + 3
        Z_tv = encode(ctx.meta, sp)
        m = fit_lowrank(ctx.X[mask], Z_tv[mask], k0, lam)
        y_te = predict_lowrank(m, encode(meta_te, sp))
        f_in = ev.flatten(ev.evaluate(predict_lowrank(m, Z_tv), ctx, val, cfg))
        out[lab] = {"n_fit": int(mask.sum()), "n_cols": sp["n_cols"],
                    "n_dead": int(m["dead"].sum()) if "dead" in m
                    else int(m["dead_cols"].sum()),
                    "val_total": f_in["total"], "y_te": y_te}
    a(f"  {'配置':<26}{'拟合行':>8}{'设计列':>8}{'无观测蛋白列':>14}"
      f"{'官方 val total':>16}")
    for lab, d in out.items():
        a(f"  {lab:<26}{d['n_fit']:>8d}{d['n_cols']:>8d}{d['n_dead']:>14d}"
          f"{d['val_total']:>16.6f}")
    a("")
    a("  ⚠ C 那一行的 val total **不是样本外分数**：val 行参与了它自己的拟合，")
    a("    所以这两个数不可直接比大小，只用来说明差异存在。")
    ya, yb = out["C 旧 · 全部 train_val 拟合"]["y_te"], out["D 新 · 只用 train 折拟合"]["y_te"]
    dd = np.abs(ya - yb)
    cc = float(np.corrcoef(ya.ravel(), yb.ravel())[0, 1])
    a("")
    a(f"  两个模型给 test 的 4,454 x 5,243 预测：")
    a(f"    逐元素绝对差 中位 {np.median(dd):.4f}  均值 {dd.mean():.4f}  "
      f"p99 {np.percentile(dd, 99):.4f}  最大 {dd.max():.4f}")
    a(f"    整体相关 {cc:.6f}")
    a("  → 这一处才是真正改变了交出去的东西：丢掉 3,038 行 val 样本（含整整一株菌），")
    a("    预测确实变了。但手册第 17 页不给选择余地，且违规后果是取消成绩。")
    a("")

    txt = "\n".join(L)
    print(txt)
    paths.ensure_dir(OUT_DIR)
    p = os.path.join(OUT_DIR, "quantify_l1_leak.txt")
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(txt)
    payload = {
        "stat_leak": {
            "legacy_cols": int(Z_old.shape[1]), "compliant_cols": int(Z_new.shape[1]),
            "lowrank_total_legacy": rows[0][1]["total"],
            "lowrank_total_compliant": rows[1][1]["total"],
            "lowrank_total_delta": d_lr,
            "fullrank_total_legacy": rows[0][2]["total"],
            "fullrank_total_compliant": rows[1][2]["total"],
            "fullrank_total_delta": d_fr,
            "prediction_abs_diff_median": float(np.median(dy)),
            "prediction_abs_diff_max": float(dy.max()),
        },
        "fitrow_leak": {
            k: {kk: vv for kk, vv in v.items() if kk != "y_te"} for k, v in out.items()
        },
        "fitrow_test_prediction_diff": {
            "median": float(np.median(dd)), "mean": float(dd.mean()),
            "p99": float(np.percentile(dd, 99)), "max": float(dd.max()),
            "corr": cc,
        },
        "_provenance": provenance.stamp(),
    }
    pj = os.path.join(OUT_DIR, "quantify_l1_leak.json")
    with open(pj, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)
    print(f"\n[写出] {p}\n[写出] {pj}")


if __name__ == "__main__":
    main()
