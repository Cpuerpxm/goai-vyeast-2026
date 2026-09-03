"""拿我们自己这份预测，试几种「别人可能这么算」的读法，看哪种能出 0.9+。

不是猜别人做了什么，是量一件事：从同一份预测出发，换一种对手册的读法，
数字能被抬到多高。能抬到 0.9 的读法如果存在，那 0.9 就不必解释成作弊。
"""
from __future__ import annotations
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import paths
from models.predict_test import build_submission_fullrank, LAM_CTX_DEFAULT, LAM_DRUG_DEFAULT
from scorer import evaluate as ev
from scorer.config import ScorerConfig


def pooled_pcc(a, b):
    """把整个 (样本 × 蛋白) 矩阵拉平算一个相关，只用两边都非缺失的位置。"""
    m = np.isfinite(a) & np.isfinite(b)
    x, y = a[m].ravel(), b[m].ravel()
    if x.size < 10:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def main():
    cfg = ScorerConfig()
    ctx = ev.build_context(verbose=False)
    val = ctx.rows(ev.VAL_SPLITS)
    _, _, _, _, f_official, y = build_submission_fullrank(ctx, LAM_CTX_DEFAULT, LAM_DRUG_DEFAULT)

    rows = np.nonzero(val & ctx.treated)[0]
    rows_all = np.nonzero(val)[0]
    D_true = ctx.D[rows]
    D_pred = y[rows].astype(np.float64) - ctx.C[rows].astype(np.float64)
    X_true = ctx.X[rows_all]
    X_pred = y[rows_all]

    print("我们对外报的六项加权总分：", f"{f_official['total']:.4f}")
    print("其中 Δ 类三项：", f"fc {f_official['fc_pcc']:.4f}  "
          f"ctx {f_official['ctx_resid']:.4f}  drug {f_official['drug_resid']:.4f}\n")

    print("同一份预测，换几种读法后 Δ 类那一项会变成多少：")
    print(f"{'读法':<44}{'值':>10}")
    print("-" * 54)

    # 1) 我们的：逐样本 PCC 求平均，逐蛋白 PCC 求平均，两轴再平均
    print(f"{'我们的：两轴分别聚合再平均':<44}{f_official['fc_pcc']:>10.4f}")

    # 2) 把整个矩阵拉平算一个 PCC
    print(f"{'把整个矩阵拉平算一个 PCC':<44}{pooled_pcc(D_true, D_pred):>10.4f}")

    # 3) 逐样本 PCC 只取样本轴
    from scorer.metrics import pcc_axis
    vs = pcc_axis(D_true, D_pred, cfg, axis=1)
    vp = pcc_axis(D_true, D_pred, cfg, axis=0)
    print(f"{'只沿样本轴':<44}{np.nanmean(vs):>10.4f}")
    print(f"{'只沿蛋白轴':<44}{np.nanmean(vp):>10.4f}")

    # 4) 若误把「原始 FC」读成绝对丰度（不减对照）
    va = pcc_axis(X_true, X_pred, cfg, axis=1)
    print(f"{'误读成不减对照（直接比绝对丰度）':<44}{np.nanmean(va):>10.4f}")
    print(f"{'同上，整个矩阵拉平':<44}{pooled_pcc(X_true, X_pred):>10.4f}")

    print("\n若六项全部按「不减对照」这一读法计分，加权总分约：")
    a = np.nanmean(va)
    print(f"  0.20×{a:.4f} + 0.25×{a:.4f} + 0.20×{a:.4f} + 0.20×{a:.4f} "
          f"+ 0.10×{a:.4f} + 0.05×0.65  ≈  {0.95*a + 0.05*0.65:.4f}")

    out = os.path.join(paths.RESULTS, "step17_convention"); paths.ensure_dir(out)
    with open(os.path.join(out, "what_gives_09.txt"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"官方六项加权 {f_official['total']:.4f}\n")
        fh.write(f"Δ 拉平 PCC {pooled_pcc(D_true, D_pred):.4f}\n")
        fh.write(f"Δ 样本轴 {np.nanmean(vs):.4f}  蛋白轴 {np.nanmean(vp):.4f}\n")
        fh.write(f"绝对丰度样本轴 {a:.4f}  拉平 {pooled_pcc(X_true, X_pred):.4f}\n")
        fh.write(f"全按绝对丰度读法的加权总分 ≈ {0.95*a + 0.05*0.65:.4f}\n")
    print(f"\n写出 {out}/what_gives_09.txt")


if __name__ == "__main__":
    main()
