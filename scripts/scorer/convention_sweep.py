"""同一份预测，换官方未定义的那几个计分约定，总分会跳到哪里。

写这个是为了回答一个具体的问题：我们报的 0.51 有多少是模型的，
多少是我们自己复刻评分器时替官方做的选择。

`scripts/scorer/config.py` 开头就写了「官方口径尚有未定项（多对照合并、缺失掩码、
常数向量、聚合方式），因此全部做成参数」。本脚本把每一项单独拨一格，看总分怎么动。
"""
from __future__ import annotations
import os, sys
from dataclasses import replace
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import paths, split_guard as sg
from models.predict_test import build_submission_fullrank, LAM_CTX_DEFAULT, LAM_DRUG_DEFAULT
from scorer import evaluate as ev
from scorer.config import ScorerConfig

COLS = ["total", "abs_pcc", "abs_r2", "fc_pcc", "ctx_resid", "drug_resid", "both_time"]


def main():
    base = ScorerConfig()
    ctx = ev.build_context(verbose=False)
    val = ctx.rows(ev.VAL_SPLITS)
    _, _, _, _, _, y = build_submission_fullrank(ctx, LAM_CTX_DEFAULT, LAM_DRUG_DEFAULT)

    variants = [
        ("我们的默认（材料里报的）", base),
        ("Δ 类只沿样本轴聚合", replace(base, axis_combine="sample_only")),
        ("Δ 类只沿蛋白轴聚合", replace(base, axis_combine="protein_only")),
        ("绝对项也走两轴平均", replace(base, absolute_axis="mean")),
        ("聚合用中位数不用均值", replace(base, agg="median")),
        ("未定义的模块踢出分母", replace(base, undefined_module="renorm")),
        ("未定义的轴踢出平均", replace(base, undefined_axis="drop")),
        ("单次 PCC 最少有效点 30→5", replace(base, min_valid_points=5)),
        ("多对照取均值不取中位数", replace(base, control_agg="mean")),
        ("指标 5 只算 FC 不含绝对项", replace(base, both_time_parts="fc_only")),
    ]

    print(f"{'计分约定':<30}" + "".join(f"{c:>11}" for c in COLS))
    print("-" * (30 + 11 * len(COLS)))
    rows = []
    for name, cfg in variants:
        if cfg.control_agg != base.control_agg:
            c2 = ev.build_context(verbose=False, cfg=cfg) if _accepts_cfg() else None
            if c2 is None:
                print(f"{name:<30}{'（需重建 ctx，跳过）':>20}")
                continue
            f = ev.flatten(ev.evaluate(y, c2, c2.rows(ev.VAL_SPLITS), cfg))
        else:
            f = ev.flatten(ev.evaluate(y, ctx, val, cfg))
        rows.append((name, f))
        print(f"{name:<30}" + "".join(f"{f[c]:>11.4f}" for c in COLS))

    lo = min(r[1]["total"] for r in rows)
    hi = max(r[1]["total"] for r in rows)
    print(f"\n同一份预测，换约定后总分区间 [{lo:.4f}, {hi:.4f}]，跨度 {hi-lo:.4f}")
    print(f"我们对外报的是 {rows[0][1]['total']:.4f}")

    out = os.path.join(paths.RESULTS, "step17_convention"); paths.ensure_dir(out)
    with open(os.path.join(out, "report.txt"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"{'计分约定':<30}" + "".join(f"{c:>11}" for c in COLS) + "\n")
        for name, f in rows:
            fh.write(f"{name:<30}" + "".join(f"{f[c]:>11.4f}" for c in COLS) + "\n")
        fh.write(f"\n区间 [{lo:.4f}, {hi:.4f}] 跨度 {hi-lo:.4f}\n")
    print(f"写出 {out}/report.txt")


def _accepts_cfg():
    import inspect
    return "cfg" in inspect.signature(ev.build_context).parameters


if __name__ == "__main__":
    main()
