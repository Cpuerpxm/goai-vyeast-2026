"""共享参照污染的可复现实验（补 backlog L2-2）。

此前「把对照错配到别的样本后 FC 从 0.185 掉到 0.006」这个数字只在一次性调用里跑过，
没有冻结脚本。Pro R2 指出：无对应代码与日志的数字只能算报告值，不能作为证据。
本脚本把它固化。

机制：评分器定义 `Δ_pred = ŷ − y_ctrl`、`Δ_true = y − y_ctrl`，两者共享同一条**真实**
对照向量。因此即使 ŷ 完全不含药物信息，残留的 −y_ctrl 也会与真值里的同一项对上。

三个对照条件：
  A 正确匹配   Δ_pred = ŷ − C_own      ← 官方口径
  B 全局错配   Δ_pred = ŷ − C_random   ← 打断共享，看还剩多少
  C 同上下文错配 Δ_pred = ŷ − C_同条件他样本 ← 只打断"同一次测量"，保留条件相关

若 A 远高于 B，说明该分数主要来自共享参照而非药物知识。
C 用来区分"共享同一次测量"与"共享同一实验条件"两种成分。

运行：python shared_reference_probe.py
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import paths
from scorer import evaluate as ev
from scorer.config import ScorerConfig
from scorer.metrics import pcc_axis

CTX_KEYS = ["Strains", "Medium", "Temperature", "pert_time"]
OUT_DIR = os.path.join(paths.RESULTS, "step3_diagnostics")


def main() -> None:
    paths.ensure_dir(OUT_DIR)
    cfg = ScorerConfig()
    ctx = ev.build_context()
    rng = np.random.default_rng(20260805)

    ok = ctx.treated & np.isfinite(ctx.D).any(axis=1)
    rows = np.nonzero(ok & ctx.rows(ev.VAL_SPLITS))[0]

    # 零知识预测：直接复用基线阶梯里的 B0，**不另起一套定义**。
    # 曾经这里自己算过一遍"训练集全局均值"，与 B0 的差别仅在于是否含对照样本，
    # 结果同一个量出现 0.1834 与 0.1845 两个值，被外部审查判为口径不一致。
    # 单一来源是唯一可靠的防线（GPT Pro R3 · L1-13）。
    from models.baselines import b0_global_mean

    y_hat = b0_global_mean(ctx)

    # 三种对照配法
    perm_global = rng.permutation(rows)
    key = ctx.meta[CTX_KEYS].astype(str).agg("\x1f".join, axis=1).to_numpy()
    perm_ctx = rows.copy()
    for k in np.unique(key[rows]):
        idx = np.nonzero(key[rows] == k)[0]
        if idx.size > 1:
            perm_ctx[idx] = rows[idx][rng.permutation(idx.size)]

    def fc(ctrl_rows):
        dp = y_hat[rows].astype(np.float64) - ctx.C[ctrl_rows].astype(np.float64)
        s = np.nanmean(pcc_axis(ctx.D[rows], dp, cfg, axis=1))
        p = np.nanmean(pcc_axis(ctx.D[rows], dp, cfg, axis=0))
        return s, p

    a_s, a_p = fc(rows)
    b_s, b_p = fc(perm_global)
    c_s, c_p = fc(perm_ctx)

    L = []
    w = L.append
    w("=" * 84)
    w("共享参照污染 · 可复现实验")
    w("=" * 84)
    w(f"预测：训练集全局均值谱（零药物知识），评估行 {len(rows)}（官方四类 val 的处理样本）")
    w("Δ_true 始终用正确匹配的对照；只改 Δ_pred 里减去的那条对照。")
    w("")
    w(f"  {'条件':<34}{'fc_pcc 样本轴':>14}{'fc_pcc 蛋白轴':>14}")
    w("  " + "-" * 62)
    w(f"  {'A 正确匹配（官方口径）':<34}{a_s:>14.4f}{a_p:>14.4f}")
    w(f"  {'B 全局错配到随机样本':<34}{b_s:>14.4f}{b_p:>14.4f}")
    w(f"  {'C 错配到同上下文的别的样本':<34}{c_s:>14.4f}{c_p:>14.4f}")
    w("")
    w(f"A − B = {a_s - b_s:+.4f}（样本轴）：这一整块来自「与真值共享同一条真实对照向量」")
    w(f"A − C = {a_s - c_s:+.4f}（样本轴）：其中来自「共享同一次具体测量」的部分")
    w(f"C − B = {c_s - b_s:+.4f}（样本轴）：来自「共享同一实验条件」的部分")
    w("")
    w("判读：一个对药物一无所知的模型在指标 2（25% 权重）上拿到 A 这么多分，")
    w("而打断共享后掉到 B。**指标 2 不可单独作为模型选择依据。**")
    w("")
    w("⚠ 这不等于评分器实现有错，也不等于参赛者在钻空子——它是评分**定义**本身")
    w("  （Δ 两侧都减同一条观测对照）的必然结果。合规的做法是：模型只用允许的输入")
    w("  产出稳定的绝对预测，让评分器自身的机制自然生效，并在报告里披露该机制。")

    txt = "\n".join(L)
    print(txt)
    p = os.path.join(OUT_DIR, "shared_reference_probe.txt")
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(txt)
    print(f"\n[写出] {p}")


if __name__ == "__main__":
    main()
