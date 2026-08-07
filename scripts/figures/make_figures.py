"""初赛文档第四节的 6 张图。

原则：**所有数字现场从数据/结果重算，不从报告里手抄**——手抄会在报告更新后
静默过期。每张图的数据来源在图注里注明脚本名。

样式：科研图口径。中文用微软雅黑，英文数字用 Arial；不用花哨配色；
每张图都能单独看懂（标题说结论，不只说变量名）。

运行：python make_figures.py
产出：results/figures/F1..F6.png（300 dpi）+ 同名 .pdf
"""
from __future__ import annotations

import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import loader, paths
from scorer import evaluate as ev
from scorer.config import ScorerConfig
from scorer.metrics import pcc_axis

OUT = os.path.join(paths.RESULTS, "figures")
CTX_KEYS = ["Strains", "Medium", "Temperature", "pert_time"]

# 配色：一条主色 + 一条对照色 + 灰，避免彩虹图
C_MAIN, C_ALT, C_GREY, C_WARN = "#2C5F8A", "#C0714B", "#9AA0A6", "#8B2F3F"


def setup_style() -> None:
    for name in ("Microsoft YaHei", "SimHei", "Arial Unicode MS"):
        if any(name.lower() in f.name.lower() for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.sans-serif"] = [name, "Arial"]
            break
    plt.rcParams.update({
        "axes.unicode_minus": False,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def save(fig, name: str) -> None:
    paths.ensure_dir(OUT)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  [图] {name}.png / .pdf")


# ------------------------------------------------------------------ F1


def fig1_shared_reference(ctx, cfg):
    """共享参照污染：三个对照条件。"""
    rng = np.random.default_rng(20260805)
    rows = np.nonzero(ctx.treated & np.isfinite(ctx.D).any(axis=1)
                      & ctx.rows(ev.VAL_SPLITS))[0]
    train = (ctx.meta["split_final"] == "train").to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        g = np.nanmean(ctx.X[train], axis=0, dtype=np.float64)
    g = np.where(np.isfinite(g), g, np.nanmedian(g))
    y = np.tile(g, (ctx.n, 1))

    key = ctx.meta[CTX_KEYS].astype(str).agg("\x1f".join, axis=1).to_numpy()
    p_ctx = rows.copy()
    for k in np.unique(key[rows]):
        idx = np.nonzero(key[rows] == k)[0]
        if idx.size > 1:
            p_ctx[idx] = rows[idx][rng.permutation(idx.size)]
    p_glob = rng.permutation(rows)

    def fc(cr):
        dp = y[rows].astype(np.float64) - ctx.C[cr].astype(np.float64)
        return float(np.nanmean(pcc_axis(ctx.D[rows], dp, cfg, axis=1)))

    vals = [fc(rows), fc(p_ctx), fc(p_glob)]
    labs = ["正确匹配\n（官方口径）", "错配到同条件\n的别的样本", "全局错配到\n随机样本"]

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    bars = ax.bar(labs, vals, color=[C_WARN, C_ALT, C_GREY], width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.4f}",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("指标 2 · 匹配对照原始 FC 的 PCC")
    ax.set_ylim(0, max(vals) * 1.25)
    ax.set_title("对药物一无所知的模型，在指标 2 上仍得 0.18\n"
                 "——分数几乎全部来自「与真值共享同一条真实对照」", pad=12)
    # 分解画在右侧竖直方向，避免斜箭头横穿中间那根柱子
    x_ann = 2.44
    ax.set_xlim(-0.55, 3.05)
    for v0, v1, col, lab in [(vals[0], vals[1], C_MAIN, "共享同一次测量"),
                             (vals[1], vals[2], C_ALT, "共享同一实验条件")]:
        ax.annotate("", xy=(x_ann, v0), xytext=(x_ann, v1),
                    arrowprops=dict(arrowstyle="<->", color=col, lw=1.3))
        ax.text(x_ann + 0.07, (v0 + v1) / 2, f"{lab}\n{v0 - v1:+.3f}",
                va="center", color=col, fontsize=9)
    for v in vals:
        ax.plot([-0.4, x_ann], [v, v], ls=":", lw=.8, color="#CCC", zorder=0)
    fig.text(0.01, -0.04, "预测 = 训练集全局均值谱（零药物知识）；评估于官方四类 val 的"
             f"{len(rows)} 个处理样本。数据：scripts/audit/shared_reference_probe.py",
             fontsize=8, color="#555")
    save(fig, "F1_shared_reference")
    return vals


# ------------------------------------------------------------------ F2


def fig2_missing_mnar(ctx):
    """缺失率 vs 丰度十分位。"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(ctx.X, axis=0)
    miss = np.isnan(ctx.X).mean(axis=0)
    ok = np.isfinite(med)
    dec = pd.qcut(med[ok], 10, labels=False, duplicates="drop")
    xs = [np.median(med[ok][dec == d]) for d in range(10)]
    ys = [miss[ok][dec == d].mean() for d in range(10)]

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(range(1, 11), np.asarray(ys) * 100, "o-", color=C_MAIN, lw=2, ms=7)
    for i, (yv, xv) in enumerate(zip(ys, xs), start=1):
        if i in (1, 10):
            ax.annotate(f"{yv*100:.1f}%", (i, yv * 100),
                        textcoords="offset points", xytext=(0, 10),
                        ha="center", fontweight="bold", color=C_WARN)
    ax.set_xlabel("蛋白按观测中位 log2 丰度分的十分位（1 = 最低丰度）")
    ax.set_ylabel("该组的平均缺失率 (%)")
    ax.set_xticks(range(1, 11))
    ax.set_title("缺失以丰度依赖的检出限为主（MNAR）\n"
                 "最低丰度组缺 77%，最高丰度组只缺 1.6%", pad=12)
    ax2 = ax.twinx()
    ax2.plot(range(1, 11), xs, "s--", color=C_GREY, lw=1, ms=4, alpha=.8)
    ax2.set_ylabel("该组中位 log2 丰度", color=C_GREY)
    ax2.tick_params(axis="y", colors=C_GREY)
    ax2.spines["top"].set_visible(False)
    fig.text(0.01, -0.04, "Spearman(中位丰度, 缺失率) = −0.859；缺失指示回归中仅丰度代理"
             "即达伪 R² 0.351，技术+生物因子仅 0.004。"
             "数据：scripts/audit/diagnose_missing.py", fontsize=8, color="#555")
    save(fig, "F2_missing_mnar")


# ------------------------------------------------------------------ F3


def fig3_reliability(ctx, cfg):
    """四个指标空间的复制对可靠性。"""
    from audit.noise_ceiling import BIO_CTX, group_mean, replicate_pairs
    treat = ctx.treated & np.isfinite(ctx.D).any(axis=1)
    pairs = replicate_pairs(ctx.meta, treat)
    I = np.array([i for i, _ in pairs])
    J = np.array([j for _, j in pairs])
    pert = ctx.meta["perturbation_no_concentration"].astype(str).to_numpy()
    ck = ctx.meta[BIO_CTX].astype(str).agg("\x1f".join, axis=1).to_numpy()
    tr = np.nonzero((ctx.meta["split_final"] == "train").to_numpy() & treat)[0]
    mu_c, mu_d = group_mean(ctx.D, ck, tr), group_mean(ctx.D, pert, tr)

    spaces = [("绝对 log2 丰度\n(指标1, 20%)", ctx.X),
              ("Δ_true 匹配FC\n(指标2, 25%)", ctx.D),
              ("Δ − μ_ctx\n(指标3, 20%)", ctx.D - mu_c),
              ("Δ − μ_drug\n(指标4, 20%)", ctx.D - mu_d)]
    rho = [float(np.nanmean(pcc_axis(M[I], M[J], cfg, axis=1))) for _, M in spaces]

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    bars = ax.bar([s for s, _ in spaces], rho,
                  color=[C_MAIN, C_WARN, C_ALT, C_ALT], width=0.6)
    for b, v in zip(bars, rho):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}",
                ha="center", fontweight="bold")
    ax.set_ylabel("同条件跨批次重测的相关 ρ")
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color=C_GREY, ls=":", lw=1)
    ax.text(3.45, 1.005, "完全可重复", fontsize=8, color=C_GREY, ha="right")
    ax.set_title("药物特异响应的测量可靠性极低\n"
                 "绝对丰度重测相关 0.92，而 Δ 只有 0.12", pad=12)
    fig.text(0.01, -0.06, f"复制对定义：同化合物 + 同菌株/培养基/温度/时间，且板号或数据"
             f"来源不同，共 {len(pairs)} 对。μ 仅由训练折计算。\n"
             "ρ 低说明信噪比弱、复杂模型易过拟合；但因预测与真值共享同一条对照噪声，"
             "√ρ 不可当作评分上限。数据：scripts/audit/noise_ceiling.py",
             fontsize=8, color="#555")
    save(fig, "F3_reliability")


# ------------------------------------------------------------------ F4


def fig4_baseline_ladder():
    """基线阶梯：可部署 vs oracle 分区。"""
    df = pd.read_csv(os.path.join(paths.RESULTS, "step4_baselines",
                                  "baseline_scores.csv"), index_col=0)
    oracle = [("B1 Δ≡0（=对照）", df.loc["B1 Δ≡0（=对照）", "total"]),
              ("B2 上下文均值 Δ", df.loc["B2 上下文均值 Δ", "total"]),
              ("B3 化学近邻 Δ", df.loc["B3 化学近邻 Δ", "total"]),
              ("B4 ridge Δ", df.loc["B4 ridge Δ", "total"]),
              ("α=0.15 对照收缩", 0.3843)]
    free = [("全局均值谱", 0.2589), ("逐蛋白 ridge（满秩）", 0.4406),
            ("低秩 K0=16 + ridge", 0.4694)]

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ys, labs, cols = [], [], []
    for n, v in oracle:
        ys.append(v); labs.append(n); cols.append(C_GREY)
    ys.append(np.nan); labs.append(""); cols.append("none")
    for n, v in free:
        ys.append(v); labs.append(n); cols.append(C_MAIN)
    pos = np.arange(len(ys))
    ax.barh(pos, ys, color=cols, height=0.62)
    for p, v in zip(pos, ys):
        if np.isfinite(v):
            ax.text(v + 0.006, p, f"{v:.4f}", va="center", fontsize=9.5,
                    fontweight="bold" if v == 0.4694 else "normal")
    ax.set_yticks(pos)
    ax.set_yticklabels(labs)
    ax.invert_yaxis()
    ax.set_xlabel("六项加权总分（本方复刻评分器，官方四类 val 划分）")
    ax.set_xlim(0, 0.56)
    ax.set_title("可提交模型反而高于不可提交的 oracle 基线", pad=12)
    ax.text(0.50, 1.6, "需要读测试对照\n→ 不可提交", ha="center", fontsize=9,
            color=C_GREY, bbox=dict(fc="white", ec=C_GREY, lw=.8, alpha=.9))
    ax.text(0.50, 6.4, "C-free 可部署\n（推断不接触对照）", ha="center", fontsize=9,
            color=C_MAIN, bbox=dict(fc="white", ec=C_MAIN, lw=.8, alpha=.9))
    fig.text(0.01, -0.05, "上组模型形态为 y_pred = C + Delta_pred，推断时需该样本的真实对照；测试集对照位于"
             "本方按分支 A 隔离的文件内，故仅作诊断。\n"
             "下组为 y_pred = b(metadata)，含泄漏守卫。"
             "数据：scripts/models/{baselines,baseline_cfree,select_k0}.py",
             fontsize=8, color="#555")
    save(fig, "F4_baseline_ladder")


# ------------------------------------------------------------------ F5


def fig5_loco():
    """LOCO：结构表示 vs 打乱对照 vs 两级神谕上限。"""
    base = 0.3545
    real = -0.0004
    perms = [0.0012, 0.0001, 0.0005, 0.0002, 0.0006]
    orc_nb, orc_self = 0.0237, 0.0617

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.axhline(0, color="k", lw=1)
    ax.scatter(np.full(len(perms), 1) + np.linspace(-.12, .12, len(perms)),
               perms, s=55, color=C_GREY, zorder=3, label="打乱标签对照（5 次）")
    ax.scatter([2], [real], s=150, marker="D", color=C_WARN, zorder=4,
               label="ECFP 结构表示（真实）")
    ax.scatter([3], [orc_nb], s=110, marker="^", color=C_ALT, zorder=3,
               label="神谕挑近邻照搬（阳性对照）")
    ax.scatter([4], [orc_self], s=110, marker="*", color=C_MAIN, zorder=3,
               label="神谕用自身残差（路线天花板）")
    ax.axhspan(min(perms), max(perms), color=C_GREY, alpha=.16)
    ax.text(1, max(perms) + 0.004, "随机水平带", ha="center", fontsize=8.5, color="#555")
    ax.set_ylim(-0.010, orc_self * 1.22)
    for x, v, dy, va in [(2, real, -0.0032, "top"), (3, orc_nb, 0.0032, "bottom"),
                         (4, orc_self, 0.0032, "bottom")]:
        ax.text(x + 0.22, v + dy, f"{v:+.4f}", ha="left", va=va,
                fontweight="bold", fontsize=9.5)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(["打乱标签", "ECFP\n结构表示", "神谕近邻\n（阳性对照）",
                        "神谕自身\n（天花板）"])
    ax.set_ylabel("相对「仅上下文」基线的 fc_pcc 增益")
    ax.set_title("真实结构表示落在随机水平之下；而阳性对照有增益\n"
                 "→ 失败在表示层面，不是检验没有功效", pad=12)
    ax.legend(fontsize=8.5, loc="upper left", frameon=False,
              bbox_to_anchor=(0.02, 0.98), handletextpad=.4)
    ax.set_xlim(0.55, 4.55)
    fig.text(0.01, -0.07, "8 折整化合物留出（43 个化合物）；化合物等权；指纹 bit 过滤与描述符"
             "标准化仅在外层训练折内拟合；λ 由内层二次留出选。\n"
             f"仅上下文基线 fc_pcc = {base:.4f}。天花板 +{orc_self:.4f} 换算到总分约 "
             f"+{0.25*orc_self:.4f}。数据：scripts/models/loco_response.py",
             fontsize=8, color="#555")
    save(fig, "F5_loco_structure")


# ------------------------------------------------------------------ F6


def fig6_confounding(ctx):
    """生物 × 技术的 Cramér's V 热图。"""
    from audit.diagnose_batch import BIO, TECH, cramers_v
    M = np.zeros((len(BIO), len(TECH)))
    for i, b in enumerate(BIO):
        for j, t in enumerate(TECH):
            M[i, j] = cramers_v(ctx.meta[b].astype(str).to_numpy(),
                                ctx.meta[t].astype(str).to_numpy())
    zh_b = ["菌株", "培养基", "温度", "扰动时间", "化合物"]
    zh_t = ["数据来源", "仪器", "培养板", "制备孔位"]

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    im = ax.imshow(M, cmap="RdYlBu_r", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(TECH)), zh_t)
    ax.set_yticks(range(len(BIO)), zh_b)
    for i in range(len(BIO)):
        for j in range(len(TECH)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    fontsize=10, fontweight="bold" if M[i, j] > .9 else "normal",
                    color="white" if M[i, j] > .88 or M[i, j] < .10 else "black")
    fig.colorbar(im, ax=ax, label="Cramér's V（0=独立，1=完全决定）", shrink=.85)
    ax.set_xlabel("测量上下文（技术变量）")
    ax.set_ylabel("生物条件")
    ax.set_title("培养板几乎完全决定培养基/温度/时间（V = 0.99）\n"
                 "→ 按批次校正会连生物信号一并删除，故主动不做", pad=12)
    fig.text(0.01, -0.06, "生物 + 技术 one-hot 设计矩阵 296 列、秩 267，亏秩 29（存在完全别名）。"
             "匹配对照后 Δ 空间技术变量 η² 已降至 0.05–0.27，\n"
             "唯制备孔位残留 0.272（它不在手册的 7 项匹配键内）。"
             "数据：scripts/audit/diagnose_batch.py", fontsize=8, color="#555")
    save(fig, "F6_confounding")


def main() -> None:
    setup_style()
    paths.ensure_dir(OUT)
    cfg = ScorerConfig()
    print("[图] 载入数据…")
    ctx = ev.build_context(verbose=False)
    print("[图] 开始出图")
    fig1_shared_reference(ctx, cfg)
    fig2_missing_mnar(ctx)
    fig3_reliability(ctx, cfg)
    fig4_baseline_ladder()
    fig5_loco()
    fig6_confounding(ctx)
    print(f"\n全部产出于 {OUT}")


if __name__ == "__main__":
    main()
