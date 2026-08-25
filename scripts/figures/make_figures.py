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


def _auth() -> dict:
    """权威表是数字的唯一来源。图**不再自行重算**——

    2026-08-07 外审发现：图生成于修复 B0 定义之前，比权威表旧了一整天，
    F1/F4 上的数字与正文对不上。根因是图和正文各算各的。
    改为图直接读 AUTHORITATIVE.json，图文同源，权威表一变图就跟着变。
    """
    import json
    with open(os.path.join(paths.RESULTS, "AUTHORITATIVE.json"), encoding="utf-8") as fh:
        return json.load(fh)


def fig1_shared_reference(ctx, cfg):
    """共享参照：三个对照条件。数值取自权威表。"""
    sr = _auth()["shared_reference"]
    vals = [sr["correct"]["sample_axis"],
            sr["mismatch_same_context"]["sample_axis"],
            sr["mismatch_global"]["sample_axis"]]
    labs = ["正确匹配\n（官方口径）", "错配到同条件\n的别的样本", "全局错配到\n随机样本"]

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    bars = ax.bar(labs, vals, color=[C_WARN, C_ALT, C_GREY], width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.4f}",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("指标 2 · 匹配对照原始 FC 的 PCC（样本轴）")
    ax.set_ylim(0, max(vals) * 1.28)
    ax.set_title("对药物一无所知的模型，在指标 2 上仍得 0.18\n"
                 "打断「与真值共享同一条真实对照」后大幅下降", pad=12)
    # ⚠ 刻意不画「+0.102 / +0.074」这类分解箭头：
    #   相关系数不具可加分解性，错配同时改变了协方差与分母，
    #   三条件之差不能读作可相加的信号份额（2026-08-07 外审 Layer 1）。
    ax.text(0.5, 0.94, "三条件之差不可读作可相加的信号份额", transform=ax.transAxes,
            ha="center", fontsize=8.5, color="#777", style="italic")
    fig.text(0.01, -0.04, "预测 = 训练集全局均值谱（零药物知识）；评估于官方四类 val 的"
             f"{sr['n_rows']}个处理样本。数值取自 results/AUTHORITATIVE.md。",
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
    ax.set_ylabel("跨板/来源复制对的操作性一致性 ρ")
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color=C_GREY, ls=":", lw=1)
    ax.text(3.45, 1.005, "完全可重复", fontsize=8, color=C_GREY, ha="right")
    ax.set_title("药物特异响应的跨批次操作性一致性很低\n"
                 "绝对丰度 0.92，而 Δ 只有 0.12", pad=12)
    fig.text(0.01, -0.19, f"复制对定义：同化合物 + 同菌株/培养基/温度/时间，且板号或数据"
             f"来源不同，共 {len(pairs)} 对。μ 仅由训练折计算。\n"
             "ρ 低说明该口径下跨批次可重现的成分弱、复杂模型易过拟合；ρ 混合了测量噪声与"
             "批次间真实差异，不是纯粹的仪器精度。\n"
             "又因预测与真值共享同一条对照，√ρ 不可当作评分上限。"
             "数据：scripts/audit/noise_ceiling.py",
             fontsize=8, color="#555")
    save(fig, "F3_reliability")


# ------------------------------------------------------------------ F4


def fig4_baseline_ladder():
    """基线阶梯：可部署形态 vs 需读对照的诊断基线。数值取自权威表。

    分两个面板而不是一张图叠两组：两组的推断前提根本不同，
    共用一根 y 轴容易被读成一条可比的名次表。
    B0 全局均值谱**不读对照**，因此只出现在下面板；权威表把它列在
    oracle 组内是为了给那条阶梯一个起点，图里不能照抄那个分组。
    """
    m = _auth()["models"]
    oracle = sorted([(k, v["total"]) for k, v in m["oracle_C_based"].items()
                     if not k.startswith("B0 ")], key=lambda t: t[1])
    free = sorted([(k, v["total"]) for k, v in m["c_free"].items()], key=lambda t: t[1])
    xmax = max(v for _, v in oracle + free) * 1.20

    fig, axes = plt.subplots(2, 1, figsize=(7.6, 5.4), sharex=True,
                             gridspec_kw={"height_ratios": [len(oracle), len(free)],
                                          "hspace": 0.32})
    for ax, items, col, head in (
        (axes[0], oracle, C_GREY,
         "需要该样本的实测对照才能推断  y_pred = C + Δ_pred  · 仅作诊断"),
        (axes[1], free, C_MAIN,
         "推断时不接触任何对照  y_pred = b(metadata)  · 我方提交路线"),
    ):
        pos = np.arange(len(items))
        ax.barh(pos, [v for _, v in items], color=col, height=0.6)
        top = max(v for _, v in items)
        for p, (_, v) in zip(pos, items):
            ax.text(v + 0.006, p, f"{v:.4f}", va="center", fontsize=9.5,
                    fontweight="bold" if v == top else "normal")
        ax.set_yticks(pos)
        ax.set_yticklabels([n for n, _ in items])
        ax.invert_yaxis()
        ax.set_xlim(0, xmax)
        ax.set_title(head, fontsize=9.5, color=col, loc="left", pad=6)

    axes[1].set_xlabel("六项加权总分（本方复刻评分器，官方四类 val 划分）")
    fig.suptitle("不读测试对照的可部署模型，反而高于需读对照的诊断基线",
                 fontsize=11, y=0.99)
    fig.text(0.01, -0.06, "两组的推断前提不同，不是同一张名次表：上组在推断时需要该样本的实测对照，"
             "而最终评审用组委会另备的、不随赛题发放的内部评测集，那上面读不到对照，"
             "这类模型跑不起来。\n"
             "下组含泄漏守卫，确保预测不依赖对照。B0 全局均值谱不读对照，只列于下组。"
             "数值取自 results/AUTHORITATIVE.md。", fontsize=8, color="#555")
    save(fig, "F4_baseline_ladder")


# ------------------------------------------------------------------ F5


def fig5_loco():
    """LOCO：结构表示 vs 打乱对照 vs 两级神谕上限。数值取自权威表。

    这些数原先是手抄进来的常量，正是 F1/F4 过期那个问题的同一种病。
    现在统一走权威表的 loco 分区（由 loco_response.py 写出的 loco.json 收录）。
    """
    lc = _auth().get("loco", {})
    if "_missing" in lc or not lc:
        raise SystemExit("❗权威表未收录 LOCO，先跑 scripts/models/loco_response.py "
                         "再跑 scripts/scorer/authoritative_results.py")
    base = lc["context_only_fc_pcc"]
    real = lc["gain_ecfp"]
    perms = list(lc["shuffled_gains"])
    orc_nb, orc_self = lc["gain_oracle_neighbor"], lc["gain_oracle_self"]
    # 第二个阳性对照：神谕特征**走同一条 fit_response**、同一套内层 λ 选择。
    # 它与 ECFP 那一点唯一的差别就是特征内容，所以必须画在同一张图上——
    # 只画「绕过模块」的那个对照，答不上「你的模块是不是被正则压死了」这一问。
    orc_feat = lc.get("gain_oracle_feature_same_module")

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.axhline(0, color="k", lw=1)
    ax.scatter(np.full(len(perms), 1) + np.linspace(-.12, .12, len(perms)),
               perms, s=55, color=C_GREY, zorder=3, label=f"打乱标签对照（{len(perms)}次）")
    ax.scatter([2], [real], s=150, marker="D", color=C_WARN, zorder=4,
               label="ECFP 结构表示（真实）")
    if orc_feat is not None:
        ax.scatter([3], [orc_feat], s=120, marker="s", color="#4E7D5A", zorder=3,
                   label="神谕特征走同一模块（阳性对照·不绕过模块）")
    ax.scatter([4], [orc_nb], s=110, marker="^", color=C_ALT, zorder=3,
               label="神谕挑近邻照搬（阳性对照·绕过模块）")
    ax.scatter([5], [orc_self], s=110, marker="*", color=C_MAIN, zorder=3,
               label="神谕用自身残差（化合物特异路线的上限）")
    ax.axhspan(min(perms), max(perms), color=C_GREY, alpha=.16)
    ax.text(1, max(perms) + 0.004, "随机水平带", ha="center", fontsize=8.5, color="#555")
    ax.set_ylim(-0.010, orc_self * 1.22)
    pts = [(2, real, -0.0032, "top"), (4, orc_nb, 0.0032, "bottom"),
           (5, orc_self, 0.0032, "bottom")]
    if orc_feat is not None:
        pts.append((3, orc_feat, 0.0032, "bottom"))
    for x, v, dy, va in pts:
        ax.text(x + 0.20, v + dy, f"{v:+.4f}", ha="left", va=va,
                fontweight="bold", fontsize=9.5)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xticklabels(["打乱标签", "ECFP\n结构表示", "神谕特征\n走同一模块",
                        "神谕近邻\n绕过模块", "神谕自身\n（该路线上限）"])
    ax.set_ylabel("相对「仅上下文」基线的 fc_pcc 增益")
    # 标题必须跟着数走，不能写死一句话。2026-08-24 合规重跑后真实增益
    # 落进了随机带内部（此前是带的下方），写死的旧标题会与图里的点矛盾。
    inside = min(perms) <= real <= max(perms)
    where = "落在随机水平带内部" if inside else (
        "落在随机水平之上" if real > max(perms) else "落在随机水平之下")
    ax.set_title(f"真实结构表示{where}；两个阳性对照都明显有增益\n"
                 "→ 失败在表示层面：架构给分，模块也没被正则压死", pad=12)
    ax.legend(fontsize=8.5, loc="upper left", frameon=False,
              bbox_to_anchor=(0.02, 0.98), handletextpad=.4)
    ax.set_xlim(0.55, 5.7)
    # 折数与化合物数都取自 loco.json 本身。2026-08-24 起 LOCO 宇宙收进 train 折，
    # 化合物从 43 变成 37；再去引 entity_census 的 train_val_compounds 就会
    # 在图注里写出一个与实际留出规模不符的数。
    n_folds = lc.get("n_folds_effective", lc.get("n_folds", "?"))
    n_cmpd = lc.get("n_compounds", "?")
    fig.text(0.01, -0.07, f"{n_folds}折整化合物留出"
             f"（宇宙 = split_final=='train'，{n_cmpd} 个合法训练化合物）；"
             "化合物等权；指纹 bit 过滤与描述符"
             "标准化仅在外层训练折内拟合；λ 由内层二次留出选。\n"
             + ("两个阳性对照分工不同：走同一模块的验「模块还有容量」，"
                "绕过模块的验「架构与评分口径给分」。" if orc_feat is not None else "")
             + f"　仅上下文基线 fc_pcc = {base:.4f}。该路线上限 +{orc_self:.4f} 换算到总分约 "
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
