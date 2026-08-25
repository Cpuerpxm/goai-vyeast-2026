"""唯一权威结果表：所有对外引用的数字都从这里出，任何文档不得手抄。

为什么要有这个（GPT Pro R3 · L1-13）：
共享参照那组数字在不同文档里出现了两个版本——0.1872/0.0058 与 0.1834/0.0068。
根因是前者产自评分器修 bug 之前的一次临时探查，后者产自冻结脚本，两版都被手抄进了文档。
只要还允许手抄，这类漂移就会反复发生。

因此本脚本：
  1. 现场重算所有对外数字，不读任何既有报告
  2. 一并冻结口径指纹（split / 对照聚合 / NaN 规则 / 代码 SHA / 种子）
  3. 输出 results/AUTHORITATIVE.md 与 .json，文档只许引用，不许复制

运行：python authoritative_results.py
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import warnings
from dataclasses import asdict
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import paths, provenance
from data import split_guard as sg
from scorer import evaluate as ev
from scorer.config import ScorerConfig
from scorer.metrics import pcc_axis

OUT_MD = os.path.join(paths.RESULTS, "AUTHORITATIVE.md")
OUT_JSON = os.path.join(paths.RESULTS, "AUTHORITATIVE.json")
CTX_KEYS = ["Strains", "Medium", "Temperature", "pert_time"]

#: 只要这些文件里有一个变了，results/step9_loco/loco.json 里的数就可能过期。
#: 权威表**不重跑** LOCO（8 折外层留出代价太高），因此改成硬核对：
#: loco.json 里记着它自己跑的时候这些文件的哈希，对不上就拒绝收录。
LOCO_DEPS = [
    "models/loco_response.py", "models/design.py", "models/lowrank.py",
    "data/split_guard.py", "data/loader.py", "data/control_match.py",
    "scorer/config.py", "scorer/metrics.py", "scorer/evaluate.py",
]


def code_fingerprint() -> dict:
    """源码摘要 + 公开仓库发布标识 + 环境。

    ❗2026-08-24（L1-3）：旧版把本地大仓库 `E:/Claude_Code` 的 `git rev-parse HEAD`
    写成 `_git_head`，记下来的是 `b0a309dce` —— 公开仓库里根本不存在这个提交。
    现在改由 `data/provenance.py` 出：tree_digest 谁 clone 都能复算，
    发布标识来自 `RELEASE.json`（`scripts/release.py` 写），没发布就明写未发布。
    """
    return provenance.stamp()


def shared_reference(ctx, cfg, seed=20260805) -> dict:
    """共享参照三条件。**这是唯一权威口径**。"""
    rng = np.random.default_rng(seed)
    rows = np.nonzero(ctx.treated & np.isfinite(ctx.D).any(axis=1)
                      & ctx.rows(ev.VAL_SPLITS))[0]
    # 复用基线阶梯的 B0，保证与 baselines.py 是同一个预测向量
    from models.baselines import b0_global_mean

    y = b0_global_mean(ctx)

    key = ctx.meta[CTX_KEYS].astype(str).agg("\x1f".join, axis=1).to_numpy()
    p_ctx = rows.copy()
    for k in np.unique(key[rows]):
        idx = np.nonzero(key[rows] == k)[0]
        if idx.size > 1:
            p_ctx[idx] = rows[idx][rng.permutation(idx.size)]

    def fc(cr):
        dp = y[rows].astype(np.float64) - ctx.C[cr].astype(np.float64)
        return {"sample_axis": float(np.nanmean(pcc_axis(ctx.D[rows], dp, cfg, axis=1))),
                "protein_axis": float(np.nanmean(pcc_axis(ctx.D[rows], dp, cfg, axis=0)))}

    return {
        "n_rows": int(rows.size),
        "correct": fc(rows),
        "mismatch_same_context": fc(p_ctx),
        "mismatch_global": fc(rng.permutation(rows)),
        "_note": "R3-L1-08：相关系数不可加性分解。三条件之差只说明"
                 "共享参照抬高了零知识基线，不得拆成可相加的信号份额。",
    }


def baselines_and_cfree(ctx, cfg) -> dict:
    """基线阶梯 + C-free 骨架 + 选秩曲线。全部现场重算。"""
    from models.baseline_cfree import masked_ridge
    from models.design import FEATURE_SETS, encode, freeze
    from models.baselines import (b0_global_mean, b1_control, b2_ctx_mean_delta,
                                  b2g_global_delta, b3o_oracle_neighbor, b4_ridge,
                                  _fill)
    from models.select_k0 import fit_lowrank_pipeline

    val = ctx.rows(ev.VAL_SPLITS)
    fb = b0_global_mean(ctx)[0]
    out = {"oracle_C_based": {}, "c_free": {}, "k0_curve": {}}

    for lab, y in [
        ("B0 全局均值谱", b0_global_mean(ctx)),
        ("B1 预测=匹配对照", b1_control(ctx, fb)),
        ("B2g 总体平均响应", b2g_global_delta(ctx, fb)),
        ("B2 上下文均值响应", b2_ctx_mean_delta(ctx, fb)),
        ("B4 ridge 响应", b4_ridge(ctx, fb)),
    ]:
        out["oracle_C_based"][lab] = ev.flatten(ev.evaluate(y, ctx, val, cfg))
    # 化学近邻（Tanimoto）也要进表。它出现在文档的基线阶梯里，
    # 却一直不在权威表内，成了没有出处的孤儿数字（2026-08-07 外审）。
    from models.baselines import b3_chem_neighbor
    smi = os.path.join(paths.DATA_EXTERNAL, "compound_smiles.csv")
    p3, why = b3_chem_neighbor(ctx, fb, smi)
    if p3 is not None:
        out["oracle_C_based"]["B3 化学近邻响应"] = ev.flatten(ev.evaluate(p3, ctx, val, cfg))
    else:
        print(f"  ⚠ 化学近邻基线跳过：{why}")

    p3o, _ = b3o_oracle_neighbor(ctx, fb, val)
    # 命名口径两次修订：
    #  2026-08-07 —— 原称「作弊」不准确，它并非违规操作，而是给定
    #    「已知每个留出样本真值、可在训练化合物里取最近邻」这一不可实现前提下的成绩。
    #  2026-08-25（GPT Pro R6 L1-02 的同类问题）—— 也不能叫「上限」。
    #    它是按**谱级相关**挑出的最近邻，而评分是**逐样本相关的平均**；
    #    前者的最大化解不等于后者的最大化解，所以它不构成任何族的上界。
    #    统一改叫「目标派生参考」：用留出真值构造出来的一个比照点，仅此而已。
    out["oracle_C_based"]["B3o 神谕近邻(目标派生参考)"] = ev.flatten(ev.evaluate(p3o, ctx, val, cfg))

    mu_g = b0_global_mean(ctx)[0]
    alpha_best, alpha_val = None, -np.inf
    for a in (0.05, 0.10, 0.15, 0.20, 0.30):
        y = _fill(ctx.C + a * (mu_g[None, :] - ctx.C), fb)
        t = ev.flatten(ev.evaluate(y, ctx, val, cfg))["total"]
        if t > alpha_val:
            alpha_val, alpha_best = t, a
    out["oracle_C_based"][f"对照收缩 α={alpha_best}"] = {"total": alpha_val}

    train = sg.train_rows(ctx.meta)
    # ❗L1-1：这一行原来是 design(ctx.meta, ...)，在**全表**建词表并算 log-time
    # 均值方差——产出 0.4694 的正是这个脚本。现在词表与标准化参数只由 train 折冻结。
    spec = freeze(ctx.meta, train, FEATURE_SETS["bio_tech"], with_drug=False)
    Z = encode(ctx.meta, spec)
    out["design_spec"] = {k: v for k, v in spec.items() if k != "levels"}
    out["design_spec"]["levels"] = spec["levels"]
    mu, W = masked_ridge(Z, ctx.X, train, 30.0, ctx.meta)
    y_full = np.nan_to_num((mu[None, :] + Z.astype(np.float64) @ W).astype(np.float32),
                           nan=float(np.nanmedian(mu)))
    out["c_free"]["全局均值谱"] = ev.flatten(ev.evaluate(b0_global_mean(ctx), ctx, val, cfg))
    out["c_free"]["逐蛋白 ridge(满秩)"] = ev.flatten(ev.evaluate(y_full, ctx, val, cfg))
    for k in (8, 16, 32, 96):
        y = np.nan_to_num(fit_lowrank_pipeline(ctx, Z, train, k, 30.0), nan=0.0)
        f = ev.flatten(ev.evaluate(y, ctx, val, cfg))
        out["k0_curve"][f"K0={k}"] = {"total": f["total"], "abs_r2": f["abs_r2"],
                                      "fc_pcc": f["fc_pcc"]}
        if k == 16:
            out["c_free"]["低秩 K0=16 + ridge"] = f
            y_best = y
    # ❗2026-08-25（GPT Pro R6 · L2-07）：60 次够支撑「差值明显为正」这个粗判断，
    # 不够支撑 2.5% / 97.5% 分位的点估计——分位数在尾部本来就最不稳。
    # 上一版还把低重抽样次数称为「保守」，那是把样本不足说成了谨慎。
    # 提到 500 次。代价是权威表这一步从约 5 分钟涨到约 40 分钟，值得。
    out["paired_bootstrap_lowrank_vs_fullrank"] = ev.grouped_bootstrap_paired(
        y_full, y_best, ctx, val, n_boot=500, cfg=cfg)

    # 分场景：总分是四类 val 混在一起的加权和，看不出模型在哪类外推上垮。
    # 评审必问"S2 未见菌株占测试集一半，你在那上面怎么样"，故单列。
    by = ev.evaluate_by_split(y_best, ctx, cfg)
    out["submitted_model_by_split"] = {
        r["split"]: {k: (None if k in ("split",) else
                         (int(r[k]) if k.startswith("n") else float(r[k])))
                     for k in by.columns if k != "split" and r[k] == r[k]}
        for _, r in by.iterrows()
    }
    return out


def control_match_and_reliability(ctx, cfg) -> dict:
    from audit.noise_ceiling import BIO_CTX, group_mean, replicate_pairs

    z = np.load(os.path.join(paths.RESULTS, "step2_control_match",
                             "delta_true_train_val_median.npz"), allow_pickle=False)
    n_ctrl = z["n_ctrl"][ctx.treated]
    treat = ctx.treated & np.isfinite(ctx.D).any(axis=1)
    pairs = replicate_pairs(ctx.meta, treat)
    I = np.array([i for i, _ in pairs]); J = np.array([j for _, j in pairs])
    pert = ctx.meta["perturbation_no_concentration"].astype(str).to_numpy()
    ck = ctx.meta[BIO_CTX].astype(str).agg("\x1f".join, axis=1).to_numpy()
    tr = np.nonzero((ctx.meta["split_final"] == "train").to_numpy() & treat)[0]
    rel = {}
    for lab, M in [("绝对 log2 丰度", ctx.X), ("Δ_true 匹配FC", ctx.D),
                   ("Δ − μ_ctx", ctx.D - group_mean(ctx.D, ck, tr)),
                   ("Δ − μ_drug", ctx.D - group_mean(ctx.D, pert, tr))]:
        rel[lab] = float(np.nanmean(pcc_axis(M[I], M[J], cfg, axis=1)))
    return {
        "n_treated": int(ctx.treated.sum()),
        "zero_match_rate": float((n_ctrl == 0).mean()),
        "multi_match_rate": float((n_ctrl > 1).mean()),
        "median_n_controls_when_multi": float(np.median(n_ctrl[n_ctrl > 1])),
        "n_replicate_pairs": len(pairs),
        "replicate_consistency_rho": rel,
    }


def confounding_and_missing(ctx) -> dict:
    from audit.diagnose_batch import BIO, TECH, cramers_v, onehot
    from scipy import stats

    v = {}
    for b in ["Medium", "Temperature", "pert_time"]:
        v[f"{b} × Yeast_cell_plate"] = round(cramers_v(
            ctx.meta[b].astype(str).to_numpy(),
            ctx.meta["Yeast_cell_plate"].astype(str).to_numpy()), 3)
    Zm = onehot(ctx.meta, BIO + TECH)
    rank = int(np.linalg.matrix_rank(Zm))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(ctx.X, axis=0)
    miss = np.isnan(ctx.X).mean(axis=0)
    ok = np.isfinite(med)
    import pandas as pd
    dec = pd.qcut(med[ok], 10, labels=False, duplicates="drop")
    rho, _ = stats.spearmanr(med[ok], miss[ok])
    return {
        "cramers_v": v,
        "design_matrix": {"cols": int(Zm.shape[1]), "rank": rank,
                          "deficiency": int(Zm.shape[1] - rank)},
        "missing_spearman_abundance_vs_rate": float(rho),
        "missing_rate_lowest_decile": float(miss[ok][dec == 0].mean()),
        "missing_rate_highest_decile": float(miss[ok][dec == 9].mean()),
    }


def entity_census(ctx) -> dict:
    """化合物计数的四个口径。

    文中37/43/46/54反复出现，读者极易当成同一个量前后矛盾。
    这里一次算清，文档只许引用本节。
    """
    from data import loader
    CONTROL_LABELS = {"Water", "DMSO", "Quality Control"}
    tv = set(ctx.meta["perturbation_no_concentration"].astype(str))
    te = set(loader.load_metadata("test")["perturbation_no_concentration"].astype(str))
    tr_mask = (ctx.meta["split_final"] == "train").to_numpy() & ctx.treated
    tr = set(ctx.meta.loc[tr_mask, "perturbation_no_concentration"].astype(str))
    tv_nc, te_nc, tr_nc = tv - CONTROL_LABELS, te - CONTROL_LABELS, tr - CONTROL_LABELS
    return {
        "train_val_labels_incl_controls": len(tv),
        "train_val_compounds": len(tv_nc),
        "train_split_compounds": len(tr_nc),
        "test_labels_incl_controls": len(te),
        "test_compounds": len(te_nc),
        "test_only_compounds": len(te_nc - tv_nc),
        "response_matrix_rows": len(tr_nc),
        "response_matrix_rank_cap": len(tr_nc) - 1,
        "task_rank_hard_cap": len(tv_nc) - 1,
        "_note": "bootstrap 的分组键是扰动标签，故报 train_val_labels_incl_controls；"
                 "整化合物留出覆盖 train_val_compounds；响应矩阵行数是 train_split_compounds。",
    }


def ingest_loco(allow_stale: bool = False) -> dict:
    """收录 LOCO 的数，但**不重跑**——那是 8 折外层留出，代价太高。

    2026-08-07 外审：文中 +0.0617 / +0.0237 / -0.0004 全不在注册范围内，
    数字核对脚本对它们完全无感。于是这里按内容哈希收录 loco.json。

    ❗2026-08-24（L1-3）补上真正的闸门：只按内容哈希收录，挡不住「loco.json 是旧
    模型跑出来的」这件事——旧结果照样会被带进新材料。现在改成硬核对：
    loco.json 里记着它运行时 LOCO_DEPS 各文件的哈希，任一对不上即**拒绝收录**。
    要强行放行必须显式传 --allow-stale-loco，且结果里会留下醒目的作废标记。
    """
    p = os.path.join(paths.RESULTS, "step9_loco", "loco.json")
    if not os.path.exists(p):
        return {"_missing": f"缺 {p}，先跑 scripts/models/loco_response.py"}
    raw = open(p, "rb").read()
    d = json.loads(raw.decode("utf-8"))
    d["_source_file"] = os.path.relpath(p, paths.PROJECT_ROOT).replace("\\", "/")
    d["_source_sha256"] = hashlib.sha256(raw).hexdigest()[:16]
    d["_source_mtime"] = datetime.fromtimestamp(os.path.getmtime(p)).isoformat(timespec="seconds")

    now = provenance.file_hashes(LOCO_DEPS)
    rec = (d.get("_provenance") or {}).get("files") or {}
    drift = {k: {"loco.json 记录": rec.get(k, "(未记录)"), "当前": now[k]}
             for k in LOCO_DEPS if rec.get(k) != now[k]}
    d["_dep_check"] = {"deps": LOCO_DEPS, "drift": drift, "stale": bool(drift)}
    if drift:
        msg = ("results/step9_loco/loco.json 是用**另一份代码**跑出来的，"
               f"以下依赖对不上：{sorted(drift)}。"
               "重跑 `python scripts/models/loco_response.py` 后再生成权威表。")
        if not allow_stale:
            raise RuntimeError("[权威表] 拒绝收录过期 LOCO 结果：" + msg)
        d["_STALE_WARNING"] = "⚠ 本节数字来自过期代码，**不得对外引用**。" + msg
    return d


#: 菌株搬运实验的依赖文件，同样按哈希核对，防止旧结论被带进新材料。
STRAIN_DEPS = [
    "models/strain_transport.py", "models/design.py", "models/lowrank.py",
    "data/strain_genome.py", "data/split_guard.py", "data/loader.py",
    "data/control_match.py", "scorer/config.py", "scorer/metrics.py",
    "scorer/evaluate.py",
]


def ingest_strain_transport(allow_stale: bool = False) -> dict:
    """收录未见菌株效应搬运的结论（含它的否定判据）。

    与 LOCO 同样的规矩：不重跑，但硬核对依赖代码指纹；对不上就拒绝收录。
    这条路线按预注册判据被判定为**不保留**，结论本身也要进权威表——
    负结果同样是结果，材料里要引它，就不能靠手抄。
    """
    p = os.path.join(paths.RESULTS, "step11_strain_transport", "strain_transport.json")
    if not os.path.exists(p):
        return {"_missing": f"缺 {p}，先跑 scripts/models/strain_transport.py"}
    raw = open(p, "rb").read()
    d = json.loads(raw.decode("utf-8"))
    d["_source_file"] = os.path.relpath(p, paths.PROJECT_ROOT).replace("\\", "/")
    d["_source_sha256"] = hashlib.sha256(raw).hexdigest()[:16]
    d["_source_mtime"] = datetime.fromtimestamp(os.path.getmtime(p)).isoformat(timespec="seconds")

    now = provenance.file_hashes(STRAIN_DEPS)
    rec = (d.get("_provenance") or {}).get("files") or {}
    drift = {k for k in STRAIN_DEPS if rec.get(k) != now[k]}
    d["_dep_check"] = {"deps": STRAIN_DEPS, "drift": sorted(drift), "stale": bool(drift)}
    if drift:
        msg = (f"strain_transport.json 与当前代码不一致，对不上的依赖：{sorted(drift)}。"
               "重跑 `python scripts/models/strain_transport.py` 后再生成权威表。")
        if not allow_stale:
            raise RuntimeError("[权威表] 拒绝收录过期的菌株搬运结果：" + msg)
        d["_STALE_WARNING"] = "⚠ 本节数字来自过期代码，**不得对外引用**。" + msg
    return d


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-stale-loco", action="store_true",
                    help="强行收录与当前代码不一致的 loco.json（会打作废标记）")
    ap.add_argument("--allow-stale-strain", action="store_true",
                    help="强行收录与当前代码不一致的 strain_transport.json")
    args = ap.parse_args()

    cfg = ScorerConfig()
    ctx = ev.build_context(verbose=False)
    val = ctx.rows(ev.VAL_SPLITS)

    res = {
        "generated_for": "GOAI 赛道三方向一 · 对外引用的唯一数字来源",
        "caliber": {
            "split": "官方 split_final；评估集 = val_chem_only + val_strain_only + val_both + val_time",
            "control_set": "Water + DMSO（质控样本单列，不作生物对照）",
            "control_agg": cfg.control_agg,
            "match_keys": "7 项，不含 protein_well",
            "undefined_pcc": cfg.undefined_pcc,
            "undefined_axis": cfg.undefined_axis,
            "undefined_module": cfg.undefined_module,
            "both_time_parts": cfg.both_time_parts,
            "const_atol": cfg.const_atol,
            "min_valid_points": cfg.min_valid_points,
            "seeds": {"shared_reference": 20260805, "loco_folds": 20260805},
        },
        "code_fingerprint": code_fingerprint(),
        "shared_reference": shared_reference(ctx, cfg),
    }
    print("[权威表] 算基线阶梯与 C-free …")
    res["models"] = baselines_and_cfree(ctx, cfg)
    print("[权威表] 算对照匹配与复制可靠性 …")
    res["control_and_reliability"] = control_match_and_reliability(ctx, cfg)
    print("[权威表] 算混杂与缺失 …")
    res["confounding_and_missing"] = confounding_and_missing(ctx)
    res["entity_census"] = entity_census(ctx)
    res["loco"] = ingest_loco(allow_stale=args.allow_stale_loco)
    res["strain_transport"] = ingest_strain_transport(
        allow_stale=args.allow_stale_strain)

    # 数据实况
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res["data"] = {
            "n_samples_train_val": int(ctx.n),
            "n_proteins": int(ctx.X.shape[1]),
            "missing_rate_abundance": float(np.isnan(ctx.X).mean()),
            "missing_rate_delta_treated": float(np.isnan(ctx.D[ctx.treated]).mean()),
            "all_missing_cols_abundance": int(np.isnan(ctx.X).all(axis=0).sum()),
            "all_missing_cols_delta": int(np.isnan(ctx.D[ctx.treated]).all(axis=0).sum()),
            "n_treated": int(ctx.treated.sum()),
            "_note": "以上仅 train_val。测试集蛋白矩阵的对应统计来自隔离决定前的"
                     "一次探查性读取，见合规披露，不在本表重算。",
        }

    paths.ensure_dir(paths.RESULTS)
    with open(OUT_JSON, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2)

    sr = res["shared_reference"]
    L = ["# 权威结果表（机器生成，勿手改）", "",
         "本文件由 `scripts/scorer/authoritative_results.py` 现场重算生成。",
         "**所有对外文档只许引用本表，不许复制数字。**",
         "口径或代码一变，重跑本脚本，全部引用同步更新。", "",
         "## 口径指纹", "", "| 项 | 值 |", "|---|---|"]
    for k, v in res["caliber"].items():
        L.append(f"| {k} | `{v}` |")
    fp = res["code_fingerprint"]
    rel = fp["release"]
    L += ["", "## 代码与发布指纹", "",
          f"- 源码摘要 `tree_digest` = **`{fp['tree_digest']}`**"
          f"（`scripts/` 下 {fp['n_source_files']} 个 .py，路径排序后逐个 SHA-256 再汇总）",
          f"- 公开仓库发布状态：**{rel['_status']}**"
          + (f"，tag `{rel.get('tag')}` · commit `{rel.get('commit')}` · "
             f"与本次源码一致：{'是' if rel.get('tree_digest_matches_release') else '**否**'}"
             if rel["_status"] == "已发布" else ""),
          f"- 运行环境：Python {fp['env']['python']} · numpy {fp['env']['numpy']} · "
          f"pandas {fp['env']['pandas']} · scipy {fp['env']['scipy']} · "
          f"scikit-learn {fp['env']['sklearn']} · rdkit {fp['env']['rdkit']}",
          "",
          "<details><summary>逐文件 SHA-256 前 16 位</summary>", "",
          "| 文件 | 摘要 |", "|---|---|"]
    for k, v in fp["files"].items():
        L.append(f"| `{k}` | `{v}` |")
    L += ["", "</details>"]
    L += ["", "## 共享参照三条件（指标 2，零药物知识预测）", "",
          f"评估样本 {sr['n_rows']}（官方四类 val 的处理样本）", "",
          "| 条件 | 样本轴 PCC | 蛋白轴 PCC |", "|---|---|---|"]
    for k, lab in [("correct", "正确匹配（官方口径）"),
                   ("mismatch_same_context", "错配到同条件的别的样本"),
                   ("mismatch_global", "全局错配到随机样本")]:
        L.append(f"| {lab} | {sr[k]['sample_axis']:.4f} | {sr[k]['protein_axis']:.4f} |")
    L += ["", f"> {sr['_note']}", "",
          "## 数据实况（train_val）", "", "| 项 | 值 |", "|---|---|"]
    for k, v in res["data"].items():
        if k.startswith("_"):
            continue
        L.append(f"| {k} | {v:.4%} |" if "rate" in k else f"| {k} | {v} |")
    m = res["models"]
    L += ["", "## 模型分数（官方四类 val，本方复刻评分器）", "",
          "### 需读测试对照者（oracle 诊断，**不可提交**）", "",
          "| 模型 | total | abs_pcc | abs_r2 | fc_pcc |", "|---|---|---|---|---|"]
    for k, f in m["oracle_C_based"].items():
        L.append(f"| {k} | {f['total']:.4f} | "
                 + (f"{f['abs_pcc']:.4f} | {f['abs_r2']:.4f} | {f['fc_pcc']:.4f} |"
                    if "abs_pcc" in f else "— | — | — |"))
    L += ["", "### C-free（推断不接触对照，**可提交**）", "",
          "| 模型 | total | abs_pcc | abs_r2 | fc_pcc | ctx_resid | drug_resid |",
          "|---|---|---|---|---|---|---|"]
    for k, f in m["c_free"].items():
        L.append(f"| {k} | {f['total']:.4f} | {f['abs_pcc']:.4f} | {f['abs_r2']:.4f} | "
                 f"{f['fc_pcc']:.4f} | {f['ctx_resid']:.4f} | {f['drug_resid']:.4f} |")
    L += ["", "### 端到端选秩曲线", "", "| K0 | total | abs_r2 | fc_pcc |", "|---|---|---|---|"]
    for k, f in m["k0_curve"].items():
        L.append(f"| {k} | {f['total']:.4f} | {f['abs_r2']:.4f} | {f['fc_pcc']:.4f} |")
    pb = m["paired_bootstrap_lowrank_vs_fullrank"]
    L += ["", "### 配对 bootstrap：低秩 K0=16 − 满秩 ridge", "",
          f"- 差值均值 **{pb['diff_mean']:+.4f}**",
          f"- 95% 区间 **[{pb['diff_ci'][0]:+.4f}, {pb['diff_ci'][1]:+.4f}]**"
          f"，{'排除 0' if pb['excludes_zero'] else '含 0'}",
          f"- {pb['n_boot']} 次重抽样，按 {pb['n_groups']} 个化合物分组，"
          f"低秩更优的比例 {pb['frac_positive']:.1%}", "",
          "> ⚠ 判据：两模型跑在同一批数据与同一批重抽样上，估计高度相关，"
          "故看**配对差值**的区间，而非比较两条边际区间是否重叠。"]

    cr = res["control_and_reliability"]
    L += ["", "## 对照匹配与复制可靠性", "", "| 项 | 值 |", "|---|---|",
          f"| 处理样本数 | {cr['n_treated']} |",
          f"| 0 匹配率 | {cr['zero_match_rate']:.1%} |",
          f"| 多匹配比例 | {cr['multi_match_rate']:.1%}（中位 {cr['median_n_controls_when_multi']:.0f} 个对照）|",
          f"| 复制对数 | {cr['n_replicate_pairs']} |", "",
          "跨板/来源复制对的**操作性响应谱一致性** ρ（勿解释为信号占比）：", "",
          "| 空间 | ρ |", "|---|---|"]
    for k, v in cr["replicate_consistency_rho"].items():
        L.append(f"| {k} | {v:.3f} |")

    cm = res["confounding_and_missing"]
    L += ["", "## 混杂与缺失", "", "| 项 | 值 |", "|---|---|"]
    for k, v in cm["cramers_v"].items():
        L.append(f"| Cramér's V · {k} | {v} |")
    dm = cm["design_matrix"]
    L += [f"| 生物+技术设计矩阵 | {dm['cols']} 列 / 秩 {dm['rank']} / 亏秩 **{dm['deficiency']}** |",
          f"| Spearman(蛋白中位丰度, 缺失率) | {cm['missing_spearman_abundance_vs_rate']:+.3f} |",
          f"| 最低丰度十分位缺失率 | {cm['missing_rate_lowest_decile']:.1%} |",
          f"| 最高丰度十分位缺失率 | {cm['missing_rate_highest_decile']:.1%} |"]

    both = (sr["correct"]["sample_axis"] + sr["correct"]["protein_axis"]) / 2
    L += ["", f"> {res['data']['_note']}", "",
          "## 同一个量的两种呈现（勿误当成两个不同结果）", "",
          f"上表按轴分列；而基线阶梯里 B0 的 `fc_pcc` 报的是**两轴平均** = {both:.4f}。",
          "两者是同一次计算的不同切面。零知识预测统一使用 `baselines.b0_global_mean`，",
          "本表与基线表不再存在第二套「全局均值谱」定义。", "",
          "## 化合物表示的 nested LOCO（train 折内，宇宙 = split_final=='train'）", ""]
    lc = res["loco"]
    if lc.get("_missing"):
        L.append(f"> {lc['_missing']}")
    else:
        if lc.get("_STALE_WARNING"):
            L.append(f"> {lc['_STALE_WARNING']}")
        L += [f"- 合法训练化合物 **{lc.get('n_compounds')}** 个；"
              f"外层 {lc.get('n_folds_effective')} 折整化合物留出；"
              f"train 行 {lc.get('n_train_rows')}（处理行 {lc.get('n_train_compound_rows')}）",
              "", "| 配置 | fc_pcc | 相对仅上下文 |", "|---|---|---|",
              f"| 仅上下文 b | {lc.get('context_only_fc_pcc', float('nan')):.4f} | — |",
              f"| + ECFP 结构表示 | {lc.get('ecfp_fc_pcc', float('nan')):.4f} | "
              f"{lc.get('gain_ecfp', float('nan')):+.4f} |",
              f"| [阳性对照2] 神谕特征走同一模块 | — | "
              f"{lc.get('gain_oracle_feature_same_module', float('nan')):+.4f} |",
              f"| [阳性对照] 神谕残差照搬 | — | {lc.get('gain_oracle_neighbor', float('nan')):+.4f} |",
              f"| [参考] 神谕自身平均残差（目标派生，**非上界**） | — | {lc.get('gain_oracle_self', float('nan')):+.4f} |",
              "",
              f"- shuffled-label 对照 {lc.get('n_perm')} 次：均值 "
              f"{(lc.get('shuffled_mean') if lc.get('shuffled_mean') is not None else float('nan')):+.4f}"
              f" / 最大 {(lc.get('shuffled_max') if lc.get('shuffled_max') is not None else float('nan')):+.4f}"
              f" → 真增益{'超过' if lc.get('passes_shuffled_control') else '**未超过**'}全部打乱对照",
              f"- 依赖代码一致性核对：{'通过' if not lc.get('_dep_check', {}).get('stale') else '**未通过（结果过期）**'}"]
        fv = lc.get("final_val_chem_only") or {}
        if fv:
            L += ["",
                  f"官方 `val_chem_only` 一次性确认（{fv['n_val_compounds']} 个化合物，"
                  f"其中 {fv['n_with_smiles']} 个有 SMILES；lambda={fv['lam']:g} 取自内层众数）：",
                  "", "| 配置 | total | fc_pcc | ctx_resid |", "|---|---|---|---|",
                  f"| 仅上下文 b | {fv['context_only']['total']:.4f} | "
                  f"{fv['context_only']['fc_pcc']:.4f} | {fv['context_only']['ctx_resid']:.4f} |",
                  f"| + ECFP | {fv['ecfp']['total']:.4f} | {fv['ecfp']['fc_pcc']:.4f} | "
                  f"{fv['ecfp']['ctx_resid']:.4f} |",
                  f"| **差** | {fv['delta_total']:+.4f} | {fv['delta_fc_pcc']:+.4f} | "
                  f"{fv['delta_ctx_resid']:+.4f} |",
                  "", "> 只有 6 个留出化合物，这一格不确定性很大，只作方向一致性核对。"]
    L += ["", "## 未见菌株的效应搬运（外部基因组资源，按预注册判据裁决）", ""]
    st = res["strain_transport"]
    if st.get("_missing"):
        L.append(f"> {st['_missing']}")
    else:
        if st.get("_STALE_WARNING"):
            L.append(f"> {st['_STALE_WARNING']}")
        ex = st.get("external_resource", {})
        zb = st.get("loso_zero_baseline", {})
        bu, bs = st.get("loso_best_uniform"), st.get("loso_best_snp")
        fv = st.get("final_val_strain_only", {})
        L += [f"- 外部资源：{ex.get('citation')}；文件 `{ex.get('matrix_file')}`，"
              f"SHA-256 `{str(ex.get('matrix_sha256'))[:16]}…`，面板 {ex.get('panel_size')} 株",
              f"- 训练菌株 {len(st.get('train_strains', []))} 株，其中有面板坐标的 "
              f"{len(st.get('donors_with_coordinates', []))} 株（供体池）",
              f"- 未见菌株到最近供体 {st.get('unseen_nearest_donor_distance', float('nan')):.3f}"
              f"（面板分位 {st.get('unseen_panel_quantile', float('nan')):.1%}，"
              f"三供体跨度 {st.get('unseen_donor_distance_spread', float('nan')):.3f}）；"
              f"val 菌株 {st.get('val_nearest_donor_distance', float('nan')):.3f}"
              f"（分位 {st.get('val_panel_quantile', float('nan')):.1%}，"
              f"跨度 {st.get('val_donor_distance_spread', float('nan')):.3f}）",
              "",
              "| 阶段 | 方案 | drug_resid | 相对现状 |", "|---|---|---|---|",
              f"| train 内 LOSO | 现状（未见菌株整块 0） | "
              f"{zb.get('drug_resid', float('nan')):.4f} | — |"]
        for lab, b in (("等权（无基因组信息）", bu), ("SNP 距离核", bs)):
            if b:
                L.append(f"| train 内 LOSO | {lab} | {b['drug_resid']:.4f} | "
                         f"{b['drug_resid'] - zb.get('drug_resid', float('nan')):+.4f} |")
        if fv:
            for name, lab in (("zero", "现状"), ("uniform", "等权"), ("snp", "SNP 距离核")):
                if name in fv:
                    L.append(f"| val_strain_only 一次性 | {lab} | "
                             f"{fv[name]['drug_resid']:.4f} | "
                             f"{fv[name]['drug_resid'] - fv['zero']['drug_resid']:+.4f} |")
            vd = fv.get("verdict", {})
            L += ["",
                  f"- 预注册判据（写在 `_handoff/CURRENT.md`，不是事后定的）："
                  f"drug_resid 增量 ≥ +0.015 或六项总分增量 ≥ +0.003，"
                  f"且 abs_r2 与 S1 abs_r2 下降不超过 0.005",
                  f"- 实测：drug_resid {vd.get('delta_drug_resid', float('nan')):+.4f} · "
                  f"总分 {vd.get('delta_total', float('nan')):+.4f} · "
                  f"abs_r2 {vd.get('delta_abs_r2', float('nan')):+.4f} · "
                  f"S1 abs_r2 {vd.get('delta_s1_abs_r2', float('nan')):+.4f}",
                  f"- **裁决：{'保留' if vd.get('passed') else '不保留，按预案永久停止该路线'}**"]
        L += ["",
              "> ⚠ 可验证性是不对称的：加权最可能起作用的那一格（未见菌株，与某训练菌株近缘）"
              "无法验证；唯一能验证的那一格（val 菌株）到三个供体近乎等距，验不出加权本身。"
              "上表能支持的结论只到「搬运机制本身在本数据上不带来增益」，"
              "不能反过来说「菌株基因组对该任务无用」。"]

    L += ["",
          "## 已知的作废数字（勿再引用）", "",
          "| 作废值 | 出处 | 现行值 |", "|---|---|---|",
          "| B0 abs_pcc 0.9535 | 未定义轴被静默踢出时的值 | 0.4768（undefined_axis=zero） |",
          "| B2 ctx_resid 0.0011 | float32 舍入噪声被当真信号 | 0.0000 |",
          "| 修评分器前的整张基线表（B0 0.2928 / B2 0.2761 / B3 0.2479 / B4 0.3026 / α 0.3473） | 未定义轴与常数判据两个 bug 修复前 | 见 docs/06 §11.2 修正后表 |",
          "| 加权上限 ≈ 0.42 | 由 √ρ 推出，前提不成立（预测与真值共享对照噪声） | **已作废，不替换为任何单一数字** |",
          "| C-free 低秩 K0=16 总分 0.4694 | 设计矩阵词表与 log-time 标准化参数在**全表**估计（L1-1） | 见本表「C-free」一节 |",
          "| LOCO Morgan 增益 −0.0004 / 神谕近邻 +0.0237 / 神谕自身 +0.0617 | 外层留出宇宙含 val 化合物（L1-2） | 见本表 LOCO 一节 |",
          "| 提交模型用全部 train_val 拟合 | `predict_test.py --fit-rows all`，违反手册第 17 页 | 选项已删除，只拟合 train 折 |"]

    with open(OUT_MD, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(L) + "\n")

    print("\n".join(L))
    print(f"\n[写出] {OUT_MD}\n[写出] {OUT_JSON}")


if __name__ == "__main__":
    main()
