"""评估台：把预测的 log2 蛋白质组向量打成官方六项加权总分。

与 metrics.py 的分工：metrics.py 只管单个指标怎么算；本模块管
「哪些样本进哪个指标、参照均值怎么冻结、总分怎么加权、不确定性怎么给」。

关键口径（docs/01 §三，手册原文）：
  Δ_pred = ŷ_treat − y_control      对照用**真实值**，不是预测值
  Δ_true = y_treat  − y_control
  μ_ctx  = 同上下文下**训练药物**的 Δ_true 均值      → 指标 3
  μ_drug = 同药物下**训练上下文**的 Δ_true 均值      → 指标 4
  所有参照统计**仅用训练数据冻结**。

指标 → 样本子集的分配：
  指标1 绝对保真度   全部样本（含对照，提交要求预测所有样本）
  指标2 匹配 FC      全部处理样本
  指标3 上下文残差   val_chem_only      （S1 新化合物）
  指标4 药物均值残差 val_strain_only    （S2 新菌株）
  指标5 双重/时间     val_both + val_time
  指标6 DEP          全部处理样本

不确定性：**按药物或菌株分组的 bootstrap**（CLAUDE.md R4），不是按行。
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import loader, paths
from scorer.config import ScorerConfig
from scorer.metrics import (combine, metric_absolute, metric_both_time, metric_dep,
                            metric_fc, metric_residual)

CTX_KEYS_DEFAULT = ["Strains", "Medium", "Temperature", "pert_time"]
PERT_COL = "perturbation_no_concentration"
VAL_SPLITS = ["val_chem_only", "val_strain_only", "val_both", "val_time"]

METRIC_ROWS = {
    "absolute": "all",
    "fc": "treated",
    "ctx_resid": ["val_chem_only"],
    "drug_resid": ["val_strain_only"],
    "both_time": ["val_both", "val_time"],
    "dep": "treated",
}


# --------------------------------------------------------------- 上下文


@dataclass
class EvalContext:
    meta: pd.DataFrame
    X: np.ndarray            # (n,p) log2 真值
    C: np.ndarray            # (n,p) 匹配对照谱（真实值）
    D: np.ndarray            # (n,p) Δ_true = X - C
    mu_ctx: np.ndarray       # (n,p) 训练冻结
    mu_drug: np.ndarray      # (n,p) 训练冻结
    proteins: np.ndarray
    is_control: np.ndarray
    is_qc: np.ndarray
    train_mask: np.ndarray
    ctx_keys: List[str] = field(default_factory=lambda: list(CTX_KEYS_DEFAULT))

    @property
    def n(self) -> int:
        return self.X.shape[0]

    def rows(self, splits) -> np.ndarray:
        return self.meta["split_final"].isin(splits).to_numpy()

    @property
    def treated(self) -> np.ndarray:
        return (~self.is_control) & (~self.is_qc)


def _group_mean_frozen(
    D: np.ndarray, keys: np.ndarray, fit_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """按 keys 分组，只用 fit_mask 的样本算逐蛋白均值；未见组回退到训练总体均值。

    返回 (mu (n,p), fallback (n,) bool)。
    """
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        glob = np.nanmean(D[fit_mask], axis=0, dtype=np.float64).astype(np.float32)
    mu = np.tile(glob, (D.shape[0], 1))
    fallback = np.ones(D.shape[0], dtype=bool)
    for k in np.unique(keys):
        sel = keys == k
        fit = sel & fit_mask
        if not fit.any():
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            m = np.nanmean(D[fit], axis=0, dtype=np.float64).astype(np.float32)
        m = np.where(np.isfinite(m), m, glob)
        mu[sel] = m
        fallback[sel] = False
    return mu, fallback


def build_context(
    delta_npz: Optional[str] = None,
    ctx_keys: Optional[List[str]] = None,
    verbose: bool = True,
) -> EvalContext:
    ctx_keys = list(ctx_keys or CTX_KEYS_DEFAULT)
    meta = loader.load_metadata("train_val")
    sid, proteins, Xr = loader.load_proteome_log2(verbose=False)
    X = loader.align_proteome_to_metadata(meta, sid, Xr)
    del Xr

    delta_npz = delta_npz or os.path.join(
        paths.RESULTS, "step2_control_match", "delta_true_train_val_median.npz")
    z = np.load(delta_npz, allow_pickle=False)
    assert list(z["sample_ids"]) == list(meta["sample_ID"].astype(str)), "样本顺序不一致"
    D = z["delta"]
    C = X - D                     # 对照谱可由此还原，step2 未重复存盘
    is_ctrl, is_qc = z["is_control"], z["is_qc"]

    treated = (~is_ctrl) & (~is_qc)
    train_mask = (meta["split_final"] == "train").to_numpy() & treated

    ctx_key = meta[ctx_keys].astype(str).agg("\x1f".join, axis=1).to_numpy()
    drug_key = meta[PERT_COL].astype(str).to_numpy()
    mu_ctx, fb_ctx = _group_mean_frozen(D, ctx_key, train_mask)
    mu_drug, fb_drug = _group_mean_frozen(D, drug_key, train_mask)

    if verbose:
        ev = ~train_mask & treated
        print(f"[eval] 参照均值冻结自 {int(train_mask.sum())} 个训练处理样本")
        print(f"[eval] 非训练处理样本中 μ_ctx 回退到总体均值的比例  "
              f"{fb_ctx[ev].mean():.1%}")
        print(f"[eval] 非训练处理样本中 μ_drug 回退到总体均值的比例 "
              f"{fb_drug[ev].mean():.1%}")

    return EvalContext(meta=meta, X=X, C=C, D=D, mu_ctx=mu_ctx, mu_drug=mu_drug,
                       proteins=proteins, is_control=is_ctrl, is_qc=is_qc,
                       train_mask=train_mask, ctx_keys=ctx_keys)


# --------------------------------------------------------------- 打分


def _subset(ctx: EvalContext, eval_mask: np.ndarray, spec) -> np.ndarray:
    if spec == "all":
        m = eval_mask
    elif spec == "treated":
        m = eval_mask & ctx.treated
    else:
        m = eval_mask & ctx.rows(spec) & ctx.treated
    return np.nonzero(m)[0]


def evaluate(
    y_pred: np.ndarray,
    ctx: EvalContext,
    eval_mask: np.ndarray,
    cfg: Optional[ScorerConfig] = None,
) -> dict:
    """y_pred : (n,p) 对**全部行**的 log2 预测；eval_mask 选出参与评估的行。"""
    cfg = cfg or ScorerConfig()
    # 差分一律走 float64：预测按 float32 落盘，float32 下 (C+μ)−C 会残留舍入噪声。
    # 单靠 float64 还不够（预测本身已被 float32 截断），常数判据必须同时用相对阈值。
    D_pred = y_pred.astype(np.float64) - ctx.C.astype(np.float64)
    parts: Dict[str, dict] = {}

    r = _subset(ctx, eval_mask, METRIC_ROWS["absolute"])
    parts["absolute"] = metric_absolute(ctx.X[r], y_pred[r], cfg) if r.size else {}

    r = _subset(ctx, eval_mask, METRIC_ROWS["fc"])
    parts["fc"] = metric_fc(ctx.D[r], D_pred[r], cfg) if r.size else {}

    r = _subset(ctx, eval_mask, METRIC_ROWS["ctx_resid"])
    parts["ctx_resid"] = (metric_residual(ctx.D[r], D_pred[r], ctx.mu_ctx[r], cfg)
                          if r.size else {})

    r = _subset(ctx, eval_mask, METRIC_ROWS["drug_resid"])
    parts["drug_resid"] = (metric_residual(ctx.D[r], D_pred[r], ctx.mu_drug[r], cfg)
                           if r.size else {})

    r = _subset(ctx, eval_mask, METRIC_ROWS["both_time"])
    parts["both_time"] = (metric_both_time(ctx.X[r], y_pred[r], ctx.D[r], D_pred[r], cfg)
                          if r.size else {})

    r = _subset(ctx, eval_mask, METRIC_ROWS["dep"])
    parts["dep"] = metric_dep(ctx.D[r], D_pred[r], cfg) if r.size else {}

    parts["total"] = combine(parts, cfg)
    return parts


def _evaluate_rows(y_pred, ctx: EvalContext, rows: np.ndarray, cfg=None) -> dict:
    """按**行索引**（允许重复）评分，供 bootstrap 用。

    与 evaluate() 的区别：evaluate() 接布尔掩码，天然去重；
    bootstrap 必须保留重抽样的重复计数，否则区间偏窄。
    """
    cfg = cfg or ScorerConfig()
    D_pred = y_pred.astype(np.float64) - ctx.C.astype(np.float64)
    treated = ctx.treated[rows]
    split = ctx.meta["split_final"].to_numpy()[rows]
    parts: Dict[str, dict] = {}

    parts["absolute"] = metric_absolute(ctx.X[rows], y_pred[rows], cfg)

    rt = rows[treated]
    parts["fc"] = metric_fc(ctx.D[rt], D_pred[rt], cfg) if rt.size else {}
    parts["dep"] = metric_dep(ctx.D[rt], D_pred[rt], cfg) if rt.size else {}

    rc = rows[treated & (split == "val_chem_only")]
    parts["ctx_resid"] = (metric_residual(ctx.D[rc], D_pred[rc], ctx.mu_ctx[rc], cfg)
                          if rc.size else {})
    rs = rows[treated & (split == "val_strain_only")]
    parts["drug_resid"] = (metric_residual(ctx.D[rs], D_pred[rs], ctx.mu_drug[rs], cfg)
                           if rs.size else {})
    rb = rows[treated & np.isin(split, ["val_both", "val_time"])]
    parts["both_time"] = (metric_both_time(ctx.X[rb], y_pred[rb], ctx.D[rb], D_pred[rb], cfg)
                          if rb.size else {})

    parts["total"] = combine(parts, cfg)
    return parts


def evaluate_by_split(y_pred, ctx: EvalContext, cfg=None) -> pd.DataFrame:
    """逐个官方 val 划分单独打指标 1 与指标 2（看模型在哪类外推上垮）。"""
    cfg = cfg or ScorerConfig()
    D_pred = y_pred - ctx.C
    out = []
    for s in VAL_SPLITS + ["train"]:
        m = ctx.rows([s])
        r_all = np.nonzero(m)[0]
        r_tr = np.nonzero(m & ctx.treated)[0]
        row = {"split": s, "n": int(m.sum()), "n_treated": int((m & ctx.treated).sum())}
        if r_all.size:
            ab = metric_absolute(ctx.X[r_all], y_pred[r_all], cfg)
            row["abs_pcc"], row["abs_r2"] = ab["pcc"], ab["r2"]
        if r_tr.size:
            row["fc_pcc"] = metric_fc(ctx.D[r_tr], D_pred[r_tr], cfg)["pcc"]
        out.append(row)
    return pd.DataFrame(out)


def flatten(parts: dict) -> dict:
    """把 evaluate 的嵌套结果压成一行，便于表格化。"""
    d = parts.get("dep", {})
    return {
        "total": parts.get("total", np.nan),
        "abs_pcc": parts.get("absolute", {}).get("pcc", np.nan),
        "abs_r2": parts.get("absolute", {}).get("r2", np.nan),
        "fc_pcc": parts.get("fc", {}).get("pcc", np.nan),
        "ctx_resid": parts.get("ctx_resid", {}).get("pcc", np.nan),
        "drug_resid": parts.get("drug_resid", {}).get("pcc", np.nan),
        "both_time": parts.get("both_time", {}).get("pcc", np.nan),
        "dep_dir": d.get("direction_acc", np.nan),
        "dep_pcc": d.get("high_effect_pcc", np.nan),
        "dep_f1": d.get("f1_at_k", np.nan),
    }


# --------------------------------------------------------------- bootstrap


def grouped_bootstrap(
    y_pred: np.ndarray,
    ctx: EvalContext,
    eval_mask: np.ndarray,
    group_by: str = PERT_COL,
    n_boot: int = 100,
    seed: int = 20260805,
    cfg: Optional[ScorerConfig] = None,
) -> dict:
    """按药物或菌株整组重抽样，给总分与各分项的 2.5/97.5 分位区间。

    ❗按行 bootstrap 会严重低估不确定性：同一化合物有上百个相关样本。
    """
    cfg = cfg or ScorerConfig()
    rng = np.random.default_rng(seed)
    rows = np.nonzero(eval_mask)[0]
    g = ctx.meta[group_by].astype(str).to_numpy()[rows]
    groups = np.unique(g)
    by_group = {k: rows[g == k] for k in groups}

    keys = ["total", "abs_pcc", "fc_pcc", "ctx_resid", "drug_resid", "both_time"]
    acc = {k: [] for k in keys}
    for _ in range(n_boot):
        pick = rng.choice(groups, size=groups.size, replace=True)
        sel = np.concatenate([by_group[k] for k in pick])
        # ❗不能用布尔掩码 + np.unique：那样会把「同一组被抽中两次」压成一次，
        # 实际退化为无放回的 cluster subsampling，不是 bootstrap，区间会系统性偏窄。
        # 正确做法是按重抽样的行索引（含重复）实际复制样本再评分。
        # 2026-08-06 由 GPT Pro R3 · L2-1 指出。
        f = flatten(_evaluate_rows(y_pred, ctx, sel, cfg))
        for k in keys:
            acc[k].append(f[k])
    out = {}
    for k in keys:
        v = np.asarray([x for x in acc[k] if np.isfinite(x)])
        out[k] = (np.nan, np.nan) if v.size < 5 else (
            float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    out["_n_boot"] = n_boot
    out["_group_by"] = group_by
    out["_n_groups"] = int(groups.size)
    return out


def grouped_bootstrap_paired(
    y_a: np.ndarray,
    y_b: np.ndarray,
    ctx: EvalContext,
    eval_mask: np.ndarray,
    group_by: str = PERT_COL,
    n_boot: int = 100,
    seed: int = 20260806,
    cfg: Optional[ScorerConfig] = None,
) -> dict:
    """两个模型的**配对** bootstrap：每次重抽样上算差值，再取差值的分位区间。

    ❗不能拿两个模型各自的边际区间去看是否重叠——两个估计跑在同一批数据、
    同一批重抽样上，高度相关；边际区间重叠**不代表**差异不显著。
    要判断 B 是否优于 A，必须看 (B−A) 这个量自身的区间是否含 0。
    """
    cfg = cfg or ScorerConfig()
    rng = np.random.default_rng(seed)
    rows = np.nonzero(eval_mask)[0]
    g = ctx.meta[group_by].astype(str).to_numpy()[rows]
    groups = np.unique(g)
    by_group = {k: rows[g == k] for k in groups}

    diffs, a_vals, b_vals = [], [], []
    for _ in range(n_boot):
        pick = rng.choice(groups, size=groups.size, replace=True)
        sel = np.concatenate([by_group[k] for k in pick])
        fa = flatten(_evaluate_rows(y_a, ctx, sel, cfg))["total"]
        fb = flatten(_evaluate_rows(y_b, ctx, sel, cfg))["total"]
        if np.isfinite(fa) and np.isfinite(fb):
            a_vals.append(fa)
            b_vals.append(fb)
            diffs.append(fb - fa)
    d = np.asarray(diffs)
    if d.size < 5:
        return {"n_boot": int(d.size), "insufficient": True}
    lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
    return {
        "n_boot": int(d.size),
        "n_groups": int(groups.size),
        "group_by": group_by,
        "diff_mean": float(d.mean()),
        "diff_ci": (lo, hi),
        "excludes_zero": bool(lo > 0 or hi < 0),
        "frac_positive": float((d > 0).mean()),
        "a_ci": (float(np.percentile(a_vals, 2.5)), float(np.percentile(a_vals, 97.5))),
        "b_ci": (float(np.percentile(b_vals, 2.5)), float(np.percentile(b_vals, 97.5))),
    }


if __name__ == "__main__":
    ctx = build_context()
    val = ctx.rows(VAL_SPLITS)
    print(f"\n自检：用真值本身当预测（应各项 ≈ 1）")
    f = flatten(evaluate(ctx.X, ctx, val))
    for k, v in f.items():
        print(f"  {k:<12}{v:>8.4f}")
