"""第 2 步 · 对照匹配管线：产出每个样本的 Δ_true。

口径（docs/05_训练方案.md §2.2，来源为官方手册）：

    Δ_true = log2(y_treat) − log2(y_control)

对照须匹配 **7 项**：data_source / Strains / Medium / Temperature / pert_time /
instrument / Yeast_cell_plate。**不含 protein_well**（手册未列 well）。

对照集合 = Water + DMSO（实测 0 匹配率 0.0%）。Quality Control 单独识别，
**不作为生物学对照**——它在语义上是质控样本。

92.9% 的处理样本匹配到多个合法对照（中位 2，最多 3），官方合并规则未定义，
故做成可切换参数，默认**逐蛋白中位数**。

缺失（27%）**不填补**：对照侧逐蛋白 nanmedian（全缺才是 NaN），处理侧原样，
Δ 在任一侧缺失处为 NaN。下游损失必须掩码。

附带产出（噪声地板）：对照样本自身也算 Δ，但**留一法排除自身**，
得到的是纯技术+生物噪声的 Δ 分布 —— 它是指标 2 可达上限的经验参照。

用法：
    python control_match.py                     # 默认 median
    python control_match.py --agg mean
    python control_match.py --controls Water DMSO "Quality Control"
"""
from __future__ import annotations

import argparse
import os
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import loader, paths

# 手册规定的对照匹配键（7 项，不含 well）
MATCH_KEYS: List[str] = [
    "data_source", "Strains", "Medium", "Temperature",
    "pert_time", "instrument", "Yeast_cell_plate",
]
PERT_COL = "perturbation_no_concentration"
SOLVENT_CONTROLS: List[str] = ["Water", "DMSO"]
QC_LABEL = "Quality Control"

OUT_DIR = os.path.join(paths.RESULTS, "step2_control_match")


@dataclass
class MatchConfig:
    match_keys: List[str] = field(default_factory=lambda: list(MATCH_KEYS))
    control_names: List[str] = field(default_factory=lambda: list(SOLVENT_CONTROLS))
    agg: str = "median"                  # median | mean
    exclude_self_for_controls: bool = True   # 对照样本用留一法，避免 Δ≡0 的自指
    pert_col: str = PERT_COL

    def __post_init__(self) -> None:
        if self.agg not in ("median", "mean"):
            raise ValueError(f"agg must be median|mean, got {self.agg}")


# --------------------------------------------------------------- 分组与索引


def group_ids(meta: pd.DataFrame, cfg: MatchConfig) -> np.ndarray:
    """按 7 个匹配键给每行一个组号。处理样本与其合法对照同组。"""
    missing = [k for k in cfg.match_keys if k not in meta.columns]
    if missing:
        raise KeyError(f"metadata 缺少匹配键：{missing}")
    key = meta[cfg.match_keys].astype(str).agg("\x1f".join, axis=1)
    return pd.factorize(key, sort=True)[0].astype(np.int64)


def control_rows_by_group(
    meta: pd.DataFrame, gid: np.ndarray, cfg: MatchConfig
) -> Dict[int, np.ndarray]:
    is_ctrl = meta[cfg.pert_col].isin(cfg.control_names).to_numpy()
    out: Dict[int, np.ndarray] = {}
    rows = np.nonzero(is_ctrl)[0]
    for g in np.unique(gid[rows]):
        out[int(g)] = rows[gid[rows] == g]
    return out


# --------------------------------------------------------------- 对照谱聚合


def _aggregate(block: np.ndarray, how: str) -> np.ndarray:
    """逐蛋白聚合多个对照。全缺列返回 NaN，不填补。"""
    if block.shape[0] == 0:
        return np.full(block.shape[1], np.nan, dtype=np.float32)
    if block.shape[0] == 1:
        return block[0].astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slice
        v = np.nanmedian(block, axis=0) if how == "median" else np.nanmean(block, axis=0)
    return v.astype(np.float32)


def control_profiles(
    X: np.ndarray, meta: pd.DataFrame, cfg: MatchConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (C, n_ctrl, is_ctrl)。

    C       : (n, p) 每个样本对应的匹配对照谱，无合法对照时整行 NaN
    n_ctrl  : (n,)   实际参与聚合的对照样本数（对照样本已扣除自身）
    is_ctrl : (n,)   该样本本身是否为溶剂对照
    """
    n, p = X.shape
    gid = group_ids(meta, cfg)
    ctrl_rows = control_rows_by_group(meta, gid, cfg)
    is_ctrl = meta[cfg.pert_col].isin(cfg.control_names).to_numpy()

    C = np.full((n, p), np.nan, dtype=np.float32)
    n_ctrl = np.zeros(n, dtype=np.int32)

    # 处理样本：同组内全部对照，按组缓存（同组多样本共用一个对照谱）
    cache: Dict[int, np.ndarray] = {}
    for g, rows in ctrl_rows.items():
        cache[g] = _aggregate(X[rows], cfg.agg)

    treat_rows = np.nonzero(~is_ctrl)[0]
    for i in treat_rows:
        g = int(gid[i])
        if g in cache:
            C[i] = cache[g]
            n_ctrl[i] = len(ctrl_rows[g])

    # 对照样本：留一法排除自身，得到噪声地板参照
    if cfg.exclude_self_for_controls:
        for i in np.nonzero(is_ctrl)[0]:
            g = int(gid[i])
            rows = ctrl_rows.get(g)
            if rows is None:
                continue
            others = rows[rows != i]
            if others.size == 0:
                continue
            C[i] = _aggregate(X[others], cfg.agg)
            n_ctrl[i] = others.size
    else:
        for i in np.nonzero(is_ctrl)[0]:
            g = int(gid[i])
            if g in cache:
                C[i] = cache[g]
                n_ctrl[i] = len(ctrl_rows[g])

    return C, n_ctrl, is_ctrl


def compute_delta(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Δ = 处理 − 对照。任一侧缺失即 NaN（成对完整，不填补）。"""
    return (X - C).astype(np.float32)


# --------------------------------------------------------------- 主流程


def run(cfg: MatchConfig, out_dir: str = OUT_DIR, verbose: bool = True) -> dict:
    meta = loader.load_metadata("train_val")
    sid, proteins, X_raw = loader.load_proteome_log2(verbose=verbose)
    X = loader.align_proteome_to_metadata(meta, sid, X_raw)
    del X_raw

    C, n_ctrl, is_ctrl = control_profiles(X, meta, cfg)
    D = compute_delta(X, C)

    is_qc = (meta[cfg.pert_col] == QC_LABEL).to_numpy()
    is_treat = ~is_ctrl & ~is_qc

    paths.ensure_dir(out_dir)
    npz = os.path.join(out_dir, f"delta_true_train_val_{cfg.agg}.npz")

    def _u(seq, width=64):
        """定宽 unicode：pandas 3 / numpy 2 的字符串数组存进 npz 会退化成 object dtype。"""
        return np.asarray([str(x) for x in seq], dtype=f"<U{width}")

    np.savez(
        npz,
        delta=D,
        n_ctrl=n_ctrl,
        is_control=is_ctrl,
        is_qc=is_qc,
        sample_ids=_u(meta["sample_ID"], 32),
        proteins=_u(proteins),
        split_final=_u(meta["split_final"], 32),
        perturbation=_u(meta[cfg.pert_col]),
        match_keys=_u(cfg.match_keys, 32),
        control_names=_u(cfg.control_names, 32),
        agg=_u([cfg.agg], 16),
    )

    report = _report(meta, X, C, D, n_ctrl, is_ctrl, is_qc, is_treat, cfg, npz)
    rpt_path = os.path.join(out_dir, f"report_{cfg.agg}.txt")
    with open(rpt_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(report)
    if verbose:
        print(report)
        print(f"[写出] {npz}\n[写出] {rpt_path}")

    return {"delta": D, "n_ctrl": n_ctrl, "meta": meta, "proteins": proteins,
            "is_control": is_ctrl, "is_qc": is_qc, "npz": npz}


def _report(meta, X, C, D, n_ctrl, is_ctrl, is_qc, is_treat, cfg, npz_path) -> str:
    L: List[str] = []
    a = L.append
    a("=" * 78)
    a("第 2 步 · 对照匹配管线 · 报告")
    a("=" * 78)
    a(f"匹配键 ({len(cfg.match_keys)})   : {' / '.join(cfg.match_keys)}")
    a(f"对照集合          : {' + '.join(cfg.control_names)}   （QC 单列，不作生物对照）")
    a(f"多对照聚合        : 逐蛋白 {cfg.agg}")
    a(f"对照样本自身      : {'留一法排除自身' if cfg.exclude_self_for_controls else '含自身'}")
    a("")

    a("-" * 78)
    a("A. 匹配覆盖率（处理样本，排除 Water/DMSO/QC）")
    a("-" * 78)
    nt = int(is_treat.sum())
    nc = n_ctrl[is_treat]
    a(f"处理样本数 {nt}")
    for lo, hi, lab in [(0, 0, "0 匹配"), (1, 1, "1 匹配"), (2, 10**9, "多匹配")]:
        m = (nc >= lo) & (nc <= hi)
        a(f"  {lab:<8} {int(m.sum()):5d}  ({100*m.mean():5.1f}%)")
    if (nc > 1).any():
        a(f"  多匹配时对照数：中位 {np.median(nc[nc>1]):.0f}  最大 {nc.max()}")
    a("")
    a("按 split_final：")
    for k, idx in meta.loc[is_treat].groupby("split_final").groups.items():
        sub = n_ctrl[np.asarray(idx)]
        a(f"  {k:<18} n={len(sub):5d}   0匹配 {int((sub==0).sum()):4d} ({100*(sub==0).mean():4.1f}%)"
          f"   中位对照数 {np.median(sub):.0f}")
    a("")

    a("-" * 78)
    a("B. Δ_true 覆盖与量级（处理样本）")
    a("-" * 78)
    Dt = D[is_treat]
    Xt = X[is_treat]
    a(f"log2 丰度缺失率        {np.isnan(Xt).mean():7.4%}")
    a(f"Δ_true 缺失率          {np.isnan(Dt).mean():7.4%}   （对照侧再损失一部分）")
    a(f"  其中处理侧有值但对照侧缺 {np.mean(np.isfinite(Xt) & np.isnan(Dt)):7.4%}")
    fin = Dt[np.isfinite(Dt)]
    a(f"Δ 有效点 {fin.size:,}")
    q = np.percentile(fin, [1, 5, 25, 50, 75, 95, 99])
    a("Δ 分位  p1 %+.3f  p5 %+.3f  p25 %+.3f  p50 %+.3f  p75 %+.3f  p95 %+.3f  p99 %+.3f" % tuple(q))
    a(f"Δ 标准差 {fin.std():.3f}   中位 |Δ| {np.median(np.abs(fin)):.3f}")
    a(f"|Δ| > 1（手册 DEP 阈值）占比 {np.mean(np.abs(fin) > 1):.2%}")
    a("")
    a("全缺失蛋白（整列无观测）:")
    a(f"  丰度矩阵 {int(np.isnan(X).all(axis=0).sum())} 列   Δ 矩阵 {int(np.isnan(Dt).all(axis=0).sum())} 列")
    a("")

    a("-" * 78)
    a("C. 噪声地板：对照 vs 对照（留一法）")
    a("-" * 78)
    a("对照样本的 Δ 在生物学上应为 0，其离散度即技术+生物噪声，")
    a("是指标 2（匹配对照原始 FC，25%）可达上限的经验参照。")
    ctrl_ok = is_ctrl & (n_ctrl > 0)
    if ctrl_ok.sum():
        Dc = D[ctrl_ok]
        finc = Dc[np.isfinite(Dc)]
        a(f"参与样本 {int(ctrl_ok.sum())}（{cfg.control_names} 中有其它同组对照的）")
        a(f"  噪声 Δ 标准差 {finc.std():.3f}   中位 |Δ| {np.median(np.abs(finc)):.3f}"
          f"   |Δ|>1 占比 {np.mean(np.abs(finc) > 1):.2%}")
        a(f"  信噪比（处理中位|Δ| / 对照中位|Δ|）= "
          f"{np.median(np.abs(fin)) / max(np.median(np.abs(finc)), 1e-9):.2f}")
        a("  注：比值接近 1 意味着单样本 Δ 主要是噪声，模型上限受限；")
        a("      >1 说明扰动信号确实高于对照间波动。")
    else:
        a("  无可用对照（所有对照组内只有 1 个对照样本）")
    a("")

    a("-" * 78)
    a("D. QC 样本（单独识别，未作对照）")
    a("-" * 78)
    a(f"QC 样本数 {int(is_qc.sum())}   已从对照集合与处理集合中双双排除")
    a("")
    a(f"产出：{npz_path}")
    a("  delta (n,p) float32 / n_ctrl / is_control / is_qc / sample_ids / proteins /")
    a("  split_final / perturbation / match_keys / control_names / agg")
    a("  对照谱可由 C = X − delta 还原，故不重复存盘。")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description="对照匹配 → Δ_true")
    ap.add_argument("--agg", choices=["median", "mean"], default="median")
    ap.add_argument("--controls", nargs="+", default=list(SOLVENT_CONTROLS))
    ap.add_argument("--include-self", action="store_true",
                    help="对照样本不做留一（会得到 Δ≈0 的自指结果，仅调试用）")
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()
    cfg = MatchConfig(control_names=list(args.controls), agg=args.agg,
                      exclude_self_for_controls=not args.include_self)
    run(cfg, out_dir=args.out)


if __name__ == "__main__":
    main()
