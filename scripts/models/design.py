"""设计矩阵：冻结（freeze）与编码（encode）分离。

**旧的 `baseline_cfree.design()` 已删除**，不是加了个参数——留着它就会被再次误用。

旧写法为什么违规（2026-08-24 复赛整改 L1-1）：

    codes, uniq = pd.factorize(meta[c].astype(str))   # 词表来自传进去的整张表
    t = (t - t.mean()) / t.std()                      # 标准化参数同上

拟合行只是后一步的掩码；到这一步，val 的水平集合与时间分布已经进了模型定义。
手册第 17 页明文：验证集与测试集「不得用于估计任何统计量（含保留蛋白列表与
归一化参数）」。而且这类泄漏**不会报错**，它只是安静地改变了设计矩阵。

现在的形态与 `predict_test.py` 一直在用的 `freeze_vocab`/`encode` 一致：

    spec = freeze(meta, fit_rows, cat_cols)   # 只看 fit_rows
    Z    = encode(meta, spec)                 # 按 spec 编码任意表

`freeze()` 内部调 `split_guard.assert_train_only`，拟合行里混进 val 直接抛异常。
未见水平 → 该块整行 0 → 岭回归自动回退到总体先验（S2 未见菌株就走这条路）。

自检：python design.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import split_guard as sg

PERT_COL = "perturbation_no_concentration"
TIME_COL = "pert_time"

#: 上下文特征集。板号与培养基/温度/时间的 Cramér's V = 0.992，
#: 加板号等于把这三个生物变量的身份背下来，对新板不可泛化，故不入。
FEATURE_SETS = {
    "bio": ["Strains", "Medium", "Temperature"],
    "bio_tech": ["Strains", "Medium", "Temperature", "data_source", "instrument"],
    # 基线阶梯 B4 用的口径（只含生物条件 + 药物 one-hot）
    "bio_drug": ["Strains", "Medium", "Temperature"],
}

SPEC_VERSION = "design/2.0-frozen"


def freeze(
    meta: pd.DataFrame,
    fit_rows,
    cat_cols,
    with_drug: bool = False,
    drug_col: str = PERT_COL,
    enforce_train_only: bool = True,
) -> dict:
    """只用 `fit_rows` 冻结词表与 log-time 标准化参数，返回可序列化的 spec。

    `enforce_train_only=False` 只留给一处：LOCO 的**折内**再切分——
    那时拟合行本身已经是 train 折的子集，外层已经守过一次。
    """
    if enforce_train_only:
        mask = sg.assert_train_only(meta, fit_rows, what="design 冻结行")
    else:
        mask = sg._as_bool_mask(fit_rows, len(meta))
        if not mask.any():
            raise ValueError("design 冻结行为空")

    cat_cols = list(cat_cols)
    sub = meta.loc[mask]
    levels = {c: sorted(sub[c].astype(str).unique().tolist()) for c in cat_cols}

    t = np.log1p(sub[TIME_COL].to_numpy(dtype=np.float64))
    t_mean, t_std = float(t.mean()), float(t.std())
    if not np.isfinite(t_std) or t_std < 1e-12:
        # 拟合行只有单一时间点时三次多项式退化。不静默除零，改成恒零块并标记。
        t_std, degenerate = 1.0, True
    else:
        degenerate = False

    spec = {
        "_version": SPEC_VERSION,
        "cat_cols": cat_cols,
        "levels": levels,
        "t_mean": t_mean,
        "t_std": t_std,
        "t_degenerate": degenerate,
        "with_drug": bool(with_drug),
        "drug_col": drug_col,
        "drugs": (sorted(sub[drug_col].astype(str).unique().tolist())
                  if with_drug else []),
        "fit_rows_digest": sg.rows_digest(meta, mask),
    }
    spec["n_cols"] = n_cols(spec)
    return spec


def n_cols(spec: dict) -> int:
    return (1 + sum(len(spec["levels"][c]) for c in spec["cat_cols"])
            + 3 + (len(spec["drugs"]) if spec["with_drug"] else 0))


def encode(meta: pd.DataFrame, spec: dict, soft_levels: dict | None = None) -> np.ndarray:
    """按冻结的 spec 编码任意表。未见水平整块归零。

    `soft_levels` 给未见水平一个**软 one-hot**：`{列名: {未见取值: {已见水平: 权重}}}`。
    默认 None 时行为与从前完全一致（未见水平整块 0）。

    这是 S2（未见菌株）那条路的唯一接口：把一株没见过的菌编码成「87% 的最近训练菌株
    + 12% 的次近 + 1% 的最远」，模型本身一个字都不用改。权重从哪来是 `models/strain_transport.py`
    的事，本模块只负责把它写进设计矩阵，并且**只允许写在已见水平上**——
    这一点由下面的检查强制，防止有人借软编码悄悄新开一列。
    """
    if spec.get("_version") != SPEC_VERSION:
        raise ValueError(f"spec 版本不符：{spec.get('_version')} ≠ {SPEC_VERSION}")
    soft_levels = soft_levels or {}
    for c, mp in soft_levels.items():
        if c not in spec["cat_cols"]:
            raise ValueError(f"soft_levels 里的列 {c} 不在 spec 的类别列内")
        seen = set(spec["levels"][c])
        for src, w in mp.items():
            if src in seen:
                raise ValueError(f"{c}={src} 是已见水平，不许用软编码覆盖它")
            bad = set(w) - seen
            if bad:
                raise ValueError(f"{c}={src} 的软权重指向未见水平 {sorted(bad)}")
    n = len(meta)
    blocks = [np.ones((n, 1), dtype=np.float32)]
    for c in spec["cat_cols"]:
        levels = spec["levels"][c]
        idx = {v: i for i, v in enumerate(levels)}
        soft = soft_levels.get(c, {})
        M = np.zeros((n, len(levels)), dtype=np.float32)
        for i, v in enumerate(meta[c].astype(str)):
            j = idx.get(v)
            if j is not None:
                M[i, j] = 1.0
            elif v in soft:
                for s, wt in soft[v].items():
                    M[i, idx[s]] = np.float32(wt)
        blocks.append(M)

    t = np.log1p(meta[TIME_COL].to_numpy(dtype=np.float64))
    t = (t - spec["t_mean"]) / spec["t_std"]
    tb = np.stack([t, t ** 2, t ** 3], axis=1).astype(np.float32)
    if spec["t_degenerate"]:
        tb[:] = 0.0
    blocks.append(tb)

    if spec["with_drug"]:
        idx = {d: i for i, d in enumerate(spec["drugs"])}
        M = np.zeros((n, len(spec["drugs"])), dtype=np.float32)
        for i, v in enumerate(meta[spec["drug_col"]].astype(str)):
            j = idx.get(v)
            if j is not None:
                M[i, j] = 1.0          # 未见化合物整行 0 → 响应项归零
        blocks.append(M)

    Z = np.hstack(blocks)
    if Z.shape[1] != spec["n_cols"]:
        raise AssertionError(f"列数 {Z.shape[1]} ≠ spec 记录的 {spec['n_cols']}")
    return Z


def unseen_report(meta: pd.DataFrame, spec: dict) -> dict:
    """哪些行落到了未见水平上——评审必问 S2 未见菌株怎么处理。"""
    out = {}
    for c in spec["cat_cols"]:
        seen = set(spec["levels"][c])
        vals = meta[c].astype(str)
        bad = ~vals.isin(seen)
        if bad.any():
            out[c] = {"n_rows": int(bad.sum()), "levels": sorted(set(vals[bad]))}
    if spec["with_drug"]:
        seen = set(spec["drugs"])
        vals = meta[spec["drug_col"]].astype(str)
        bad = ~vals.isin(seen)
        if bad.any():
            out[spec["drug_col"]] = {"n_rows": int(bad.sum()),
                                     "levels": sorted(set(vals[bad]))}
    return out


def _selftest() -> None:
    meta = pd.DataFrame({
        "sample_ID": [f"s{i}" for i in range(8)],
        "split_final": ["train"] * 5 + ["val_strain_only"] * 3,
        "Strains": ["A", "A", "B", "B", "A", "Z", "Z", "Z"],
        "Medium": ["m1", "m2", "m1", "m2", "m1", "m1", "m2", "m1"],
        "Temperature": [30, 30, 37, 37, 30, 30, 37, 30],
        "pert_time": [15, 30, 60, 120, 240, 15, 60, 240],
        PERT_COL: ["c1", "c2", "c1", "c3", "c2", "c9", "c1", "c9"],
    })
    tr = sg.train_rows(meta)
    ok = 0

    spec = freeze(meta, tr, ["Strains", "Medium", "Temperature"])
    assert spec["levels"]["Strains"] == ["A", "B"], "未见菌株 Z 不该进词表"; ok += 1
    Z = encode(meta, spec)
    assert Z.shape == (8, 1 + 2 + 2 + 2 + 3), Z.shape; ok += 1
    assert Z[5:, 1:3].sum() == 0.0, "未见菌株行的菌株块必须整行 0"; ok += 1

    # 标准化参数只能由拟合行决定：改动 val 行的任何字段都不许改变 spec 与拟合行编码
    m2 = meta.copy()
    m2.loc[5:, "pert_time"] = 99999
    m2.loc[5:, "Strains"] = "QQQ"
    spec2 = freeze(m2, tr, ["Strains", "Medium", "Temperature"])
    assert spec2["t_mean"] == spec["t_mean"] and spec2["t_std"] == spec["t_std"]; ok += 1
    assert spec2["levels"] == spec["levels"]; ok += 1
    assert np.array_equal(encode(m2, spec2)[tr], Z[tr]), "拟合行编码被 val 行改动影响"; ok += 1

    specd = freeze(meta, tr, ["Strains"], with_drug=True)
    assert specd["drugs"] == ["c1", "c2", "c3"], specd["drugs"]; ok += 1
    Zd = encode(meta, specd)
    assert Zd[5, -3:].sum() == 0.0, "未见化合物 c9 的药物块必须整行 0"; ok += 1
    assert unseen_report(meta, specd)["Strains"]["levels"] == ["Z"]; ok += 1

    try:
        freeze(meta, np.ones(8, dtype=bool), ["Strains"])
    except sg.TrainBoundaryError:
        ok += 1
    else:
        raise AssertionError("freeze 没拦住含 val 的拟合行")

    # 单一时间点 → 退化块归零而不是除零
    m3 = meta.copy()
    m3["pert_time"] = 60
    sp3 = freeze(m3, tr, ["Strains"])
    assert sp3["t_degenerate"] and np.isfinite(encode(m3, sp3)).all(); ok += 1

    assert n_cols(spec) == encode(meta, spec).shape[1]; ok += 1

    # ---- 软 one-hot（S2 未见菌株的编码接口）----
    soft = {"Strains": {"Z": {"A": 0.75, "B": 0.25}}}
    Zs = encode(meta, spec, soft_levels=soft)
    assert Zs.shape == Z.shape, "软编码不许改变列数"; ok += 1
    assert np.allclose(Zs[tr], Z[tr]), "软编码不许动已见水平的行"; ok += 1
    assert np.allclose(Zs[5, 1:3], [0.75, 0.25]), Zs[5, 1:3]; ok += 1
    for bad_soft, why in [
        ({"Strains": {"A": {"B": 1.0}}}, "拿软编码覆盖已见水平"),
        ({"Strains": {"Z": {"QQ": 1.0}}}, "软权重指向未见水平"),
        ({"Medium": {"Z": {"A": 1.0}}}, "软权重指向未见水平（列错配）"),
        ({"NotACol": {"Z": {"A": 1.0}}}, "列不在 spec 内"),
    ]:
        try:
            encode(meta, spec, soft_levels=bad_soft)
        except ValueError:
            ok += 1
        else:
            raise AssertionError(f"软编码没拦住：{why}")
    print(f"[design] selftest {ok} 项全部通过")


if __name__ == "__main__":
    _selftest()
