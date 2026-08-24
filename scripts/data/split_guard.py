"""训练边界守卫：任何进入拟合的行，都必须是 `split_final == 'train'`。

为什么要有这个模块（2026-08-24 复赛整改 L1-1）：

官方手册（2026-08 修订版）第 17 页：

> 训练仅可使用 **train 划分**的蛋白质组标签，验证集用于模型选择；
> **验证集与测试集均不得参与训练，也不得用于估计任何统计量**
> （含保留蛋白列表与归一化参数）。

复赛门槛是「晋级队伍须通过代码复现核验，核查不通过者不计成绩」。
靠「记得只传 train 行」是守不住的——旧版 `design()` 就是这么破的：
拟合行确实只有 train，但**词表与标准化参数**在传进去的整张表上算，
于是 val 的水平集合与时间分布已经进了模型定义，而且不报错。

因此边界靠调用即抛异常的守卫实现，不靠注释。所有拟合入口都必须先过
`assert_train_only()`；拿训练行统一走 `train_rows()`，不再各写各的掩码。

自检：python split_guard.py
"""
from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np
import pandas as pd

TRAIN_SPLIT = "train"
SPLIT_COL = "split_final"
#: 官方四类验证划分。它们只许用于模型选择与最终一次性评估，绝不进拟合。
VAL_SPLITS = ("val_chem_only", "val_strain_only", "val_both", "val_time")


class TrainBoundaryError(RuntimeError):
    """拟合行里混进了 train 折以外的样本。"""


def _as_bool_mask(rows, n: int) -> np.ndarray:
    """把布尔掩码 / 整数索引 / 索引列表统一成长度 n 的布尔掩码。"""
    r = np.asarray(rows)
    if r.dtype == bool:
        if r.shape != (n,):
            raise ValueError(f"布尔掩码长度 {r.shape} ≠ 表长 {n}")
        return r
    m = np.zeros(n, dtype=bool)
    m[r.astype(np.int64)] = True
    return m


def train_rows(
    meta: pd.DataFrame,
    treated: Optional[np.ndarray] = None,
    treated_only: bool = False,
) -> np.ndarray:
    """唯一的训练行定义：`split_final == 'train'`。

    `treated_only=True` 时再与处理样本掩码取交（对照与质控排除）。
    对照样本本身是合法的绝对丰度监督，所以默认**保留**；
    只有响应类模型（Δ 空间）才需要 treated_only。
    """
    m = (meta[SPLIT_COL].astype(str) == TRAIN_SPLIT).to_numpy()
    if treated_only:
        if treated is None:
            raise ValueError("treated_only=True 时必须传 treated 掩码")
        m = m & np.asarray(treated, dtype=bool)
    return m


def assert_train_only(meta: pd.DataFrame, rows, what: str = "拟合行") -> np.ndarray:
    """守卫：rows 里出现任何非 train 折样本即抛异常。返回规范化的布尔掩码。"""
    mask = _as_bool_mask(rows, len(meta))
    split = meta[SPLIT_COL].astype(str).to_numpy()
    bad = mask & (split != TRAIN_SPLIT)
    if bad.any():
        from collections import Counter

        c = Counter(split[bad])
        raise TrainBoundaryError(
            f"{what}里混进 {int(bad.sum())} 行非 train 折样本：{dict(c)}。"
            f"手册第 17 页：验证集与测试集不得参与训练，也不得用于估计任何统计量。"
        )
    if not mask.any():
        raise TrainBoundaryError(f"{what}为空，无法拟合")
    return mask


def rows_digest(meta: pd.DataFrame, rows) -> str:
    """拟合行的内容指纹（排序后的 sample_ID）——写进 spec，供复现核验比对。"""
    import hashlib

    mask = _as_bool_mask(rows, len(meta))
    ids = sorted(meta.loc[mask, "sample_ID"].astype(str).tolist())
    h = hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()
    return f"{int(mask.sum())}:{h[:16]}"


def _selftest() -> None:
    import sys

    meta = pd.DataFrame({
        "sample_ID": [f"s{i}" for i in range(6)],
        SPLIT_COL: ["train", "train", "val_chem_only", "train",
                    "val_strain_only", "val_time"],
    })
    treated = np.array([True, False, True, True, True, True])

    ok = 0
    tr = train_rows(meta)
    assert tr.tolist() == [True, True, False, True, False, False]; ok += 1
    trt = train_rows(meta, treated=treated, treated_only=True)
    assert trt.tolist() == [True, False, False, True, False, False]; ok += 1
    assert assert_train_only(meta, tr).sum() == 3; ok += 1
    assert assert_train_only(meta, np.array([0, 1, 3])).sum() == 3; ok += 1
    for bad in (np.array([0, 2]), np.ones(6, dtype=bool)):
        try:
            assert_train_only(meta, bad)
        except TrainBoundaryError:
            ok += 1
        else:
            raise AssertionError("守卫没拦住混入 val 的拟合行")
    try:
        assert_train_only(meta, np.zeros(6, dtype=bool))
    except TrainBoundaryError:
        ok += 1
    else:
        raise AssertionError("空拟合行没被拦")
    d1, d2 = rows_digest(meta, tr), rows_digest(meta, np.array([3, 1, 0]))
    assert d1 == d2, "指纹应与行顺序无关"; ok += 1
    assert rows_digest(meta, np.array([0, 1])) != d1; ok += 1
    print(f"[split_guard] selftest {ok} 项全部通过")


if __name__ == "__main__":
    _selftest()
