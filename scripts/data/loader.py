"""数据加载：metadata + log2 蛋白质组矩阵（缺失保留为 NaN，不填补）。

所有读取都过 `paths.assert_readable` 守卫。CSV 解析一次后缓存成 npz，
后续步骤（基线、低秩、响应模型）直接命中缓存。
"""
from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import paths

_CACHE_FILE = "proteome_train_val_log2.npz"


def load_metadata(which: str = "train_val") -> pd.DataFrame:
    """which ∈ {'train_val', 'test'}。test 侧只有 metadata，没有蛋白丰度入口。"""
    if which == "train_val":
        p = paths.META_TRAIN_VAL
    elif which == "test":
        p = paths.META_TEST
    else:
        raise ValueError(f"unknown metadata split: {which}")
    return pd.read_csv(paths.assert_readable(p))


def load_proteome_log2(
    use_cache: bool = True, force: bool = False, verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (sample_ids, protein_names, X_log2)。

    X_log2 : float32, shape (n_samples, n_proteins)，缺失为 NaN。
    ❗不做任何填补：目标缺失位置不得当作已知值参与损失（docs/05 §2.1）。
    """
    cache = os.path.join(paths.CACHE, _CACHE_FILE)
    if use_cache and not force and os.path.exists(cache):
        z = np.load(cache, allow_pickle=False)
        if verbose:
            print(f"[loader] 命中缓存 {cache}")
        return z["sample_ids"].astype(str), z["proteins"].astype(str), z["X"]

    src = paths.assert_readable(paths.PROT_TRAIN_VAL)
    if verbose:
        print(f"[loader] 解析 {src} …（首次约 30-60 秒）")
    df = pd.read_csv(src)
    # 显式定宽 unicode：pandas 3 / numpy 2 的字符串数组存进 npz 会变成 object dtype，
    # 再以 allow_pickle=False 读回会报错。
    sample_ids = np.asarray(df.iloc[:, 0].astype(str).tolist(), dtype="<U32")
    proteins = np.asarray([str(c) for c in df.columns[1:]], dtype="<U64")
    raw = df.iloc[:, 1:].to_numpy(dtype=np.float64)
    del df

    n_nonpos = int(np.sum(raw <= 0))
    if n_nonpos:
        # docs/05 §2.1 第 1 步：确认是否存在 0 值。有则置 NaN 并记录，不做伪计数。
        print(f"[loader] ⚠ 发现 {n_nonpos} 个非正值 → 置 NaN（log2 无定义），未做伪计数填补")
        raw[raw <= 0] = np.nan

    ok = np.isfinite(raw)
    X = np.full(raw.shape, np.nan, dtype=np.float32)
    X[ok] = np.log2(raw[ok]).astype(np.float32)
    del raw, ok

    paths.ensure_dir(paths.CACHE)
    np.savez(cache, sample_ids=sample_ids, proteins=proteins, X=X)
    if verbose:
        print(f"[loader] X {X.shape}  缺失率 {np.isnan(X).mean():.4%}  → 缓存 {cache}")
    return sample_ids, proteins, X


def align_proteome_to_metadata(
    meta: pd.DataFrame, sample_ids: np.ndarray, X: np.ndarray
) -> np.ndarray:
    """把蛋白矩阵按 metadata 的行序重排，返回重排后的 X。"""
    pos = {s: i for i, s in enumerate(sample_ids)}
    missing = [s for s in meta["sample_ID"].astype(str) if s not in pos]
    if missing:
        raise KeyError(f"{len(missing)} 个 metadata 样本在蛋白矩阵中缺失，例：{missing[:5]}")
    order = np.asarray([pos[s] for s in meta["sample_ID"].astype(str)], dtype=np.int64)
    return X[order]


if __name__ == "__main__":
    meta = load_metadata()
    sid, prot, X = load_proteome_log2()
    Xa = align_proteome_to_metadata(meta, sid, X)
    print(f"metadata {meta.shape}  proteome {Xa.shape}")
    print(f"全缺失蛋白列数 {int(np.isnan(Xa).all(axis=0).sum())}")
    fin = Xa[np.isfinite(Xa)]
    print(
        "log2 分布  min %.2f  p1 %.2f  median %.2f  p99 %.2f  max %.2f"
        % (fin.min(), *np.percentile(fin, [1, 50, 99]), fin.max())
    )
