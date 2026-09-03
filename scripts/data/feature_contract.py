"""4,422 蛋白建模空间的唯一入口。

来源：《虚拟细胞方向材料提交说明》第 11 页与第 14 页。
    · 仅使用 split_final == train 的样本计算缺失率，删除缺失率 >= 80% 的蛋白
    · 当前标准建模空间为 4,422 个蛋白
    · prediction.csv 为 4,454 行 ×（sample_ID + 4,422 蛋白列）

主办方未随赛题下发 contract 文件，`configs/feature_contract.json` 由
`scripts/submission/build_feature_contract.py` 按上述规则本地重建，
且与文档给出的 4,422 对账通过。

设计约束（GPT Pro 会诊 2026-09-02 L1-00）：
    · 不允许各脚本各自读 JSON 各自切片；一切经由本模块
    · 切片必须发生在计算 PCA / C / μ_ctx / μ_drug **之前**
    · 切片按蛋白名精确匹配，不依赖列序假设

关掉契约（回到 5,243 全维）只用于诊断对照：设环境变量 VYEAST_NO_CONTRACT=1。
正式候选一律在契约空间内训练与评估。
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import List, Sequence, Tuple

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PATH = os.path.join(_ROOT, "configs", "feature_contract.json")

EXPECTED_N = 4422
ENV_DISABLE = "VYEAST_NO_CONTRACT"


def enabled() -> bool:
    """契约是否生效。关掉只允许用于诊断对照。"""
    return os.environ.get(ENV_DISABLE, "") not in ("1", "true", "True")


@lru_cache(maxsize=1)
def load() -> dict:
    if not os.path.exists(_PATH):
        raise FileNotFoundError(
            f"找不到 {_PATH}；先跑 scripts/submission/build_feature_contract.py --contract-only"
        )
    with open(_PATH, encoding="utf-8") as f:
        c = json.load(f)
    prots = c["proteins"]
    if len(prots) != EXPECTED_N:
        raise ValueError(f"契约里有 {len(prots)} 个蛋白，应为 {EXPECTED_N}")
    if len(set(prots)) != EXPECTED_N:
        raise ValueError("契约蛋白名有重复")
    return c


@lru_cache(maxsize=4)
def _index_for(proteins_key: Tuple[str, ...]) -> np.ndarray:
    """按蛋白名精确匹配，返回契约顺序对应的列下标。"""
    contract = load()["proteins"]
    pos = {p: i for i, p in enumerate(proteins_key)}
    missing = [p for p in contract if p not in pos]
    if missing:
        raise KeyError(
            f"契约里有 {len(missing)} 个蛋白不在输入列中，例如 {missing[:3]}"
        )
    idx = np.asarray([pos[p] for p in contract], dtype=np.int64)
    if idx.size != EXPECTED_N or len(set(idx.tolist())) != EXPECTED_N:
        raise ValueError("索引长度或唯一性检查失败")
    return idx


def index(proteins: Sequence[str]) -> np.ndarray:
    return _index_for(tuple(str(p) for p in proteins))


def apply(proteins: np.ndarray, *arrays: np.ndarray, verbose: bool = False):
    """把蛋白名数组与若干 (n_samples, n_proteins) 数组一起切到契约空间。

    返回 (proteins_kept, arr1_kept, arr2_kept, ...)。
    契约关闭时原样返回，只打一行提示。
    """
    if not enabled():
        if verbose:
            print(f"[contract] {ENV_DISABLE} 已设，跳过切片，保持 {len(proteins)} 维（仅供诊断）")
        return (proteins, *arrays)

    idx = index(proteins)
    kept = np.asarray([proteins[i] for i in idx], dtype=proteins.dtype)

    out = []
    for a in arrays:
        if a is None:
            out.append(None)
            continue
        if a.ndim != 2 or a.shape[1] != len(proteins):
            raise ValueError(
                f"数组形状 {a.shape} 与蛋白数 {len(proteins)} 不匹配，无法按列切片"
            )
        out.append(a[:, idx])

    contract = load()
    if list(kept) != contract["proteins"]:
        raise ValueError("切片后蛋白顺序与契约不一致")
    if verbose:
        print(f"[contract] {len(proteins)} → {len(kept)} 维"
              f"（删除 {len(proteins) - len(kept)} 个高缺失蛋白）")
    return (kept, *out)


def n_proteins() -> int:
    """当前生效的蛋白维度。契约关闭时返回 None，交由调用方按实际列数处理。"""
    return EXPECTED_N if enabled() else None


def assert_prediction_shape(n_rows: int, n_protein_cols: int) -> None:
    """提交前的形状硬校验（说明第 14 页）。"""
    if n_rows != 4454:
        raise ValueError(f"prediction 行数 {n_rows} != 4454")
    want = n_proteins()
    if want is not None and n_protein_cols != want:
        raise ValueError(f"prediction 蛋白列数 {n_protein_cols} != {want}")
