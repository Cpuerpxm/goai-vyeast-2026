"""菌株基因组坐标：1011 酵母基因组计划的 SNP 距离矩阵。

来源与版本见 `data/external/yeast1011/SOURCE.md`（手册修订对照表表一第 1 条要求
「外部公开资源可用于实体特征构建，须披露来源与版本」）。

本模块只做三件事：读矩阵、按角色取子矩阵、把 provenance 一并带出来。
任何建模都在 `models/strain_transport.py` 里，不混在这。

**菌株代号不写进源码。** 它们是非公开数据集的一部分，公开仓库一律走
`data/desensitize.py` 换成占位符；代号一律**运行时从 metadata 读**，
所以这个文件在公开与不公开两种场合是同一份。

已核实的事实（2026-08-24 实测，运行 `python strain_genome.py` 可复现）：

- 训练菌株共 4 株，其中 **3 株**能在 1011 面板里按代号全词命中，**1 株命不中**
  ——那株是实验室菌株命名，不属野生分离株面板。面板里也找不到任何可辨认的
  S288C / BY4741 / 参考株条目（名字里不含 288 / S28 / REF / BY47 的任何一个），
  所以「把它当成某个参考株的近亲」没有证据支撑，不走。因此**核函数的供体池只有 3 株**。
- 整株落在 val 的那一株、以及只出现在 test 的那一株（占 test 50.1%），都能命中面板。

⚠ 一个必须写进材料的不对称：**未见菌株**与某一株训练菌株的 SNP 距离落在面板全体
距离分布的低分位，属近缘，加权在这一格最可能起作用；而唯一可验证的那一株
（整株在 val）到三个供体近乎等距，对加权本身不敏感。也就是说，
**加权最可能起作用的那一格恰恰无法验证**，能验证的那一格又验不出加权。
不能拿后者的结果替前者背书。

自检：python strain_genome.py
"""
from __future__ import annotations

import gzip
import hashlib
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import loader, paths
from data import split_guard as sg

MATRIX = os.path.join(paths.DATA_EXTERNAL, "yeast1011",
                      "1011DistanceMatrixBasedOnSNPs.tab.gz")
SOURCE_MD = os.path.join(paths.DATA_EXTERNAL, "yeast1011", "SOURCE.md")

#: SOURCE.md 里登记的 SHA-256。读矩阵时核对，文件被换掉要立刻发现。
EXPECTED_SHA256 = "140da4e5193584c01e60c554a2ba5075a542d925be540afe7c7a92b7377af928"

STRAIN_COL = "Strains"


class GenomeDataError(RuntimeError):
    pass


def file_sha256(path: str = MATRIX) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def strain_roles() -> dict:
    """运行时从 metadata 推出每株菌的角色，不在源码里写代号。

    返回 {"train": [...], "val_only": [...], "test_only": [...], "all": [...]}
    """
    tv = loader.load_metadata("train_val")
    te = loader.load_metadata("test")
    tv_s = set(tv[STRAIN_COL].astype(str))
    te_s = set(te[STRAIN_COL].astype(str))
    tr_s = set(tv.loc[sg.train_rows(tv), STRAIN_COL].astype(str))
    return {
        "train": sorted(tr_s),
        "val_only": sorted(tv_s - tr_s),
        "test_only": sorted(te_s - tv_s),
        "all": sorted(tv_s | te_s),
    }


def load_matrix(path: str = MATRIX, verify: bool = True) -> pd.DataFrame:
    """读整张 1011 x 1011 SNP 距离矩阵。

    文件格式（实测）：首行 1012 个字段，第 0 个是行名列的表头 `STD`，
    其余 1011 个是菌株名；此后每行 1 个菌株名 + 1011 个距离。
    """
    if not os.path.exists(path):
        raise GenomeDataError(f"缺 {path}，见 data/external/yeast1011/SOURCE.md")
    if verify:
        got = file_sha256(path)
        if got != EXPECTED_SHA256:
            raise GenomeDataError(
                f"SNP 距离矩阵的 SHA-256 与 SOURCE.md 登记的不一致：\n"
                f"  登记 {EXPECTED_SHA256}\n  实际 {got}\n"
                "文件被替换过，先核对来源再继续。")
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        names = header[1:]
        rows, index = [], []
        for line in fh:
            f = line.rstrip("\n").split("\t")
            index.append(f[0])
            rows.append(np.asarray(f[1:], dtype=np.float64))
    D = np.vstack(rows)
    if D.shape != (len(names), len(names)):
        raise GenomeDataError(f"矩阵形状 {D.shape} 与表头 {len(names)} 列对不上")
    if index != names:
        raise GenomeDataError("行名与列名顺序不一致，不能直接当对称矩阵用")
    if not np.allclose(np.diag(D), 0.0, atol=1e-9):
        raise GenomeDataError("对角线不为 0，这不是距离矩阵")
    if not np.allclose(D, D.T, atol=1e-9):
        raise GenomeDataError("矩阵不对称")
    return pd.DataFrame(D, index=names, columns=names)


def contest_submatrix(strains: List[str] | None = None,
                      path: str = MATRIX) -> tuple[pd.DataFrame, dict]:
    """取赛题菌株的子矩阵，并返回命中/未命中情况与面板分位数。"""
    roles = strain_roles()
    strains = list(strains or roles["all"])
    D = load_matrix(path)
    hit = [s for s in strains if s in D.index]
    miss = [s for s in strains if s not in D.index]
    sub = D.loc[hit, hit]
    flat = D.values[np.triu_indices(len(D), k=1)]
    info = {
        "matrix_file": os.path.relpath(path, paths.PROJECT_ROOT).replace("\\", "/"),
        "matrix_sha256": file_sha256(path),
        "panel_size": int(len(D)),
        "roles": roles,
        "hit": hit,
        "miss": miss,
        "n_train_with_coordinates": len([s for s in roles["train"] if s in D.index]),
        "panel_distance_percentiles": {
            f"p{q}": float(np.percentile(flat, q)) for q in (1, 5, 25, 50, 75, 95, 99)
        },
        "source_doc": os.path.relpath(SOURCE_MD, paths.PROJECT_ROOT).replace("\\", "/"),
        "citation": "Peter et al., Nature 2018, 556, 339-344（1,011 株酿酒酵母基因组）",
    }
    return sub, info


def panel_quantile(distance: float, path: str = MATRIX) -> float:
    """给定距离在面板全体两两距离中的分位（0-1），用来说明「有多近」。"""
    D = load_matrix(path)
    flat = D.values[np.triu_indices(len(D), k=1)]
    return float((flat < distance).mean())


def kernel_weights(target: str, donors: List[str], D: pd.DataFrame,
                   bandwidth: float) -> Dict[str, float]:
    """高斯核：w(s*, s) = exp(-(d/h)^2)，归一化到和为 1。

    donors 里若没有任何一株有坐标，或 target 没坐标，返回空字典（调用方据此回退）。
    """
    if bandwidth <= 0:
        raise ValueError("bandwidth 必须为正")
    use = [s for s in donors if s in D.index and target in D.index]
    if not use:
        return {}
    d = np.asarray([D.loc[target, s] for s in use], dtype=np.float64)
    w = np.exp(-(d / bandwidth) ** 2)
    tot = float(w.sum())
    if not np.isfinite(tot) or tot <= 0:
        return {}
    return {s: float(x / tot) for s, x in zip(use, w)}


def uniform_weights(donors: List[str], D: pd.DataFrame | None = None) -> Dict[str, float]:
    """不含任何基因组信息的对照：训练菌株等权。

    这是**必须有的对照**。未见菌株现在被编码成整块 0，预测因此只剩截距；
    而训练菌株拿的是「截距 + 该株系数」。岭回归下这些系数的平均并不为 0，
    所以「把整块 0 换成等权平均」本身就可能提分，与基因组毫无关系。
    不设这个对照，就会把这部分收益错记到 SNP 距离头上。
    """
    if not donors:
        return {}
    return {s: 1.0 / len(donors) for s in donors}


def _selftest() -> None:
    ok = 0
    D = load_matrix()
    assert D.shape == (1011, 1011); ok += 1
    sub, info = contest_submatrix()
    roles = info["roles"]
    assert len(roles["train"]) == 4, roles["train"]; ok += 1
    assert len(roles["val_only"]) == 1 and len(roles["test_only"]) == 1; ok += 1
    assert info["n_train_with_coordinates"] == 3, info["n_train_with_coordinates"]; ok += 1
    assert len(info["miss"]) == 1 and info["miss"][0] in roles["train"]; ok += 1

    donors = [s for s in roles["train"] if s in D.index]
    unseen = roles["test_only"][0]
    val_only = roles["val_only"][0]

    w = kernel_weights(unseen, donors, sub, bandwidth=1.0)
    assert abs(sum(w.values()) - 1.0) < 1e-9; ok += 1
    # 带宽越大越平，越小越尖
    w_wide = kernel_weights(unseen, donors, sub, bandwidth=100.0)
    assert max(w_wide.values()) - min(w_wide.values()) < 0.01, w_wide; ok += 1
    w_narrow = kernel_weights(unseen, donors, sub, bandwidth=0.3)
    assert max(w_narrow.values()) > 0.99, w_narrow; ok += 1
    # 没坐标的菌株拿不到权重
    assert kernel_weights(info["miss"][0], donors, sub, bandwidth=1.0) == {}; ok += 1
    u = uniform_weights(donors)
    assert all(abs(v - 1 / len(donors)) < 1e-12 for v in u.values()); ok += 1

    d_unseen = min(sub.loc[unseen, donors])
    d_val = min(sub.loc[val_only, donors])
    q_unseen, q_val = panel_quantile(d_unseen), panel_quantile(d_val)
    spread_val = float(max(sub.loc[val_only, donors]) - min(sub.loc[val_only, donors]))
    spread_unseen = float(max(sub.loc[unseen, donors]) - min(sub.loc[unseen, donors]))
    assert q_unseen < q_val, "未见菌株应比 val 菌株更贴近某个训练菌株"; ok += 1

    print(f"[strain_genome] selftest {ok} 项全部通过")
    print(f"  面板 {info['panel_size']} 株；训练菌株 {len(roles['train'])} 株，"
          f"其中有坐标 {info['n_train_with_coordinates']} 株")
    print(f"  面板距离分位：p5 {info['panel_distance_percentiles']['p5']:.3f} · "
          f"中位 {info['panel_distance_percentiles']['p50']:.3f} · "
          f"p95 {info['panel_distance_percentiles']['p95']:.3f}")
    print(f"  未见菌株（占 test 50.1%）到最近供体 {d_unseen:.3f}，"
          f"面板分位 {q_unseen:.1%}；三个供体的距离跨度 {spread_unseen:.3f}")
    print(f"  val 菌株到最近供体 {d_val:.3f}，面板分位 {q_val:.1%}；"
          f"跨度 {spread_val:.3f}  ← 跨度小 = 加权在这一格几乎不起作用")
    print("\n  赛题菌株两两 SNP 距离（代号见运行输出，源码不含）：")
    print(sub.round(4).to_string())


if __name__ == "__main__":
    _selftest()
