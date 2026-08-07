"""对照匹配管线的单元测试（合成数据，不读真实数据）。

运行：python test_control_match.py     期望 PASS 全通过，exit 0
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import paths
from data.control_match import (
    MATCH_KEYS, QC_LABEL, MatchConfig, compute_delta, control_profiles, group_ids,
)

_OK, _FAIL = [], []


def check(name: str, cond: bool, extra: str = "") -> None:
    (_OK if cond else _FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {extra}" if extra else ""))


def _meta(rows) -> pd.DataFrame:
    """rows: list of dict，缺省字段用默认值补齐。"""
    base = dict(data_source="WAYB", Strains="STRAIN_ALPHA", Medium="glu", Temperature=30,
                pert_time=60, instrument="QE1", Yeast_cell_plate="P1",
                protein_well="A1", split_final="train")
    out = []
    for i, r in enumerate(rows):
        d = dict(base)
        d.update(r)
        d.setdefault("sample_ID", f"S{i}")
        out.append(d)
    return pd.DataFrame(out)


# --------------------------------------------------------------------------

print("\n[1] 匹配键 = 手册 7 项，不含 protein_well")
cfg = MatchConfig()
check("匹配键恰为手册 7 项", cfg.match_keys == MATCH_KEYS and len(MATCH_KEYS) == 7,
      str(MATCH_KEYS))
check("protein_well 不在匹配键内", "protein_well" not in cfg.match_keys)

m = _meta([
    {"perturbation_no_concentration": "Water", "protein_well": "A1"},
    {"perturbation_no_concentration": "DrugX", "protein_well": "H9"},   # well 不同
])
g = group_ids(m, cfg)
check("well 不同仍同组", g[0] == g[1], f"gid={g.tolist()}")

m = _meta([
    {"perturbation_no_concentration": "Water", "Yeast_cell_plate": "P1"},
    {"perturbation_no_concentration": "DrugX", "Yeast_cell_plate": "P2"},  # 板不同
])
g = group_ids(m, cfg)
check("板号不同则不同组", g[0] != g[1], f"gid={g.tolist()}")


# --------------------------------------------------------------------------

print("\n[2] 多对照逐蛋白 median 聚合")
m = _meta([
    {"perturbation_no_concentration": "Water"},
    {"perturbation_no_concentration": "Water"},
    {"perturbation_no_concentration": "DMSO"},
    {"perturbation_no_concentration": "DrugX"},
])
X = np.array([
    [1.0, 10.0, 100.0],
    [3.0, 20.0, 200.0],
    [8.0, 60.0, 900.0],
    [5.0, 30.0, 300.0],
], dtype=np.float32)
C, n_ctrl, is_ctrl = control_profiles(X, m, cfg)
check("处理样本对照数 = 3", n_ctrl[3] == 3, f"n_ctrl={n_ctrl.tolist()}")
check("逐蛋白 median = [3,20,200]", np.allclose(C[3], [3.0, 20.0, 200.0]), str(C[3]))
D = compute_delta(X, C)
check("Δ = 处理 − median", np.allclose(D[3], [2.0, 10.0, 100.0]), str(D[3]))

cfg_mean = MatchConfig(agg="mean")
Cm, _, _ = control_profiles(X, m, cfg_mean)
check("mean 可切换且结果不同", (not np.allclose(Cm[3], C[3])) and np.allclose(Cm[3], [4.0, 30.0, 400.0]),
      str(Cm[3]))


# --------------------------------------------------------------------------

print("\n[3] 缺失传播：不填补")
m = _meta([
    {"perturbation_no_concentration": "Water"},
    {"perturbation_no_concentration": "DMSO"},
    {"perturbation_no_concentration": "DrugX"},
])
X = np.array([
    [1.0, np.nan, np.nan],
    [3.0, 20.0, np.nan],
    [5.0, 30.0, 40.0],
], dtype=np.float32)
C, n_ctrl, _ = control_profiles(X, m, cfg)
check("对照部分缺失 → 用剩余对照", np.isclose(C[2, 1], 20.0), f"C[2,1]={C[2,1]}")
check("对照全缺失 → 对照谱 NaN", np.isnan(C[2, 2]), f"C[2,2]={C[2,2]}")
D = compute_delta(X, C)
check("对照全缺 → Δ 为 NaN（不填补）", np.isnan(D[2, 2]))
check("对照有值 → Δ 正常", np.isclose(D[2, 0], 3.0) and np.isclose(D[2, 1], 10.0),
      str(D[2]))

X2 = X.copy()
X2[2, 0] = np.nan          # 处理侧缺失
D2 = compute_delta(X2, C)
check("处理侧缺失 → Δ 为 NaN", np.isnan(D2[2, 0]))


# --------------------------------------------------------------------------

print("\n[4] 0 匹配的处理样本")
m = _meta([
    {"perturbation_no_concentration": "Water", "Yeast_cell_plate": "P1"},
    {"perturbation_no_concentration": "DrugX", "Yeast_cell_plate": "P9"},
])
X = np.array([[1.0, 2.0], [5.0, 6.0]], dtype=np.float32)
C, n_ctrl, _ = control_profiles(X, m, cfg)
check("无同组对照 → n_ctrl=0", n_ctrl[1] == 0)
check("无同组对照 → 整行 NaN", np.isnan(C[1]).all())
check("无同组对照 → Δ 整行 NaN", np.isnan(compute_delta(X, C)[1]).all())


# --------------------------------------------------------------------------

print("\n[5] QC 不作生物学对照")
m = _meta([
    {"perturbation_no_concentration": QC_LABEL},
    {"perturbation_no_concentration": "DrugX"},
])
X = np.array([[1.0, 2.0], [5.0, 6.0]], dtype=np.float32)
C, n_ctrl, is_ctrl = control_profiles(X, m, cfg)
check("QC 不计入对照集合", n_ctrl[1] == 0 and not is_ctrl[0])
check("默认对照集合 = Water+DMSO", cfg.control_names == ["Water", "DMSO"])

cfg_qc = MatchConfig(control_names=["Water", "DMSO", QC_LABEL])
_, n_qc, _ = control_profiles(X, m, cfg_qc)
check("显式加入后 QC 才被当对照", n_qc[1] == 1)


# --------------------------------------------------------------------------

print("\n[6] 对照样本走留一法（噪声地板）")
m = _meta([
    {"perturbation_no_concentration": "Water"},
    {"perturbation_no_concentration": "Water"},
    {"perturbation_no_concentration": "DMSO"},
])
X = np.array([[1.0], [3.0], [11.0]], dtype=np.float32)
C, n_ctrl, is_ctrl = control_profiles(X, m, cfg)
check("对照自身被排除（n_ctrl = 组内对照数−1）", (n_ctrl[:3] == 2).all(), f"n_ctrl={n_ctrl.tolist()}")
D = compute_delta(X, C)
check("对照 Δ 非恒 0（否则是自指）", not np.isclose(D[0, 0], 0.0), f"Δ[0]={D[0,0]}")
check("留一 median 正确：样本0 → median(3,11)=7", np.isclose(C[0, 0], 7.0), f"C[0,0]={C[0,0]}")

cfg_self = MatchConfig(exclude_self_for_controls=False)
Cs, ns, _ = control_profiles(X, m, cfg_self)
check("关闭留一后含自身（n_ctrl=3）", (ns[:3] == 3).all(), f"n_ctrl={ns.tolist()}")

m1 = _meta([{"perturbation_no_concentration": "Water"},
            {"perturbation_no_concentration": "DrugX"}])
X1 = np.array([[1.0], [5.0]], dtype=np.float32)
C1, n1, _ = control_profiles(X1, m1, cfg)
check("组内只有 1 个对照 → 该对照留一后 n_ctrl=0 且 NaN",
      n1[0] == 0 and np.isnan(C1[0, 0]))
check("同组处理样本仍拿到该对照", n1[1] == 1 and np.isclose(C1[1, 0], 1.0))


# --------------------------------------------------------------------------

print("\n[7] 测试标签物理隔离守卫（CLAUDE.md R2）")
try:
    paths.assert_readable("data/raw/proteome_raw_test.csv")
    check("读 proteome_raw_test.csv 被拦截", False, "未抛异常")
except paths.TestLabelLeakError:
    check("读 proteome_raw_test.csv 被拦截", True)
check("paths 模块不暴露测试蛋白路径常量", not hasattr(paths, "PROT_TEST"))
check("metadata_test.csv 可读（测试侧只读元数据）",
      paths.assert_readable(paths.META_TEST).endswith("metadata_test.csv"))


# --------------------------------------------------------------------------

print("\n" + "=" * 60)
print(f"PASS {len(_OK)} / {len(_OK) + len(_FAIL)}")
if _FAIL:
    print("FAILED: " + ", ".join(_FAIL))
sys.exit(1 if _FAIL else 0)
