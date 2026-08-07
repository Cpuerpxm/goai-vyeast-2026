import os
import numpy as np
import pandas as pd

D = r"E:\TMP\claude\E--Claude-Code-X-DIGEST\4ddbb19b-c849-4ded-811b-45de0270be24\scratchpad\goai_track3\extracted\input"
COL = "perturbation_no_concentration"
# 手册: 对照须匹配 来源/菌株/培养基/温度/时间/仪器/板号  (不含 well)
KEY = ["data_source", "Strains", "Medium", "Temperature", "pert_time",
       "instrument", "Yeast_cell_plate"]
CTRL_NAMES = ["Water", "DMSO", "Quality Control"]

meta = pd.read_csv(os.path.join(D, "WAYB_WAYC_metadata_train_val(1).csv"))

print("=" * 80)
print("A. 对照 exact-match 覆盖率审计（train_val，匹配键 = 手册规定的 7 项）")
print("=" * 80)

treat = meta[~meta[COL].isin(CTRL_NAMES)].copy()
print("处理样本数（排除 Water/DMSO/QC）: %d" % len(treat))
print()

for ctrl_set, label in [(["Water"], "仅 Water"),
                        (["DMSO"], "仅 DMSO"),
                        (["Water", "DMSO"], "Water+DMSO"),
                        (["Water", "DMSO", "Quality Control"], "Water+DMSO+QC")]:
    ctrl = meta[meta[COL].isin(ctrl_set)]
    cnt = ctrl.groupby(KEY).size().rename("n_ctrl").reset_index()
    j = treat.merge(cnt, on=KEY, how="left")
    n = j["n_ctrl"].fillna(0).astype(int)
    z = int((n == 0).sum()); one = int((n == 1).sum()); many = int((n > 1).sum())
    print("对照集合 = %-22s  0匹配 %5d (%5.1f%%)   1匹配 %5d (%5.1f%%)   多匹配 %5d (%5.1f%%)"
          % (label, z, 100*z/len(treat), one, 100*one/len(treat), many, 100*many/len(treat)))
    if many:
        print("      多匹配时每样本的对照数: 中位 %.0f  最大 %d" % (n[n > 1].median(), n.max()))
print()

# 按 split 看 Water+DMSO 的覆盖
ctrl = meta[meta[COL].isin(["Water", "DMSO"])]
cnt = ctrl.groupby(KEY).size().rename("n_ctrl").reset_index()
j = treat.merge(cnt, on=KEY, how="left")
j["n_ctrl"] = j["n_ctrl"].fillna(0).astype(int)
print("按 split_final（对照集合 = Water+DMSO）:")
for k, g in j.groupby("split_final"):
    z = int((g["n_ctrl"] == 0).sum())
    print("  %-18s n=%5d   0匹配 %5d (%5.1f%%)" % (k, len(g), z, 100*z/len(g)))
print()

print("=" * 80)
print("B. 化合物响应矩阵的秩上限验证（Pro 的 Q3.4 论证）")
print("=" * 80)
n_comp_train = meta[~meta[COL].isin(CTRL_NAMES)][COL].nunique()
print("训练集非对照化合物数            : %d" % n_comp_train)
print("蛋白维度                        : 5243")
print("化合物×蛋白 响应矩阵中心化后秩上限: %d" % (n_comp_train - 1))
print("→ 5243 维输出中，最多只有 %d 个线性独立的化合物响应方向" % (n_comp_train - 1))
