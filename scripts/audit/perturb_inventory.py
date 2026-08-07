import os
import pandas as pd

D = r"E:\TMP\claude\E--Claude-Code-X-DIGEST\4ddbb19b-c849-4ded-811b-45de0270be24\scratchpad\goai_track3\extracted\input"

tr = pd.read_csv(os.path.join(D, "WAYB_WAYC_metadata_train_val(1).csv"))
te = pd.read_csv(os.path.join(D, "WAYB_WAYC_metadata_test(1).csv"))

COL = "perturbation_no_concentration"

set_tr = set(tr[COL].unique())
set_te = set(te[COL].unique())

print("### 化合物总览")
print("train_val 唯一化合物: %d" % len(set_tr))
print("test      唯一化合物: %d" % len(set_te))
print("仅出现在 test（未见化合物）: %d" % len(set_te - set_tr))
print("仅出现在 train_val: %d" % len(set_tr - set_te))
print("两边都有: %d" % len(set_tr & set_te))
print()

ctr = tr[COL].value_counts()
cte = te[COL].value_counts()
allc = sorted(set_tr | set_te)

rows = []
for c in allc:
    n_tr = int(ctr.get(c, 0))
    n_te = int(cte.get(c, 0))
    if n_tr == 0:
        tag = "TEST-ONLY(未见)"
    elif n_te == 0:
        tag = "TRAIN-ONLY"
    else:
        tag = "both"
    rows.append((c, n_tr, n_te, tag))

print("### 逐化合物明细（按 tag, 名称排序）")
print("%-42s %8s %8s   %s" % ("compound", "n_train", "n_test", "tag"))
print("-" * 78)
for tag in ["TEST-ONLY(未见)", "both", "TRAIN-ONLY"]:
    sub = [r for r in rows if r[3] == tag]
    for c, a, b, t in sorted(sub, key=lambda x: -x[1] - x[2]):
        print("%-42s %8d %8d   %s" % (c[:42], a, b, t))
    print()

print("### 菌株")
for name, df in [("train_val", tr), ("test", te)]:
    vc = df["Strains"].value_counts()
    print(name, dict(vc))
print("未见菌株:", sorted(set(te["Strains"]) - set(tr["Strains"])))
print()

print("### split_final 分布")
print("train_val:", dict(tr["split_final"].value_counts()))
print("test     :", dict(te["split_final"].value_counts()))
print()

print("### 对照样本（Water / DMSO 类）")
for name, df in [("train_val", tr), ("test", te)]:
    mask = df[COL].str.contains("water|dmso|control|untreat", case=False, na=False)
    print(name, "对照样本数 =", int(mask.sum()), " 名称:", sorted(df.loc[mask, COL].unique()))
