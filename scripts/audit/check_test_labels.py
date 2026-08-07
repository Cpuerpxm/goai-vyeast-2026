"""【已封存】发现 test 集含完整蛋白丰度的那次探查。

这个脚本是分支 A（测试标签隔离）决定的证据来源，保留作记录。
但它本身会读 proteome_raw_test.csv，与 CLAUDE.md R2 冲突，因此默认拒绝运行。
若组委会书面确认标签可用，删掉下面的 guard 并同步更新 R2。
"""
import os
import sys

import numpy as np
import pandas as pd

if os.environ.get("GOAI_ALLOW_TEST_LABELS") != "1":
    sys.exit("【已封存】本脚本会读 proteome_raw_test.csv，违反 CLAUDE.md R2（分支 A 隔离）。\n"
             "它的结论已记入 docs/02 与 _handoff/CURRENT.md，无需重跑。")

D = r"E:\TMP\claude\E--Claude-Code-X-DIGEST\4ddbb19b-c849-4ded-811b-45de0270be24\scratchpad\goai_track3\extracted\input"

for tag, fn, mfn in [
    ("TRAIN_VAL", "WAYB_WAYC_proteome_raw_train_val.csv", "WAYB_WAYC_metadata_train_val(1).csv"),
    ("TEST", "WAYB_WAYC_proteome_raw_test.csv", "WAYB_WAYC_metadata_test(1).csv"),
]:
    p = os.path.join(D, fn)
    df = pd.read_csv(p)
    meta = pd.read_csv(os.path.join(D, mfn))
    vals = df.drop(columns=["sample_ID"])
    total = vals.shape[0] * vals.shape[1]
    nan = int(vals.isna().sum().sum())
    print("=" * 78)
    print("%s  shape=%s" % (tag, vals.shape))
    print("  总单元格 %d, NaN %d, 缺失率 %.2f%%" % (total, nan, 100.0 * nan / total))
    per_row = vals.notna().sum(axis=1)
    print("  每行非空蛋白数: min=%d  median=%d  max=%d  mean=%.0f"
          % (per_row.min(), int(per_row.median()), per_row.max(), per_row.mean()))
    n_empty_rows = int((per_row == 0).sum())
    print("  完全空行(全NaN)数量: %d" % n_empty_rows)
    per_col = vals.notna().sum(axis=0)
    print("  完全空列(该蛋白全缺)数量: %d" % int((per_col == 0).sum()))
    print("  每列非空样本数: min=%d  median=%d  max=%d"
          % (per_col.min(), int(per_col.median()), per_col.max()))

    # 按 split 看缺失
    m = meta[["sample_ID", "split_final"]].merge(
        pd.DataFrame({"sample_ID": df["sample_ID"], "n_obs": per_row.values}),
        on="sample_ID", how="left")
    print("  按 split_final 的每行非空蛋白数(均值):")
    for k, v in m.groupby("split_final")["n_obs"].agg(["mean", "min", "max", "count"]).iterrows():
        print("    %-20s mean=%8.0f min=%6d max=%6d n=%d" % (k, v["mean"], v["min"], v["max"], v["count"]))
    print()
