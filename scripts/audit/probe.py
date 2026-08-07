"""【已封存】最初的列结构探查。

清单里含 proteome_raw_test.csv，与 CLAUDE.md R2（分支 A 隔离）冲突，默认拒绝运行。
train_val 侧的等价探查见 scripts/data/loader.py 的 __main__ 段。
"""
import os
import sys

import pandas as pd

if os.environ.get("GOAI_ALLOW_TEST_LABELS") != "1":
    sys.exit("【已封存】本脚本清单含 proteome_raw_test.csv，违反 CLAUDE.md R2。\n"
             "结论已记入 docs/02_数据实况与阅读清单.md，无需重跑。")

D = r"E:\TMP\claude\E--Claude-Code-X-DIGEST\4ddbb19b-c849-4ded-811b-45de0270be24\scratchpad\goai_track3\extracted\input"

META = ["WAYB_WAYC_metadata_train_val(1).csv", "WAYB_WAYC_metadata_test(1).csv"]
PROT = ["WAYB_WAYC_proteome_raw_train_val.csv", "WAYB_WAYC_proteome_raw_test.csv"]

for f in META:
    p = os.path.join(D, f)
    df = pd.read_csv(p)
    print("=" * 78)
    print(f, "shape =", df.shape)
    print("-" * 78)
    for c in df.columns:
        s = df[c]
        nu = s.nunique(dropna=True)
        print("  %-28s nunique=%-6d dtype=%-10s nulls=%d" % (c, nu, str(s.dtype), s.isna().sum()))
        if nu <= 30:
            vals = sorted(map(str, s.dropna().unique()))
            print("      values: %s" % vals)
    print()
    print("head:")
    print(df.head(3).to_string())
    print()

for f in PROT:
    p = os.path.join(D, f)
    print("=" * 78)
    hdr = pd.read_csv(p, nrows=3)
    with open(p, "r", encoding="utf-8", errors="ignore") as fh:
        nrows = sum(1 for _ in fh) - 1
    print(f, "rows =", nrows, " cols =", hdr.shape[1])
    print("  first 10 cols:", list(hdr.columns[:10]))
    print("  last 5 cols :", list(hdr.columns[-5:]))
    print("  sample block:")
    print(hdr.iloc[:3, :6].to_string())
    print()
