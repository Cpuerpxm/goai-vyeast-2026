"""按主办方《虚拟细胞方向材料提交说明》第 11 页的规则重建 feature contract，
并把已有的全维 prediction.csv 裁到该契约的列集合。

规则原文（说明第 11 页）：
    仅使用 split_final == train 的样本计算缺失率，删除缺失率达到或超过 80% 的蛋白；
    当前标准建模空间为 4,422 个蛋白，名称和顺序以主办方 feature contract 为准。

主办方未随赛题下发 feature contract 文件，因此本脚本按上述规则自行重建，
并把重建结果与说明中的 4,422 这一数字对账；对不上即报错退出，不静默通过。

列顺序：沿用官方 proteome 原始 CSV 的列顺序（去掉被删的蛋白后保持相对次序），
这是在没有拿到官方 contract 文件时唯一有据可依的顺序约定。

用法：
    python scripts/submission/build_feature_contract.py            # 重建契约 + 裁列
    python scripts/submission/build_feature_contract.py --contract-only
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_META = os.path.join(ROOT, "data", "raw", "metadata_train_val.csv")
RAW_PROT = os.path.join(ROOT, "data", "raw", "proteome_raw_train_val.csv")
TEST_META = os.path.join(ROOT, "data", "raw", "metadata_test.csv")
PRED_IN = os.path.join(ROOT, "results", "step10_submission", "prediction.csv")
OUT_DIR = os.path.join(ROOT, "results", "step11_contract")

MISSING_THRESHOLD = 0.80        # 说明：缺失率 >= 80% 的蛋白删除
EXPECTED_KEPT = 4422            # 说明第 11 页与第 14 页给出的标准建模空间
EXPECTED_ROWS = 4454            # 说明第 14 页：prediction.csv 行数


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def build_contract() -> dict:
    meta = pd.read_csv(RAW_META, usecols=["sample_ID", "split_final"])
    train_ids = set(meta.loc[meta.split_final == "train", "sample_ID"])
    if not train_ids:
        sys.exit("FAIL: metadata 中没有 split_final == train 的样本")

    miss = None
    cols = None
    n_rows = 0
    for chunk in pd.read_csv(RAW_PROT, chunksize=500):
        sub = chunk[chunk["sample_ID"].isin(train_ids)]
        if sub.empty:
            continue
        vals = sub.drop(columns=["sample_ID"])
        if cols is None:
            cols = list(vals.columns)
            miss = np.zeros(len(cols), dtype=np.int64)
        miss += vals.isna().sum().to_numpy()
        n_rows += len(sub)

    if n_rows != len(train_ids):
        sys.exit(f"FAIL: 统计到 {n_rows} 行 train，metadata 声明 {len(train_ids)} 行")

    rate = miss / n_rows
    keep_mask = rate < MISSING_THRESHOLD
    kept = [c for c, k in zip(cols, keep_mask) if k]
    dropped = [c for c, k in zip(cols, keep_mask) if not k]

    if len(kept) != EXPECTED_KEPT:
        sys.exit(
            f"FAIL: 按规则算出 {len(kept)} 个蛋白，说明文档要求 {EXPECTED_KEPT} 个。"
            " 口径不一致，停止，不要用这份契约裁列。"
        )

    return {
        "_rule": "split_final==train 上缺失率 >= 0.80 的蛋白删除（方向说明第 11 页）",
        "_source": "本地按规则重建；主办方未随赛题下发 contract 文件",
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_train_rows": n_rows,
        "n_proteins_raw": len(cols),
        "n_proteins_kept": len(kept),
        "n_proteins_dropped": len(dropped),
        "missing_threshold": MISSING_THRESHOLD,
        "proteins": kept,
        "dropped_proteins": dropped,
    }


def trim_prediction(contract: dict) -> dict:
    keep = contract["proteins"]
    keep_set = set(keep)
    out_path = os.path.join(OUT_DIR, "prediction.csv")

    with open(PRED_IN, newline="", encoding="utf-8") as fin:
        reader = csv.reader(fin)
        header = next(reader)
        if header[0] != "sample_ID":
            sys.exit(f"FAIL: 输入首列是 {header[0]}，应为 sample_ID")
        src_prot = header[1:]
        missing = keep_set - set(src_prot)
        if missing:
            sys.exit(f"FAIL: 契约里有 {len(missing)} 个蛋白不在输入预测中，例如 {sorted(missing)[:3]}")

        idx = [0] + [src_prot.index(p) + 1 for p in keep]

        os.makedirs(OUT_DIR, exist_ok=True)
        n_out = 0
        sample_ids = []
        bad_cells = 0
        with open(out_path, "w", newline="", encoding="utf-8") as fout:
            writer = csv.writer(fout, lineterminator="\n")
            writer.writerow(["sample_ID"] + keep)
            for row in reader:
                sample_ids.append(row[0])
                vals = [row[i] for i in idx]
                for v in vals[1:]:
                    if v == "" or v.lower() in ("nan", "inf", "-inf"):
                        bad_cells += 1
                writer.writerow(vals)
                n_out += 1

    if n_out != EXPECTED_ROWS:
        sys.exit(f"FAIL: 输出 {n_out} 行，说明要求 {EXPECTED_ROWS} 行")
    if bad_cells:
        sys.exit(f"FAIL: 输出中有 {bad_cells} 个 NA/Inf 单元格，说明要求全部有限")

    test_meta = pd.read_csv(TEST_META, usecols=["sample_ID"])
    official = list(test_meta["sample_ID"])
    order_ok = official == sample_ids
    set_ok = set(official) == set(sample_ids)

    return {
        "output": out_path,
        "rows": n_out,
        "protein_cols": len(keep),
        "sha256": sha256_file(out_path),
        "size_bytes": os.path.getsize(out_path),
        "sample_id_set_matches_official": set_ok,
        "sample_id_order_matches_official": order_ok,
        "na_or_inf_cells": bad_cells,
        "prediction_scale": "log2",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contract-only", action="store_true")
    args = ap.parse_args()

    print("[1/2] 按方向说明第 11 页规则重建 feature contract ...")
    contract = build_contract()
    os.makedirs(OUT_DIR, exist_ok=True)
    cpath = os.path.join(OUT_DIR, "feature_contract.json")
    with open(cpath, "w", encoding="utf-8", newline="\n") as f:
        json.dump(contract, f, ensure_ascii=False, indent=1)
    print(f"      原始 {contract['n_proteins_raw']} 个 → 保留 {contract['n_proteins_kept']} 个"
          f"，删除 {contract['n_proteins_dropped']} 个")
    print(f"      与说明文档的 {EXPECTED_KEPT} 对账通过")
    print(f"      写出 {cpath}")

    if args.contract_only:
        return

    print("[2/2] 按契约裁列 prediction.csv ...")
    res = trim_prediction(contract)
    mpath = os.path.join(OUT_DIR, "prediction_manifest.json")
    with open(mpath, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"contract": cpath, **res}, f, ensure_ascii=False, indent=1)

    print(f"      行数 {res['rows']}  蛋白列 {res['protein_cols']}")
    print(f"      sample_ID 集合与官方一致: {res['sample_id_set_matches_official']}")
    print(f"      sample_ID 顺序与官方一致: {res['sample_id_order_matches_official']}")
    print(f"      NA/Inf 单元格: {res['na_or_inf_cells']}")
    print(f"      SHA256 {res['sha256']}")
    print(f"      大小 {res['size_bytes']/1048576:.1f} MB")
    print(f"      写出 {res['output']}")
    print(f"      写出 {mpath}")

    if not (res["sample_id_set_matches_official"] and res["sample_id_order_matches_official"]):
        sys.exit("FAIL: sample_ID 与官方测试 metadata 不一致，说明第 14 页要求集合与顺序都一致")
    print("\n[OK] 全部检查通过")


if __name__ == "__main__":
    main()
