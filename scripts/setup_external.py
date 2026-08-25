"""外部资源的一次性准备：下载、校验、确定性再生。

为什么需要它（2026-08-25 · GPT Pro R6 · L1-03）：

此前 README 声称「把组委会的四个文件放进 `data/raw/`，然后一条命令 `run_all.py`」。
实测不成立——干净 clone 里**连 `data/` 目录都没有**，而 `run_all.py` 会跑到
需要化合物结构（`compound_smiles.csv`）与菌株基因组（1011 SNP 距离矩阵）的步骤，
两者都不在仓库里。评审照着文档做，必然在 step6b / step9 / step11 失败。

所以把外部资源的准备**显式化成一个入口**，并把「主流程离线」这句话的前提说清楚：
**先跑一次本脚本联网取回资源，之后 `run_all.py` 全程离线。**

三类资源，处理方式各不相同：

| 资源 | 来源 | 本脚本怎么做 |
|---|---|---|
| 1011 SNP 距离矩阵 | 1002genomes 公开地址 | 下载 + 核对登记的 SHA-256 |
| PubChem 名称/结构转储 | NCBI FTP 公开地址 | 下载（2.4 GB）+ 记录 SHA-256 |
| 化合物名 → SMILES | 由上两者本地解析得到 | 调 `resolve_smiles.py` 重建 |

⚠ 有 4 个化合物的名称在 PubChem 同义词表里查不到（数据集用了缩写或同系物混合物名），
需要一份 4 条的**别名映射**才能确定性重建全部 54 条。该文件含赛事化合物名，
按参赛协议不进公开仓库，**随提交物一并交给组委会**。
组委会把它放到 `data/external/compound_aliases.json` 即可完整重建；
没有它时本脚本会明确报出还差哪几条，而不是静默产出一个少 4 行的表。

用法：
    python scripts/setup_external.py            # 全部准备
    python scripts/setup_external.py --check    # 只检查现状，不下载
    python scripts/setup_external.py --skip-pubchem   # 已有 compound_smiles.csv 时
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import paths

PROVENANCE = os.path.join(paths.DATA_EXTERNAL, "PROVENANCE.json")

SNP_DIR = os.path.join(paths.DATA_EXTERNAL, "yeast1011")
SNP_FILE = os.path.join(SNP_DIR, "1011DistanceMatrixBasedOnSNPs.tab.gz")
SNP_URL = "http://1002genomes.u-strasbg.fr/files/1011DistanceMatrixBasedOnSNPs.tab.gz"
SNP_SHA = "140da4e5193584c01e60c554a2ba5075a542d925be540afe7c7a92b7377af928"

# 2026-08-25 实测：NCBI 的 Extras/ 目录**滚动更新**，不是归档快照。
# 我方 2026-08-05 取回时 CID-Synonym-filtered.gz 是 964,716,803 字节；
# 08-25 再取，上游报 968,456,680 字节、Last-Modified 就是当天早上，SHA-256 自然不符。
# 也就是说**任何人都无法从这个 URL 重新取回我方用过的那一版**。
# 所以化合物结构表改为「随提交物直接交付 + 校验哈希」为主路径，
# 从 PubChem 重建降为回退路径，并明确告知重建结果可能与我方的不一致。
PUBCHEM_DIR = os.path.join(paths.PROJECT_ROOT, "data", "_pubchem_dump")
PUBCHEM = [
    ("CID-Synonym-filtered.gz",
     "https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-Synonym-filtered.gz",
     "c6d57604dc0fe746a7f5672586b1828dc77d05591e142ce0d0a55c3348ad3a46"),
    ("CID-SMILES.gz",
     "https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-SMILES.gz",
     "b62cf763f38b0d4812340ed7285137ab305be0a7de1ccc16bd564e4429d176b0"),
]

SMILES_CSV = os.path.join(paths.DATA_EXTERNAL, "compound_smiles.csv")
ALIAS_JSON = os.path.join(paths.DATA_EXTERNAL, "compound_aliases.json")

#: 我方实际用于产出报告数字的那一份化合物结构表的校验值。
#: 评委拿到提交包后可据此确认拿对了文件。
SMILES_SHA = "3863eed5d961856b5f0435e94382f760fd2eb7a24d0b3aa9429cf29a011e8ff9"

#: 手册第 15 页的对照名，不算化合物实体
_NON_COMPOUND = {"Water", "DMSO", "Quality Control"}


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: str, want_sha: str | None, label: str) -> dict:
    """下载并核对 SHA-256。已存在且校验通过就跳过——本脚本可重复运行。"""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest):
        got = sha256(dest)
        if want_sha is None or got == want_sha:
            print(f"  [已就绪] {label}  SHA-256 {got[:16]}…")
            return {"url": url, "sha256": got, "bytes": os.path.getsize(dest),
                    "status": "already_present"}
        print(f"  [重下] {label}：本地 SHA-256 与登记值不符（{got[:16]}… ≠ {want_sha[:16]}…）")

    print(f"  [下载] {label}  <- {url}")
    tmp = dest + ".part"
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as fh:
        total = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
            total += len(chunk)
            if total % (100 << 20) < (1 << 20):
                print(f"       … {total / 1e6:.0f} MB")
    got = sha256(tmp)
    if want_sha and got != want_sha:
        os.remove(tmp)
        raise SystemExit(
            f"❗{label} 校验失败：\n  期望 {want_sha}\n  实际 {got}\n"
            "上游文件可能已更新。请核对来源后更新本脚本里登记的 SHA-256，"
            "不要在校验不符的情况下继续——那样复现出来的数字与我方报告的不是同一份输入。")
    os.replace(tmp, dest)
    print(f"       完成 {total / 1e6:.1f} MB  SHA-256 {got[:16]}…")
    return {"url": url, "sha256": got, "bytes": total, "status": "downloaded"}


def compound_names() -> list:
    """赛事化合物名单，运行时从 metadata 读，不写进源码。"""
    from data import loader

    names = set()
    for which in ("train_val", "test"):
        m = loader.load_metadata(which)
        names |= {str(x) for x in m["perturbation_no_concentration"].unique()}
    return sorted(names - _NON_COMPOUND)


def smiles_matches_registered() -> bool:
    """手上这份结构表是不是我方跑出那批数字时用的那一份。"""
    return os.path.exists(SMILES_CSV) and sha256(SMILES_CSV) == SMILES_SHA


def check_smiles() -> tuple[bool, list]:
    """返回 (是否完整, 未解析出结构的化合物名)。"""
    if not os.path.exists(SMILES_CSV):
        return False, compound_names()
    import pandas as pd

    tab = pd.read_csv(SMILES_CSV)
    if not {"compound", "smiles"}.issubset(tab.columns):
        return False, compound_names()
    have = {str(r["compound"]) for _, r in tab.iterrows()
            if str(r["smiles"]).strip() and str(r["smiles"]).lower() != "nan"}
    missing = [c for c in compound_names() if c not in have]
    return (not missing), missing


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只报现状，不下载")
    ap.add_argument("--skip-pubchem", action="store_true",
                    help="已有 compound_smiles.csv，跳过 2.4 GB 转储")
    args = ap.parse_args()

    print("=" * 78)
    print("外部资源准备（跑完这一次之后，run_all.py 全程离线）")
    print("=" * 78)

    ok_smiles, missing = check_smiles()
    have_snp = os.path.exists(SNP_FILE) and sha256(SNP_FILE) == SNP_SHA

    if args.check:
        print(f"\n1011 SNP 距离矩阵    : {'就绪' if have_snp else '缺失或校验不符'}")
        print(f"化合物 SMILES 表     : {'就绪（54/54）' if ok_smiles else f'缺 {len(missing)} 条'}")
        if ok_smiles and not smiles_matches_registered():
            print("   ⚠ 该表与登记校验值不符——多半是从更新过的 PubChem 转储重建的，")
            print(f"     期望 {SMILES_SHA[:16]}…  实际 {sha256(SMILES_CSV)[:16]}…")
            print("     能跑，但产出数字与我方报告的不保证一致。")
        if missing:
            print(f"  未解析：{missing}")
        print(f"别名映射             : {'存在' if os.path.exists(ALIAS_JSON) else '缺失（随提交物由组委会提供）'}")
        sys.exit(0 if (have_snp and ok_smiles) else 1)

    rec = {"_note": "外部资源取回记录。只含地址、日期与校验值，不含数据本身。",
           "resources": {}}

    print("\n[1/3] 1011 Yeast Genomes · SNP 距离矩阵")
    rec["resources"]["1011_snp_matrix"] = fetch(SNP_URL, SNP_FILE, SNP_SHA, "SNP 距离矩阵")
    rec["resources"]["1011_snp_matrix"]["citation"] = \
        "Peter et al., Nature 2018, 556, 339-344"

    print("\n[2/3] 化合物名 → SMILES")
    if smiles_matches_registered():
        print("  [就绪] compound_smiles.csv 与登记校验值一致")
        print(f"         SHA-256 {SMILES_SHA[:16]}… —— 这就是我方跑出那批数字用的那一份")
        rec["resources"]["compound_smiles.csv"] = {
            "sha256": SMILES_SHA,
            "status": "supplied_with_submission",
            "note": "随提交物交付，非从公开地址取回；上游 PubChem 转储是滚动更新的",
        }
    elif ok_smiles and args.skip_pubchem:
        print("  [跳过] compound_smiles.csv 已完整（54/54），但与登记校验值不符")
        print(f"         期望 {SMILES_SHA[:16]}…  实际 {sha256(SMILES_CSV)[:16]}…")
        print("         产出的数字与我方报告的不保证一致，不要据此声称复现成功。")
    elif args.skip_pubchem:
        print(f"  ⚠ --skip-pubchem 但表不完整，仍缺 {len(missing)} 条：{missing}")
    else:
        print("  ⚠ 没有随提交物交付的 compound_smiles.csv，回退到从 PubChem 重建。")
        print("     注意：NCBI 的 Extras/ 目录滚动更新，**取不回我方用过的那一版**")
        print("     （实测 2026-08-05 是 964,716,803 字节，08-25 已变成 968,456,680）。")
        print("     重建能跑通，但结果与我方报告的数字不保证一致。")
        for name, url, want in PUBCHEM:
            # want 传 None：上游滚动更新，不能拿旧 SHA 当门槛，
            # 只如实记录这次实际取到的是哪一版，以及我方当初用的是哪一版。
            rec["resources"][name] = fetch(url, os.path.join(PUBCHEM_DIR, name),
                                           None, f"PubChem {name}")
            rec["resources"][name]["sha256_when_we_used_it"] = want
            rec["resources"][name]["upstream_is_rolling"] = True
        print("  [解析] 调 resolve_smiles.py 本地重建（化合物名不出本机）")
        import subprocess

        r = subprocess.run([sys.executable,
                            os.path.join(paths.SCRIPTS_ROOT, "data", "resolve_smiles.py"),
                            "--dump-dir", PUBCHEM_DIR, "--out", SMILES_CSV],
                           capture_output=True, text=True, encoding="utf-8")
        print("   " + "\n   ".join((r.stdout or "").strip().splitlines()[-6:]))
        if r.returncode != 0:
            print("   " + (r.stderr or "")[-800:])
            raise SystemExit("❗SMILES 重建失败")

    ok_smiles, missing = check_smiles()
    print("\n[3/3] 完整性核对")
    print(f"  SNP 矩阵      : {'OK' if os.path.exists(SNP_FILE) else 'FAIL'}")
    print(f"  SMILES 54/54  : {'OK' if ok_smiles else f'缺 {len(missing)} 条'}")
    print(f"  SMILES 校验    : "
          f"{'与登记值一致' if smiles_matches_registered() else '**与登记值不符或缺失**'}")
    if not ok_smiles:
        print(f"    未解析：{missing}")
        if not os.path.exists(ALIAS_JSON):
            print("\n  ⚠ 这几条要靠别名映射才能确定性重建。该文件含赛事化合物名，")
            print("    按《选手参赛协议》第八条不进公开仓库，**随提交物一并交给组委会**。")
            print(f"    把它放到 {ALIAS_JSON} 后重跑本脚本即可补齐。")
            print("    没有它时，依赖化合物结构的步骤（step6b / step9）会少这 4 个化合物，")
            print("    结果与我方报告的数字**不一致**——不要在这种状态下声称复现成功。")

    os.makedirs(paths.DATA_EXTERNAL, exist_ok=True)
    from datetime import datetime

    rec["prepared_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    rec["smiles_complete"] = bool(ok_smiles)
    rec["smiles_missing"] = missing
    with open(PROVENANCE, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(rec, fh, ensure_ascii=False, indent=2)
    print(f"\n[写出] {PROVENANCE}")
    print("\n准备完成。接下来： python scripts/run_all.py"
          if ok_smiles else "\n⚠ 未完全就绪，见上。")
    sys.exit(0 if ok_smiles else 1)


if __name__ == "__main__":
    main()
