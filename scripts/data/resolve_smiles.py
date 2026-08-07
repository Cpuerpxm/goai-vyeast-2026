"""化合物名 → SMILES：**离线**解析 PubChem 公开转储。

合规依据（CLAUDE.md R1）：不得把完整化合物名单发送给任何外部服务。
因此这里不查 API，而是下载 PubChem 全量公开文件后在本机做匹配——
化合物名单一个字都不出本机，出去的只有对两个固定公开 URL 的 GET。

两个文件（ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/）：
    CID-Synonym-filtered.gz   名称 / 同义词 → CID     约 0.96 GB
    CID-SMILES.gz             CID → 规范 SMILES        约 1.48 GB

匹配策略（逐级放宽，全部在本机）：
    1. 名称规范化后精确匹配（小写、去空格/连字符/括号空格）
    2. 去掉盐/水合物后缀再匹配（hydrochloride / dihydrate / isethionate ...）
    3. 仍失败的进 unresolved 清单，由人工补

产出 data/external/compound_smiles.csv：compound, smiles, cid, match_level, source

用法：
    python resolve_smiles.py --download      # 首次：下载两个转储（约 2.4 GB）
    python resolve_smiles.py                 # 用已下载的转储做解析
"""
from __future__ import annotations

import argparse
import csv
import gzip
import os
import re
import sys
import urllib.request

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import loader, paths

DUMP_DIR = os.path.join(paths.PROJECT_ROOT, "data", "_pubchem_dump")
SYN_URL = "https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-Synonym-filtered.gz"
SMI_URL = "https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-SMILES.gz"
OUT_CSV = os.path.join(paths.DATA_EXTERNAL, "compound_smiles.csv")

# 盐 / 水合物 / 立体前缀等后缀，去掉后再试一次
SALT_SUFFIX = [
    "hydrochloride", "dihydrochloride", "hydrobromide", "sulfate", "sulphate",
    "phosphate", "maleate", "citrate", "mesylate", "tosylate", "acetate",
    "tartrate", "fumarate", "succinate", "isethionate", "hyclate", "besylate",
    "sodium salt", "potassium salt", "sodium", "potassium", "calcium",
    "monohydrate", "dihydrate", "trihydrate", "hydrate", "anhydrous",
]
NON_COMPOUND = {"water", "dmso", "quality control"}

# 数据集里用了缩写或同系物混合物名，PubChem 同义词表查不到。
# 别名表本身含赛事化合物名，按《选手参赛协议》第八条不入库，
# 放在 data/external/compound_aliases.json（已 gitignore）。
# 只断言「名称别名」，SMILES 仍由 PubChem 给出；match_level 标 alias 供复核。
ALIAS_FILE = os.path.join(paths.DATA_EXTERNAL, "compound_aliases.json")


def _load_alias() -> dict:
    if os.path.exists(ALIAS_FILE):
        import json
        return json.load(open(ALIAS_FILE, encoding="utf-8"))
    print(f"[提示] 未找到 {ALIAS_FILE}，跳过别名解析。"
          "该文件为 {缩写或混合物名: 规范名} 的映射，需自行提供。")
    return {}


ALIAS = _load_alias()


def norm(s: str) -> str:
    s = str(s).strip().lower()
    s = s.replace("β", "beta").replace("α", "alpha")
    s = re.sub(r"[\s\-_,]+", "", s)
    s = re.sub(r"[（）]", "", s)
    return s


def strip_salt(s: str) -> str:
    t = str(s).strip().lower()
    changed = True
    while changed:
        changed = False
        for suf in SALT_SUFFIX:
            if t.endswith(" " + suf):
                t = t[: -(len(suf) + 1)].strip()
                changed = True
    return t


def download(url: str, dst: str) -> None:
    if os.path.exists(dst) and os.path.getsize(dst) > 1_000_000:
        print(f"[skip] 已存在 {dst}（{os.path.getsize(dst)/1e9:.2f} GB）")
        return
    print(f"[下载] {url}\n     → {dst}")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".part"

    def hook(blocks, bs, total):
        if total > 0 and blocks % 2000 == 0:
            print(f"      {blocks*bs/1e9:.2f} / {total/1e9:.2f} GB", flush=True)

    urllib.request.urlretrieve(url, tmp, reporthook=hook)
    os.replace(tmp, dst)
    print(f"[完成] {os.path.getsize(dst)/1e9:.2f} GB")


def compound_names() -> list[str]:
    """训练 + 测试元数据里的全部扰动名（只读本机文件）。"""
    names = set()
    for w in ("train_val", "test"):
        m = loader.load_metadata(w)
        names |= set(m["perturbation_no_concentration"].astype(str))
    return sorted(n for n in names if n.strip().lower() not in NON_COMPOUND)


def resolve(names: list[str], syn_gz: str, smi_gz: str) -> pd.DataFrame:
    # 目标键：原名 / 去盐名，各自的规范化形式
    want: dict[str, list[tuple[str, int]]] = {}
    for n in names:
        want.setdefault(norm(n), []).append((n, 1))
        sn = norm(strip_salt(n))
        if sn != norm(n):
            want.setdefault(sn, []).append((n, 2))
        if n in ALIAS:
            want.setdefault(norm(ALIAS[n]), []).append((n, 3))
    print(f"[解析] 目标 {len(names)} 个化合物，{len(want)} 个匹配键")

    hits: dict[str, tuple[int, int]] = {}       # 原名 → (cid, level)
    print(f"[解析] 扫描 {syn_gz} …")
    seen = 0
    with gzip.open(syn_gz, "rt", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            seen += 1
            if seen % 20_000_000 == 0:
                print(f"      已扫 {seen/1e6:.0f}M 行，命中 {len(hits)}/{len(names)}", flush=True)
            tab = line.find("\t")
            if tab < 0:
                continue
            key = norm(line[tab + 1:].rstrip("\n"))
            tgt = want.get(key)
            if not tgt:
                continue
            cid = int(line[:tab])
            for orig, lvl in tgt:
                cur = hits.get(orig)
                # 同名多 CID：取等级更优者，同等级取更小 CID（PubChem 惯例：更早收录 = 母体）
                if cur is None or lvl < cur[1] or (lvl == cur[1] and cid < cur[0]):
                    hits[orig] = (cid, lvl)
    print(f"[解析] 同义词扫描完成，命中 {len(hits)}/{len(names)}")

    need_cid = {c for c, _ in hits.values()}
    smiles: dict[int, str] = {}
    print(f"[解析] 扫描 {smi_gz} 取 {len(need_cid)} 个 CID 的 SMILES …")
    with gzip.open(smi_gz, "rt", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            tab = line.find("\t")
            if tab < 0:
                continue
            cid = int(line[:tab])
            if cid in need_cid:
                smiles[cid] = line[tab + 1:].strip()
                if len(smiles) == len(need_cid):
                    break

    rows = []
    for n in names:
        h = hits.get(n)
        if h and h[0] in smiles:
            lvl = {1: "exact", 2: "salt_stripped", 3: f"alias:{ALIAS.get(n, '')}"}[h[1]]
            rows.append({"compound": n, "smiles": smiles[h[0]], "cid": h[0],
                         "match_level": lvl,
                         "source": "PubChem CID-Synonym-filtered + CID-SMILES (offline)"})
        else:
            rows.append({"compound": n, "smiles": "", "cid": "",
                         "match_level": "UNRESOLVED", "source": ""})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--dump-dir", default=DUMP_DIR)
    ap.add_argument("--out", default=OUT_CSV)
    args = ap.parse_args()

    syn = os.path.join(args.dump_dir, "CID-Synonym-filtered.gz")
    smi = os.path.join(args.dump_dir, "CID-SMILES.gz")
    if args.download:
        download(SYN_URL, syn)
        download(SMI_URL, smi)
    for p in (syn, smi):
        if not os.path.exists(p):
            print(f"❗缺 {p}，先跑 python resolve_smiles.py --download")
            sys.exit(2)

    names = compound_names()
    df = resolve(names, syn, smi)
    paths.ensure_dir(paths.DATA_EXTERNAL)
    df.to_csv(args.out, index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL)

    ok = df["match_level"] != "UNRESOLVED"
    print(f"\n解析成功 {int(ok.sum())}/{len(df)}")
    print(df["match_level"].value_counts().to_string())
    if (~ok).any():
        print("\n未解析（需人工补 smiles 列）：")
        for n in df.loc[~ok, "compound"]:
            print(f"  - {n}")

    try:
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.*")
        bad = [r["compound"] for _, r in df[ok].iterrows()
               if Chem.MolFromSmiles(str(r["smiles"])) is None]
        print(f"\nRDKit 可解析 {int(ok.sum()) - len(bad)}/{int(ok.sum())}")
        if bad:
            print("  RDKit 解析失败：" + ", ".join(bad))
    except ImportError:
        pass
    print(f"\n[写出] {args.out}")
    print("❗人工抽查建议：打开 csv 核对几个关键化合物的 cid 是否指向母体而非盐/异构体。")


if __name__ == "__main__":
    main()
