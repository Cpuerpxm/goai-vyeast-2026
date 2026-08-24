"""拿测试集真值给自己打分。**只在模型冻结之后跑，结果不许回流。**

手册 2026-08 修订版第 15 页：

> 测试集蛋白质组真值随数据包一并发布，**供参赛队自评参考，不作最终排名依据**。

第 17 页同时说最终评审用组委会另备的一组独立内部评测集，不随赛题发放。
所以 test 上的分数只有一个用途：写进实验结果报告，说明我们在四类外推场景上
各自处于什么水平。它**不能**用来选模型、调参数、挑配置——那样等于拿评测集训练。

这条纪律靠三件事强制，不靠自觉：

1. 本文件是唯一被允许读 `proteome_raw_test.csv` 的地方
   （`paths.assert_readable_selfeval` 校验调用方路径）；
2. 必须显式传 `paths.SELF_EVAL_TOKEN`；
3. 必须已经存在 `results/step10_submission/submission_manifest.json`
   ——那是模型冻结的凭证。没冻结就不许看真值。

打分口径与官方 val 完全一致：把 test 的四类划分改名成对应的 val 划分，
直接喂给同一个评估台，六项指标的行分配、参照均值、缺失规则一个字都不改。
参照均值 mu_ctx / mu_drug 仍然**只由 train 折冻结**，再按上下文键与药物键映射到
test 行；映射不到的回退到 train 总体均值，回退比例会报出来。

运行（先跑 predict_test.py）：
    python self_eval.py
    python self_eval.py --prediction <别的 prediction.csv>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import control_match as cm
from data import loader, paths, provenance
from data import split_guard as sg
from scorer import evaluate as ev
from scorer.config import ScorerConfig

OUT_DIR = os.path.join(paths.RESULTS, "step12_self_eval")
CACHE = os.path.join(paths.CACHE, "proteome_test_log2.npz")

#: test 的四类划分 -> 官方 val 的同名角色。改名只为复用同一个评估台。
SPLIT_MAP = {
    "test_chem_only": "val_chem_only",
    "test_strain_only": "val_strain_only",
    "test_both": "val_both",
    "test_time": "val_time",
}


def load_test_proteome(verbose: bool = True):
    """读测试集蛋白丰度并转 log2。走的是自评专用入口，不是常规入口。"""
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=False)
        if verbose:
            print(f"[自评] 命中缓存 {CACHE}")
        return z["sample_ids"].astype(str), z["proteins"].astype(str), z["X"]

    src = paths.assert_readable_selfeval(
        paths.PROT_TEST_SELFEVAL, paths.SELF_EVAL_TOKEN, __file__)
    if verbose:
        print(f"[自评] 解析 {src} …（首次约 30-60 秒）")
    df = pd.read_csv(src)
    sample_ids = np.asarray(df.iloc[:, 0].astype(str).tolist(), dtype="<U32")
    proteins = np.asarray([str(c) for c in df.columns[1:]], dtype="<U64")
    raw = df.iloc[:, 1:].to_numpy(dtype=np.float64)
    del df
    n_nonpos = int(np.sum(raw <= 0))
    if n_nonpos:
        print(f"[自评] ⚠ 非正值 {n_nonpos} 个 → 置 NaN（log2 无定义），不做伪计数")
        raw[raw <= 0] = np.nan
    ok = np.isfinite(raw)
    X = np.full(raw.shape, np.nan, dtype=np.float32)
    X[ok] = np.log2(raw[ok]).astype(np.float32)
    del raw, ok
    paths.ensure_dir(paths.CACHE)
    np.savez(CACHE, sample_ids=sample_ids, proteins=proteins, X=X)
    if verbose:
        print(f"[自评] X {X.shape}  缺失率 {np.isnan(X).mean():.4%}  → 缓存 {CACHE}")
    return sample_ids, proteins, X


def frozen_reference(train_ctx, meta_te: pd.DataFrame, keys) -> tuple[np.ndarray, float]:
    """把 train 折冻结的分组均值映射到 test 行，映射不到就回退到 train 总体均值。"""
    tr = sg.train_rows(train_ctx.meta) & train_ctx.treated
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        glob = np.nanmean(train_ctx.D[tr], axis=0, dtype=np.float64).astype(np.float32)
    glob = np.where(np.isfinite(glob), glob, 0.0)

    def _key(df):
        return df[list(keys)].astype(str).agg("\x1f".join, axis=1).to_numpy()

    k_tr, k_te = _key(train_ctx.meta), _key(meta_te)
    table = {}
    for k in np.unique(k_tr[tr]):
        sel = tr & (k_tr == k)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            m = np.nanmean(train_ctx.D[sel], axis=0, dtype=np.float64).astype(np.float32)
        table[k] = np.where(np.isfinite(m), m, glob)
    mu = np.tile(glob, (len(meta_te), 1))
    hit = np.zeros(len(meta_te), dtype=bool)
    for i, k in enumerate(k_te):
        v = table.get(k)
        if v is not None:
            mu[i] = v
            hit[i] = True
    return mu, float(1.0 - hit.mean())


def build_test_context(train_ctx, cfg: ScorerConfig):
    """造一个 test 侧的 EvalContext，划分改名成 val_* 以复用同一个评估台。"""
    import dataclasses

    meta_te = loader.load_metadata("test")
    sid, proteins, Xr = load_test_proteome()
    if list(proteins) != list(train_ctx.proteins):
        raise RuntimeError("测试集蛋白列与训练集不一致，不能直接比较")
    pos = {s: i for i, s in enumerate(sid)}
    order = np.asarray([pos[s] for s in meta_te["sample_ID"].astype(str)], dtype=np.int64)
    X = Xr[order]
    del Xr

    mcfg = cm.MatchConfig()
    C, n_ctrl, is_ctrl = cm.control_profiles(X, meta_te, mcfg)
    is_qc = (meta_te[mcfg.pert_col].astype(str) == cm.QC_LABEL).to_numpy()
    D = cm.compute_delta(X, C)

    mu_ctx, fb_ctx = frozen_reference(train_ctx, meta_te, ev.CTX_KEYS_DEFAULT)
    mu_drug, fb_drug = frozen_reference(train_ctx, meta_te, [ev.PERT_COL])

    # ❗test 侧对照覆盖率必须逐划分报出来，不能只报一个总数。
    # 2026-08-24 实测：test 里只有 202 行溶剂对照，其中 188 行属于**未见菌株**那一株。
    # 于是 Δ_true 在 test 上大面积无定义——而指标 2/3/4/5/6 全都建立在 Δ 上，
    # 合计 80% 权重。只看总匹配率会以为「一半有对照」，实际是「几乎全部集中在一格」。
    treated_te = (~is_ctrl) & (~is_qc)
    cov = {}
    for s in sorted(meta_te["split_final"].astype(str).unique()):
        sel = treated_te & (meta_te["split_final"].astype(str) == s).to_numpy()
        cov[s] = {"n_treated": int(sel.sum()),
                  "with_control": int((n_ctrl[sel] > 0).sum()),
                  "coverage": float((n_ctrl[sel] > 0).mean()) if sel.any() else float("nan")}
    ctrl_by_strain = (meta_te.loc[is_ctrl, "Strains"].astype(str)
                      .value_counts().to_dict())

    meta_scored = meta_te.copy()
    meta_scored["split_final"] = meta_scored["split_final"].astype(str).map(
        lambda s: SPLIT_MAP.get(s, s))
    unknown = set(meta_scored["split_final"]) - set(ev.VAL_SPLITS)
    if unknown:
        raise RuntimeError(f"test 出现了预期外的划分：{sorted(unknown)}")

    ctx = ev.EvalContext(
        meta=meta_scored, X=X, C=C, D=D, mu_ctx=mu_ctx, mu_drug=mu_drug,
        proteins=proteins, is_control=is_ctrl, is_qc=is_qc,
        train_mask=np.zeros(len(meta_te), dtype=bool))
    info = {
        "n_samples": int(len(meta_te)),
        "n_treated": int(ctx.treated.sum()),
        "missing_rate_abundance": float(np.isnan(X).mean()),
        "missing_rate_delta_treated": float(np.isnan(D[ctx.treated]).mean()),
        "zero_control_match_rate": float((n_ctrl[ctx.treated] == 0).mean()),
        "multi_control_match_rate": float((n_ctrl[ctx.treated] > 1).mean()),
        "mu_ctx_fallback_rate": fb_ctx,
        "mu_drug_fallback_rate": fb_drug,
        "split_counts": meta_te["split_final"].value_counts().to_dict(),
        "n_solvent_controls": int(is_ctrl.sum()),
        "controls_by_strain_masked": {f"strain#{i+1}": v for i, v in
                                      enumerate(ctrl_by_strain.values())},
        "control_coverage_by_split": cov,
    }
    return ctx, meta_te, info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prediction", default=os.path.join(
        paths.RESULTS, "step10_submission", "prediction.csv"))
    args = ap.parse_args()

    if not os.path.exists(paths.SUBMISSION_MANIFEST):
        raise SystemExit(
            f"缺 {paths.SUBMISSION_MANIFEST}：模型还没冻结，不许看 test 真值。"
            "先跑 python scripts/models/predict_test.py")
    manifest = json.load(open(paths.SUBMISSION_MANIFEST, encoding="utf-8"))
    if not os.path.exists(args.prediction):
        raise SystemExit(f"缺预测文件 {args.prediction}")

    import hashlib
    sha = hashlib.sha256(open(args.prediction, "rb").read()).hexdigest()
    same = (sha == manifest.get("prediction_sha256"))

    cfg = ScorerConfig()
    print("[自评] 载入 train_val（只为拿 train 折冻结的参照均值）…")
    train_ctx = ev.build_context(verbose=False)
    print("[自评] 造 test 侧评估上下文 …")
    ctx, meta_te, info = build_test_context(train_ctx, cfg)

    pred = pd.read_csv(args.prediction)
    if list(pred.iloc[:, 0].astype(str)) != list(meta_te["sample_ID"].astype(str)):
        raise SystemExit("预测文件的样本顺序与 metadata_test.csv 不一致")
    if list(pred.columns[1:]) != list(ctx.proteins):
        raise SystemExit("预测文件的蛋白列与数据不一致")
    y = pred.iloc[:, 1:].to_numpy(dtype=np.float32)
    del pred

    all_rows = np.ones(ctx.n, dtype=bool)
    f_all = ev.flatten(ev.evaluate(y, ctx, all_rows, cfg))
    by = ev.evaluate_by_split(y, ctx, cfg)

    L: list[str] = []
    a = L.append
    a("=" * 96)
    a("测试集自评（手册允许，供参考；最终排名用组委会另备的内部评测集）")
    a("=" * 96)
    a(f"预测文件 {os.path.relpath(args.prediction, paths.PROJECT_ROOT)}")
    a(f"  SHA-256 {sha[:16]}…   与提交清单登记的"
      f"{'一致' if same else '**不一致**（说明这份预测不是冻结的那一份）'}")
    a(f"  冻结时的配置：拟合行 {manifest.get('n_fit_rows')} 行（{manifest.get('fit_split')} 折）"
      f"  K0={manifest.get('k0')}  lam={manifest.get('lam')}")
    a("")
    a("test 侧数据实况（现算，不引用 train_val 的数）：")
    a(f"  样本 {info['n_samples']}（处理样本 {info['n_treated']}）"
      f"  划分 {info['split_counts']}")
    a(f"  丰度缺失率 {info['missing_rate_abundance']:.2%}；"
      f"处理样本的 Δ 缺失率 {info['missing_rate_delta_treated']:.2%}")
    a(f"  对照匹配：0 匹配率 {info['zero_control_match_rate']:.1%}，"
      f"多匹配 {info['multi_control_match_rate']:.1%}")
    a(f"  参照均值回退到 train 总体均值的比例："
      f"mu_ctx {info['mu_ctx_fallback_rate']:.1%} / mu_drug {info['mu_drug_fallback_rate']:.1%}")
    a("")
    a("❗test 侧对照覆盖率（这一段比总分更值得看）")
    a(f"  test 全表只有 {info['n_solvent_controls']} 行溶剂对照，"
      f"按菌株分布 {list(info['controls_by_strain_masked'].values())}"
      "（已按数量排序脱去代号）——绝大多数集中在**未见菌株**那一株。")
    a("")
    a(f"  {'场景':>18}{'处理样本':>10}{'有匹配对照':>12}{'覆盖率':>10}")
    for s, d in info["control_coverage_by_split"].items():
        a(f"  {s:>18}{d['n_treated']:>10d}{d['with_control']:>12d}{d['coverage']:>10.1%}")
    a("")
    a("  为什么要紧：Δ_true = 处理 − 匹配对照。没有匹配对照的样本，Δ 整行无定义，")
    a("  于是**指标 2/3/4/5/6 全都算不到它头上**——那是 80% 的权重。")
    a("  train_val 上 0 匹配率是 0.0%，test 上却过半，两边的评估基础根本不同。")
    a("  实际后果：test 上的 Δ 类指标由「未见菌株」那一格主导，")
    a("  而指标 3（上下文残差）只落在很薄的一层样本上，其读数不确定性远大于 val。")
    a("")
    a("六项加权总分与分项（与官方 val 同一套口径）")
    a(f"  {'total':>9}{'abs_pcc':>10}{'abs_r2':>10}{'fc_pcc':>10}"
      f"{'ctx_resid':>11}{'drug_resid':>12}{'both_time':>11}{'dep_dir':>10}")
    a(f"  {f_all['total']:>9.4f}{f_all['abs_pcc']:>10.4f}{f_all['abs_r2']:>10.4f}"
      f"{f_all['fc_pcc']:>10.4f}{f_all['ctx_resid']:>11.4f}"
      f"{f_all['drug_resid']:>12.4f}{f_all['both_time']:>11.4f}{f_all['dep_dir']:>10.4f}")
    a("")
    a("逐场景（划分名已换回 test 侧口径）")
    inv = {v: k for k, v in SPLIT_MAP.items()}
    a("    " + "  ".join(f"{c:>16}" for c in ["场景", "n", "abs_pcc", "abs_r2", "fc_pcc"]))
    for _, r in by.iterrows():
        name = inv.get(r["split"], r["split"])
        if name == "train":
            continue
        a("    " + "  ".join([f"{name:>16}", f"{int(r['n']):>16d}",
                              f"{r.get('abs_pcc', np.nan):>16.4f}",
                              f"{r.get('abs_r2', np.nan):>16.4f}",
                              f"{r.get('fc_pcc', np.nan):>16.4f}"]))
    a("")
    a("⚠ 这张表只许写进报告，不许用来选模型、调参数或挑配置。")
    a("  手册明说它不作排名依据；拿它做取舍就等于用评测集训练。")

    txt = "\n".join(L)
    print(txt)
    paths.ensure_dir(OUT_DIR)
    p = os.path.join(OUT_DIR, "self_eval.txt")
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(txt)
    payload = {
        "_schema": "self_eval/1.0",
        "_warning": "test 成绩仅供自评，不作排名依据，不得用于任何模型选择",
        "prediction_file": os.path.relpath(args.prediction, paths.PROJECT_ROOT).replace("\\", "/"),
        "prediction_sha256": sha,
        "matches_frozen_manifest": bool(same),
        "frozen_config": {k: manifest.get(k) for k in
                          ("k0", "lam", "fit_split", "n_fit_rows", "n_dead_cols")},
        "test_data_facts": info,
        "overall": f_all,
        "by_split": {inv.get(r["split"], r["split"]):
                     {k: (int(r[k]) if k.startswith("n") else float(r[k]))
                      for k in by.columns if k != "split" and r[k] == r[k]}
                     for _, r in by.iterrows() if r["split"] != "train"},
        "_provenance": provenance.stamp(),
    }
    pj = os.path.join(OUT_DIR, "self_eval.json")
    with open(pj, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)
    print(f"\n[写出] {p}\n[写出] {pj}")


if __name__ == "__main__":
    main()
