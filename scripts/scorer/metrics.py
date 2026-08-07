"""六项评分指标的实现。

所有函数在缺失处采用成对完整（pairwise complete）。
Δ_pred = ŷ_treat - y_control   (对照用真实值，手册原文)
Δ_true = y_treat  - y_control
"""
import numpy as np

try:                                    # 直接在 scripts/scorer/ 下运行
    from config import ScorerConfig
except ImportError:                     # 从 scripts/ 作为包导入
    from scorer.config import ScorerConfig


# ----------------------------------------------------------------- 基础统计


def _pairwise_mask(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.isfinite(a) & np.isfinite(b)


def _is_const(sd, vals_rms, cfg: ScorerConfig):
    """常数判据用**相对**阈值。见 config.const_rtol 的注释（float32 舍入噪声）。"""
    return sd <= np.maximum(cfg.const_atol, cfg.const_rtol * vals_rms)


def pcc(a: np.ndarray, b: np.ndarray, cfg: ScorerConfig) -> float:
    """成对完整的 Pearson 相关。常数向量按配置处理。"""
    m = _pairwise_mask(a, b)
    n = int(m.sum())
    if n < cfg.min_valid_points:
        return np.nan
    x, y = a[m].astype(np.float64), b[m].astype(np.float64)
    sx, sy = x.std(), y.std()
    rx, ry = np.sqrt((x ** 2).mean()), np.sqrt((y ** 2).mean())
    if _is_const(sx, rx, cfg) or _is_const(sy, ry, cfg):
        return 0.0 if cfg.undefined_pcc == "zero" else np.nan
    return float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy))


def r2(y_true: np.ndarray, y_pred: np.ndarray, cfg: ScorerConfig) -> float:
    """决定系数，可为负（刻画幅度与校准）。"""
    m = _pairwise_mask(y_true, y_pred)
    n = int(m.sum())
    if n < cfg.min_valid_points:
        return np.nan
    t, p = y_true[m], y_pred[m]
    ss_tot = float(((t - t.mean()) ** 2).sum())
    if ss_tot < 1e-12:
        return np.nan
    ss_res = float(((t - p) ** 2).sum())
    return 1.0 - ss_res / ss_tot


# ------------------------------------------------- 向量化版本（与上面逐条版等价）
# 逐条 python 循环在 5,243 蛋白 × 数千样本 × bootstrap 下太慢。
# 下面两个函数一次算完一整条轴，语义必须与 pcc / r2 完全一致，
# test_metrics.py 里有等价性断言看着。


def pcc_axis(A: np.ndarray, B: np.ndarray, cfg: ScorerConfig, axis: int) -> np.ndarray:
    """沿 axis 逐条算成对完整 PCC。axis=1 → 逐样本；axis=0 → 逐蛋白。"""
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if axis == 0:
        A, B = A.T, B.T
    m = np.isfinite(A) & np.isfinite(B)
    a = np.where(m, A, 0.0)
    b = np.where(m, B, 0.0)
    n = m.sum(axis=1).astype(np.float64)
    # ❗必须走两遍法（先算均值再算离差）。一遍法的 Σx²/n − (Σx/n)² 在
    # log2 丰度这种「均值 ~20、离差 ~2」的数据上会灾难性相消：常数向量的方差
    # 会算成 ~1e-11 而不是 0，于是「常数向量 → PCC 未定义」的判据失效，
    # 返回一个纯粹由舍入误差决定的假相关。逐条版用的 .std() 本来就是两遍法。
    with np.errstate(invalid="ignore", divide="ignore"):
        ma = a.sum(1) / n
        mb = b.sum(1) / n
        ac = np.where(m, A - ma[:, None], 0.0)
        bc = np.where(m, B - mb[:, None], 0.0)
        va = (ac * ac).sum(1) / n
        vb = (bc * bc).sum(1) / n
        cov = (ac * bc).sum(1) / n
        sda = np.sqrt(np.maximum(va, 0.0))
        sdb = np.sqrt(np.maximum(vb, 0.0))
        rmsa = np.sqrt((a * a).sum(1) / n)
        rmsb = np.sqrt((b * b).sum(1) / n)
        r = cov / (sda * sdb)
    out = np.where(n >= cfg.min_valid_points, r, np.nan)
    const = _is_const(sda, rmsa, cfg) | _is_const(sdb, rmsb, cfg)
    out = np.where(const & (n >= cfg.min_valid_points),
                   0.0 if cfg.undefined_pcc == "zero" else np.nan, out)
    return out


def r2_axis(T: np.ndarray, P: np.ndarray, cfg: ScorerConfig, axis: int) -> np.ndarray:
    """沿 axis 逐条算成对完整 R²。"""
    T = np.asarray(T, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    if axis == 0:
        T, P = T.T, P.T
    m = np.isfinite(T) & np.isfinite(P)
    t = np.where(m, T, 0.0)
    p = np.where(m, P, 0.0)
    n = m.sum(axis=1).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        mt = t.sum(1) / n                       # 同 pcc_axis：两遍法防相消
        tc = np.where(m, T - mt[:, None], 0.0)
        ss_tot = (tc * tc).sum(1)
        ss_res = ((t - p) ** 2 * m).sum(1)
        r = 1.0 - ss_res / ss_tot
    out = np.where((n >= cfg.min_valid_points) & (ss_tot >= 1e-12), r, np.nan)
    return out


# 每次 _both_axes 调用后写入两轴原值与有效计数，供报告层取用
# （Pro R2 L2-5：所有结果都要能看到两轴原值、有效向量数和未定义比例）。
_LAST_AXIS: dict = {}


def last_axis_detail() -> dict:
    return dict(_LAST_AXIS)


def _agg(vals, cfg: ScorerConfig) -> float:
    v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return np.nan
    return float(np.median(v) if cfg.agg == "median" else v.mean())


def _both_axes(y_true: np.ndarray, y_pred: np.ndarray, cfg: ScorerConfig, fn) -> float:
    """样本轴与蛋白轴分别聚合，再按配置组合。

    y_true / y_pred : (n_samples, n_proteins)
    手册：样本轴与蛋白轴分别聚合，避免高丰度易预测蛋白掩盖困难样本。
    """
    vec = {pcc: pcc_axis, r2: r2_axis}.get(fn)
    if vec is not None:
        vs = vec(y_true, y_pred, cfg, axis=1)
        vp = vec(y_true, y_pred, cfg, axis=0)
    else:
        vs = np.asarray([fn(y_true[i], y_pred[i], cfg) for i in range(y_true.shape[0])])
        vp = np.asarray([fn(y_true[:, j], y_pred[:, j], cfg) for j in range(y_true.shape[1])])
    by_sample, by_protein = _agg(vs, cfg), _agg(vp, cfg)
    _LAST_AXIS.update(sample=by_sample, protein=by_protein,
                      n_sample_valid=int(np.isfinite(vs).sum()), n_sample=int(vs.size),
                      n_protein_valid=int(np.isfinite(vp).sum()), n_protein=int(vp.size))
    if cfg.axis_combine == "sample_only":
        return by_sample
    if cfg.axis_combine == "protein_only":
        return by_protein
    # ❗整条轴未定义时不能静默踢出：全局均值谱模型在每个蛋白上都是常数，
    # 蛋白轴全 nan，"踢出"会让它只按样本轴计分而白拿一档（2026-08-05 Pro R2 L1-02）。
    if cfg.undefined_axis == "zero":
        s = 0.0 if not np.isfinite(by_sample) else by_sample
        p = 0.0 if not np.isfinite(by_protein) else by_protein
        return float((s + p) / 2.0)
    vals = [v for v in (by_sample, by_protein) if np.isfinite(v)]
    return float(np.mean(vals)) if vals else np.nan


# ----------------------------------------------------------------- 六项指标


def metric_absolute(y_true, y_pred, cfg):
    """指标 1 · 绝对保真度 (20%)。逐样本 + 逐蛋白的 corr 与 R²。"""
    return {
        "pcc": _both_axes(y_true, y_pred, cfg, pcc),
        "r2": _both_axes(y_true, y_pred, cfg, r2),
    }


def metric_fc(d_true, d_pred, cfg):
    """指标 2 · 匹配对照原始 FC (25%)。所有 OOD 划分的核心。"""
    v = _both_axes(d_true, d_pred, cfg, pcc)
    return {"pcc": v, "axis": last_axis_detail()}


def metric_both_time(y_true, y_pred, d_true, d_pred, cfg):
    """指标 5 · 双重未知 / 时间外推 (10%)。

    手册："test_both 以原始 FC + 绝对保真度为主；test_time 以绝对保真度 + 原始 FC 为主"
    → **两个分量都要算**。此前只算了 FC，漏掉绝对保真度（2026-08-05 Pro R2 L1-03）。
    """
    fc = _both_axes(d_true, d_pred, cfg, pcc)
    if cfg.both_time_parts == "fc_only":
        return {"pcc": fc, "fc": fc, "abs_pcc": np.nan, "abs_r2": np.nan}
    a_pcc = _both_axes(y_true, y_pred, cfg, pcc)
    a_r2 = _both_axes(y_true, y_pred, cfg, r2)
    absolute = np.nanmean([v for v in (a_pcc, a_r2) if np.isfinite(v)]) \
        if np.isfinite([a_pcc, a_r2]).any() else np.nan
    parts = [v for v in (fc, absolute) if np.isfinite(v)]
    return {"pcc": float(np.mean(parts)) if parts else np.nan,
            "fc": fc, "abs_pcc": a_pcc, "abs_r2": a_r2}


def metric_residual(d_true, d_pred, mu, cfg):
    """指标 3/4 · 残差 (各 20%)。

    mu : (n_samples, n_proteins) 逐样本对应的参照均值
         指标 3 用 μ_ctx（同上下文下训练药物的 Δ_true 均值）
         指标 4 用 μ_drug（同药物下训练上下文的 Δ_true 均值）
    """
    return {"pcc": _both_axes(d_true - mu, d_pred - mu, cfg, pcc)}


def metric_dep(d_true, d_pred, cfg):
    """指标 6 · 高效应蛋白与 DEP 检出 (5%)。

    对 |Δ_true| > threshold 的蛋白算方向准确率、高效应 PCC、Recall@K、F1、AUPRC。
    手册明确：Recall 不单用，须与 precision/F1/AUPRC 组合，避免全报阳性刷召回。
    """
    hits, precs, recs, f1s, pccs = [], [], [], [], []
    for i in range(d_true.shape[0]):
        t, p = d_true[i], d_pred[i]
        m = _pairwise_mask(t, p)
        if m.sum() < cfg.min_valid_points:
            continue
        t, p = t[m], p[m]
        pos = np.abs(t) > cfg.dep_threshold
        if pos.sum() == 0:
            continue

        # 方向准确率（仅在真实大效应蛋白上）
        hits.append(float((np.sign(t[pos]) == np.sign(p[pos])).mean()))
        # 高效应 PCC
        pccs.append(pcc(t[pos], p[pos], cfg))

        # Recall@K / precision / F1：按 |Δ_pred| 取 top-K 作为预测阳性
        k = min(cfg.dep_top_k, p.size)
        topk = np.argpartition(-np.abs(p), k - 1)[:k]
        pred_pos = np.zeros(p.size, dtype=bool)
        pred_pos[topk] = True
        tp = int((pred_pos & pos).sum())
        prec = tp / max(int(pred_pos.sum()), 1)
        rec = tp / max(int(pos.sum()), 1)
        precs.append(prec)
        recs.append(rec)
        f1s.append(0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec))

    return {
        "direction_acc": _agg(hits, cfg),
        "high_effect_pcc": _agg(pccs, cfg),
        "precision_at_k": _agg(precs, cfg),
        "recall_at_k": _agg(recs, cfg),
        "f1_at_k": _agg(f1s, cfg),
    }


# ----------------------------------------------------------------- 总分


def combine(parts: dict, cfg: ScorerConfig) -> float:
    """按手册权重组合。parts 的键见 score_all 的返回。

    每个模块先归一到单一标量：
      absolute  = mean(pcc, r2)
      fc/ctx/drug/both_time = pcc
      dep       = mean(direction_acc, high_effect_pcc, f1_at_k)   ← 不单用 recall
    """
    cfg.check_weights()

    def _m(*vals):
        v = [x for x in vals if x is not None and np.isfinite(x)]
        return float(np.mean(v)) if v else np.nan

    s_abs = _m(parts.get("absolute", {}).get("pcc"), parts.get("absolute", {}).get("r2"))
    s_fc = parts.get("fc", {}).get("pcc", np.nan)
    s_ctx = parts.get("ctx_resid", {}).get("pcc", np.nan)
    s_drug = parts.get("drug_resid", {}).get("pcc", np.nan)
    s_bt = parts.get("both_time", {}).get("pcc", np.nan)
    d = parts.get("dep", {})
    s_dep = _m(d.get("direction_acc"), d.get("high_effect_pcc"), d.get("f1_at_k"))

    terms = [
        (cfg.w_absolute, s_abs), (cfg.w_fc, s_fc), (cfg.w_ctx_resid, s_ctx),
        (cfg.w_drug_resid, s_drug), (cfg.w_both_time, s_bt), (cfg.w_dep, s_dep),
    ]
    if cfg.undefined_module == "zero":
        # 未定义的模块按 0 计入，分母是满权重。
        # 必须有这个选项：否则「让某个指标未定义」本身就能提分——
        # 例如 Δ≡0 的基线使 FC 变成常数向量 → PCC 未定义 → 被踢出分母 →
        # 只靠它最擅长的绝对保真度拿分，总分反而领先。
        return float(sum(w * (v if np.isfinite(v) else 0.0) for w, v in terms))
    num = sum(w * v for w, v in terms if np.isfinite(v))
    den = sum(w for w, v in terms if np.isfinite(v))
    return float(num / den) if den > 0 else np.nan
