"""评分器单元测试（方案第 1 天的交付判据）。

五项：完美预测 / 常数预测 / 缩放预测 / 缺失掩码 / 多对照合并可切换。
运行：python test_metrics.py
"""
import sys
import numpy as np

from config import ScorerConfig
from metrics import (metric_absolute, metric_fc, metric_residual, metric_dep,
                     combine, pcc, r2)

RNG = np.random.default_rng(0)
PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("  [PASS] " if cond else "  [FAIL] ") + name + ("  " + detail if detail else ""))


def make_data(n=40, p=300, missing=0.0):
    y = RNG.normal(10, 2, size=(n, p))
    c = RNG.normal(10, 2, size=(n, p))
    if missing > 0:
        mask = RNG.random((n, p)) < missing
        y = y.copy(); y[mask] = np.nan
        mask2 = RNG.random((n, p)) < missing
        c = c.copy(); c[mask2] = np.nan
    return y, c


# ------------------------------------------------------------------ T1
print("\nT1 · 完美预测：各可定义项应为 1")
cfg = ScorerConfig(min_valid_points=5)
y, c = make_data()
yhat = y.copy()
a = metric_absolute(y, yhat, cfg)
check("absolute.pcc == 1", np.isclose(a["pcc"], 1.0), f"got {a['pcc']:.6f}")
check("absolute.r2  == 1", np.isclose(a["r2"], 1.0), f"got {a['r2']:.6f}")

d_true, d_pred = y - c, yhat - c
f = metric_fc(d_true, d_pred, cfg)
check("fc.pcc == 1", np.isclose(f["pcc"], 1.0), f"got {f['pcc']:.6f}")

mu = np.tile(d_true.mean(axis=0), (d_true.shape[0], 1))
rres = metric_residual(d_true, d_pred, mu, cfg)
check("residual.pcc == 1", np.isclose(rres["pcc"], 1.0), f"got {rres['pcc']:.6f}")

dep = metric_dep(d_true, d_pred, cfg)
check("dep.direction_acc == 1", np.isclose(dep["direction_acc"], 1.0), f"got {dep['direction_acc']:.6f}")

total = combine({"absolute": a, "fc": f, "ctx_resid": rres, "drug_resid": rres,
                 "both_time": f, "dep": dep}, cfg)
check("combine 总分接近 1", total > 0.95, f"got {total:.6f}")

# ------------------------------------------------------------------ T2
print("\nT2 · 常数预测：PCC 未定义，按配置分支")
const = np.full_like(y, 7.0)
cfg_nan = ScorerConfig(min_valid_points=5, undefined_pcc="nan")
cfg_zero = ScorerConfig(min_valid_points=5, undefined_pcc="zero")
v_nan = pcc(y[0], const[0], cfg_nan)
v_zero = pcc(y[0], const[0], cfg_zero)
check("undefined_pcc='nan' 返回 nan", np.isnan(v_nan))
check("undefined_pcc='zero' 返回 0", v_zero == 0.0)
check("两种配置结果不同（参数确实生效）", (np.isnan(v_nan) and v_zero == 0.0))

# ------------------------------------------------------------------ T3
print("\nT3 · 缩放预测 ŷ=2y：PCC 不变，R² 变差")
scaled = 2.0 * y
p_scaled = pcc(y[0], scaled[0], cfg)
r_scaled = r2(y[0], scaled[0], cfg)
check("PCC 仍为 1（对幅度不敏感）", np.isclose(p_scaled, 1.0), f"got {p_scaled:.6f}")
check("R² 明显 < 1（对幅度敏感）", r_scaled < 0.5, f"got {r_scaled:.4f}")

# ------------------------------------------------------------------ T4
print("\nT4 · 缺失掩码：成对完整，且有效点不足时返回 nan")
ym, cm = make_data(missing=0.3)
yhat_m = ym.copy()
a_m = metric_absolute(ym, yhat_m, cfg)
check("含 30% 缺失时完美预测仍为 1", np.isclose(a_m["pcc"], 1.0), f"got {a_m['pcc']:.6f}")

a_vec = np.array([1.0, 2.0, np.nan, 4.0])
b_vec = np.array([1.0, np.nan, 3.0, 4.0])
cfg_hi = ScorerConfig(min_valid_points=10)
check("有效点(2) < 阈值(10) 时返回 nan", np.isnan(pcc(a_vec, b_vec, cfg_hi)))
cfg_lo = ScorerConfig(min_valid_points=2)
check("有效点(2) >= 阈值(2) 时可计算", np.isfinite(pcc(a_vec, b_vec, cfg_lo)))

# ------------------------------------------------------------------ T5
print("\nT5 · 多对照合并：median / mean 可切换且结果不同")
ctrls = np.array([[1.0, 2.0, 3.0],
                  [1.0, 2.0, 9.0],
                  [1.0, 2.0, 3.0]])          # 第三列有离群值
med = np.nanmedian(ctrls, axis=0)
mea = np.nanmean(ctrls, axis=0)
check("median 抗离群 (=3.0)", np.isclose(med[2], 3.0), f"got {med[2]}")
check("mean  受离群影响 (=5.0)", np.isclose(mea[2], 5.0), f"got {mea[2]}")
check("两种合并方式结果不同", not np.allclose(med, mea))

ctrls_nan = ctrls.copy(); ctrls_nan[0, 0] = np.nan
check("合并时忽略 NaN", np.isclose(np.nanmedian(ctrls_nan, axis=0)[0], 1.0))

# ------------------------------------------------------------------ T6
print("\nT6 · 向量化 pcc_axis / r2_axis 必须与逐条 pcc / r2 完全等价")
from metrics import pcc_axis, r2_axis          # noqa: E402

for tag, (nn, pp, miss) in {
    "无缺失": (25, 120, 0.0),
    "30%缺失": (25, 120, 0.30),
    "高缺失": (25, 120, 0.70),
}.items():
    A = RNG.normal(0, 1, size=(nn, pp))
    B = 0.6 * A + RNG.normal(0, 1, size=(nn, pp))
    if miss > 0:
        A[RNG.random(A.shape) < miss] = np.nan
        B[RNG.random(B.shape) < miss] = np.nan
    for cfg_t in (ScorerConfig(min_valid_points=10),
                  ScorerConfig(min_valid_points=10, undefined_pcc="zero")):
        for ax, lab in ((1, "样本轴"), (0, "蛋白轴")):
            n_it = A.shape[0] if ax == 1 else A.shape[1]
            ref_p = np.array([pcc(A[i], B[i], cfg_t) if ax == 1
                              else pcc(A[:, i], B[:, i], cfg_t) for i in range(n_it)])
            got_p = pcc_axis(A, B, cfg_t, axis=ax)
            ref_r = np.array([r2(A[i], B[i], cfg_t) if ax == 1
                              else r2(A[:, i], B[:, i], cfg_t) for i in range(n_it)])
            got_r = r2_axis(A, B, cfg_t, axis=ax)
            okp = np.allclose(ref_p, got_p, equal_nan=True, atol=1e-9)
            okr = np.allclose(ref_r, got_r, equal_nan=True, atol=1e-7)
            check(f"pcc_axis 等价 [{tag}/{lab}/undef={cfg_t.undefined_pcc}]", okp,
                  "" if okp else f"maxdiff={np.nanmax(np.abs(ref_p-got_p)):.2e}")
            check(f"r2_axis  等价 [{tag}/{lab}/undef={cfg_t.undefined_pcc}]", okr,
                  "" if okr else f"maxdiff={np.nanmax(np.abs(ref_r-got_r)):.2e}")

# ---- 回归测试：log2 丰度量级（均值 ~20）下的数值相消 ----
# 一遍法 Σx²/n − (Σx/n)² 在这里会把常数向量的方差算成 ~1e-11 而不是 0，
# 于是「常数 → 未定义」判据失效，返回纯舍入误差决定的假相关。2026-08-05 实际踩过。
n_s, n_p = 200, 400
truth = RNG.normal(20.0, 2.8, size=(n_s, n_p))       # 真实 log2 丰度量级
const_pred = np.tile(truth.mean(axis=0), (n_s, 1))   # 全局均值谱：逐蛋白恒定
cfg_r = ScorerConfig(min_valid_points=30)
got = pcc_axis(truth, const_pred, cfg_r, axis=0)     # 蛋白轴：预测是常数
ref = np.array([pcc(truth[:, j], const_pred[:, j], cfg_r) for j in range(n_p)])
check("常数预测在 log2 量级下蛋白轴全部未定义（不得返回假相关）",
      np.all(np.isnan(got)),
      f"finite={int(np.isfinite(got).sum())}/{n_p}")
check("与逐条版一致（log2 量级 / 常数预测）", np.allclose(ref, got, equal_nan=True))

truth_m = truth.copy()
truth_m[RNG.random(truth.shape) < 0.3] = np.nan
for ax in (0, 1):
    n_it = truth.shape[1] if ax == 0 else truth.shape[0]
    ref_p = np.array([pcc(truth_m[:, i] if ax == 0 else truth_m[i],
                          truth_m[:, i] * 1.3 + 5 if ax == 0 else truth_m[i] * 1.3 + 5,
                          cfg_r) for i in range(n_it)])
    got_p = pcc_axis(truth_m, truth_m * 1.3 + 5, cfg_r, axis=ax)
    ref_r = np.array([r2(truth_m[:, i] if ax == 0 else truth_m[i],
                         truth_m[:, i] * 1.3 + 5 if ax == 0 else truth_m[i] * 1.3 + 5,
                         cfg_r) for i in range(n_it)])
    got_r = r2_axis(truth_m, truth_m * 1.3 + 5, cfg_r, axis=ax)
    check(f"log2 量级 + 30%缺失下 pcc_axis 等价 [axis={ax}]",
          np.allclose(ref_p, got_p, equal_nan=True, atol=1e-9))
    check(f"log2 量级 + 30%缺失下 r2_axis 等价 [axis={ax}]",
          np.allclose(ref_r, got_r, equal_nan=True, atol=1e-6))

# 常数向量：两条路径都要落到 undefined_pcc 分支
Ac = np.ones((6, 60)); Bc = RNG.normal(0, 1, size=(6, 60))
cfg_z = ScorerConfig(min_valid_points=10, undefined_pcc="zero")
check("常数行：向量化也返回 0.0",
      np.allclose(pcc_axis(Ac, Bc, cfg_z, axis=1), 0.0))
cfg_n = ScorerConfig(min_valid_points=10, undefined_pcc="nan")
check("常数行：向量化也返回 nan",
      np.all(np.isnan(pcc_axis(Ac, Bc, cfg_n, axis=1))))

# ------------------------------------------------------------------ T7
print("\nT7 · Pro R2 L1-02/L1-03 三处修复（每条都曾产出错误分数）")
from metrics import metric_both_time, _both_axes, last_axis_detail   # noqa: E402
import os, tempfile                                                   # noqa: E402

# --- L1-03a：float32 落盘后 (C+μ)−C 的残差必须判为常数，不得算出假 PCC ---
n_s, n_p = 60, 800
C = RNG.normal(20.0, 2.8, size=(n_s, n_p)).astype(np.float32)
mu = RNG.normal(0.0, 0.35, size=n_p).astype(np.float32)
with tempfile.TemporaryDirectory() as td:
    fp = os.path.join(td, "pred.npy")
    np.save(fp, (C + mu).astype(np.float32))       # 真实提交也是有限精度
    y_pred = np.load(fp)
D_pred = y_pred.astype(np.float64) - C.astype(np.float64)
resid_pred = D_pred - mu                            # 理论恒为 0
resid_true = RNG.normal(0, 0.3, size=(n_s, n_p))
cfg_c = ScorerConfig(min_valid_points=30)
mx = float(np.abs(resid_pred).max())
check("float32 往返后残差确实非零（说明这个坑真实存在）", mx > 0,
      f"max|resid| = {mx:.2e}")
got = pcc_axis(resid_true, resid_pred, cfg_c, axis=1)
check("相对阈值下：恒零残差判为常数 → 全部未定义，不产出假相关",
      np.all(np.isnan(got)), f"finite={int(np.isfinite(got).sum())}/{n_s}")
cfg_abs = ScorerConfig(min_valid_points=30, const_atol=1e-12, const_rtol=0.0)  # 旧阈值
old = pcc_axis(resid_true, resid_pred, cfg_abs, axis=1)
check("旧的 1e-12 绝对阈值确实会产出假相关（回归对照）",
      int(np.isfinite(old).sum()) > 0, f"finite={int(np.isfinite(old).sum())}/{n_s}")
# 真实 Δ 尺度（sd≈0.4）必须不受新地板影响
real = RNG.normal(0, 0.4, size=(n_s, n_p))
check("真实 Δ 尺度不被新地板误伤",
      int(np.isfinite(pcc_axis(real, real * 1.7, cfg_c, axis=1)).sum()) == n_s)

# --- L1-02：整条轴未定义不得被静默踢出 ---
truth = RNG.normal(20.0, 2.8, size=(n_s, n_p))
const_pred = np.tile(truth.mean(axis=0), (n_s, 1))    # 逐蛋白恒定 → 蛋白轴全未定义
v_zero = _both_axes(truth, const_pred, ScorerConfig(min_valid_points=30,
                                                    undefined_axis="zero"), pcc)
det = last_axis_detail()
v_drop = _both_axes(truth, const_pred, ScorerConfig(min_valid_points=30,
                                                    undefined_axis="drop"), pcc)
check("undefined_axis=drop 时退化模型只按样本轴计分（旧行为）",
      np.isclose(v_drop, det["sample"]), f"drop={v_drop:.4f} 样本轴={det['sample']:.4f}")
check("undefined_axis=zero 时蛋白轴记 0 后等权平均",
      np.isclose(v_zero, det["sample"] / 2), f"zero={v_zero:.4f}")
check("zero 严格低于 drop（退化模型不再白拿一档）", v_zero < v_drop,
      f"{v_zero:.4f} < {v_drop:.4f}")
check("两轴原值与有效计数可取用", det["n_protein_valid"] == 0 and det["n_sample_valid"] > 0,
      f"蛋白轴有效 {det['n_protein_valid']}/{det['n_protein']}，"
      f"样本轴有效 {det['n_sample_valid']}/{det['n_sample']}")

# --- L1-03b：指标 5 必须同时含 FC 与绝对保真度 ---
y_t = RNG.normal(20.0, 2.8, size=(40, 600))
y_p = y_t + RNG.normal(0, 0.5, size=y_t.shape)
d_t = RNG.normal(0, 0.4, size=y_t.shape)
d_p = d_t + RNG.normal(0, 0.4, size=y_t.shape)
cfg_bt = ScorerConfig(min_valid_points=30)
bt = metric_both_time(y_t, y_p, d_t, d_p, cfg_bt)
check("指标 5 含绝对分量（abs_pcc / abs_r2 非 nan）",
      np.isfinite(bt["abs_pcc"]) and np.isfinite(bt["abs_r2"]))
check("指标 5 的合成值 ≠ 单纯 FC", not np.isclose(bt["pcc"], bt["fc"]),
      f"合成 {bt['pcc']:.4f} vs 仅FC {bt['fc']:.4f}")
bt2 = metric_both_time(y_t, y_p, d_t, d_p,
                       ScorerConfig(min_valid_points=30, both_time_parts="fc_only"))
check("fc_only 可切回旧口径", np.isclose(bt2["pcc"], bt2["fc"]))

# ------------------------------------------------------------------ 汇总
print("\n" + "=" * 60)
print("PASS %d / %d" % (len(PASS), len(PASS) + len(FAIL)))
if FAIL:
    print("FAILED:")
    for n in FAIL:
        print("  - " + n)
    sys.exit(1)
print("全部通过")
