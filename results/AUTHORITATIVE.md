# 权威结果表（机器生成，勿手改）

本文件由 `scripts/scorer/authoritative_results.py` 现场重算生成。
**所有对外文档只许引用本表，不许复制数字。**
口径或代码一变，重跑本脚本，全部引用同步更新。

## 口径指纹

| 项 | 值 |
|---|---|
| split | `官方 split_final；评估集 = val_chem_only + val_strain_only + val_both + val_time` |
| control_set | `Water + DMSO（质控样本单列，不作生物对照）` |
| control_agg | `median` |
| match_keys | `7 项，不含 protein_well` |
| undefined_pcc | `nan` |
| undefined_axis | `zero` |
| undefined_module | `zero` |
| both_time_parts | `fc_and_absolute` |
| const_atol | `1e-05` |
| min_valid_points | `30` |
| seeds | `{'shared_reference': 20260805, 'loco_folds': 20260805}` |

## 代码与发布指纹

- 源码摘要 `tree_digest` = **`46a32e9d2723ff45`**（`scripts/` 下 42 个 .py，路径排序后逐个 SHA-256 再汇总）
- 公开仓库发布状态：**已发布**，tag `semifinal-v1` · commit `f45c93294278e332fa3e1b74cff6e80f436d2b4f` · 与本次源码一致：**否**
- 运行环境：Python 3.13.12 · numpy 2.4.4 · pandas 3.0.2 · scipy 1.17.1 · scikit-learn 1.9.0 · rdkit 2024.09.5

<details><summary>逐文件 SHA-256 前 16 位</summary>

| 文件 | 摘要 |
|---|---|
| `audit/audit_control_match.py` | `5b5371a17c4a80cc` |
| `audit/check_test_labels.py` | `cf9b4255d04969aa` |
| `audit/diagnose_batch.py` | `13d8ca44b74df4a4` |
| `audit/diagnose_missing.py` | `5977e1465caa13e1` |
| `audit/noise_ceiling.py` | `6998b92fdb0c407c` |
| `audit/perturb_inventory.py` | `bbb44a403f50c7dc` |
| `audit/probe.py` | `5945504af6ce0ced` |
| `audit/quantify_l1_leak.py` | `10131ca0205c6292` |
| `audit/read_docx.py` | `ddf7c538e5cf5e8e` |
| `audit/shared_reference_probe.py` | `9f48ca2784e1ae20` |
| `audit/train_boundary_probe.py` | `7919f5b481cb85ed` |
| `data/control_match.py` | `bf63ab055a32f7fd` |
| `data/desensitize.py` | `30fc801e30ffdcfb` |
| `data/doc_number_check.py` | `02e63df6a5337131` |
| `data/fix_table_widths.py` | `e7af1530135e473e` |
| `data/loader.py` | `9551b0b77a3afeea` |
| `data/paths.py` | `012dddce52c7f8ad` |
| `data/pkg_leak_scan.py` | `4beecda54c318439` |
| `data/provenance.py` | `063995f20623437c` |
| `data/resolve_smiles.py` | `eb74476cdac63ced` |
| `data/split_guard.py` | `495e823d20e4bdca` |
| `data/stale_number_scan.py` | `f916541569dbde86` |
| `data/strain_genome.py` | `4ccd0041680392d4` |
| `data/test_control_match.py` | `5f7f03d99dff7ae2` |
| `eval_test/self_eval.py` | `4f78f3c134cfbf07` |
| `figures/make_figures.py` | `c33e250c3cbd905f` |
| `models/baseline_cfree.py` | `b0c6418afaa3f631` |
| `models/baselines.py` | `8afc1137f5a9c326` |
| `models/design.py` | `837b76beef58a555` |
| `models/loco_response.py` | `7ea5dfa078f86f50` |
| `models/lowrank.py` | `56df6a42963a9e17` |
| `models/predict_test.py` | `9d01988f37fbf3ac` |
| `models/response.py` | `bdb873a4a6c84480` |
| `models/select_k0.py` | `a289a8ca02bd59a3` |
| `models/strain_transport.py` | `84870811d44abe64` |
| `release.py` | `5bf504a8227ab8a4` |
| `run_all.py` | `eb7afa2a3e079c97` |
| `scorer/authoritative_results.py` | `9f25c90d6b00bbc6` |
| `scorer/config.py` | `bf372f32805b5918` |
| `scorer/evaluate.py` | `5b7ef2d00f41bd7d` |
| `scorer/metrics.py` | `89e06dccb4207170` |
| `scorer/test_metrics.py` | `43765ed05028e061` |

</details>

## 共享参照三条件（指标 2，零药物知识预测）

评估样本 2806（官方四类 val 的处理样本）

| 条件 | 样本轴 PCC | 蛋白轴 PCC |
|---|---|---|
| 正确匹配（官方口径） | 0.1845 | 0.1898 |
| 错配到同条件的别的样本 | 0.0825 | 0.0855 |
| 全局错配到随机样本 | 0.0076 | 0.0038 |

> R3-L1-08：相关系数不可加性分解。三条件之差只说明共享参照抬高了零知识基线，不得拆成可相加的信号份额。

## 数据实况（train_val）

| 项 | 值 |
|---|---|
| n_samples_train_val | 8958 |
| n_proteins | 5243 |
| missing_rate_abundance | 27.3534% |
| missing_rate_delta_treated | 28.9292% |
| all_missing_cols_abundance | 186 |
| all_missing_cols_delta | 213 |
| n_treated | 7884 |

## 模型分数（官方四类 val，本方复刻评分器）

### 需读测试对照者（oracle 诊断，**不可提交**）

| 模型 | total | abs_pcc | abs_r2 | fc_pcc |
|---|---|---|---|---|
| B0 全局均值谱 | 0.2587 | 0.4768 | 0.4138 | 0.1872 |
| B1 预测=匹配对照 | 0.2836 | 0.9036 | 0.8228 | 0.0000 |
| B2g 总体平均响应 | 0.3087 | 0.9040 | 0.8229 | 0.0502 |
| B2 上下文均值响应 | 0.3111 | 0.9049 | 0.8254 | 0.1720 |
| B4 ridge 响应 | 0.3381 | 0.9040 | 0.8062 | 0.1697 |
| B3 化学近邻响应 | 0.2898 | 0.8974 | 0.8071 | 0.0311 |
| B3o 神谕近邻(响应空间上限) | 0.3220 | 0.9006 | 0.8142 | 0.1150 |
| 对照收缩 α=0.15 | 0.3843 | — | — | — |

### C-free（推断不接触对照，**可提交**）

| 模型 | total | abs_pcc | abs_r2 | fc_pcc | ctx_resid | drug_resid |
|---|---|---|---|---|---|---|
| 全局均值谱 | 0.2587 | 0.4768 | 0.4138 | 0.1872 | 0.1324 | 0.2252 |
| 逐蛋白 ridge(满秩) | 0.4406 | 0.8947 | 0.7088 | 0.3052 | 0.2718 | 0.3417 |
| 低秩 K0=16 + ridge | 0.4694 | 0.8984 | 0.7985 | 0.3295 | 0.3115 | 0.3509 |

### 端到端选秩曲线

| K0 | total | abs_r2 | fc_pcc |
|---|---|---|---|
| K0=8 | 0.4505 | 0.7760 | 0.3108 |
| K0=16 | 0.4694 | 0.7985 | 0.3295 |
| K0=32 | 0.4691 | 0.7976 | 0.3290 |
| K0=96 | 0.4674 | 0.7941 | 0.3277 |

### 配对 bootstrap：低秩 K0=16 − 满秩 ridge

- 差值均值 **+0.0282**
- 95% 区间 **[+0.0211, +0.0346]**，排除 0
- 60 次重抽样，按 46 个化合物分组，低秩更优的比例 100.0%

> ⚠ 判据：两模型跑在同一批数据与同一批重抽样上，估计高度相关，故看**配对差值**的区间，而非比较两条边际区间是否重叠。

## 对照匹配与复制可靠性

| 项 | 值 |
|---|---|
| 处理样本数 | 7884 |
| 0 匹配率 | 0.0% |
| 多匹配比例 | 92.9%（中位 2 个对照）|
| 复制对数 | 4384 |

跨板/来源复制对的**操作性响应谱一致性** ρ（勿解释为信号占比）：

| 空间 | ρ |
|---|---|
| 绝对 log2 丰度 | 0.921 |
| Δ_true 匹配FC | 0.115 |
| Δ − μ_ctx | 0.157 |
| Δ − μ_drug | 0.059 |

## 混杂与缺失

| 项 | 值 |
|---|---|
| Cramér's V · Medium × Yeast_cell_plate | 0.992 |
| Cramér's V · Temperature × Yeast_cell_plate | 0.992 |
| Cramér's V · pert_time × Yeast_cell_plate | 0.992 |
| 生物+技术设计矩阵 | 296 列 / 秩 267 / 亏秩 **29** |
| Spearman(蛋白中位丰度, 缺失率) | -0.859 |
| 最低丰度十分位缺失率 | 77.4% |
| 最高丰度十分位缺失率 | 1.6% |

> 以上仅 train_val。测试集蛋白矩阵的对应统计来自隔离决定前的一次探查性读取，见合规披露，不在本表重算。

## 同一个量的两种呈现（勿误当成两个不同结果）

上表按轴分列；而基线阶梯里 B0 的 `fc_pcc` 报的是**两轴平均** = 0.1872。
两者是同一次计算的不同切面。零知识预测统一使用 `baselines.b0_global_mean`，
本表与基线表不再存在第二套「全局均值谱」定义。

## 化合物表示的 nested LOCO（train 折内，宇宙 = split_final=='train'）

- 合法训练化合物 **37** 个；外层 8 折整化合物留出；train 行 5920（处理行 5078）

| 配置 | fc_pcc | 相对仅上下文 |
|---|---|---|
| 仅上下文 b | 0.3521 | — |
| + ECFP 结构表示 | 0.3526 | +0.0004 |
| [阳性对照2] 神谕特征走同一模块 | — | +0.0217 |
| [阳性对照] 神谕残差照搬 | — | +0.0300 |
| [上限] 神谕自身平均残差 | — | +0.0674 |

- shuffled-label 对照 5 次：均值 +0.0002 / 最大 +0.0007 → 真增益**未超过**全部打乱对照
- 依赖代码一致性核对：通过

官方 `val_chem_only` 一次性确认（6 个化合物，其中 6 个有 SMILES；lambda=2000 取自内层众数）：

| 配置 | total | fc_pcc | ctx_resid |
|---|---|---|---|
| 仅上下文 b | 0.3547 | 0.3505 | 0.3115 |
| + ECFP | 0.3549 | 0.3509 | 0.3118 |
| **差** | +0.0002 | +0.0004 | +0.0003 |

> 只有 6 个留出化合物，这一格不确定性很大，只作方向一致性核对。

## 未见菌株的效应搬运（外部基因组资源，按预注册判据裁决）

- 外部资源：Peter et al., Nature 2018, 556, 339-344（1,011 株酿酒酵母基因组）；文件 `data/external/yeast1011/1011DistanceMatrixBasedOnSNPs.tab.gz`，SHA-256 `140da4e5193584c0…`，面板 1011 株
- 训练菌株 4 株，其中有面板坐标的 3 株（供体池）
- 未见菌株到最近供体 0.398（面板分位 14.6%，三供体跨度 1.701）；val 菌株 1.362（分位 73.8%，跨度 0.414）

| 阶段 | 方案 | drug_resid | 相对现状 |
|---|---|---|---|
| train 内 LOSO | 现状（未见菌株整块 0） | 0.2986 | — |
| train 内 LOSO | 等权（无基因组信息） | 0.2981 | -0.0005 |
| train 内 LOSO | SNP 距离核 | 0.2982 | -0.0005 |
| val_strain_only 一次性 | 现状 | 0.3509 | +0.0000 |
| val_strain_only 一次性 | 等权 | 0.3496 | -0.0013 |
| val_strain_only 一次性 | SNP 距离核 | 0.3499 | -0.0010 |

- 预注册判据（写在 `_handoff/CURRENT.md`，不是事后定的）：drug_resid 增量 ≥ +0.015 或六项总分增量 ≥ +0.003，且 abs_r2 与 S1 abs_r2 下降不超过 0.005
- 实测：drug_resid -0.0010 · 总分 -0.0008 · abs_r2 -0.0014 · S1 abs_r2 +0.0000
- **裁决：不保留，按预案永久停止该路线**

> ⚠ 可验证性是不对称的：加权最可能起作用的那一格（未见菌株，与某训练菌株近缘）无法验证；唯一能验证的那一格（val 菌株）到三个供体近乎等距，验不出加权本身。上表能支持的结论只到「搬运机制本身在本数据上不带来增益」，不能反过来说「菌株基因组对该任务无用」。

## 已知的作废数字（勿再引用）

| 作废值 | 出处 | 现行值 |
|---|---|---|
| B0 abs_pcc 0.9535 | 未定义轴被静默踢出时的值 | 0.4768（undefined_axis=zero） |
| B2 ctx_resid 0.0011 | float32 舍入噪声被当真信号 | 0.0000 |
| 修评分器前的整张基线表（B0 0.2928 / B2 0.2761 / B3 0.2479 / B4 0.3026 / α 0.3473） | 未定义轴与常数判据两个 bug 修复前 | 见 docs/06 §11.2 修正后表 |
| 加权上限 ≈ 0.42 | 由 √ρ 推出，前提不成立（预测与真值共享对照噪声） | **已作废，不替换为任何单一数字** |
| C-free 低秩 K0=16 总分 0.4694 | 设计矩阵词表与 log-time 标准化参数在**全表**估计（L1-1） | 见本表「C-free」一节 |
| LOCO Morgan 增益 −0.0004 / 神谕近邻 +0.0237 / 神谕自身 +0.0617 | 外层留出宇宙含 val 化合物（L1-2） | 见本表 LOCO 一节 |
| 提交模型用全部 train_val 拟合 | `predict_test.py --fit-rows all`，违反手册第 17 页 | 选项已删除，只拟合 train 折 |
