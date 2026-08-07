"""评分器配置。

官方口径尚有未定项（多对照合并、缺失掩码、常数向量、聚合方式），
因此全部做成参数，不写死。默认值是当前的最佳猜测，见 docs/05_训练方案.md。
"""
from dataclasses import dataclass, field
from typing import List, Literal


# 手册规定的对照匹配键：来源/菌株/培养基/温度/时间/仪器/板号（不含 well）
MATCH_KEYS: List[str] = [
    "data_source", "Strains", "Medium", "Temperature",
    "pert_time", "instrument", "Yeast_cell_plate",
]

# 非化合物条目
SOLVENT_CONTROLS: List[str] = ["Water", "DMSO"]
QC_LABEL: str = "Quality Control"
NON_COMPOUND: List[str] = SOLVENT_CONTROLS + [QC_LABEL]


@dataclass
class ScorerConfig:
    # ---- 对照 ----
    control_names: List[str] = field(default_factory=lambda: list(SOLVENT_CONTROLS))
    # 多个合法对照如何合并（官方未定义）
    control_agg: Literal["median", "mean"] = "median"

    # ---- 缺失 ----
    # 绝对指标：只用真值非缺失位置
    # FC / 残差：只用 treat 与 control 共同非缺失位置
    drop_all_missing_proteins: bool = True   # 186-189 个全缺失蛋白是否排除
    min_valid_points: int = 30               # 单次 PCC 所需最少有效点

    # ---- 常数向量导致 PCC 未定义时（官方未定义）----
    undefined_pcc: Literal["nan", "zero"] = "nan"

    # 判「常数向量」的阈值：sd <= max(const_atol, const_rtol * RMS)
    #
    # ❗const_atol 必须按 float32 在 log2 丰度量级上的表示精度来定，不能用 1e-12。
    # 预测按 float32 落盘（真实提交也是有限精度），(C + μ) − C 会残留舍入噪声：
    # float32 eps ≈ 1.2e-7，乘以 log2 丰度量级（~20）≈ 2.4e-6，取 1e-5 留裕量。
    # Δ 的真实尺度是 sd ≈ 0.4，比这个地板大四个数量级，不会误伤真信号。
    #
    # 注意：不能用「相对自身 RMS」做判据——纯舍入噪声向量的 sd ≈ RMS，
    # 比值恒为 1，永远触发不了。const_rtol 只用于兜住量级远大于 log2 丰度的输入。
    #
    # 2026-08-05 实际踩过：B2 的上下文残差理论恒为 0，却在舍入噪声上算出 0.0011。
    const_atol: float = 1e-5
    const_rtol: float = 1e-9

    # 某个评分模块整体未定义时，总分怎么算（官方未定义）
    #   "renorm" = 踢出分母（可能奖励「让指标未定义」的退化模型）
    #   "zero"   = 按 0 计入，分母保持满权重
    # 做模型比较时必须用 "zero"，否则 Δ≡0 这类基线会靠未定义指标虚高。
    undefined_module: Literal["renorm", "zero"] = "zero"

    # 某一整条轴（样本轴或蛋白轴）全未定义时怎么办（官方未定义）
    #   "drop" = 踢出平均（会奖励"让整条轴未定义"的退化模型）
    #   "zero" = 记 0 后与另一轴等权平均
    # 同 undefined_module 的道理：全局均值谱模型在每个蛋白上都是常数，
    # 蛋白轴全部未定义，"drop" 会让它只按样本轴计分而白拿一档。
    undefined_axis: Literal["drop", "zero"] = "zero"

    # 指标 5（双重未知 / 时间外推，10%）的构成。
    # 手册："test_both 以原始 FC + 绝对保真度为主；test_time 以绝对保真度 + 原始 FC 为主"
    # → 两个分量都要算，不能只算 FC。
    both_time_parts: Literal["fc_and_absolute", "fc_only"] = "fc_and_absolute"

    # ---- 聚合 ----
    # 样本轴与蛋白轴分别聚合后如何组合（官方未定义）
    axis_combine: Literal["mean", "sample_only", "protein_only"] = "mean"
    agg: Literal["mean", "median"] = "mean"

    # ---- DEP ----
    dep_threshold: float = 1.0   # |Δ_true| > 1
    dep_top_k: int = 100         # Recall@K

    # ---- 权重（手册固定，不应修改）----
    w_absolute: float = 0.20
    w_fc: float = 0.25
    w_ctx_resid: float = 0.20
    w_drug_resid: float = 0.20
    w_both_time: float = 0.10
    w_dep: float = 0.05

    def check_weights(self) -> None:
        s = (self.w_absolute + self.w_fc + self.w_ctx_resid
             + self.w_drug_resid + self.w_both_time + self.w_dep)
        assert abs(s - 1.0) < 1e-9, f"weights must sum to 1, got {s}"
