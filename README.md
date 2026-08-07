# 虚拟酵母扰动蛋白质组预测

GOAI 世界人工智能开源大赛 · 赛道三 AI for Research · 算法赛题方向一

给定酿酒酵母在特定菌株、化合物、培养基、温度、时间与测量上下文下的扰动蛋白质组数据，
预测独立测试样本的完整 5,243 维 log2 蛋白丰度向量。

方案说明见 [`初赛方案说明文档.md`](初赛方案说明文档.md)。

## 这个仓库里有什么

| 目录 | 内容 |
|---|---|
| `scripts/scorer/` | 官方六项指标的复刻实现、评估台、权威结果表生成器（61 项单元测试） |
| `scripts/data/` | 数据管线、对照匹配、诊断脚本、三类合规与一致性扫描器（28 项单元测试） |
| `scripts/models/` | 基线阶梯、低秩分解、响应模型、C-free 可部署骨架、整化合物留出验证 |
| `scripts/figures/` | 出图 |
| `results/logs/` | 全部脚本的实际运行日志 |
| `results/AUTHORITATIVE.md` | 对外引用的唯一数字来源，机器生成，附口径与代码指纹 |
| `results/figures/` | 方案说明文档中的六张图（PNG + PDF） |

## 三条也许对别人有用的东西

**1 · 评分口径的复刻与参数化。** 官方六项指标有多处未定义（多对照如何合并、
常数向量导致相关系数未定义时怎么办、整条聚合轴未定义时怎么办、未定义模块如何进总分）。
我们把这些全做成参数而非写死，并配 61 项测试。开发过程中这套测试捕获了三类
会系统性改变模型排名的实现缺陷，其中两类记录在方案文档 4.1.1。

**2 · 共享参照效应。** 评分定义中预测与真值两侧减去同一条实测对照，
这会让一个对药物一无所知的模型在扰动指标上拿到不低的分数。
`scripts/audit/shared_reference_probe.py` 是这个现象的可复现实验。

**3 · 三道数字防线。** 数字靠手抄进文档，源头一改抄件不会跟着改。
`authoritative_results.py` 定义哪个数是对的，`stale_number_scan.py` 查作废值是否还在流通，
`doc_number_check.py` 查文档与权威表是否一致。

## 复现

```bash
python scripts/scorer/test_metrics.py           # 61/61
python scripts/data/test_control_match.py       # 28/28
python scripts/scorer/authoritative_results.py  # 重算权威数字表
python scripts/data/doc_number_check.py         # 核对文档与权威表一致
```

依赖：Python 3.13 + numpy / pandas / scipy / scikit-learn / RDKit / matplotlib。
无 GPU 依赖，无商业 API 调用。所有随机过程使用固定种子。

## 数据

**本仓库不含任何赛事数据。** 赛事数据受《选手参赛协议》第八条约束，不得再分发。
仓库内的运行日志已经过自动化脱敏扫描，不含蛋白丰度数值、完整化合物名单与菌株代号；
日志中出现的 `STRAIN_A` 等为占位符。

要复现需自行从组委会获取数据并置于 `data/raw/`。
`scripts/data/paths.py` 对测试集蛋白质组文件设有读取守卫，详见方案文档 5.3。

## 许可证

MIT
