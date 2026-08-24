# 虚拟酵母扰动蛋白质组预测

GOAI 世界人工智能开源大赛 · 赛道三 AI for Research · 算法赛题方向一

给定酿酒酵母在特定菌株、化合物、培养基、温度、时间与测量上下文下的扰动蛋白质组数据，
预测独立测试样本的完整 5,243 维 log2 蛋白丰度向量。

**所有对外数字只有一个来源**：[`results/AUTHORITATIVE.md`](results/AUTHORITATIVE.md)，
由 `scripts/scorer/authoritative_results.py` 现场重算生成。本 README 与任何文档
都只引用它，不复制数值——复制过的数字在源头更新后不会跟着改，这类漂移我们踩过。

## 快速开始

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows；Linux/macOS 用 source .venv/bin/activate
pip install -r requirements.txt

# 把组委会发的四个文件放进 data/raw/：
#   metadata_train_val.csv  proteome_raw_train_val.csv
#   metadata_test.csv       proteome_raw_test.csv

python scripts/run_all.py         # 一条命令跑完全流程，产出 prediction.csv
```

`run_all.py` 的步骤清单用 `python scripts/run_all.py --list` 看，
中途接续用 `--from step7`，单跑一步用 `--only step9`。
每步的完整 stdout 落在 `results/_run_logs/<步骤>.log`。

只想要提交文件：

```bash
python scripts/models/predict_test.py
# -> results/step10_submission/prediction.csv  （4,454 x 5,243）
# -> results/step10_submission/submission_manifest.json  （含预测文件 SHA-256 与冻结的设计 spec）
```

## 环境与依赖

| 项 | 值 |
|---|---|
| Python | 3.13.12（3.11+ 应可运行，只在 3.13.12 上实测） |
| 直接依赖 | numpy · pandas · scipy · scikit-learn · matplotlib · rdkit |
| 版本锁定 | [`requirements.txt`](requirements.txt) |
| GPU | 不需要 |
| 联网 | 不需要（外部资源已随仓库披露来源与校验值，见下） |
| 随机性 | 全部固定种子；`scripts/data/provenance.py` 可复算源码摘要与环境快照 |

外部公开资源（手册要求披露来源与版本）：
[`data/external/yeast1011/SOURCE.md`](data/external/yeast1011/SOURCE.md)
——1011 Yeast Genomes Project 的 SNP 距离矩阵，含下载地址、日期、SHA-256、
以及它在本项目里被用来做什么、**没有**被用来做什么。

## 训练边界（这次整改的重点）

手册第 17 页：

> 训练仅可使用 **train 划分**的蛋白质组标签，验证集用于模型选择；
> **验证集与测试集均不得参与训练，也不得用于估计任何统计量**
> （含保留蛋白列表与归一化参数）。

我们此前有两处不合规，都已修掉，并且把「合规」做成了可运行的检验而不是一句声明：

```bash
python scripts/audit/train_boundary_probe.py
```

它做两件事：

1. **静态扫描**——挡三类已经踩过的写法（全表建词表、全表算标准化参数、
   允许拟合行取 train 以外的值）。扫描器自己先过阳性对照（三类违规全中）
   与阴性对照（只在文字里谈论违规的干净文件零命中）；第一版就是栽在阴性对照上，
   把讲规则的文档字符串当成了违规代码。
2. **经验探针**——把 `split_final != 'train'` 的每一行都破坏掉
   （丰度换随机数、缺失掩码重画、类别字段换假水平、时间换乱数），从头再拟合一次。
   若模型真的只由 train 折决定，冻结的设计 spec、`mu` / `U` / `W`、保留蛋白列表，
   以及最终交出去的 4,454 × 5,243 预测矩阵，**必须逐位相同**。
   任意一处不同即退出码非 0。实际读数见
   [`results/logs/step0_compliance_train_boundary_probe.txt`](results/logs/step0_compliance_train_boundary_probe.txt)。

两处违规各值多少分，也量给出来了（`scripts/audit/quantify_l1_leak.py`）：
一处是程序性的、对成绩几乎没有影响，另一处真的改变了交出去的预测。
两者量级差很远，不能混着讲——具体数字见权威表与该脚本的输出。

## 这个仓库里有什么

| 目录 | 内容 |
|---|---|
| `scripts/scorer/` | 官方六项指标的复刻实现、评估台、权威结果表生成器（61 项单元测试） |
| `scripts/data/` | 数据管线、对照匹配、训练边界守卫、源码指纹、菌株基因组坐标、脱敏与泄漏扫描 |
| `scripts/models/` | 设计矩阵冻结/编码、基线阶梯、低秩分解、响应模型、C-free 可部署骨架、整化合物留出验证、未见菌株效应搬运 |
| `scripts/audit/` | 诊断脚本与两个合规检验（训练边界探针、违规量化） |
| `scripts/figures/` | 出图 |
| `scripts/run_all.py` | 全流程入口 |
| `scripts/release.py` | 发布到本仓库并写下可核验的 tag |
| `results/logs/` | 全部脚本的实际运行日志 |
| `results/AUTHORITATIVE.md` | 对外引用的唯一数字来源，附口径指纹、源码摘要与发布标识 |
| `results/figures/` | 方案说明文档中的图（PNG + PDF） |

## 四条也许对别人有用的东西

**1 · 评分口径的复刻与参数化。** 官方六项指标有多处未定义（多对照如何合并、
常数向量导致相关系数未定义时怎么办、整条聚合轴未定义时怎么办、未定义模块如何进总分）。
我们把这些全做成参数而非写死，并配 61 项测试。开发过程中这套测试捕获了三类
会系统性改变模型排名的实现缺陷。

**2 · 共享参照效应。** 评分定义中预测与真值两侧减去同一条实测对照，
这会让一个对药物一无所知的模型在扰动指标上拿到不低的分数。
`scripts/audit/shared_reference_probe.py` 是这个现象的可复现实验。

**3 · 把「合规」写成可运行的检验。** 见上一节。静态读代码看不出词表与标准化参数
的泄漏——上一轮我们就是这么漏掉的：拟合行确实只有 train，泄漏藏在别处。
破坏-重拟合-逐位比对是能判定的，谁都能跑一遍。

**4 · 三道数字防线。** 数字靠手抄进文档，源头一改抄件不会跟着改。
`authoritative_results.py` 定义哪个数是对的，`stale_number_scan.py` 查作废值是否还在流通，
`doc_number_check.py` 查文档与权威表是否一致。权威表还会硬核对
`results/step9_loco/loco.json` 的依赖代码指纹——对不上就**拒绝收录**，
免得旧模型的结论被带进新材料。

## 复现自检

```bash
python scripts/scorer/test_metrics.py            # 评分器单元测试
python scripts/data/test_control_match.py        # 对照匹配单元测试
python scripts/data/split_guard.py               # 训练边界守卫自检
python scripts/models/design.py                  # 设计矩阵冻结/编码自检
python scripts/data/provenance.py                # 源码摘要与环境
python scripts/data/strain_genome.py             # 外部基因组资源读取与核权重自检
python scripts/audit/train_boundary_probe.py     # 训练边界合规探针（全量）
```

## 数据

**本仓库不含任何赛事数据。** 赛事数据受《选手参赛协议》第八条约束，不得再分发。
仓库内的运行日志与文档在发布前会过 `scripts/data/desensitize.py`，
菌株代号与化合物名换成稳定占位符（`STRAIN_A` / `COMPOUND_01`，编号与初赛公开版一致），
再由 `scripts/data/pkg_leak_scan.py` 复扫一遍确认。源码本身不含任何赛事实体名。

要复现需自行从组委会获取数据并置于 `data/raw/`。

## 许可证

MIT
