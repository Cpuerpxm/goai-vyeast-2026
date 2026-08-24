"""路径常量 + 测试标签物理隔离守卫。

`data/raw/proteome_raw_test.csv` 含完整蛋白丰度。手册 2026-08 修订版第 15 页已明确：

> 测试集蛋白质组真值随数据包一并发布，**供参赛队自评参考，不作最终排名依据**。

所以它**可以**用于自评（CLAUDE.md R2 已按此重写），但绝不能回流进拟合或选参。
隔离因此从「一律禁读」改成「按用途分两个入口」，并且靠代码结构实现，不靠记性：

- `assert_readable()` —— 所有常规入口走它，见到测试蛋白文件一律抛异常。
  建模、评估台、出图、提交脚本全部只能走这条路。
- `assert_readable_selfeval()` —— 唯一的例外口子，且有三重限制：
  调用方必须在 `scripts/eval_test/` 下、必须显式传令牌、必须已经存在冻结好的
  提交清单（即模型已经定死）。三条缺一就抛异常。

这样「模型冻结之后才允许看 test 真值」这件事是被强制的，而不是写在注释里的承诺。
"""
from __future__ import annotations

import os

_HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_ROOT = os.path.dirname(_HERE)                 # .../scripts
PROJECT_ROOT = os.path.dirname(SCRIPTS_ROOT)          # .../GOAI_VYEAST

DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_EXTERNAL = os.path.join(PROJECT_ROOT, "data", "external")
RESULTS = os.path.join(PROJECT_ROOT, "results")
CACHE = os.path.join(RESULTS, "_cache")

META_TRAIN_VAL = os.path.join(DATA_RAW, "metadata_train_val.csv")
PROT_TRAIN_VAL = os.path.join(DATA_RAW, "proteome_raw_train_val.csv")
META_TEST = os.path.join(DATA_RAW, "metadata_test.csv")

# 刻意不定义 PROT_TEST。若将来组委会书面确认测试标签可用，
# 在此处新增常量并同步更新 CLAUDE.md R2 与 PROJECT_CHARTER 变更日志。

FORBIDDEN_BASENAMES = ("proteome_raw_test.csv",)


class TestLabelLeakError(RuntimeError):
    """试图读取被隔离的测试集真值文件。"""


def assert_readable(path: str) -> str:
    """任何数据读取入口都必须先过这道守卫。"""
    base = os.path.basename(str(path)).lower()
    if base in FORBIDDEN_BASENAMES:
        raise TestLabelLeakError(
            f"{base} 在分支 A 下被物理隔离（CLAUDE.md R2）：它含完整蛋白丰度，"
            "与手册「真值标签保留」矛盾，组委会书面答复到达前不得进入任何环节。"
            "测试侧只读 metadata_test.csv。"
        )
    return path


#: 自评入口的令牌。写成一句完整的话，是为了让任何人在代码里看到它就知道前提是什么。
SELF_EVAL_TOKEN = "MODEL_ALREADY_FROZEN_TEST_TRUTH_FOR_SELF_EVAL_ONLY"

#: 冻结凭证：`predict_test.py` 成功落盘后写的清单。没有它就说明模型还没定死。
SUBMISSION_MANIFEST = os.path.join(
    RESULTS, "step10_submission", "submission_manifest.json")

PROT_TEST_SELFEVAL = os.path.join(DATA_RAW, "proteome_raw_test.csv")


class SelfEvalGateError(RuntimeError):
    """在模型冻结之前、或从不该碰它的地方，试图读取测试集真值。"""


def assert_readable_selfeval(path: str, token: str, caller_file: str) -> str:
    """测试集真值的唯一读取口。三重限制，缺一即抛异常。

    手册允许拿 test 真值自评；不允许的是让它影响模型。所以这里卡的不是「能不能读」，
    而是「在什么时点、由谁读」：模型必须先冻结（提交清单已落盘），
    读取方必须是 `scripts/eval_test/` 下的自评脚本，且必须显式写出令牌。
    """
    if token != SELF_EVAL_TOKEN:
        raise SelfEvalGateError(
            "读取测试集真值必须显式传 paths.SELF_EVAL_TOKEN。"
            "传不出这个令牌，说明调用方并不清楚自己在做什么。")
    caller = os.path.abspath(str(caller_file)).replace("\\", "/")
    allowed = os.path.join(SCRIPTS_ROOT, "eval_test").replace("\\", "/") + "/"
    if not caller.startswith(allowed):
        raise SelfEvalGateError(
            f"只有 scripts/eval_test/ 下的自评脚本可以读测试集真值，"
            f"当前调用方是 {caller}。建模与评估台一律走 assert_readable()。")
    if not os.path.exists(SUBMISSION_MANIFEST):
        raise SelfEvalGateError(
            f"缺 {SUBMISSION_MANIFEST}：模型还没冻结。"
            "先跑 scripts/models/predict_test.py 把提交文件与清单落盘，再自评。")
    return path


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def bootstrap_sys_path() -> None:
    """让 scripts/ 下的模块能互相 import（scripts 作为 import 根）。"""
    import sys

    if SCRIPTS_ROOT not in sys.path:
        sys.path.insert(0, SCRIPTS_ROOT)
