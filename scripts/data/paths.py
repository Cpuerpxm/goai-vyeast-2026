"""路径常量 + 测试标签物理隔离守卫。

CLAUDE.md R2：`data/raw/proteome_raw_test.csv` 含完整蛋白丰度，与手册「真值标签
保留」的表述矛盾，在组委会书面答复到达前不得进入任何环节。

本模块**刻意不提供该文件的路径常量**，并对任何试图读取它的调用抛异常。
隔离靠代码结构实现，不靠「记得不用」。
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


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def bootstrap_sys_path() -> None:
    """让 scripts/ 下的模块能互相 import（scripts 作为 import 根）。"""
    import sys

    if SCRIPTS_ROOT not in sys.path:
        sys.path.insert(0, SCRIPTS_ROOT)
