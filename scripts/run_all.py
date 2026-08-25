"""一条命令从官方数据跑到 prediction.csv。

复赛提交物第 1 项要求「可运行代码仓库（含 README、环境配置、运行说明）」，
且晋级队伍要过代码复现核验。所以入口只有一个，顺序写死在这里，不靠 README
里的一串手抄命令——手抄的顺序会过期，代码不会。

前置：`data/raw/` 下有官方四个文件（metadata_train_val.csv /
proteome_raw_train_val.csv / metadata_test.csv / proteome_raw_test.csv）。

其中 `proteome_raw_test.csv` 在**建模与评估的每一步都读不到**：
`data/paths.assert_readable` 一律拦截。只有最后一步 `step12`（自评）能打开它，
而那一步的门禁要求提交清单已经落盘——也就是模型已经冻结。
手册允许拿 test 真值自评，不允许它影响模型；这个先后顺序就是那条界线。

用法：
    python run_all.py                 # 全流程
    python run_all.py --from step7    # 从某一步接着跑
    python run_all.py --only step9    # 只跑一步
    python run_all.py --list          # 看步骤清单
"""
from __future__ import annotations

import argparse
import datetime
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from data import paths

#: (步骤号, 说明, 相对 scripts/ 的脚本路径, 额外参数)
STEPS = [
    ("step2", "对照匹配 -> Delta_true", "data/control_match.py", []),
    ("step3a", "批次混杂诊断", "audit/diagnose_batch.py", []),
    ("step3b", "缺失机制诊断", "audit/diagnose_missing.py", []),
    ("step3c", "复制一致性与噪声地板", "audit/noise_ceiling.py", []),
    ("step3d", "共享参照探针", "audit/shared_reference_probe.py", []),
    ("step4", "基线阶梯 B0-B4", "models/baselines.py", []),
    ("step5", "掩码低秩分解与秩上限核对", "models/lowrank.py", []),
    ("step6a", "响应模型 phi=onehot", "models/response.py", ["--phi", "onehot"]),
    ("step6b", "响应模型 phi=morgan", "models/response.py", ["--phi", "morgan"]),
    ("step7", "C-free 可部署骨架", "models/baseline_cfree.py", []),
    ("step8", "端到端选秩 K0", "models/select_k0.py", []),
    ("step9", "nested LOCO（train 折内 37 个化合物）", "models/loco_response.py", []),
    ("step11", "未见菌株效应搬运（外部基因组资源）", "models/strain_transport.py", []),
    ("step0", "训练边界合规探针", "audit/train_boundary_probe.py", []),
    ("leak", "两处违规各值多少分", "audit/quantify_l1_leak.py", []),
    ("auth", "权威结果表", "scorer/authoritative_results.py", []),
    ("fig", "六张图", "figures/make_figures.py", []),
    ("step10", "提交文件 prediction.csv", "models/predict_test.py", []),
    # 自评必须排在 step10 之后：它读 test 真值，而门禁要求
    # `results/step10_submission/submission_manifest.json` 已经存在（模型已冻结）。
    ("step12", "test 真值自评（手册允许，不作排名依据）", "eval_test/self_eval.py", []),
    # 数字防线放最后：前面每一步都跑完、日志都落盘了，才谈得上"文档里的数能追到出处"。
    # 2026-08-25 补进流水线——此前这两个脚本谁都没调，第三道防线的默认目标还指着
    # 初赛冻结稿，等于复赛材料一次没查过，而报告里写着"每个数都经程序化比对"。
    ("stale", "作废数字是否还在流通", "data/stale_number_scan.py", []),
    ("docnum", "文档数字能否追到出处", "data/doc_number_check.py", []),
]

LOG_DIR = os.path.join(paths.RESULTS, "_run_logs")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", help="从哪一步开始（含）")
    ap.add_argument("--only", help="只跑这一步")
    ap.add_argument("--list", action="store_true")
    # 默认一步失败就停。写成 --keep-going 而不是 --stop-on-error，
    # 是因为 `action="store_true", default=True` 那种写法**永远关不掉**，
    # 看起来像个开关其实不是（2026-08-24 自查发现）。
    ap.add_argument("--keep-going", action="store_true",
                    help="某步失败也继续往下跑（默认失败即停）")
    ap.add_argument("--skip-external", action="store_true",
                    help="跳过依赖外部资源的步骤（化合物结构 / 菌株基因组）")
    args = ap.parse_args()
    stop_on_error = not args.keep_going

    if args.list:
        for k, desc, script, extra in STEPS:
            print(f"  {k:<8}{desc:<34}{script} {' '.join(extra)}")
        return

    todo = STEPS
    if args.only:
        todo = [s for s in STEPS if s[0] == args.only]
        if not todo:
            raise SystemExit(f"没有这一步：{args.only}")
    elif args.start:
        names = [s[0] for s in STEPS]
        if args.start not in names:
            raise SystemExit(f"没有这一步：{args.start}")
        todo = STEPS[names.index(args.start):]
    # ❗过滤必须放在 --only / --from 之后：那两个分支会整个重置 todo，
    # 放前面的话 `--skip-external --from step4` 会把过滤悄悄丢掉。
    if args.skip_external:
        # 这四步要么要化合物结构（Morgan 指纹），要么要 1011 SNP 距离矩阵
        drop = {"step4", "step6b", "step9", "step11"}
        todo = [s for s in todo if s[0] not in drop]
        print(f"[--skip-external] 跳过 {sorted(drop)}；"
              "权威表会因缺 LOCO/菌株结果而拒绝收录，属预期")

    # ---- 外部资源前置检查 ----
    # ❗2026-08-25（GPT Pro R6 · L1-03）：此前干净 clone 照着 README 做，会一路跑到
    # step6b / step9 / step11 才因为缺 compound_smiles.csv 或 SNP 矩阵而失败，
    # 而那时已经烧掉半小时。失败要早、要指名道姓说缺什么、怎么补。
    need_external = [k for k, _, _, _ in todo
                     if k in ("step4", "step6b", "step9", "step11")]
    if need_external:
        import subprocess as _sp

        chk = _sp.run([sys.executable, os.path.join(HERE, "setup_external.py"), "--check"],
                      capture_output=True, text=True, encoding="utf-8")
        if chk.returncode != 0:
            print(chk.stdout)
            print("=" * 88)
            print("❗外部资源未就绪，以下步骤会失败：" + " / ".join(need_external))
            print("   1) 先把随提交物交付的三个文件放进 data/external/：")
            print("        compound_smiles.csv  compound_aliases.json  entity_alias.json")
            print("   2) 再跑一次（需联网，只这一次；只剩 SNP 矩阵要下载）：")
            print("       python scripts/setup_external.py")
            print("   只想跑不依赖外部资源的部分：")
            print("       python scripts/run_all.py --skip-external")
            print("=" * 88)
            sys.exit(2)
        print("[前置] 外部资源已就绪（1011 SNP 矩阵 + 54/54 化合物结构）", flush=True)

    os.makedirs(LOG_DIR, exist_ok=True)
    t_all = time.time()
    results = []
    for key, desc, script, extra in todo:
        p = os.path.join(HERE, script)
        print("=" * 88)
        print(f"[{key}] {desc}   ->   {script} {' '.join(extra)}")
        print("=" * 88, flush=True)
        t0 = time.time()
        log = os.path.join(LOG_DIR, f"{key}.log")
        with open(log, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(f"# {datetime.datetime.now().isoformat(timespec='seconds')}  "
                     f"{script} {' '.join(extra)}\n")
            fh.flush()
            # `-u`：stdout 重定向到文件时 Python 默认整块缓冲，跑十几分钟的步骤
            # 在日志里一个字都看不到，没法判断是在算还是卡死（2026-08-24 实测）。
            r = subprocess.run([sys.executable, "-u", p] + extra, stdout=fh,
                               stderr=subprocess.STDOUT, text=True, encoding="utf-8")
        dt = time.time() - t0
        ok = r.returncode == 0
        results.append((key, ok, dt, log))
        print(f"  {'PASS' if ok else 'FAIL'}  {dt:.0f}s   日志 {log}", flush=True)
        if not ok:
            print("  ---- 日志末尾 ----")
            tail = open(log, encoding="utf-8").read().splitlines()[-25:]
            print("\n".join("  " + x for x in tail), flush=True)
            if stop_on_error:
                break

    print("=" * 88)
    print(f"总耗时 {time.time() - t_all:.0f}s")
    for key, ok, dt, log in results:
        print(f"  {key:<8}{'PASS' if ok else 'FAIL':<6}{dt:>7.0f}s   {log}")
    sys.exit(0 if all(ok for _, ok, _, _ in results) else 1)


if __name__ == "__main__":
    main()
