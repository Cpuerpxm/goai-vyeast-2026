"""把当前代码与结果发布到公开仓库，并写下可核验的发布标识。

为什么需要它（2026-08-24 复赛整改 L1-3）：

复赛门槛是「晋级队伍须通过代码复现核验」。核验的人手里只有
`https://github.com/Cpuerpxm/goai-vyeast-2026`。此前权威表里记的
`_git_head = b0a309dce` 是本地大仓库 `E:/Claude_Code` 的提交号——
公开仓库解析不了，等于没记。而且本地 `GOAI_VYEAST/` 根本不是独立 git 仓库，
不存在「在这里 git tag」这个动作。

所以发布走这个脚本：把要公开的文件同步进公开仓库的一份克隆，打 tag 推上去，
再把 tag 与 commit 写回项目根的 `RELEASE.json`。之后
`data/provenance.release_info()` 读它，权威表 / 图 / prediction 就都指向同一个实例。

发布前先脱敏，再过三道闸门，任何一道不过就中止：
  0. 脱敏（`data/desensitize.py`）：菌株代号与化合物名换成稳定占位符。
     公开 = 面向不特定第三方分发，与「给顾问做技术把关」性质不同（CLAUDE.md R1）；
     初赛那版公开仓库就是这么处理的，占位符编号保持一致。
  1. 训练边界静态扫描（`audit/train_boundary_probe.py --static-only`）
  2. 赛事数据泄漏扫描（`data/pkg_leak_scan.py`，CLAUDE.md R1 强制）
  3. 待发布文件里不得出现 data/raw 下的任何文件

用法：
    python release.py --tag semifinal-v1                 # 只做本地演练，不推送
    python release.py --tag semifinal-v1 --push          # 真的推
    python release.py --show                             # 看当前 RELEASE.json
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import paths, provenance

REMOTE = "https://github.com/Cpuerpxm/goai-vyeast-2026.git"
RELEASE_FILE = os.path.join(paths.PROJECT_ROOT, "RELEASE.json")

#: 要公开的东西。目录整体同步；`data/raw` 与 `results/*.npz` 之类永远不在此列。
PUBLISH_DIRS = [
    ("scripts", "scripts"),
    ("results/figures", "results/figures"),
]
PUBLISH_FILES = [
    ("README.md", "README.md"),
    ("requirements.txt", "requirements.txt"),
    ("results/AUTHORITATIVE.md", "results/AUTHORITATIVE.md"),
    # ❗`data/external/` **整目录不发**：`compound_smiles.csv` 就是完整化合物名单
    # （54 行，带 SMILES 与 PubChem CID），那正是协议禁止再分发的东西；
    # 初赛公开版也没有它。只发外部资源的来源与版本披露这一份。
    ("data/external/yeast1011/SOURCE.md", "data/external/yeast1011/SOURCE.md"),
]
#: 各步骤的运行日志：源路径 -> 仓库内路径
PUBLISH_LOGS = [
    ("results/step0_compliance/train_boundary_probe.txt",
     "results/logs/step0_compliance_train_boundary_probe.txt"),
    ("results/step2_control_match/report_median.txt",
     "results/logs/step2_control_match_report_median.txt"),
    ("results/step3_diagnostics/batch_confounding.txt",
     "results/logs/step3_diagnostics_batch_confounding.txt"),
    ("results/step3_diagnostics/missing_mechanism.txt",
     "results/logs/step3_diagnostics_missing_mechanism.txt"),
    ("results/step3_diagnostics/noise_ceiling.txt",
     "results/logs/step3_diagnostics_noise_ceiling.txt"),
    ("results/step3_diagnostics/shared_reference_probe.txt",
     "results/logs/step3_diagnostics_shared_reference_probe.txt"),
    ("results/step4_baselines/report.txt", "results/logs/step4_baselines_report.txt"),
    ("results/step5_lowrank/report.txt", "results/logs/step5_lowrank_report.txt"),
    ("results/step7_cfree/report_bio_tech.txt",
     "results/logs/step7_cfree_report_bio_tech.txt"),
    ("results/step8_select_k0/report.txt", "results/logs/step8_select_k0_report.txt"),
    ("results/step9_loco/report.txt", "results/logs/step9_loco_report.txt"),
    ("results/step11_strain_transport/report.txt",
     "results/logs/step11_strain_transport_report.txt"),
    ("results/step0_compliance/quantify_l1_leak.txt",
     "results/logs/step0_compliance_quantify_l1_leak.txt"),
    ("results/step12_self_eval/self_eval.txt",
     "results/logs/step12_self_eval.txt"),
    ("results/step10_submission/submission_manifest.json",
     "results/logs/step10_submission_manifest.json"),
]
SKIP_SUFFIX = (".pyc", ".bak", ".tmp", ".npz", ".npy", ".csv.gz")
SKIP_DIRS = {"__pycache__", ".ipynb_checkpoints", "_cache"}


def run(cmd, cwd=None, check=True) -> str:
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8")
    if check and r.returncode != 0:
        raise RuntimeError(f"命令失败 {' '.join(cmd)}\n{r.stdout}\n{r.stderr}")
    return (r.stdout or "").strip()


def gate_boundary() -> None:
    p = os.path.join(paths.SCRIPTS_ROOT, "audit", "train_boundary_probe.py")
    r = subprocess.run([sys.executable, p, "--static-only"],
                       capture_output=True, text=True, encoding="utf-8")
    print(r.stdout)
    if r.returncode != 0:
        raise SystemExit("[发布中止] 训练边界静态扫描未通过")


def gate_leak(stage_dir: str) -> None:
    """`--force` 是必需的：pkg_leak_scan 默认只肯扫 `_pkg_for_gpt_pro/` 下的会诊包，
    因为本机内部文档带化合物名与菌株代号是正常的。这里扫的是**待推公开仓库**的
    暂存目录，同样属于外发场景，所以显式强制。"""
    p = os.path.join(paths.SCRIPTS_ROOT, "data", "pkg_leak_scan.py")
    r = subprocess.run([sys.executable, p, stage_dir, "--force"],
                       capture_output=True, text=True, encoding="utf-8")
    print(r.stdout[-4000:])
    if r.returncode != 0:
        raise SystemExit("[发布中止] 赛事数据泄漏扫描未通过（CLAUDE.md R1）")


def gate_no_raw(stage_dir: str) -> None:
    """待发布目录里不得出现 data/raw 下的任何文件名。"""
    raw = set()
    if os.path.isdir(paths.DATA_RAW):
        raw = {f.lower() for f in os.listdir(paths.DATA_RAW)}
    bad = []
    for root, dirs, files in os.walk(stage_dir):
        dirs[:] = [d for d in dirs if d != ".git"]
        for f in files:
            if f.lower() in raw:
                bad.append(os.path.relpath(os.path.join(root, f), stage_dir))
    print(f"  [{'FAIL' if bad else 'PASS'}] 待发布目录不含 data/raw 文件"
          + (f"：{bad}" if bad else ""))
    if bad:
        raise SystemExit("[发布中止] 赛事原始数据混进了待发布目录")


def sync(stage_dir: str) -> list:
    """把 PUBLISH_* 同步进 stage_dir，返回实际写入的相对路径清单。"""
    written = []

    def _copy(src, dst_rel):
        dst = os.path.join(stage_dir, dst_rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        written.append(dst_rel.replace("\\", "/"))

    for src_rel, dst_rel in PUBLISH_DIRS:
        src_root = os.path.join(paths.PROJECT_ROOT, src_rel)
        if not os.path.isdir(src_root):
            print(f"  ⚠ 跳过不存在的目录 {src_rel}")
            continue
        # 先清空仓库内的对应目录，避免删掉的文件留在公开仓库里
        dst_root = os.path.join(stage_dir, dst_rel)
        if os.path.isdir(dst_root):
            shutil.rmtree(dst_root)
        for root, dirs, files in os.walk(src_root):
            dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
            for f in sorted(files):
                if f.endswith(SKIP_SUFFIX):
                    continue
                s = os.path.join(root, f)
                rel = os.path.relpath(s, src_root).replace("\\", "/")
                _copy(s, f"{dst_rel}/{rel}")

    for src_rel, dst_rel in PUBLISH_FILES + PUBLISH_LOGS:
        s = os.path.join(paths.PROJECT_ROOT, src_rel)
        if os.path.exists(s):
            _copy(s, dst_rel)
        else:
            print(f"  ⚠ 跳过不存在的文件 {src_rel}")
    return sorted(written)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", help="要打的 tag，如 semifinal-v1")
    ap.add_argument("--message", default="", help="tag 说明")
    ap.add_argument("--push", action="store_true", help="真的推送到公开仓库")
    ap.add_argument("--show", action="store_true", help="打印当前 RELEASE.json 后退出")
    ap.add_argument("--workdir", default=None, help="克隆目录，默认用临时目录")
    args = ap.parse_args()

    if args.show:
        print(json.dumps(provenance.release_info(), ensure_ascii=False, indent=2))
        return
    if not args.tag:
        raise SystemExit("必须给 --tag")

    print("=" * 88)
    print(f"发布 {args.tag} -> {REMOTE}")
    print("=" * 88)
    print("\n[闸门 1/3] 训练边界静态扫描")
    gate_boundary()

    work = args.workdir or os.path.join(
        os.environ.get("TEMP", "/tmp"), f"goai_release_{args.tag}")
    if os.path.isdir(work):
        shutil.rmtree(work, ignore_errors=True)
    os.makedirs(os.path.dirname(work) or ".", exist_ok=True)
    print(f"\n[克隆] {REMOTE} -> {work}")
    run(["git", "clone", REMOTE, work])
    head_before = run(["git", "rev-parse", "--short", "HEAD"], cwd=work)
    print(f"  远端 HEAD {head_before}")

    print("\n[同步] 写入待发布文件")
    written = sync(work)
    print(f"  写入 {len(written)} 个文件")

    print("\n[脱敏] 把菌株代号与化合物名换成稳定占位符")
    from data import desensitize

    alias = desensitize.build_alias()
    # 只脱敏本轮写进去的文件；仓库里上一轮那份文稿已核验与提交定稿逐字节一致，不动它
    rep = desensitize.scrub_dir(work, alias, only=written)
    print(f"  已登记：菌株 {len(alias['strains'])} 个 / 化合物 {len(alias['compounds'])} 个")
    print(f"  改写 {len(rep['per_file'])} 个文件，共 {sum(rep['total'].values())} 处")

    print("\n[闸门 2/3] 赛事数据泄漏扫描")
    gate_leak(work)
    print("\n[闸门 3/3] 待发布目录不含原始数据")
    gate_no_raw(work)

    status = run(["git", "status", "--porcelain"], cwd=work)
    print("\n[改动]")
    print(status or "  （无改动）")

    tree = provenance.tree_digest()
    msg = args.message or f"复赛整改：训练边界合规重跑（源码摘要 {tree}）"

    if not args.push:
        print("\n[演练] 未加 --push，到此为止。上面就是会被推上去的改动。")
        print(f"      克隆目录保留在 {work}，可自行核对。")
        return

    run(["git", "add", "-A"], cwd=work)
    if run(["git", "status", "--porcelain"], cwd=work):
        run(["git", "commit", "-m", msg], cwd=work)
    commit = run(["git", "rev-parse", "HEAD"], cwd=work)
    run(["git", "tag", "-a", args.tag, "-m", msg], cwd=work)
    run(["git", "push", "origin", "HEAD"], cwd=work)
    run(["git", "push", "origin", args.tag], cwd=work)
    print(f"\n[已推送] commit {commit[:9]}  tag {args.tag}")

    rel = {
        "remote": REMOTE.replace(".git", ""),
        "tag": args.tag,
        "commit": commit,
        "commit_short": commit[:9],
        "tree_digest": tree,
        "released_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "n_published_files": len(written),
        "message": msg,
    }
    with open(RELEASE_FILE, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(rel, fh, ensure_ascii=False, indent=2)
    print(f"[写出] {RELEASE_FILE}")

    # RELEASE.json 本身也推一份，让公开仓库自带发布标识。
    # 它不在 scripts/ 下，不影响 tree_digest，所以放在 tag 之后的第二个提交里。
    shutil.copy2(RELEASE_FILE, os.path.join(work, "RELEASE.json"))
    run(["git", "add", "RELEASE.json"], cwd=work)
    run(["git", "commit", "-m", f"记录发布标识 {args.tag}"], cwd=work)
    run(["git", "push", "origin", "HEAD"], cwd=work)
    print("[已推送] RELEASE.json")
    print("\n发布完成。接下来重跑 authoritative_results.py，让权威表带上这个 tag。")


if __name__ == "__main__":
    main()
