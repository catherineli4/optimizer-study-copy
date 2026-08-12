#!/usr/bin/env python3
"""Clear node-local HuggingFace cache for OLMo-2-0425-1B on Slurm compute nodes.

Babel blocks bare SSH to nodes without an active job (pam_slurm_adopt), so this
script resolves nodes from past launcher log files via ``sacct`` and submits one
short ``sbatch`` job per node to remove the partial cache directory.

Log filenames must look like ``<prefix>_<jobid>_<task>.out``, e.g.
``tier-0_9138715_3.out``.

Examples:
    # List nodes touched by a failed c4-divergence array job
    python3 scripts/clear_hf_1b_cache_from_logs.py \\
        --log-glob 'tier-0_9138715_*.out' --dry-run

    # Submit cleanup sbatch jobs (one per unique node)
    python3 scripts/clear_hf_1b_cache_from_logs.py \\
        --log-glob 'tier-0_9138715_*.out' --submit

    # Or pass job ids directly
    python3 scripts/clear_hf_1b_cache_from_logs.py --job-ids 9138715,9138581 --submit
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
import sys
from collections import defaultdict

DEFAULT_LOG_DIR = os.path.expanduser("~/.experiments/logs")
DEFAULT_MODEL_ID = "allenai/OLMo-2-0425-1B"
DEFAULT_HUB_CACHE = "/scratch/catheri4/huggingface/hub"
LOG_STEP_RE = re.compile(r"^.+_(\d+)_(\d+)\.out$")


def hub_cache_dir(model_id: str, hub_cache: str) -> str:
    return os.path.join(hub_cache, "models--" + model_id.replace("/", "--"))


def parse_log_steps(log_paths: list[str]) -> list[tuple[int, int]]:
    steps: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for path in sorted(log_paths):
        base = os.path.basename(path)
        match = LOG_STEP_RE.match(base)
        if not match:
            print(f"[skip] not a job log: {base}", file=sys.stderr)
            continue
        step = (int(match.group(1)), int(match.group(2)))
        if step not in seen:
            seen.add(step)
            steps.append(step)
    return steps


def sacct_node(job_id: int, task_id: int) -> str | None:
    step = f"{job_id}_{task_id}"
    proc = subprocess.run(
        ["sacct", "-j", step, "-n", "-P", "--format=JobID,NodeList"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        print(f"[warn] sacct failed for {step}: {proc.stderr.strip()}", file=sys.stderr)
        return None
    for line in proc.stdout.strip().splitlines():
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) < 2:
            continue
        jid, nodelist = parts[0].strip(), parts[1].strip()
        if jid == step and nodelist:
            return nodelist
    print(f"[warn] no NodeList for {step}", file=sys.stderr)
    return None


def expand_nodelist(nodelist: str) -> list[str]:
    """Split sacct NodeList; sacct already expands ranges on babel."""
    nodes: list[str] = []
    for part in nodelist.split(","):
        part = part.strip()
        if part:
            nodes.append(part)
    return nodes


def collect_nodes(steps: list[tuple[int, int]]) -> dict[str, list[str]]:
    node_to_logs: dict[str, list[str]] = defaultdict(list)
    for job_id, task_id in steps:
        log_tag = f"{job_id}_{task_id}"
        nodelist = sacct_node(job_id, task_id)
        if not nodelist:
            continue
        for node in expand_nodelist(nodelist):
            node_to_logs[node].append(log_tag)
    return dict(node_to_logs)


def submit_cleanup(
    node: str,
    cache_path: str,
    *,
    partition: str,
    log_dir: str,
    exclude: str | None,
) -> str:
    os.makedirs(log_dir, exist_ok=True)
    wrap = (
        f"TARGET={cache_path!r}; "
        "if [ -d \"$TARGET\" ]; then rm -rf \"$TARGET\" && echo \"cleared $TARGET on $(hostname)\"; "
        "else echo \"missing $TARGET on $(hostname)\"; fi"
    )
    cmd = [
        "sbatch",
        f"--job-name=clr-hf-1b",
        f"--partition={partition}",
        "--gres=gpu:1",
        f"--nodelist={node}",
        "--time=00:05:00",
        "--mem=1G",
        "--cpus-per-task=1",
        f"--output={log_dir}/clear_hf_1b_%j.out",
        f"--wrap={wrap}",
    ]
    if exclude:
        cmd.insert(-1, f"--exclude={exclude}")
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"sbatch failed for {node}: {proc.stderr.strip() or proc.stdout.strip()}"
        )
    job_id = proc.stdout.strip().split()[-1]
    return job_id


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--logs-dir",
        default=DEFAULT_LOG_DIR,
        help=f"Directory containing launcher .out files (default: {DEFAULT_LOG_DIR})",
    )
    ap.add_argument(
        "--log-glob",
        default="tier-0_*.out",
        help="Glob under --logs-dir for log files (default: tier-0_*.out)",
    )
    ap.add_argument(
        "--job-ids",
        default="",
        help="Comma-separated Slurm job ids; uses all array tasks found in sacct.",
    )
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--hub-cache", default=DEFAULT_HUB_CACHE)
    ap.add_argument("--partition", default="general")
    ap.add_argument("--exclude", default="babel-m9-16")
    ap.add_argument(
        "--submit-log-dir",
        default=os.path.expanduser("~/logs/clear_hf_1b"),
        help="Where to write sbatch stdout for cleanup jobs.",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print nodes only.")
    ap.add_argument(
        "--submit",
        action="store_true",
        help="Submit one sbatch cleanup job per node.",
    )
    args = ap.parse_args()

    steps: list[tuple[int, int]] = []
    if args.job_ids:
        for job_s in args.job_ids.split(","):
            job_s = job_s.strip()
            if not job_s:
                continue
            job_id = int(job_s)
            proc = subprocess.run(
                [
                    "sacct",
                    "-j",
                    str(job_id),
                    "-n",
                    "-P",
                    "--format=JobID,NodeList",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            task_ids: set[int] = set()
            step_re = re.compile(rf"^{job_id}_(\d+)$")
            for line in proc.stdout.strip().splitlines():
                jid = line.split("|", 1)[0].strip()
                m = step_re.match(jid)
                if m:
                    task_ids.add(int(m.group(1)))
            if not task_ids:
                print(f"[warn] no array tasks in sacct for job {job_id}", file=sys.stderr)
            steps.extend((job_id, tid) for tid in sorted(task_ids))
    else:
        pattern = os.path.join(args.logs_dir, args.log_glob)
        log_paths = sorted(glob.glob(pattern))
        if not log_paths:
            raise SystemExit(f"No logs match {pattern!r}")
        steps = parse_log_steps(log_paths)

    if not steps:
        raise SystemExit("No job steps to resolve.")

    cache_path = hub_cache_dir(args.model_id, args.hub_cache)
    node_to_logs = collect_nodes(steps)
    if not node_to_logs:
        raise SystemExit("No nodes resolved from sacct.")

    print(f"Model cache target: {cache_path}")
    print(f"Unique nodes ({len(node_to_logs)}):")
    for node in sorted(node_to_logs):
        logs = ", ".join(sorted(set(node_to_logs[node])))
        print(f"  {node}  <- steps {logs}")

    if args.dry_run and not args.submit:
        return
    if not args.submit:
        print("\nPass --submit to queue cleanup jobs, or --dry-run to only list nodes.")
        return

    print("\nSubmitting cleanup jobs:")
    for node in sorted(node_to_logs):
        job_id = submit_cleanup(
            node,
            cache_path,
            partition=args.partition,
            log_dir=args.submit_log_dir,
            exclude=args.exclude or None,
        )
        print(f"  {node}: sbatch job {job_id}")


if __name__ == "__main__":
    main()
