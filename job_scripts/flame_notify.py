#!/usr/bin/env python3
"""Email an alert when a flame-earlybirds node-holder job gets allocated.

Polls squeue for the user's jobs in the flame-earlybirds partition and emails
once per job as soon as it flips to RUNNING. Intended to be driven by cron:

    */2 * * * * /usr/bin/python3 ~/optimizer-study-copy/job_scripts/flame_notify.py

Mail goes straight to the recipient domain's MX. No credentials are needed: the
babel login nodes sit in 128.2.0.0/16, which CMU's own SPF record authorizes for
andrew.cmu.edu, so a message with envelope sender catheri4@andrew.cmu.edu passes
SPF. (The local sendmail is a dead end -- /usr/sbin/sendmail is an esmtp wrapper
with no relay configured, and smtp.andrew.cmu.edu requires Kerberos/PLAIN auth.)

State lives in ~/.flame_notify/: one marker file per notified job id, so a job is
announced exactly once. A send failure leaves the marker unwritten, so the next
tick retries.

    flame_notify.py            # one poll (what cron runs)
    flame_notify.py --test     # send a test mail and exit
    flame_notify.py --status   # show what it would consider, send nothing
    flame_notify.py --reset    # clear markers (re-announce running jobs)
"""

from __future__ import annotations

import argparse
import os
import pwd
import smtplib
import socket
import subprocess
import sys
import time
from email.message import EmailMessage
from pathlib import Path

PARTITION = os.environ.get("FLAME_PARTITION", "flame-earlybirds")
TO_ADDR = os.environ.get("FLAME_MAIL_TO", "catheri4@andrew.cmu.edu")
FROM_ADDR = os.environ.get("FLAME_MAIL_FROM", "catheri4@andrew.cmu.edu")
STATE_DIR = Path(os.environ.get("FLAME_STATE_DIR", Path.home() / ".flame_notify"))
USER = pwd.getpwuid(os.getuid()).pw_name

SQUEUE_FMT = "%i|%T|%N|%P|%j|%S|%e"


def log(msg: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        with (STATE_DIR / "log").open("a") as fh:
            fh.write(line + "\n")
    except OSError:
        pass


def running_jobs() -> list[dict]:
    """Jobs of this user in the watched partition, as dicts."""
    out = subprocess.run(
        ["squeue", "-u", USER, "-h", "-P", "-o", SQUEUE_FMT],
        capture_output=True, text=True, timeout=60,
    )
    if out.returncode != 0:
        raise RuntimeError(f"squeue failed: {out.stderr.strip()}")

    jobs = []
    for line in out.stdout.splitlines():
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) < 7:
            continue
        jobid, state, node, partition, name, start, end = (p.strip() for p in parts[:7])
        if partition != PARTITION:
            continue
        jobs.append({
            "jobid": jobid, "state": state, "node": node,
            "partition": partition, "name": name, "start": start, "end": end,
        })
    return jobs


def mx_hosts(domain: str) -> list[str]:
    """MX hosts for a domain, best preference first."""
    out = subprocess.run(
        ["dig", "+short", "MX", domain], capture_output=True, text=True, timeout=30,
    )
    entries = []
    for line in out.stdout.strip().splitlines():
        bits = line.split()
        if len(bits) == 2 and bits[0].isdigit():
            entries.append((int(bits[0]), bits[1].rstrip(".")))
    return [host for _, host in sorted(entries)]


def send_mail(subject: str, body: str) -> None:
    """Deliver straight to the recipient domain's MX. Raises on total failure."""
    msg = EmailMessage()
    msg["From"] = FROM_ADDR
    msg["To"] = TO_ADDR
    msg["Subject"] = subject
    msg.set_content(body)

    hosts = mx_hosts(TO_ADDR.split("@", 1)[1])
    if not hosts:
        raise RuntimeError(f"no MX records for {TO_ADDR}")

    errors = []
    for host in hosts:
        try:
            server = smtplib.SMTP(host, 25, timeout=30)
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.send_message(msg)
            server.quit()
            log(f"sent {subject!r} via {host}")
            return
        except Exception as exc:  # try the next MX
            errors.append(f"{host}: {type(exc).__name__}: {exc}")
    raise RuntimeError("all MX hosts failed -- " + "; ".join(errors))


def body_for(job: dict) -> str:
    node = job["node"] or "(unknown)"
    return f"""\
Your {PARTITION} node has been allocated.

  job      {job['jobid']}  ({job['name']})
  node     {node}
  started  {job['start']}
  ends     {job['end']}

Get on it (pam_slurm_adopt puts the ssh session in the job's cgroup, so you
inherit its CPUs, memory and all 8 GPUs):

  ssh {node}

Then run pre-generated launch_jolmo work inside the allocation:

  cd ~/optimizer-study-copy
  OPTIM_SIZE=300M OPTIM_NUM_PROCESSES=8 \\
    python -m launch_jolmo.launcher print pretrain-all-wsd --head 1 | bash

Use `print` / `printlines`, never `launch` -- launch submits new sbatch jobs and
hits the 8-GPU normal-QOS cap that this allocation exists to bypass.

Notes: /scratch on the node is empty, so DCLM re-downloads (~74 GiB/part). The
holder job has --requeue, so preemption restarts it and kills anything you
started. Details: ~/optimizer-study-copy/README_FLAME.md

-- flame_notify.py on {socket.gethostname()}
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", action="store_true", help="send a test mail and exit")
    ap.add_argument("--status", action="store_true", help="show state, send nothing")
    ap.add_argument("--reset", action="store_true", help="clear markers and exit")
    args = ap.parse_args()

    STATE_DIR.mkdir(parents=True, exist_ok=True)

    if args.reset:
        removed = 0
        for marker in STATE_DIR.glob("notified-*"):
            marker.unlink()
            removed += 1
        log(f"cleared {removed} marker(s)")
        return 0

    if args.test:
        send_mail(
            f"[babel] {PARTITION} watcher test",
            f"Test from {socket.gethostname()} at {time.strftime('%Y-%m-%d %H:%M:%S')}.\n"
            f"Watching partition {PARTITION} for user {USER}.\n",
        )
        return 0

    try:
        jobs = running_jobs()
    except Exception as exc:
        log(f"squeue error: {exc}")
        return 1

    if args.status:
        if not jobs:
            print(f"no {PARTITION} jobs for {USER}")
        for job in jobs:
            marker = STATE_DIR / f"notified-{job['jobid']}"
            print(f"{job['jobid']:>12}  {job['state']:<10} {job['node'] or '-':<14} "
                  f"notified={marker.exists()}  starts={job['start']}")
        return 0

    for job in jobs:
        if job["state"] != "RUNNING":
            continue
        marker = STATE_DIR / f"notified-{job['jobid']}"
        if marker.exists():
            continue
        node = job["node"] or "unknown"
        try:
            send_mail(f"[babel] {PARTITION} node allocated: {node} (job {job['jobid']})",
                      body_for(job))
        except Exception as exc:
            # Leave the marker unwritten so the next tick retries.
            log(f"send failed for job {job['jobid']}: {exc}")
            continue
        marker.write_text(f"{job['jobid']} {node} {time.time():.0f}\n")
        log(f"notified for job {job['jobid']} on {node}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
