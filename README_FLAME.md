# Running launch_jolmo inside a flame-earlybirds node-holder

Notes on getting `launch_jolmo` work onto a `flame-earlybirds` node that is held
by an idle `node.sh` job. Written 2026-08-05.

## The situation

- `~/node.sh` is a **node-holder**: `#SBATCH --partition=flame-earlybirds
  --qos=earlybird_qos --gpus-per-node=H100:8 --cpus-per-task=96 --mem=2063000M
  --time=2-00:00:00 --requeue`, then `nvidia-smi` and `while true; do sleep 1h; done`.
  It reserves a whole node for 2 days and idles.
- Jobs `9745059` and `9745061` (both named `i`) are queued copies of it, submitted
  2026-08-04, estimated start **2026-08-06 17:36**.
- The partition has 3 nodes — `babel-s5-16`, `babel-t5-16`, `babel-u5-16` — all
  currently `alloc`.

## Why you cannot sbatch onto it

- `earlybird_qos` allows **1 job / 8 GPUs per user**. The holder *is* that one job,
  so a second submission to `flame-earlybirds` will never start. That also means
  `9745059` and `9745061` can't run concurrently — the second is a spare.
- Targeting the node from `general` / `preempt` (`--slurm nodelist=babel-s5-16`)
  fails too: your own holder has 100% of its CPUs, GPUs and memory.
- Editing `node.sh` now does **not** change the queued jobs — Slurm copies the
  batch script to the controller at submit time. Changing it means cancel +
  resubmit, which forfeits the queue position (pending since Aug 4).

**The only way in is a job step inside the existing allocation:**
`srun --jobid=<id> --overlap`.

## Why this is worth doing

- The `normal` QOS caps you at **8 GPUs total**. That is why `9782781_1` (the
  300M chin-2 AdamW lr2.5e-3 cell) sits pending on `QOSMaxGRESPerUser` while
  `9782781_0` occupies all 8.
- Work run *inside* the flame allocation does not consume that quota — the GPUs
  are already charged to the holder job. It's 8 extra H100s in parallel with the
  `general`-partition sweep.

## Mechanism: `print`, not `launch`

- `python -m launch_jolmo.launcher print <stage>` emits a **self-contained bash
  script** — the `conda activate optim-study` line, `set -euo pipefail`, the
  project config export, then every task block (download → torchrun → unshard →
  upload-with-retry) for each artifact not already on GCS. No sbatch.
- **Never run `launcher launch` on the node** — it submits fresh SLURM jobs and
  hits the QOS cap again.
- `print` emits *every* missing artifact sequentially under `set -e`, so one
  failure aborts the rest. Slice with `--head` / `--tail`; preview with `drylaunch`.
- `printlines --output-dir DIR --jobs N` writes one script per job and prints one
  `bash …` line each — feed it to `xargs -P N` for parallel work. Evals are
  single-process `validate.py` on one GPU, so pin `CUDA_VISIBLE_DEVICES` per line
  or they all pile onto GPU 0.

## Recipe

1. **On the login node**, generate the script (the `exists` checks are slow GCS
   lookups; do them off the allocation):

   ```bash
   cd ~/optimizer-study-copy
   OPTIM_SIZE=300M python -m launch_jolmo.launcher drylaunch pretrain-all-wsd
   OPTIM_SIZE=300M OPTIM_NUM_PROCESSES=8 \
     python -m launch_jolmo.launcher print pretrain-all-wsd --head 1 > ~/flame_job.sh
   ```

   `OPTIM_NUM_PROCESSES=8` bakes `--nproc_per_node=8` into the emitted torchrun.
   Global batch is fixed at 1M tokens, so only the per-rank microbatch changes —
   the training math is unaffected.

2. **Wait and fire** — `job_scripts/flame_run.sh` polls until the holder is
   `RUNNING`, verifies an overlapping step sees the GPUs, then runs the script
   inside the allocation:

   ```bash
   tmux new -s flame
   cd ~/optimizer-study-copy
   ./job_scripts/flame_run.sh 9745059 ~/flame_job.sh
   ```

   Interactive equivalent:

   ```bash
   srun --jobid=9745059 --overlap --pty bash
   cd ~/optimizer-study-copy
   OPTIM_SIZE=300M OPTIM_NUM_PROCESSES=8 \
     python -m launch_jolmo.launcher print pretrain-all-wsd --head 1 | bash
   ```

## Gotchas

- **GPU inheritance.** Steps do not reliably inherit the job's GRES. Check with
  `srun --jobid=<id> --overlap nvidia-smi -L` before the real run; if it comes
  back empty, add `--gpus-per-node=8` to the `srun`. `flame_run.sh` probes this
  automatically.
- **`/scratch` is node-local** — it does not exist on the login node. The flame
  node starts with an empty `/scratch/catheri4/cache`, so DCLM re-downloads
  (~74 GiB/part; 300M chin-2 = 12B tokens ≈ 1 part). Budget ~an hour for the pull.
  Same for `/scratch/catheri4-outputs/<project>`.
- **`--requeue` is set on the holder.** Preemption restarts the job and kills your
  step. Artifact `exists` checks make a re-run idempotent, but in-flight training
  loses everything since the last `save_interval` (1000 steps).
- **Run under tmux/nohup.** `flame_run.sh` blocks until allocation — possibly days
  — and the step dies with your shell.
- **Don't use `run_local.py` for anything downstream consumes.** It never calls
  `_postprocess`, so it skips the unshard + upload of `final-unsharded/model.pt`;
  CPT / perturb / eval artifacts won't find the model afterward.
- **The step dies when the holder ends** (2-day wall clock), regardless of what
  it was doing.

## Allocation alert (email)

`job_scripts/flame_notify.py` emails **catheri4@andrew.cmu.edu** as soon as a
`flame-earlybirds` job flips to `RUNNING`. Installed in cron on **login1**:

```
*/2 * * * * /usr/bin/python3 /home/catheri4/optimizer-study-copy/job_scripts/flame_notify.py >> /home/catheri4/.flame_notify/cron.log 2>&1
```

The mail names the node and includes the `ssh` command plus the `print | bash`
recipe, so the message alone is enough to get started.

How it delivers, and why it looks the way it does:

- **Straight to the recipient domain's MX** (`ASPMX.L.GOOGLE.COM` — andrew.cmu.edu
  is on Google Workspace), no credentials. The babel login nodes are in
  `128.2.0.0/16`, which CMU's `_spf.cmu.edu` record authorizes for
  `andrew.cmu.edu`, so envelope sender `catheri4@andrew.cmu.edu` passes SPF.
- **The local MTA is a dead end.** `/usr/sbin/sendmail` → `/usr/bin/esmtp-wrapper`
  with no `/etc/esmtprc` or `~/.esmtprc`, so mail silently queues in
  `~/.esmtp_queue` and never leaves.
- **`smtp.andrew.cmu.edu` needs auth.** After STARTTLS it offers
  `GS2-KRB5 GSSAPI PLAIN LOGIN` and rejects unauthenticated senders with
  `554 5.7.1 Sender address rejected`. A Kerberos ticket would satisfy GSSAPI,
  but the cached one (`catheri4@ANDREW.CMU.EDU`) expired 2025-08-25 and would
  need an interactive `kinit` plus renewal to survive a multi-day wait.

State is `~/.flame_notify/`: one `notified-<jobid>` marker per announced job, so
each is announced exactly once. A failed send leaves the marker unwritten and the
next tick retries.

```bash
job_scripts/flame_notify.py --status   # what it sees; sends nothing
job_scripts/flame_notify.py --test     # send a test mail
job_scripts/flame_notify.py --reset    # clear markers (re-announce)
tail ~/.flame_notify/log               # send history
```

Second, independent signal: `MailType=BEGIN` / `MailUser` are now set on jobs
`9745059` and `9745061` via `scontrol update` (`StartTime` unchanged — mail
fields don't affect scheduling), and `~/node.sh` carries `#SBATCH --mail-type` /
`--mail-user` for future submissions. This path is unverified: `MailProg` is
`/usr/bin/smail`, which calls `/bin/mail` — absent on login1. If the controller
lacks it too, these silently no-op, which is why the cron watcher exists.

Caveats:

- The crontab lives on **login1 only** (`/var/spool/cron` is node-local). Jobs
  submitted from login2 are still seen — `squeue` is cluster-wide — but if login1
  reboots or you expect to work elsewhere, reinstall it there too.
- It watches for `RUNNING`, so it fires on requeue-restarts as well (new job id →
  new mail). That's usually what you want.

## Useful commands

```bash
squeue -u catheri4 -o "%.10i %.20j %.16P %.8T %.10M %R"   # queue + reasons
scontrol show job 9745059                                  # StartTime estimate
sacctmgr -n show qos format=name,maxtresperuser%20,maxjobsperuser
srun --jobid=9745059 --overlap nvidia-smi                  # what's on the node
squeue -j 9745059 -h -o %N                                 # which node it got
```
