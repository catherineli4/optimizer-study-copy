#!/usr/bin/env bash
# Wait for a flame-earlybirds node-holder job (node.sh) to be allocated, then run
# a pre-generated launch_jolmo script as a job STEP inside that allocation.
#
# The holder job reserves the whole node (96 CPU / 8xH100 / 2 TB) and then idles,
# so nothing else can be scheduled onto it -- and earlybird_qos allows only one
# job per user. `srun --jobid=<id> --overlap` is the only way in. Work run this
# way does NOT count against the `normal` QOS 8-GPU cap.
#
# Generate the script first, on the login node (the exists checks hit GCS):
#   cd ~/optimizer-study-copy
#   OPTIM_SIZE=300M OPTIM_NUM_PROCESSES=8 \
#     python -m launch_jolmo.launcher print pretrain-all-wsd --head 1 > ~/flame_job.sh
#
# Then:
#   ./job_scripts/flame_run.sh 9745059 ~/flame_job.sh
#
# Run it under tmux/nohup -- it blocks until the holder is allocated, which may
# be days, and the step dies with your shell otherwise.

set -euo pipefail

JID=${1:?usage: flame_run.sh <jobid> <script.sh> [cpus]}
SCRIPT=${2:?usage: flame_run.sh <jobid> <script.sh> [cpus]}
CPUS=${3:-32}
LOG=${FLAME_LOG:-$HOME/flame_job.$JID.log}
POLL=${FLAME_POLL:-60}

[ -r "$SCRIPT" ] || { echo "no such script: $SCRIPT" >&2; exit 1; }

state() { squeue -j "$JID" -h -o %T 2>/dev/null | head -1; }

echo "waiting for job $JID to be allocated (polling every ${POLL}s)..."
while :; do
  s=$(state)
  case "$s" in
    RUNNING) break ;;
    "")      echo "job $JID is no longer in the queue -- cancelled or finished" >&2; exit 1 ;;
    *)       sleep "$POLL" ;;
  esac
done

NODE=$(squeue -j "$JID" -h -o %N | head -1)
echo "job $JID is RUNNING on $NODE"

# Steps inside an allocation do not always inherit the job's GPUs. Check before
# committing the real run; if this prints nothing, re-run the srun below with
# --gpus-per-node=8 added.
echo "--- GPUs visible to an overlapping step ---"
srun --jobid="$JID" --overlap --nodes=1 --ntasks=1 nvidia-smi -L || {
  echo "could not start a step in job $JID" >&2; exit 1; }
echo "-------------------------------------------"

echo "launching $SCRIPT on $NODE -> $LOG"
exec srun --jobid="$JID" --overlap \
     --nodes=1 --ntasks=1 --cpus-per-task="$CPUS" \
     --job-name=jolmo-flame \
     bash "$SCRIPT" > "$LOG" 2>&1
