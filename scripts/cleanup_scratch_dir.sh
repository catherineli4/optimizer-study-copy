#!/usr/bin/env bash
# Usage: cleanup_scratch_dir.sh <scratch_dir_to_clean> [sentinel_file]
#
# Finds all compute nodes that appear in experiment logs, submits one sbatch
# cleanup job per node, then waits and prints results.
#
# The job checks: if <scratch_dir> exists but <sentinel_file> is missing,
# delete it so the next training run will re-download it.
#
# Args:
#   scratch_dir    Local path on each compute node to check/clean
#   sentinel_file  File inside scratch_dir whose presence means "download OK"
#                  (default: 00000.npy)

set -euo pipefail

SCRATCH_DIR="${1:?Usage: $0 <scratch_dir> [sentinel_file]}"
SENTINEL="${2:-00000.npy}"

LOGS_DIR="${LOGS_DIR:-$HOME/.experiments/logs}"
PARTITION="${PARTITION:-general}"
GRES="${GRES:-gpu:1}"
TIME_LIMIT="${TIME_LIMIT:-00:05:00}"
CLEANUP_LOGS_DIR="${CLEANUP_LOGS_DIR:-$LOGS_DIR}"

# ---------------------------------------------------------------------------
# 1. Collect unique node names from experiment logs
# ---------------------------------------------------------------------------
echo "Scanning logs in $LOGS_DIR for compute node names..."

mapfile -t NODES < <(
    grep -h "host" "$LOGS_DIR"/tier-0_*.out 2>/dev/null \
    | grep -oP 'babel-\S+\.ib' \
    | sed 's/\.ib$//' \
    | sort -u
)

if [[ ${#NODES[@]} -eq 0 ]]; then
    echo "No nodes found in logs. Exiting."
    exit 1
fi

echo "Found ${#NODES[@]} node(s): ${NODES[*]}"
echo

# ---------------------------------------------------------------------------
# 2. Submit one cleanup job per node
# ---------------------------------------------------------------------------
declare -A JOB_IDS  # node -> job id

for node in "${NODES[@]}"; do
    job_id=$(sbatch \
        --partition="$PARTITION" \
        --nodelist="$node" \
        --nodes=1 --ntasks=1 \
        --gres="$GRES" \
        --time="$TIME_LIMIT" \
        --job-name="cleanup_$(echo "$node" | tr '-' '_')" \
        --output="$CLEANUP_LOGS_DIR/cleanup_${node}_%j.out" \
        --wrap="
DIR='$SCRATCH_DIR'
SENTINEL=\"\$DIR/$SENTINEL\"
if [ -d \"\$DIR\" ] && [ ! -f \"\$SENTINEL\" ]; then
    rm -rf \"\$DIR\"
    echo \"\$(hostname): CLEANED\"
elif [ -d \"\$DIR\" ]; then
    echo \"\$(hostname): ok (files present)\"
else
    echo \"\$(hostname): dir missing, nothing to do\"
fi
" \
        2>&1 | awk '{print $NF}')
    echo "Submitted job $job_id for $node"
    JOB_IDS[$node]=$job_id
done

echo
echo "Waiting for jobs to complete..."
echo

# ---------------------------------------------------------------------------
# 3. Wait for all jobs to finish and print results
# ---------------------------------------------------------------------------
ALL_JOB_IDS=$(IFS=,; echo "${JOB_IDS[*]}")

while true; do
    # Count how many are still running/pending
    still_running=$(squeue -j "$ALL_JOB_IDS" -h 2>/dev/null | wc -l)
    if [[ "$still_running" -eq 0 ]]; then
        break
    fi
    sleep 5
done

echo "Results:"
echo "--------"
for node in "${NODES[@]}"; do
    job_id="${JOB_IDS[$node]}"
    out_file="$CLEANUP_LOGS_DIR/cleanup_${node}_${job_id}.out"
    if [[ -f "$out_file" ]]; then
        cat "$out_file"
    else
        echo "$node (job $job_id): no output file found"
    fi
done
