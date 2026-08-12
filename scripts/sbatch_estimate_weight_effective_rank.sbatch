#!/bin/bash
#SBATCH --job-name=effrank60m
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/catheri4/optimizer-study-copy/logs/slurm_%x_%j.out
#SBATCH --error=/home/catheri4/optimizer-study-copy/logs/slurm_%x_%j.err

set -euo pipefail
mkdir -p /home/catheri4/optimizer-study-copy/logs
source ~/miniconda3/etc/profile.d/conda.sh
conda activate optim-study 2>/dev/null || true
cd /home/catheri4/optimizer-study-copy
export PYTHONPATH="/home/catheri4/optimizer-study-copy/JOLMo/src:/home/catheri4/optimizer-study-copy:${PYTHONPATH:-}"

CACHE="${LOCAL_CACHE:-/scratch/catheri4-outputs/Optim-60M-tuning/JolmoModel}"
OUT=results/effective_rank_60M_bestlr
mkdir -p "$CACHE" "$OUT"

echo "=== host=$(hostname) date=$(date) ==="

# Clear GCS auth check with a useful error (do not rely on set -e alone).
if ! gsutil ls "gs://cmu-gpucloud-catheri4/Optim-60M-tuning/JolmoModel/" >/tmp/gsutil_ls_$$.txt 2>/tmp/gsutil_err_$$.txt; then
  echo "ERROR: cannot list GCS JolmoModel bucket." >&2
  echo "----- gsutil stderr -----" >&2
  cat /tmp/gsutil_err_$$.txt >&2
  if grep -qi 'Reauthentication required\|RefreshError\|invalid_grant' /tmp/gsutil_err_$$.txt; then
    echo >&2
    echo "Fix on a LOGIN node (interactive), then resubmit this job:" >&2
    echo "  gcloud auth login" >&2
    echo "  gcloud auth application-default login" >&2
    echo "  sbatch scripts/sbatch_estimate_weight_effective_rank.sbatch" >&2
  fi
  exit 42
fi
head -5 /tmp/gsutil_ls_$$.txt
rm -f /tmp/gsutil_ls_$$.txt /tmp/gsutil_err_$$.txt

PY=$(command -v python)
echo "Python: $PY $($PY -c 'import torch; print(torch.__version__)')"

$PY scripts/estimate_weight_effective_rank.py \
  --out-dir "$OUT" \
  --cache-dir "$CACHE" \
  --gram-max 4096

echo "=== done ==="
ls -la "$OUT" | head -40
