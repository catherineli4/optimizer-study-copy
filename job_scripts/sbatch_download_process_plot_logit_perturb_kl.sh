#!/bin/bash
#SBATCH --job-name=plot-logit-perturb-kl
#SBATCH --output=/home/catheri4/optimizer-study-copy/logs/logit_perturb_kl_dl_plot/%j.out
#SBATCH --error=/home/catheri4/optimizer-study-copy/logs/logit_perturb_kl_dl_plot/%j.err
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=general
#SBATCH --gres=gpu:1
# Babel preempt QoS requires ≥1 GPU; this job is CPU-only but must request one.

# Download LogitPerturbKlEvaluation JSONs from GCS and plot mean/ΔKL vs σ.
# Usage:
#   sbatch job_scripts/sbatch_download_process_plot_logit_perturb_kl.sh

set -euo pipefail

REPO=/home/catheri4/optimizer-study-copy
GCS_DIR="${GCS_DIR:-gs://cmu-gpucloud-catheri4/Optim-60M-tuning/LogitPerturbKlEvaluation}"
LOCAL_DIR="${REPO}/results/LogitPerturbKlEvaluation"
OUT_DIR="${REPO}/results/logit_perturb_kl"

mkdir -p "${REPO}/logs/logit_perturb_kl_dl_plot" "${LOCAL_DIR}" "${OUT_DIR}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate optim-study
cd "${REPO}"

echo "Downloading from ${GCS_DIR} → ${LOCAL_DIR}"
gsutil -m rsync -r "${GCS_DIR}/" "${LOCAL_DIR}/" || true

python -m new_utils.plot_logit_perturb_kl \
  --local-dir "${LOCAL_DIR}" \
  --out-dir "${OUT_DIR}"

echo "Done. Plots in ${OUT_DIR}/"
ls -la "${OUT_DIR}/"
