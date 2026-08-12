#!/bin/bash
#SBATCH --job-name=plot-logit-angle-bins
#SBATCH --output=/home/catheri4/optimizer-study-copy/logs/logit_angle_bins_dl_plot/%j.out
#SBATCH --error=/home/catheri4/optimizer-study-copy/logs/logit_angle_bins_dl_plot/%j.err
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=general
#SBATCH --gres=gpu:1

set -euo pipefail

REPO=/home/catheri4/optimizer-study-copy
GCS_DIR="${GCS_DIR:-gs://cmu-gpucloud-catheri4/Optim-60M-tuning/LogitAngleBinEvaluation}"
LOCAL_DIR="${REPO}/results/LogitAngleBinEvaluation"
OUT_DIR="${REPO}/results/logit_angle_bins"

mkdir -p "${REPO}/logs/logit_angle_bins_dl_plot" "${LOCAL_DIR}" "${OUT_DIR}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate optim-study
cd "${REPO}"

echo "Downloading from ${GCS_DIR} → ${LOCAL_DIR}"
gsutil -m rsync -r "${GCS_DIR}/" "${LOCAL_DIR}/" || true

python -m new_utils.plot_logit_angle_bins \
  --local-dir "${LOCAL_DIR}" \
  --out-dir "${OUT_DIR}"

echo "Done. Plots + examples in ${OUT_DIR}/"
ls -la "${OUT_DIR}/"
ls -la "${OUT_DIR}/examples/" 2>/dev/null || true
