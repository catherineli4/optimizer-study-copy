#!/bin/bash
#SBATCH --job-name=jolmo-sharp-vs-step
#SBATCH --output=/home/catheri4/optimizer-study-copy/logs/sharpness_vs_step/%j.out
#SBATCH --error=/home/catheri4/optimizer-study-copy/logs/sharpness_vs_step/%j.err
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=general
#SBATCH --gres=gpu:1
# Babel preempt QoS requires ≥1 GPU; this job is CPU-only but must request one.

# Download SharpnessEvaluation → plot λ_max vs step (one fig per chinchilla).
#
# Submit:
#   sbatch /home/catheri4/optimizer-study-copy/job_scripts/sbatch_download_plot_sharpness_vs_step.sh

set -euo pipefail

REPO=/home/catheri4/optimizer-study-copy
GS=gs://cmu-gpucloud-catheri4/Optim-60M-tuning/SharpnessEvaluation
LOCAL="${REPO}/results/SharpnessEvaluation"
OUT="${REPO}/results/sharpness_vs_step"

mkdir -p "${REPO}/logs/sharpness_vs_step" "${LOCAL}" "${OUT}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate optim-study

cd "${REPO}"

echo "================================================================"
echo "1) Download SharpnessEvaluation from GCS"
echo "================================================================"
gsutil -m rsync -r "${GS}/" "${LOCAL}/"
echo "Local files: $(ls -1 "${LOCAL}"/*.json 2>/dev/null | wc -l)"

echo "================================================================"
echo "2) Plot sharpness vs step → ${OUT}"
echo "================================================================"
python -m new_utils.plot_jolmo_sharpness_vs_step \
  --local-dir "${LOCAL}" \
  --out-dir "${OUT}"

echo "================================================================"
echo "Done."
echo "  JSONs:  ${LOCAL}/"
echo "  Plots:  ${OUT}/chinchilla-*-max_eigenvalue.png"
echo "          ${OUT}/chinchilla-*-max_eigenvalue_preconditioned.png"
echo "================================================================"
