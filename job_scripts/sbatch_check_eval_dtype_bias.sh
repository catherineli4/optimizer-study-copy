#!/bin/bash
#SBATCH --job-name=eval-dtype-bias
#SBATCH --output=/home/catheri4/optimizer-study-copy/logs/eval_dtype_bias/%j.out
#SBATCH --error=/home/catheri4/optimizer-study-copy/logs/eval_dtype_bias/%j.err
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=general
#SBATCH --gres=gpu:1

# Measure how much the offline-eval bf16 cast inflates C4_val CE, separately for a
# Muon and an AdamW 0.1B chinchilla-4 checkpoint.
#
# Submit:
#   sbatch /home/catheri4/optimizer-study-copy/job_scripts/sbatch_check_eval_dtype_bias.sh

set -euo pipefail

REPO=/home/catheri4/optimizer-study-copy
GS=gs://cmu-gpucloud-catheri4/Optim-100M-tuning/JolmoModel
VAL_GS=gs://cmu-gpucloud-jspringe/outputs/diversity.pretraining.v2/ValidationDataset/Tokenized/C4_val.bin
WORK=/scratch/${USER:-catheri4}/eval_dtype_bias
[ -w /scratch ] 2>/dev/null || WORK=/tmp/eval_dtype_bias

MUON=MuonExpt3-0.1B-chinchilla-4-muon-muonlr7.0e-3-adamwlr1.0e-2-wsd
ADAMW=MuonExpt3-0.1B-chinchilla-4-adamw-lr1.0e-2-wsd

mkdir -p "${REPO}/logs/eval_dtype_bias" "${WORK}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate optim-study

for run in "${MUON}" "${ADAMW}"; do
  dst="${WORK}/${run}/final-unsharded"
  mkdir -p "${dst}"
  gsutil -m cp -n "${GS}/${run}/final-unsharded/model.pt" "${dst}/"
  gsutil -m cp    "${GS}/${run}/final-unsharded/config.json" "${dst}/"
done
gsutil -m cp -n "${VAL_GS}" "${WORK}/C4_val.bin"

cd "${REPO}/JOLMo"
python "${REPO}/scripts/check_eval_dtype_bias.py" \
  --checkpoint "muon=${WORK}/${MUON}/final-unsharded" \
  --checkpoint "adamw=${WORK}/${ADAMW}/final-unsharded" \
  --val-path "${WORK}/C4_val.bin" \
  --chunk-size 4096 \
  --batch-size 4 \
  --max-instances 256 \
  --output "${REPO}/results/eval_dtype_bias.json"

echo "wandb in-loop C4_val CE for reference: muon 3.720, adamw (chin-4) see run summary"
