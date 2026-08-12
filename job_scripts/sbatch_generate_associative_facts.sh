#!/bin/bash
#SBATCH --job-name=gen-assoc-facts
#SBATCH --output=/home/catheri4/optimizer-study-copy/logs/associative_facts/%j.out
#SBATCH --error=/home/catheri4/optimizer-study-copy/logs/associative_facts/%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=general
#SBATCH --gres=gpu:1
# Babel QoS often requires a GPU even for CPU jobs.

# Generate associative-fact memmaps (<bos> v[126] r u[126] <eos> + pad → 256)
# and upload to GCS.
#
# Default scale: 6M train facts ≈ 1.536B tokens (entity vocab 50k).
# Override with env vars NUM_FACTS / ENTITY_VOCAB / OUT_DIR / GCS_URI.
#
# Submit (create log dir first if needed):
#   mkdir -p logs/associative_facts
#   sbatch job_scripts/sbatch_generate_associative_facts.sh

set -euo pipefail

REPO=/home/catheri4/optimizer-study-copy
# Use node-local scratch under the known Babel user prefix (login nodes
# do not mount /scratch; compute nodes do under catheri4 / catheri4-outputs).
OUT_DIR="${OUT_DIR:-/scratch/catheri4-outputs/associative_facts_v3}"
GCS_URI="${GCS_URI:-gs://cmu-gpucloud-catheri4/datasets/associative_facts_v3}"
NUM_FACTS="${NUM_FACTS:-6000000}"
NUM_VAL_FACTS="${NUM_VAL_FACTS:-8192}"
ENTITY_VOCAB="${ENTITY_VOCAB:-50000}"
SHARD_FACTS="${SHARD_FACTS:-1000000}"

mkdir -p "${REPO}/logs/associative_facts" "${OUT_DIR}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate optim-study
cd "${REPO}"

echo "Host: $(hostname)"
echo "Generating ${NUM_FACTS} facts → ${OUT_DIR}"
echo "Upload destination: ${GCS_URI}"
df -h "${OUT_DIR}" | tail -1 || df -h /scratch | tail -1 || true

python scripts/generate_associative_facts.py \
  --out-dir "${OUT_DIR}" \
  --num-facts "${NUM_FACTS}" \
  --num-val-facts "${NUM_VAL_FACTS}" \
  --entity-vocab-size "${ENTITY_VOCAB}" \
  --shard-facts "${SHARD_FACTS}" \
  --upload "${GCS_URI}"

echo "Done. metadata:"
cat "${OUT_DIR}/metadata.json" | head -60
echo
echo "Verify GCS:"
gsutil ls "${GCS_URI}/" || true
gsutil ls "${GCS_URI}/train/" | wc -l
gsutil ls "${GCS_URI}/train_label_mask/" | wc -l
echo
echo "Train with:"
echo "  ASSOCIATIVE_FACTS_TRAIN_SHARDS=\$(( (${NUM_FACTS} + ${SHARD_FACTS} - 1) / ${SHARD_FACTS} )) \\"
echo "    python -m launch_jolmo.launcher launch pretrain-associative-facts"
