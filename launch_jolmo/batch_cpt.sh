#!/bin/bash
#SBATCH --job-name=batch_cpt
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:1  
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
set -euo pipefail

gsutil ls gs://cmu-gpucloud-catheri4/Optim-60M-tuning/CPTModel

