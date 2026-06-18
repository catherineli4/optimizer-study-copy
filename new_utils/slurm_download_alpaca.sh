#!/usr/bin/env bash

#SBATCH --job-name=download_alpaca
#SBATCH --output=logs/download_alpaca_%j.out
#SBATCH --error=logs/download_alpaca_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1 
#SBATCH --partition=general
#SBATCH --requeue

source ~/miniconda3/etc/profile.d/conda.sh
conda activate myenv310

# Run the downloader
python catastrophic-forgetting/new_utils/download_alpaca.py