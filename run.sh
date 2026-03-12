#!/bin/bash -l
#SBATCH --time=10:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-gpu=64
#SBATCH --gpus=1
#SBATCH --partition=gpu-h200-141g-ellis

source .venv/bin/activate

uv run train.py