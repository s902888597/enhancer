#!/bin/bash
#SBATCH --job-name=top1000atsp
#SBATCH --output=/taiga/illinois/vetmed/cb/kwang222/enhancer/log_top1000_ypca_attention_summary_%j.out
#SBATCH --error=/taiga/illinois/vetmed/cb/kwang222/enhancer/log_top1000_ypca_attention_summary_%j.err
#SBATCH --account=bgcm-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=24:00:00

set -euo pipefail

source /taiga/illinois/vetmed/cb/kwang222/mingyang/codx/bin/activate
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/tmp/mplconfig

python -u /taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/average_embed/summarize_three_cancer_top1000_ypca_attention_compare.py
