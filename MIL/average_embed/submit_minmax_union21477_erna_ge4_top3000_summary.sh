#!/bin/bash
#SBATCH --job-name=mm3ge4sum
#SBATCH --account=bgcm-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/average_embed/minmax_union21477_erna_ge4_top3000_fixedsplit/logs/summary_%j.out
#SBATCH --error=/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/average_embed/minmax_union21477_erna_ge4_top3000_fixedsplit/logs/summary_%j.err

set -euo pipefail

ROOT=/taiga/illinois/vetmed/cb/kwang222/enhancer
PY=/taiga/illinois/vetmed/cb/kwang222/mingyang/codx/bin/python

"${PY}" "${ROOT}/MIL/average_embed/summarize_minmax_union21477_erna_ge4_top3000_fixedsplit.py"
