#!/bin/bash
#SBATCH --job-name=kvar3kdh
#SBATCH --account=bgcm-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/average_embed/kenneth_mergedtrain_zscore_variance_top3000_fixedsplit/logs/mixed_dualhead_%j.out
#SBATCH --error=/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/average_embed/kenneth_mergedtrain_zscore_variance_top3000_fixedsplit/logs/mixed_dualhead_%j.err

set -euo pipefail

ROOT=/taiga/illinois/vetmed/cb/kwang222/enhancer
RUN_ROOT=${ROOT}/MIL/average_embed/kenneth_mergedtrain_zscore_variance_top3000_fixedsplit
PY=/taiga/illinois/vetmed/cb/kwang222/mingyang/codx/bin/python

mkdir -p "${RUN_ROOT}/logs"

"${PY}" "${ROOT}/MIL/average_embed/run_dualhead_three_cancer_attention_masked.py" \
  --matrix-dir "${RUN_ROOT}/matrix" \
  --feat-root "${RUN_ROOT}/features_by_cancer_case_symlinks" \
  --out-dir "${RUN_ROOT}/mixed_dualhead_attention_k300_mixup" \
  --epochs 50 \
  --batch-size 4 \
  --num-workers 4 \
  --max-patches 300 \
  --mixup-alpha 1.0 \
  --mixup-prob 0.5 \
  --early-patience 5 \
  --seed 44
