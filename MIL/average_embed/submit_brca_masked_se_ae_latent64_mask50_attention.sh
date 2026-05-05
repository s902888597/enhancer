#!/bin/bash
#SBATCH --job-name=brcaseae
#SBATCH --account=bgcm-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/average_embed/zscore_union21477_erna_ge4_top3000_fixedsplit/logs/brca_masked_se_ae_latent64_mask50_%j.out
#SBATCH --error=/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/average_embed/zscore_union21477_erna_ge4_top3000_fixedsplit/logs/brca_masked_se_ae_latent64_mask50_%j.err

set -euo pipefail

ROOT=/taiga/illinois/vetmed/cb/kwang222/enhancer
RUN_ROOT=${ROOT}/MIL/average_embed/zscore_union21477_erna_ge4_top3000_fixedsplit
PY=/taiga/illinois/vetmed/cb/kwang222/mingyang/codx/bin/python
OUT=${RUN_ROOT}/brca_masked_se_ae_latent64_mask50_attention_k300_mixup

mkdir -p "${RUN_ROOT}/logs" "${OUT}"

"${PY}" "${ROOT}/MIL/average_embed/run_brca_masked_se_ae_attention.py" \
  --train-csv "${RUN_ROOT}/labels/BRCA_train_zscore_top3000_shared.csv" \
  --val-csv "${RUN_ROOT}/labels/BRCA_validation_zscore_top3000_shared.csv" \
  --test-csv "${RUN_ROOT}/labels/BRCA_test_zscore_top3000_shared.csv" \
  --feat-root "${RUN_ROOT}/features_by_cancer_case_symlinks" \
  --out-dir "${OUT}" \
  --latent-dim 64 \
  --mask-ratio 0.5 \
  --ae-epochs 300 \
  --ae-batch-size 32 \
  --ae-early-patience 30 \
  --epochs 50 \
  --batch-size 4 \
  --num-workers 4 \
  --max-patches 300 \
  --mixup-alpha 1.0 \
  --mixup-prob 0.5 \
  --recon-lambda 1.0 \
  --early-patience 5 \
  --seed 44
