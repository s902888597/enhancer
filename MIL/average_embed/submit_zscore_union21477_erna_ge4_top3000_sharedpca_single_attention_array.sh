#!/bin/bash
#SBATCH --job-name=zs3shpcas
#SBATCH --account=bgcm-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --output=/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/average_embed/zscore_union21477_erna_ge4_top3000_fixedsplit/logs/sharedpca_single_%A_%a.out
#SBATCH --error=/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/average_embed/zscore_union21477_erna_ge4_top3000_fixedsplit/logs/sharedpca_single_%A_%a.err

set -euo pipefail

ROOT=/taiga/illinois/vetmed/cb/kwang222/enhancer
RUN_ROOT=${ROOT}/MIL/average_embed/zscore_union21477_erna_ge4_top3000_fixedsplit
PY=/taiga/illinois/vetmed/cb/kwang222/mingyang/codx/bin/python
PCA_K=${PCA_K:-5}
CANCERS=(BRCA LUAD SKCM)
CANCER=${CANCERS[$SLURM_ARRAY_TASK_ID]}
PCA_DIR=${RUN_ROOT}/shared_pca_labels/pca${PCA_K}

mkdir -p "${RUN_ROOT}/logs"

"${PY}" "${ROOT}/MIL/average_embed/run_skcm_top1000_attention_regression.py" \
  --cancer "${CANCER}" \
  --train-csv "${PCA_DIR}/${CANCER}_train_sharedpca${PCA_K}.csv" \
  --val-csv "${PCA_DIR}/${CANCER}_validation_sharedpca${PCA_K}.csv" \
  --test-csv "${PCA_DIR}/${CANCER}_test_sharedpca${PCA_K}.csv" \
  --feat-root "${RUN_ROOT}/features_by_cancer_case_symlinks" \
  --out-dir "${RUN_ROOT}/single_attention_k300_mixup_sharedpca${PCA_K}/${CANCER}" \
  --epochs 50 \
  --batch-size 8 \
  --num-workers 4 \
  --max-patches-train 300 \
  --max-patches-eval 300 \
  --mixup-alpha 1.0 \
  --mixup-prob 0.5 \
  --early-patience 5 \
  --seed 44
