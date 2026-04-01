#!/bin/bash
#SBATCH --job-name=top1000att
#SBATCH --output=/taiga/illinois/vetmed/cb/kwang222/enhancer/log_top1000_attention_compare_%A_%a.out
#SBATCH --error=/taiga/illinois/vetmed/cb/kwang222/enhancer/log_top1000_attention_compare_%A_%a.err
#SBATCH --account=bgcm-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --array=0-3

set -euo pipefail

source /taiga/illinois/vetmed/cb/kwang222/mingyang/codx/bin/activate
export PYTHONUNBUFFERED=1

SCRIPT=/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/average_embed/run_skcm_top1000_attention_regression.py
FEAT_ROOT=/taiga/illinois/vetmed/cb/kwang222/enhancer/our_code_data/data/features_patches_pan_cancer_npy

TASKS=(
  "BRCA full"
  "BRCA 300"
  "LUAD full"
  "LUAD 300"
)
read -r CANCER MODE <<< "${TASKS[$SLURM_ARRAY_TASK_ID]}"

if [[ "$CANCER" == "BRCA" ]]; then
  TRAIN_CSV=/taiga/illinois/vetmed/cb/kwang222/enhancer/shared_data/BRCA_192train_63val_64test_SE_eRNA_zscore_3eRNAremoved_csv_for_AI/Top1000_SEs_192train_eRNA_zscore_3eRNAremoved.csv
  VAL_CSV=/taiga/illinois/vetmed/cb/kwang222/enhancer/shared_data/BRCA_192train_63val_64test_SE_eRNA_zscore_3eRNAremoved_csv_for_AI/Top1000_SEs_63val_eRNA_zscore_3eRNAremoved.csv
  TEST_CSV=/taiga/illinois/vetmed/cb/kwang222/enhancer/shared_data/BRCA_192train_63val_64test_SE_eRNA_zscore_3eRNAremoved_csv_for_AI/Top1000_SEs_64test_eRNA_zscore_3eRNAremoved.csv
else
  TRAIN_CSV=/taiga/illinois/vetmed/cb/kwang222/enhancer/shared_data/LUAD_189train_63val_64test_SE_eRNA_zscore_3eRNAremoved_csv_for_AI_noleak/Top1000_SEs_189train_eRNA_zscore_3eRNAremoved.csv
  VAL_CSV=/taiga/illinois/vetmed/cb/kwang222/enhancer/shared_data/LUAD_189train_63val_64test_SE_eRNA_zscore_3eRNAremoved_csv_for_AI_noleak/Top1000_SEs_63val_eRNA_zscore_3eRNAremoved.csv
  TEST_CSV=/taiga/illinois/vetmed/cb/kwang222/enhancer/shared_data/LUAD_189train_63val_64test_SE_eRNA_zscore_3eRNAremoved_csv_for_AI_noleak/Top1000_SEs_64test_eRNA_zscore_3eRNAremoved.csv
fi

if [[ "$MODE" == "full" ]]; then
  MAX_PATCHES=0
  OUT_DIR=/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/average_embed/${CANCER,,}_top1000_attention_rawpatch_seed44
else
  MAX_PATCHES=300
  OUT_DIR=/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/average_embed/${CANCER,,}_top1000_attention_rawpatch_k300_seed44
fi

python -u "${SCRIPT}" \
  --cancer "${CANCER}" \
  --train-csv "${TRAIN_CSV}" \
  --val-csv "${VAL_CSV}" \
  --test-csv "${TEST_CSV}" \
  --feat-root "${FEAT_ROOT}" \
  --out-dir "${OUT_DIR}" \
  --epochs 50 \
  --batch-size 8 \
  --num-workers 8 \
  --lr 1e-4 \
  --weight-decay 1e-5 \
  --embed-dim 256 \
  --attn-dim 128 \
  --hidden-dim 512 \
  --dropout 0.2 \
  --max-patches-train "${MAX_PATCHES}" \
  --max-patches-eval "${MAX_PATCHES}" \
  --seed 44 \
  --early-patience 5 \
  --min-delta 0.0
