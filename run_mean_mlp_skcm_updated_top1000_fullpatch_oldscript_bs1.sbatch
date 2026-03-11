#!/bin/bash
#SBATCH --account=kwang222-ic
#SBATCH --partition=bioshare
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --output=/taiga/illinois/vetmed/cb/kwang222/enhancer/python_mean_mlp_skcm_updated_top1000_fullpatch_oldscript_bs1.o%j
#SBATCH --error=/taiga/illinois/vetmed/cb/kwang222/enhancer/python_mean_mlp_skcm_updated_top1000_fullpatch_oldscript_bs1.e%j

set -euo pipefail

source /taiga/illinois/vetmed/cb/kwang222/mingyang/codx/bin/activate
cd /taiga/illinois/vetmed/cb/kwang222/enhancer

SCRIPT=/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/average_embed/run_mean_regression.py
FEAT_ROOT=/taiga/illinois/vetmed/cb/kwang222/enhancer/our_code_data/data/features_patches
OUT_BASE=/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/average_embed/skcm_updated_top1000_mean_fullpatch_rerun_bs1

TRAIN_CSV=/taiga/illinois/vetmed/cb/kwang222/enhancer/super_enhancer/input/skcm_updated_top1000_samplewise/train.csv
VAL_CSV=/taiga/illinois/vetmed/cb/kwang222/enhancer/super_enhancer/input/skcm_updated_top1000_samplewise/validation.csv
TEST_CSV=/taiga/illinois/vetmed/cb/kwang222/enhancer/super_enhancer/input/skcm_updated_top1000_samplewise/test.csv

CACHE_DIR=/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/average_embed/skcm_updated_top1000_mean_fullpatch_rerun/cache
PCA_LIST=(0 2 5 10 20 50)

for K in "${PCA_LIST[@]}"; do
  if [ "${K}" -eq 0 ]; then
    TAG="noPCA"
  else
    TAG="PCAk${K}"
  fi

  echo "=== SKCM updated ${TAG} bs1 ==="
  python "${SCRIPT}" \
    --train-csv "${TRAIN_CSV}" \
    --val-csv "${VAL_CSV}" \
    --test-csv "${TEST_CSV}" \
    --feat-root "${FEAT_ROOT}" \
    --out-dir "${OUT_BASE}/SKCM_${TAG}_seed44" \
    --mean-cache-dir "${CACHE_DIR}" \
    --epochs 100 \
    --batch-size 1 \
    --lr 1e-3 \
    --weight-decay 1e-4 \
    --hidden-dim 512 \
    --dropout 0.2 \
    --seed 44 \
    --pca-k "${K}"
done

echo "Done SKCM updated fullpatch old-script bs1 runs."
