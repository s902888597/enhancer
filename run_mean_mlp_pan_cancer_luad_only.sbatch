#!/bin/bash
#SBATCH --account=kwang222-ic
#SBATCH --partition=bioshare
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --output=/taiga/illinois/vetmed/cb/kwang222/enhancer/python_mean_mlp_luad_only.o%j
#SBATCH --error=/taiga/illinois/vetmed/cb/kwang222/enhancer/python_mean_mlp_luad_only.e%j

set -euo pipefail

source /taiga/illinois/vetmed/cb/kwang222/mingyang/codx/bin/activate
cd /taiga/illinois/vetmed/cb/kwang222/enhancer

SCRIPT=/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/average_embed/run_mean_regression_pan_cancer.py
FEAT_ROOT=/taiga/illinois/vetmed/cb/kwang222/enhancer/our_code_data/data/features_patches_pan_cancer_npy
OUT_BASE=/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/average_embed/pan_cancer_mean_mlp

LUAD_DIR=/taiga/illinois/vetmed/cb/kwang222/enhancer/sanity_check/clean_labels_rows/LUAD_189train_63val_64test_noleak_rows
COMMON_ARGS="--feat-root ${FEAT_ROOT} --epochs 100 --batch-size 32 --lr 1e-3 --weight-decay 1e-4 --hidden-dim 512 --dropout 0.2 --seed 44 --mean-cache-dir ${OUT_BASE}/cache"

PCA_LIST=(0 2 5 10 20 50)

for K in "${PCA_LIST[@]}"; do
  if [ "${K}" -eq 0 ]; then
    TAG="noPCA"
  else
    TAG="PCAk${K}"
  fi

  echo "=== LUAD ${TAG} ==="
  python "${SCRIPT}" \
    --cancer LUAD \
    --train-csv "${LUAD_DIR}/Top1000_SEs_189train_eRNA_zscore_3eRNAremoved.csv" \
    --val-csv "${LUAD_DIR}/Top1000_SEs_63val_eRNA_zscore_3eRNAremoved.csv" \
    --test-csv "${LUAD_DIR}/Top1000_SEs_64test_eRNA_zscore_3eRNAremoved.csv" \
    --out-dir "${OUT_BASE}/LUAD_${TAG}_seed44" \
    --pca-k "${K}" \
    ${COMMON_ARGS}
done

echo "Done LUAD runs."
