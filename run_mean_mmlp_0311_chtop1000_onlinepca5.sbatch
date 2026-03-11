#!/bin/bash
#SBATCH --account=kwang222-ic
#SBATCH --partition=bioshare
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --output=/taiga/illinois/vetmed/cb/kwang222/enhancer/log_mmlp_0311_0110.o%j
#SBATCH --error=/taiga/illinois/vetmed/cb/kwang222/enhancer/log_mmlp_0311_0110.e%j

set -euo pipefail

source /taiga/illinois/vetmed/cb/kwang222/mingyang/codx/bin/activate
cd /taiga/illinois/vetmed/cb/kwang222/enhancer

SCRIPT=/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/average_embed/run_mean_regression_pan_cancer.py
FEAT_ROOT=/taiga/illinois/vetmed/cb/kwang222/enhancer/our_code_data/data/features_patches
OUT_BASE=/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/average_embed/skcm_updated_top1000_reusecache_onlinepca
CACHE_DIR=/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/average_embed/skcm_updated_top1000_mean_fullpatch_rerun/cache

COMMON_ARGS="--cancer SKCM --feat-root ${FEAT_ROOT} --feature-layout split_dirs --label-layout samples_as_rows --sample-id-mode identity --epochs 100 --batch-size 32 --lr 1e-3 --weight-decay 1e-4 --hidden-dim 512 --dropout 0.2 --seed 44 --mean-cache-dir ${CACHE_DIR}"

echo "=== SKCM updated reusecache onlinePCA PCAk5 ==="
python "${SCRIPT}" \
  --train-csv "/taiga/illinois/vetmed/cb/kwang222/enhancer/super_enhancer/input/skcm_updated_top1000_samplewise/train.csv" \
  --val-csv "/taiga/illinois/vetmed/cb/kwang222/enhancer/super_enhancer/input/skcm_updated_top1000_samplewise/validation.csv" \
  --test-csv "/taiga/illinois/vetmed/cb/kwang222/enhancer/super_enhancer/input/skcm_updated_top1000_samplewise/test.csv" \
  --out-dir "${OUT_BASE}/SKCM_PCAk5_seed44" \
  --pca-k 5 \
  ${COMMON_ARGS}

echo "Done SKCM updated reusecache onlinePCA PCAk5 run."
