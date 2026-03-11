#!/bin/bash
#SBATCH --account=kwang222-ic
#SBATCH --partition=bioshare
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --output=/taiga/illinois/vetmed/cb/kwang222/enhancer/log_leg_0311_0000.o%j
#SBATCH --error=/taiga/illinois/vetmed/cb/kwang222/enhancer/log_leg_0311_0000.e%j

set -euo pipefail

source /taiga/illinois/vetmed/cb/kwang222/mingyang/codx/bin/activate
cd /taiga/illinois/vetmed/cb/kwang222/enhancer

SCRIPT=/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/average_embed/run_mean_regression_legacy.py
FEAT_ROOT=/taiga/illinois/vetmed/cb/kwang222/enhancer/our_code_data/data/features_patches
OUT_BASE=/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/average_embed/skcm_updated_top1000_reusepca_legacy

for K in 2 5 10 20 50; do
  echo "=== SKCM updated legacy reusepca PCAk${K} ==="
  python "${SCRIPT}" \
    --train-csv "/taiga/illinois/vetmed/cb/kwang222/enhancer/super_enhancer/input/skcm_updated_top1000_pca_k${K}/train_pca.csv" \
    --val-csv "/taiga/illinois/vetmed/cb/kwang222/enhancer/super_enhancer/input/skcm_updated_top1000_pca_k${K}/validation_pca.csv" \
    --test-csv "/taiga/illinois/vetmed/cb/kwang222/enhancer/super_enhancer/input/skcm_updated_top1000_pca_k${K}/test_pca.csv" \
    --feat-root "${FEAT_ROOT}" \
    --out-dir "${OUT_BASE}/SKCM_PCAk${K}_seed44" \
    --epochs 100 \
    --batch-size 32 \
    --lr 1e-3 \
    --weight-decay 1e-4 \
    --hidden-dim 512 \
    --dropout 0.2 \
    --seed 44
done

echo "Done SKCM updated legacy reusepca runs."
