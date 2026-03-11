#!/bin/bash
#SBATCH --account=kwang222-ic			## account to be charged
#SBATCH --partition=bioshare			## queue
#SBATCH --gres=gpu:A40
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=python_luad_job.o%j 			## file that will receive output from execution
#SBATCH --error=python_luad_job.e%j			## file that will receive any error messages

set -euo pipefail

source /taiga/illinois/vetmed/cb/kwang222/mingyang/codx/bin/activate
cd /taiga/illinois/vetmed/cb/kwang222/enhancer

OUT_DIR=/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/average_embed/skcm_updated_top1500_transformer_pca_k5_mp300_tta5_seed44
mkdir -p "$OUT_DIR"

python /taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/average_embed/run_transformer_regression.py \
  --train-csv /taiga/illinois/vetmed/cb/kwang222/enhancer/super_enhancer/input/skcm_updated_top1500_pca_k5/train_pca.csv \
  --val-csv /taiga/illinois/vetmed/cb/kwang222/enhancer/super_enhancer/input/skcm_updated_top1500_pca_k5/validation_pca.csv \
  --test-csv /taiga/illinois/vetmed/cb/kwang222/enhancer/super_enhancer/input/skcm_updated_top1500_pca_k5/test_pca.csv \
  --feat-root /taiga/illinois/vetmed/cb/kwang222/enhancer/our_code_data/data/features_patches \
  --out-dir "$OUT_DIR" \
  --epochs 100 \
  --batch-size 1 \
  --max-patches 300 \
  --test-avg-runs 5 \
  --lr 1e-4 \
  --weight-decay 1e-5 \
  --num-layers 4 \
  --dim-feedforward 512 \
  --d-model 256 \
  --nhead 8 \
  --dropout 0.1 \
  --seed 44
