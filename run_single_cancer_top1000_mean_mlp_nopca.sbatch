#!/bin/bash
#SBATCH --job-name=meanmlp0
#SBATCH --account=bgcm-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --array=0-2
#SBATCH --output=/taiga/illinois/vetmed/cb/kwang222/enhancer/log_single_cancer_top1000_mean_mlp_nopca_%A_%a.out

set -euo pipefail

source /taiga/illinois/vetmed/cb/kwang222/mingyang/codx/bin/activate
cd /taiga/illinois/vetmed/cb/kwang222/enhancer

CANCERS=(BRCA LUAD SKCM)
CANCER=${CANCERS[$SLURM_ARRAY_TASK_ID]}
OUT_DIR=/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/average_embed/${CANCER,,}_top1000_mean_mlp_nopca_seed44

echo "hostname=$(hostname)"
echo "task_id=${SLURM_ARRAY_TASK_ID}"
echo "cancer=${CANCER}"
echo "target_top_n=1000"
echo "pca_k=0"
echo "out_dir=${OUT_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi || true

python -u /taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/average_embed/run_single_cancer_top1000_ypca_mean_mlp.py \
  --cancer "${CANCER}" \
  --target-top-n 1000 \
  --device cuda \
  --out-dir "${OUT_DIR}" \
  --pca-k 0 \
  --epochs 50 \
  --batch-size 32 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --hidden-dim 512 \
  --dropout 0.2 \
  --early-patience 5 \
  --min-delta 0.0 \
  --seed 44
