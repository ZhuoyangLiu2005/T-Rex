#!/bin/bash
# ---------------------------------------------------------------------------
# Train the F6-only tactile VQ-VAE on a merged midtrain root.
# Mirrors the env-var pattern used by scripts/midtrain.sh.
#
# Required env vars:
#   DATA_ROOT       midtrain merged root (same layout used by midtrain trainer)
#
# Optional env vars:
#   OUTPUT_DIR=<repo_root>/outputs/tactile_vqvae
#   RUN_NAME=vqvae_f6_w16_k${CODEBOOK}_${GRANULARITY}_$(date +%m%d_%H%M)
#   WINDOW=16, STRIDE=4, CODEBOOK=64, EMBED=256
#   EPOCHS=30, BATCH=256, LR=3e-4
#   GRANULARITY=finger        hand | finger
#   USE_WANDB=1, WANDB_API_KEY=
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# ---------------------------------------------------------------------------
set -euo pipefail

PARENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PARENT_DIR}"

: "${DATA_ROOT:?set DATA_ROOT to the merged midtrain data root}"

: "${OUTPUT_DIR:=${PARENT_DIR}/outputs/tactile_vqvae}"
: "${GRANULARITY:=finger}"
: "${WINDOW:=16}"
: "${STRIDE:=4}"
: "${CODEBOOK:=64}"
: "${EMBED:=256}"
: "${EPOCHS:=30}"
: "${BATCH:=256}"
: "${LR:=3e-4}"
: "${RUN_NAME:=vqvae_f6_w${WINDOW}_k${CODEBOOK}_${GRANULARITY}_$(date +%m%d_%H%M)}"

: "${USE_WANDB:=0}"
: "${WANDB_API_KEY:=}"
: "${WANDB_MODE:=${WANDB_API_KEY:+online}}"
: "${WANDB_MODE:=offline}"

: "${CUDA_VISIBLE_DEVICES:=0,1,2,3,4,5,6,7}"
N_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)

export CUDA_VISIBLE_DEVICES WANDB_API_KEY WANDB_MODE
export PYTHONPATH="${PARENT_DIR}:${PYTHONPATH:-}"

LOCAL_LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${LOCAL_LOG_DIR}"
LOCAL_LOG_FILE="${LOCAL_LOG_DIR}/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"
echo ">>> Mirroring training output to ${LOCAL_LOG_FILE}"
echo ">>> Using ${N_GPUS} GPU(s): ${CUDA_VISIBLE_DEVICES}"
echo ">>> data_root  = ${DATA_ROOT}"
echo ">>> output_dir = ${OUTPUT_DIR}/${RUN_NAME}"

accelerate launch \
    --num_processes "${N_GPUS}" \
    --num_machines 1 \
    --machine_rank 0 \
    --mixed_precision bf16 \
    -m tactile_vqvae.train \
    --data_root        "${DATA_ROOT}" \
    --output_dir       "${OUTPUT_DIR}" \
    --run_name         "${RUN_NAME}" \
    --window           "${WINDOW}" \
    --stride           "${STRIDE}" \
    --codebook_size    "${CODEBOOK}" \
    --embed_dim        "${EMBED}" \
    --epochs           "${EPOCHS}" \
    --batch_size       "${BATCH}" \
    --lr               "${LR}" \
    --num_workers      4 \
    --val_every        2000 \
    --use_wandb        "${USE_WANDB}" \
    --granularity      "${GRANULARITY}" \
    2>&1 | tee -a "${LOCAL_LOG_FILE}"

echo ">>> Done. Checkpoint: ${OUTPUT_DIR}/${RUN_NAME}/latest.pt"
