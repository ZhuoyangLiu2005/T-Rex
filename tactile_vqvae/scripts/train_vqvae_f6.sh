#!/bin/bash
# Train the F6-only Tactile VQ-VAE on the merged midtrain root.
# Mirrors the env-var pattern used by train_qwen3vl_midtrain_flare.sh.

set -e
set -o pipefail

PARENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PARENT_DIR}"

source /mnt/amlfs-01/home/dniu/anaconda3/bin/activate /mnt/amlfs-01/home/dniu/anaconda3/envs/dex_mot
export PATH=/mnt/amlfs-01/home/dniu/anaconda3/envs/dex_mot/bin:$PATH
export PYTHONPATH=${PARENT_DIR}:${PYTHONPATH}

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
N_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)

DATA_ROOT="${DATA_ROOT:-/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/data/midtrain/merged}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/ckpts/dex_mot_expert/tactile_vqvae}"
RUN_NAME="${RUN_NAME:-vqvae_f6_w16_k1024_$(date +%m%d_%H%M)}"

WINDOW=${WINDOW:-16}
STRIDE=${STRIDE:-4}
CODEBOOK=${CODEBOOK:-1024}
EMBED=${EMBED:-256}
EPOCHS=${EPOCHS:-30}
BATCH=${BATCH:-256}
LR=${LR:-3e-4}

# Logging
USE_WANDB=1
LOCAL_LOG_DIR="${PARENT_DIR}/tactile_vqvae/logs"
mkdir -p "${LOCAL_LOG_DIR}"
LOCAL_LOG_FILE="${LOCAL_LOG_DIR}/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"
echo ">>> Mirroring training output to ${LOCAL_LOG_FILE}"
echo ">>> Using ${N_GPUS} GPU(s): ${CUDA_VISIBLE_DEVICES}"
echo ">>> data_root  = ${DATA_ROOT}"
echo ">>> output_dir = ${OUTPUT_DIR}/${RUN_NAME}"

accelerate launch \
    --num_processes ${N_GPUS} \
    --num_machines 1 \
    --machine_rank 0 \
    --mixed_precision bf16 \
    -m tactile_vqvae.train \
    --data_root        "${DATA_ROOT}" \
    --output_dir       "${OUTPUT_DIR}" \
    --run_name         "${RUN_NAME}" \
    --window           ${WINDOW} \
    --stride           ${STRIDE} \
    --codebook_size    ${CODEBOOK} \
    --embed_dim        ${EMBED} \
    --epochs           ${EPOCHS} \
    --batch_size       ${BATCH} \
    --lr               ${LR} \
    --num_workers      4 \
    --val_every        2000 \
    --use_wandb        ${USE_WANDB} \
    2>&1 | tee -a "${LOCAL_LOG_FILE}"

echo ">>> Done. Checkpoint: ${OUTPUT_DIR}/${RUN_NAME}/latest.pt"
