#!/bin/bash
# Pre-bake VQ-VAE tactile codes into a post-training JSON (adds a tactile_codes field).
set -e

PROJECT_ROOT=/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/code/T-Rex
cd ${PROJECT_ROOT}

source /mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/code/miniconda3/bin/activate /mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/code/miniconda3/envs/dex_mot
export PYTHONPATH=${PROJECT_ROOT}:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0

INPUT_JSON="/path/to/training_data/three_full_json/place_card_train.json"
VQVAE_CKPT="/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/ckpts/dex_mot_qwen/vqvae/vqvae_f6_w16_k64_finger/latest.pt"
OUTPUT_JSON="${INPUT_JSON%.json}_vqvae_k64.json"
BATCH_SIZE=512

python -m utils.encode_vqvae_codes_to_json \
    --input_json ${INPUT_JSON} \
    --output_json ${OUTPUT_JSON} \
    --vqvae_ckpt ${VQVAE_CKPT} \
    --batch_size ${BATCH_SIZE} \
    --cuda 0

echo ">>> Done. Wrote ${OUTPUT_JSON}"
