#!/bin/bash
# Bake a standalone VQ-VAE checkpoint into a trained VLA checkpoint (embedded, on-the-fly tokenizer).
set -e

PROJECT_ROOT=/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/code/T-Rex
cd ${PROJECT_ROOT}

source /mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/code/miniconda3/bin/activate /mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/code/miniconda3/envs/dex_mot
export PYTHONPATH=${PROJECT_ROOT}:$PYTHONPATH

VLA_CKPT="/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/ckpts/dex_mot_expert/exp/t-rex_posttrain_flare/checkpoint-99-12345"
VQVAE_CKPT="/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/ckpts/dex_mot_qwen/vqvae/vqvae_f6_w16_k64_finger/latest.pt"
OUTPUT="${VLA_CKPT}-vqvae"

python utils/merge_vqvae_into_ckpt.py \
    --vla_ckpt ${VLA_CKPT} \
    --vqvae_ckpt ${VQVAE_CKPT} \
    --output ${OUTPUT}

echo ">>> Merged checkpoint: ${OUTPUT}"
