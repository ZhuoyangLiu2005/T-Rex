#!/bin/bash
# Real-world inference — ZMQ REP server (slow/fast cascaded protocol).
# Auto-detects training_args.json from the checkpoint (tactile_intermediate_size,
# use_tactile_code, vqvae_codebook_size, cascaded_{total,split}_step, ...).
set -e

PROJECT_ROOT=/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/code/T-Rex
cd ${PROJECT_ROOT}/scripts

source /mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/code/miniconda3/bin/activate /mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/code/miniconda3/envs/dex_mot
export PATH=/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/code/miniconda3/envs/dex_mot/bin:$PATH
export HF_HOME=/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/huggingface
export PYTHONPATH=${PROJECT_ROOT}:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0

MODEL_PATH="/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/ckpts/dex_mot_expert/exp/t-rex_posttrain_flare/checkpoint-99-12345"
# Only needed if the checkpoint was trained with --use_tactile_code 1 (external VQ-VAE).
VQVAE_CKPT="/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/ckpts/dex_mot_qwen/vqvae/vqvae_f6_w16_k64_finger/latest.pt"
PORT=5678
ACTION_DIM=62
ACTION_CHUNK=16
DATASET_NAME=rlbench

python test.py \
    --checkpoint_path ${MODEL_PATH} \
    --dataset_name ${DATASET_NAME} \
    --cuda 0 \
    --use_robot_state 0 \
    --use_tactile_vec 1 \
    --use_tactile_deform 1 \
    --use_tactile_code 1 \
    --action_dim ${ACTION_DIM} \
    --action_chunk ${ACTION_CHUNK} \
    --port ${PORT} \
    --image_size 384 288 \
    --vqvae_codebook_size 64 \
    --vqvae_ckpt ${VQVAE_CKPT}
