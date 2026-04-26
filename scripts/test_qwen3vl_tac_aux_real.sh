#!/bin/bash
set -e

cd /mnt/amlfs-01/home/dniu/Project/dex-mot/mot/dex_mot_qwen/scripts
source /mnt/amlfs-01/home/dniu/anaconda3/bin/activate /mnt/amlfs-01/home/dniu/anaconda3/envs/dex_mot
export PATH=/mnt/amlfs-01/home/dniu/anaconda3/envs/dex_mot/bin:$PATH
export HF_HOME=/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/huggingface
export PYTHONPATH=/mnt/amlfs-01/home/dniu/Project/dex-mot/mot/dex_mot_qwen:$PYTHONPATH

export CUDA_VISIBLE_DEVICES=0

# Bimanual erase_whiteboard (62d, chunk=16), trained with tac_aux
MODEL_PATH="/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/ckpts/dex_mot_qwen/exp/qwen3vl_mot_tac_aux/TODO_RUN_NAME/checkpoint-199-TODO"
ACTION_DIM=62
ACTION_CHUNK=16

python test_qwen3vl_tac_aux_real.py \
  --checkpoint_path ${MODEL_PATH} \
  --dataset_name 'rlbench' \
  --cuda 0 \
  --use_robot_state 0 \
  --use_tactile_vec 1 \
  --use_tactile_deform 1 \
  --action_dim ${ACTION_DIM} \
  --action_chunk ${ACTION_CHUNK} \
  --n_fingers 10 \
  --include_tactile_queries 0 \
  --port 5678 \
  --image_size 384 288
