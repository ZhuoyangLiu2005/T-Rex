#!/bin/bash
set -e

cd /mnt/amlfs-01/home/dniu/Project/dex-mot/mot/last0_img/scripts
source /mnt/amlfs-01/home/dniu/anaconda3/bin/activate /mnt/amlfs-01/home/dniu/anaconda3/envs/last0
export PATH=/mnt/amlfs-01/home/dniu/anaconda3/envs/last0/bin:$PATH
export HF_HOME=/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/huggingface
export PYTHONPATH=/mnt/amlfs-01/home/dniu/Project/dex-mot/mot/last0_img:/mnt/amlfs-01/home/dniu/Project/dex-mot/mot/last0_img/transformers:$PYTHONPATH

export CUDA_VISIBLE_DEVICES=0 

MODEL_PATH="/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/ckpts/last0_img/exp/janus_img_mot_flow_sharpa/janus_pro_siglip_1B_1e-4_tri_mot_pretrainvlm_pick_cube_view3_vanilla_f1s1_0223/stage0/checkpoint-99-33200/tfmr"

python test_janus_real.py \
  --model_path ${MODEL_PATH} \
  --dataset_name 'rlbench' \
  --cuda 0 \
  --test_json_path /mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/data/bkl_inlab/training_data/three_json/pick_orange_cube_train.json \
  --use_robot_state 1 \
  --use_rtc 0 \
  --use_pred 0 \
  --action_dim 58 \
  --action_chunk 32 \
  --port 5555