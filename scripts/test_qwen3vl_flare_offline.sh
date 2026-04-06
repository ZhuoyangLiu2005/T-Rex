#!/bin/bash
set -e

cd /mnt/amlfs-01/home/dniu/Project/dex-mot/mot/dex_mot_qwen/scripts
source /mnt/amlfs-01/home/dniu/anaconda3/bin/activate /mnt/amlfs-01/home/dniu/anaconda3/envs/dex_mot
export PATH=/mnt/amlfs-01/home/dniu/anaconda3/envs/dex_mot/bin:$PATH
export HF_HOME=/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/huggingface
export PYTHONPATH=/mnt/amlfs-01/home/dniu/Project/dex-mot/mot/dex_mot_qwen:$PYTHONPATH

export CUDA_VISIBLE_DEVICES=0

MODEL_PATH="/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/ckpts/dex_mot_qwen/exp/qwen3vl_mot_flare/qwen3vl_2b_tri_mot_pretrainvlm_remove_card_0405view2_tacdeform_wostate_deltabase_eef_stride2_f1s1_res_flare_resize_0405/checkpoint-49-41500"
python test_qwen3vl_flare_offline.py \
  --checkpoint_path ${MODEL_PATH} \
  --dataset_name 'rlbench' \
  --cuda 0 \
  --test_json_path /mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/data/bkl_inlab/training_data/three_full_json/remove_card_0405_deltabase_axis_eef_bimanual_stride2_train.json \
  --use_robot_state 0 \
  --use_tactile_deform 1 \
  --action_dim 62 \
  --action_chunk 16 \
  --save_dir ./test_output_flare \
  --num_test_samples 300 \
  --port 5678 \
  --image_size 384 288 \


