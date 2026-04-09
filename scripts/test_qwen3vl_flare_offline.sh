#!/bin/bash
set -e

cd /mnt/amlfs-01/home/dniu/Project/dex-mot/mot/dex_mot_qwen/scripts
source /mnt/amlfs-01/home/dniu/anaconda3/bin/activate /mnt/amlfs-01/home/dniu/anaconda3/envs/dex_mot
export PATH=/mnt/amlfs-01/home/dniu/anaconda3/envs/dex_mot/bin:$PATH
export HF_HOME=/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/huggingface
export PYTHONPATH=/mnt/amlfs-01/home/dniu/Project/dex-mot/mot/dex_mot_qwen:$PYTHONPATH

export CUDA_VISIBLE_DEVICES=0

# ── Task config: uncomment ONE section ──────────────────────────────────────

# --- Single-arm: flip_book_page (right hand only, 1 wrist camera) ---
# MODEL_PATH="/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/ckpts/dex_mot_qwen/exp/qwen3vl_mot_flare/stage2_future_v1/checkpoint-199-70400"
# DATA_JSON="/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/data/bkl_inlab/training_data/three_full_json/flip_book_page_0405_deltabase_axis_eef_newclip_right_stride2_train.json"
# ACTION_DIM=31
# ACTION_CHUNK=8

# --- Dual-arm: remove_card (bimanual, 2 wrist cameras: right + left) ---
MODEL_PATH="/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/ckpts/dex_mot_qwen/exp/qwen3vl_mot_flare/qwen3vl_2b_tri_mot_pretrainvlm_remove_card_0405view2_tacdeform_wostate_deltabase_eef_stride2_f1s1_res_flare_resize_0405/checkpoint-199-166000"
DATA_JSON="/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/data/bkl_inlab/training_data/three_full_json/remove_card_0405_deltabase_axis_eef_bimanual_stride2_train.json"
ACTION_DIM=62
ACTION_CHUNK=16

# ────────────────────────────────────────────────────────────────────────────

python test_qwen3vl_flare_offline.py \
  --checkpoint_path ${MODEL_PATH} \
  --dataset_name 'rlbench' \
  --cuda 0 \
  --test_json_path ${DATA_JSON} \
  --use_robot_state 0 \
  --use_tactile_deform 1 \
  --action_dim ${ACTION_DIM} \
  --action_chunk ${ACTION_CHUNK} \
  --save_dir ./test_output_flare \
  --num_test_samples 300 \
  --port 5678 \
  --n_flare_tokens_per_frame 1 \
  --n_flare_steps 8 \
  --flare_frame_stride 4 \
  --image_size 384 288 \


