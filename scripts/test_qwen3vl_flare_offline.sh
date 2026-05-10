#!/bin/bash
set -e

cd /mnt/amlfs-01/home/dniu/Project/dex-mot/mot/dex_mot_expert/scripts
source /mnt/amlfs-01/home/dniu/anaconda3/bin/activate /mnt/amlfs-01/home/dniu/anaconda3/envs/dex_mot
export PATH=/mnt/amlfs-01/home/dniu/anaconda3/envs/dex_mot/bin:$PATH
export HF_HOME=/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/huggingface
export PYTHONPATH=/mnt/amlfs-01/home/dniu/Project/dex-mot/mot/dex_mot_expert:$PYTHONPATH

export CUDA_VISIBLE_DEVICES=0

# # --- Single-arm: flip_book_page (right hand only, 1 wrist camera) ---
# MODEL_PATH="/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/ckpts/dex_mot_qwen/exp/qwen3vl_mot_flare/qwen3vl_2b_mot[3]_pretrain[none]_midtrain[bkl+nv_e8]_task[pick_egg_0411]_traj[100]_view[3]_tac[force+deform]_state[wo]_stride[1]_flare[tpf4step8stride4]_0427/checkpoint-49-13050"
# DATA_JSON="/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/data/bkl_inlab/training_data/three_full_json/pick_egg_0411_deltabase_axis_eef_right_crop_stride1_train.json"
# ACTION_DIM=31
# ACTION_CHUNK=16

# --- Dual-arm: remove_card (bimanual, 2 wrist cameras: right + left) ---
MODEL_PATH="/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/ckpts/dex_mot_qwen/exp/qwen3vl_mot_expert/qwen3vl_2b_mot[3]_pretrain[mecka0507]_midtrain[none]_task[remove_card_0412+0413+0501]_traj[130]_view[3]_tac[force+deform]_state[wo]_stride[1]_flare[tpf4step8stride4]_vae[64]_0507/checkpoint-149-47550"
# DATA_JSON is the *VQ-VAE encoded* JSON; tactile_codes length matches the
# checkpoint that produced it (2 for per-hand, 10 for per-finger).
DATA_JSON="/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/data/bkl_inlab/training_data/three_full_json/remove_card_0412+0413+0501_deltabase_axis_eef_lr_bimanual_crop_stride1_train_vqvae_k64.json"
ACTION_DIM=62
ACTION_CHUNK=16

python test_qwen3vl_flare_offline.py \
  --checkpoint_path ${MODEL_PATH} \
  --dataset_name 'rlbench' \
  --cuda 0 \
  --test_json_path ${DATA_JSON} \
  --use_robot_state 0 \
  --use_tactile_vec 1 \
  --use_tactile_deform 1 \
  --action_dim ${ACTION_DIM} \
  --action_chunk ${ACTION_CHUNK} \
  --save_dir ./test_output_flare \
  --num_test_samples 200 \
  --port 5678 \
  --n_flare_tokens_per_frame 4 \
  --n_flare_steps 8 \
  --flare_frame_stride 4 \
  --image_size 384 288 \
  --use_tactile_refine_flow 1 \
  --action_flow_eval_steps 10 \
  --tactile_refine_flow_steps 4 \
  --tactile_refine_noise_scale 1.0 \
  --use_tactile_code 1 \
  --vqvae_codebook_size 64 \
  --eval_smoothness 1 \
  --eval_async_consistency 1 \
  --refine_offsets 0 4 8 12 \
  --tactile_zero_init_noise 1 \


