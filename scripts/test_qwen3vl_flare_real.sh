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
MODEL_PATH="/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/ckpts/dex_mot_qwen/exp/qwen3vl_mot_flare/stage2_future_v1/checkpoint-199-70400"
ACTION_DIM=31
ACTION_CHUNK=8

# --- Dual-arm: remove_card (bimanual, 2 wrist cameras: right + left) ---
# MODEL_PATH="/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/ckpts/dex_mot_qwen/exp/qwen3vl_mot_flare/qwen3vl_2b_tri_mot_pretrainvlm_remove_card_0405view2_tacdeform_wostate_deltabase_eef_stride2_f1s1_res_flare_resize_0405/checkpoint-199-166000"
# ACTION_DIM=62
# ACTION_CHUNK=16

# ────────────────────────────────────────────────────────────────────────────

python test_qwen3vl_flare_real.py \
  --checkpoint_path ${MODEL_PATH} \
  --dataset_name 'rlbench' \
  --cuda 0 \
  --use_robot_state 0 \
  --use_tactile_deform 1 \
  --action_dim ${ACTION_DIM} \
  --action_chunk ${ACTION_CHUNK} \
  --port 5678 \
  --image_size 384 288 \
  --use_tactile_refine_flow 1 \
  --action_flow_eval_steps 10 \
  --tactile_refine_flow_steps 4 \
  --tactile_refine_noise_scale 1.0 \
  --use_tactile_code 0 \
  --vqvae_codebook_size 64 \
  --vqvae_ckpt /mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/ckpts/dex_mot_expert/tactile_vqvae/vqvae_f6_w16_k64_finger_0507_0939/latest.pt
# To toggle:
#   --use_tactile_code 0 → server skips VQ-VAE encoding (back to F6+deform only).
#   --use_tactile_code 1 → server runs VQ-VAE on a rolling 16-frame F6 buffer
#     and feeds 2 (hand) or 10 (finger) code tokens to the tactile expert.
#     Granularity is auto-detected from the ckpt's config.
