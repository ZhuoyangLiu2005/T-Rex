#!/bin/bash
set -e

# Real-world ZMQ inference server (cascaded flow matching).
# Reads training_args.json from the checkpoint to auto-detect
# use_tactile_code, vqvae_codebook_size, and cascaded_{total,split}_step.

cd /mnt/amlfs-01/home/dniu/Project/dex-mot/mot/dex_mot_final/scripts
source /mnt/amlfs-01/home/dniu/anaconda3/bin/activate /mnt/amlfs-01/home/dniu/anaconda3/envs/dex_mot
export PATH=/mnt/amlfs-01/home/dniu/anaconda3/envs/dex_mot/bin:$PATH
export HF_HOME=/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/huggingface
export PYTHONPATH=/mnt/amlfs-01/home/dniu/Project/dex-mot/mot/dex_mot_final:$PYTHONPATH

export CUDA_VISIBLE_DEVICES=0

# --- Dual-arm cascaded expert: remove_card (bimanual, 2 wrist cameras) ---
MODEL_PATH="/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/ckpts/dex_mot_qwen/exp/qwen3vl_mot_expert/qwen3vl_2b_mot[3]_pretrain[mecka0507]_midtrain[none]_task[remove_card_0412+0413+0501]_traj[130]_view[3]_tac[force+deform]_state[wo]_stride[1]_flare[tpf4step8stride4]_vae[64]_cascaded_0511/checkpoint-99-31700"
ACTION_DIM=62
ACTION_CHUNK=16

python test_qwen3vl_flare_real.py \
  --checkpoint_path ${MODEL_PATH} \
  --dataset_name 'rlbench' \
  --cuda 0 \
  --use_robot_state 0 \
  --use_tactile_vec 1 \
  --use_tactile_deform 1 \
  --action_dim ${ACTION_DIM} \
  --action_chunk ${ACTION_CHUNK} \
  --port 5678 \
  --image_size 384 288 \
  --use_tactile_code 1 \
  --vqvae_codebook_size 64 \
  --vqvae_ckpt /mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/ckpts/dex_mot_expert/tactile_vqvae/vqvae_f6_w16_k64_finger_0507_0939/latest.pt
# To toggle:
#   --use_tactile_code 0 → server skips VQ-VAE encoding (back to F6+deform only).
#   --use_tactile_code 1 → server runs VQ-VAE on a rolling 16-frame F6 buffer
#     and feeds 2 (hand) or 10 (finger) code tokens to the tactile expert.
#     Granularity is auto-detected from the ckpt's config.
