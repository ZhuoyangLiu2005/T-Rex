#!/bin/bash
set -e

cd /mnt/amlfs-01/home/dniu/Project/dex-mot/mot/dex_mot_qwen/scripts
source /mnt/amlfs-01/home/dniu/anaconda3/bin/activate /mnt/amlfs-01/home/dniu/anaconda3/envs/dex_mot
export PATH=/mnt/amlfs-01/home/dniu/anaconda3/envs/dex_mot/bin:$PATH
export HF_HOME=/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/huggingface
export PYTHONPATH=/mnt/amlfs-01/home/dniu/Project/dex-mot/mot/dex_mot_qwen:$PYTHONPATH

export CUDA_VISIBLE_DEVICES=0

# Bimanual erase_whiteboard (62d, chunk=16), trained with tac_aux
MODEL_PATH="/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/ckpts/dex_mot_qwen/exp/qwen3vl_mot_tac_aux/qwen3vl_2b_tri_mot_flip_book_page_0405_view2_tac[force+deform]_histT8_tflare[tpf4step4stride2]_ctc_frc_flare[tpf4step8stride4]_resize_lr_0424/checkpoint-99-57800"
DATA_JSON="/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/data/bkl_inlab/training_data/three_full_json/flip_book_page_0405_deltabase_axis_eef_newclip_right_stride2_train.json"
ACTION_DIM=31
ACTION_CHUNK=16

python test_qwen3vl_tac_aux_offline.py \
  --checkpoint_path ${MODEL_PATH} \
  --dataset_name 'rlbench' \
  --cuda 0 \
  --test_json_path ${DATA_JSON} \
  --use_robot_state 0 \
  --use_tactile_vec 1 \
  --use_tactile_deform 1 \
  --action_dim ${ACTION_DIM} \
  --action_chunk ${ACTION_CHUNK} \
  --tactile_history_len 8 \
  --n_fingers 5 \
  --contact_force_threshold 0.5 \
  --force_scale 2.0 \
  --include_tactile_queries 1 \
  --save_dir ./test_output_tac_aux \
  --num_test_samples 300 \
  --port 5678 \
  --image_size 384 288

# Outputs:
#   action_dim_{i}.png             per-dim pred vs GT
#   action_trajectory.npz
#   contact_per_finger.png          per-finger contact accuracy + AUC
#   contact_vs_force.png            scatter of predicted prob vs raw |F_xyz|
#   contact_diagnostics.npz
#   force_mae_per_finger.png        per-finger force regression MAE
#   force_diagnostics.npz
#   tflare_sim_per_step.png         cos_sim per future tactile step
#   tflare_similarity.npz
