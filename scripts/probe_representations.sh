#!/bin/bash
set -e

cd /mnt/amlfs-01/home/dniu/Project/dex-mot/mot/dex_mot_qwen/scripts
source /mnt/amlfs-01/home/dniu/anaconda3/bin/activate /mnt/amlfs-01/home/dniu/anaconda3/envs/dex_mot
export PATH=/mnt/amlfs-01/home/dniu/anaconda3/envs/dex_mot/bin:$PATH
export HF_HOME=/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/huggingface
export PYTHONPATH=/mnt/amlfs-01/home/dniu/Project/dex-mot/mot/dex_mot_qwen:$PYTHONPATH

export CUDA_VISIBLE_DEVICES=0

# --- Bimanual erase_whiteboard (62d, chunk=16), trained with tflare_gate ---
MODEL_PATH="/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/ckpts/dex_mot_qwen/exp/qwen3vl_mot_tflare_gate/qwen3vl_2b_tri_mot_flip_book_page_0405_traj[100]_view2_tac[force+deform]_gate[perdim]_tflare[tpf4step4stride2]_solo_flare[tpf4step8stride4]_resize_lr_0422/checkpoint-49-14450"
DATA_JSON="/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/data/bkl_inlab/training_data/three_full_json/flip_book_page_0405_deltabase_axis_eef_newclip_right_stride2_train.json"
ACTION_DIM=31
ACTION_CHUNK=16

python probe_representations.py \
  --checkpoint_path ${MODEL_PATH} \
  --test_json_path ${DATA_JSON} \
  --num_samples 500 \
  --cuda 0 \
  --use_robot_state 0 \
  --use_tactile_vec 1 \
  --use_tactile_deform 1 \
  --action_dim ${ACTION_DIM} \
  --action_chunk ${ACTION_CHUNK} \
  --save_dir ./probe_output_tflare_gate \
  --image_size 384 288

# Saved outputs (under save_dir):
#   probe_heatmap.png            expert × target linear-probe score matrix
#   cka_matrix.png               linear CKA across experts + vision/tactile refs
#   gate_vs_contact.png          mean gate vs raw F6 magnitude
#   vtac_vact_diff_vs_contact.png   expert output divergence vs contact
#   gate_per_action_dim.png      bar plot of per-dim gate openness
#   probing_raw.npz              raw arrays for re-plotting
