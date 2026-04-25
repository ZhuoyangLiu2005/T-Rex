#!/bin/bash
set -e

cd /mnt/amlfs-01/home/dniu/Project/dex-mot/mot/dex_mot_qwen/scripts

source /mnt/amlfs-01/home/dniu/anaconda3/bin/activate /mnt/amlfs-01/home/dniu/anaconda3/envs/dex_mot
export PATH=/mnt/amlfs-01/home/dniu/anaconda3/envs/dex_mot/bin:$PATH
export HF_HOME=/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/huggingface
export PYTHONPATH=/mnt/amlfs-01/home/dniu/Project/dex-mot/mot/dex_mot_qwen:$PYTHONPATH

export WANDB_MODE=offline
export WANDB_API_KEY=5bdc90c568050775a6d10650e64857fbbc76742e
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

ORIGIN_MODEL_PATH="/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/ckpts/Qwen3-VL-2B-Instruct"
OUTPUT_ROOT_DIR="/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/ckpts/dex_mot_qwen/exp"
DATA_JSON="/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/data/bkl_inlab/training_data/nv_inlab_json/nv_inlab_deltabase_eef_bimanual_crop_stride1_train_traj100_seed42.json"
DEFORM_ENCODER_PATH="/mnt/amlfs-01/home/dniu/Project/dex-mot/mot/bi-mot/janus/DeformEncoder/ckpt/sharpa_wave_deform_encoder.pth"

EXPERIMENT_NAME="qwen3vl_mot_tac_aux"
RUN_NAME="qwen3vl_2b_tri_mot_test_nvinlab_view2_tac[force+deform]_histT8_tflare[tpf4step4stride2]_ctc_frc_flare[tpf4step8stride4]_resize_lr_$(date +%m%d)"
RESUME_CHECKPOINT="/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/ckpts/dex_mot_qwen/exp/qwen3vl_egodex_pretrain_flare/qwen3vl_2b_egodex_pretrain_bimanual_62d_stage1_handabs_flare_0407/checkpoint-0-115665"

MASTER_ADDR=${MASTER_ADDR:-10.244.165.78}
MASTER_PORT=${MASTER_PORT:-29500}
NUM_MACHINES=${NUM_MACHINES:-1}
MACHINE_RANK=${MACHINE_RANK:-0}
NUM_PROCESSES=$((NUM_MACHINES * 8))

TRAIN_BSZ=8
LR=1e-4

accelerate launch \
    --config_file ../config/sft_qwen.yaml \
    --num_processes ${NUM_PROCESSES} \
    --num_machines ${NUM_MACHINES} \
    --machine_rank ${MACHINE_RANK} \
    --main_process_ip ${MASTER_ADDR} \
    --main_process_port ${MASTER_PORT} \
    --deepspeed_multinode_launcher standard \
    train_qwen3vl_tac_aux.py \
    --model_path ${ORIGIN_MODEL_PATH} \
    --data_path ${DATA_JSON} \
    --n_epochs 100 \
    --save_freq 25 \
    --action_dim 62 \
    --action_chunk 16 \
    --train_bsz_per_gpu ${TRAIN_BSZ} \
    --learning_rate ${LR} \
    --min_lr_ratio 0 \
    --weight_decay 0 \
    --gradient_accumulation_steps 1 \
    --output_dir ${OUTPUT_ROOT_DIR} \
    --log_dir ${OUTPUT_ROOT_DIR} \
    --experiment_name ${EXPERIMENT_NAME} \
    --run_name ${RUN_NAME} \
    --use_robot_state 0 \
    --use_tactile_vec 1 \
    --use_tactile_deform 1 \
    --deform_encoder_ckpt ${DEFORM_ENCODER_PATH} \
    --tactile_intermediate_size 1536 \
    --training_stage 2 \
    --resume_checkpoint "${RESUME_CHECKPOINT}" \
    --use_flare 1 \
    --n_flare_tokens_per_frame 4 \
    --n_flare_steps 8 \
    --flare_loss_weight 0.5 \
    --flare_frame_stride 4 \
    --flare_layer_index -1 \
    --image_size 384 288 \
    --use_tactile_flare 1 \
    --n_tfl_tokens_per_step 4 \
    --n_tfl_steps 8 \
    --tactile_flare_stride 2 \
    --tflare_loss_weight 0.5 \
    --tactile_history_len 8 \
    --n_fingers 10 \
    --contact_loss_weight 0.5 \
    --force_loss_weight 0.3 \
    --contact_force_threshold 0.5 \
    --force_scale 2.0 \
    --log_freq 1 \
    --num_workers 24 \
    --prefetch_factor 4

# ── Notes ─────────────────────────────────────────────────────────────────
# Sequence: [slow | flare_q | fast | state | tac_f6 | tac_deform | t | x_t
#            | tflare_q | contact_q | force_q]
# Action expert reads tactile directly via causal attention (tac_f6 + tac_deform
# sit inside the action block). Tactile expert has no v_tac; only three sets of
# learnable query tokens routed to its block, each with a private head:
#   tflare_q  -> future tactile prediction (cosine, frozen-init target encoder)
#   contact_q -> per-finger contact classification (BCE, thresholded F6 label)
#   force_q   -> per-finger force regression (MSE, normalized |F_xyz|)
# Tactile history T=8 (current + 7 past) compressed via TacTemporalPool before
# entering the action block; same n_fingers tokens regardless of T.
# ──────────────────────────────────────────────────────────────────────────

echo ">>> Training (tactile-aux: tflare + contact + force) finished."
