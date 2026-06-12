#!/bin/bash
# Stage 3 / post-train — task-specific fine-tune, resuming from a midtrain ckpt.
# Two data formats: DATA_FORMAT=json (task JSON) or DATA_FORMAT=lerobot (LeRobot v3.0 dir).
set -e

PROJECT_ROOT=/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/code/T-Rex
cd ${PROJECT_ROOT}/scripts

source /mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/code/miniconda3/bin/activate /mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/code/miniconda3/envs/dex_mot
export PATH=/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/code/miniconda3/envs/dex_mot/bin:$PATH
export HF_HOME=/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/huggingface
export PYTHONPATH=${PROJECT_ROOT}:$PYTHONPATH
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

ORIGIN_MODEL_PATH="/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/ckpts/Qwen3-VL-2B-Instruct"
DEFORM_ENCODER_PATH="/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/ckpts/dex_mot_qwen/deform/sharpa_wave_deform_encoder.pth"
# Tactile codes are encoded ON THE FLY by the embedded VQ-VAE — no pre-baked
# codes needed.  If RESUME_CHECKPOINT was merged with an embedded VQ-VAE the
# trainer auto-detects it (VQVAE_CKPT then optional); otherwise it builds the
# module from VQVAE_CKPT.
VQVAE_CKPT="/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/ckpts/dex_mot_qwen/vqvae/vqvae_f6_w16_k64_finger/latest.pt"
RESUME_CHECKPOINT="/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/ckpts/dex_mot_expert/exp/qwen3vl_midtrain_flare/checkpoint-5-19464"
RESUME_SOURCE="midtrain"
OUTPUT_DIR="/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/ckpts/dex_mot_expert/exp"

EXPERIMENT_NAME="t-rex_posttrain_flare"
RUN_NAME="${EXPERIMENT_NAME}_$(date +%m%d_%H%M)"

ACTION_DIM=62

# ── data source: "json" or "lerobot" ──
DATA_FORMAT="json"
DATA_JSON="/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/data/bkl_inlab/training_data/three_full_json/remove_card_0413_deltabase_axis_eef_lr_bimanual_crop_stride1_train_vqvae_k64.json"
LEROBOT_ROOT="/path/to/lerobot/place_card"
if [ "${DATA_FORMAT}" = "lerobot" ]; then
    DATA_ARG="--data_format lerobot --lerobot_root ${LEROBOT_ROOT}"
else
    DATA_ARG="--data_format json --data_path ${DATA_JSON}"
fi

MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
NUM_MACHINES=${NUM_MACHINES:-1}
MACHINE_RANK=${MACHINE_RANK:-0}
NUM_PROCESSES=$((NUM_MACHINES * 8))

TRAIN_BSZ=16
LR=1e-4

accelerate launch \
    --config_file ../config/sft_qwen.yaml \
    --num_processes ${NUM_PROCESSES} \
    --num_machines ${NUM_MACHINES} \
    --machine_rank ${MACHINE_RANK} \
    --main_process_ip ${MASTER_ADDR} \
    --main_process_port ${MASTER_PORT} \
    --deepspeed_multinode_launcher standard \
    train.py \
    --model_path ${ORIGIN_MODEL_PATH} \
    ${DATA_ARG} \
    --n_epochs 100 \
    --save_freq 50 \
    --action_dim ${ACTION_DIM} \
    --action_chunk 16 \
    --train_bsz_per_gpu ${TRAIN_BSZ} \
    --learning_rate ${LR} \
    --min_lr_ratio 0 \
    --weight_decay 0 \
    --gradient_accumulation_steps 1 \
    --output_dir ${OUTPUT_DIR} \
    --log_dir ${OUTPUT_DIR} \
    --experiment_name ${EXPERIMENT_NAME} \
    --run_name ${RUN_NAME} \
    --use_robot_state 0 \
    --use_tactile_vec 1 \
    --use_tactile_deform 1 \
    --use_tactile_vqvae 1 \
    --vqvae_ckpt ${VQVAE_CKPT} \
    --deform_encoder_ckpt ${DEFORM_ENCODER_PATH} \
    --tactile_intermediate_size 1536 \
    --training_stage 2 \
    --tactile_loss_weight 1.0 \
    --cascaded_total_steps 10 \
    --cascaded_split_step 6 \
    --cascaded_tactile_dropout 0.1 \
    --cascaded_loss_weight 1.0 \
    --resume_checkpoint ${RESUME_CHECKPOINT} \
    --resume_source ${RESUME_SOURCE} \
    --use_flare 1 \
    --n_flare_tokens_per_frame 4 \
    --n_flare_steps 8 \
    --flare_loss_weight 0.5 \
    --flare_frame_stride 4 \
    --flare_layer_index -1 \
    --image_size 384 288 \
    --val_ratio 0.05 \
    --val_freq 500 \
    --max_val_batches 30

echo ">>> Post-training finished."
