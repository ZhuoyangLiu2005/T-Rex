#!/bin/bash

set -e

cd /mnt/amlfs-01/home/dniu/Project/dex-mot/mot/dex_mot/scripts
source /mnt/amlfs-01/home/dniu/anaconda3/bin/activate /mnt/amlfs-01/home/dniu/anaconda3/envs/last0
export PATH=/mnt/amlfs-01/home/dniu/anaconda3/envs/last0/bin:$PATH
export HF_HOME=/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/huggingface
export PYTHONPATH=/mnt/amlfs-01/home/dniu/Project/dex-mot/mot/dex_mot:/mnt/amlfs-01/home/dniu/Project/dex-mot/mot/dex_mot/transformers:$PYTHONPATH
export WANDB_MODE=online
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

BASE_RUN_NAME="janus_pro_siglip_1B_1e-4_tri_mot_pretrainvlm_pick_cube_0303_view2_tacdeform_state_deltacurr_joint_stride2_right_crop_f1s1_0310"
EXPERIMENT_NAME="janus_img_mot_flow_sharpa"
OUTPUT_ROOT_DIR="/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/ckpts/dex_mot/exp"

DATA_JSON="/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/data/bkl_inlab/training_data/three_full_json/pick_orange_cube_0303_deltacurr_joint_clip_right_stride2_train.json"
ORIGIN_MODEL_PATH="/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/Janus-pro-1B"
ACTION_MODEL_PATH="/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/Janus-pro-1B"
DEFORM_ENCODER_PATH="/mnt/amlfs-01/home/dniu/Project/dex-mot/mot/bi-mot/janus/DeformEncoder/ckpt/sharpa_wave_deform_encoder.pth"

NUM_PROCESSES=8
TRAIN_BSZ=8
LR=1e-4

STAGE1_SUB_NAME="${BASE_RUN_NAME}/stage0"

accelerate launch --config_file ../config/sft.yaml \
    --num_processes ${NUM_PROCESSES}  \
    --num_machines 1 \
    --machine_rank 0 \
    --deepspeed_multinode_launcher standard train_janus.py \
    --model_path ${ORIGIN_MODEL_PATH} \
    --action_model_path ${ACTION_MODEL_PATH} \
    --data_path ${DATA_JSON} \
    --data_root "" \
    --n_epochs 200 \
    --save_freq 50 \
    --action_dim 29 \
    --action_chunk 8 \
    --train_bsz_per_gpu ${TRAIN_BSZ} \
    --learning_rate ${LR} \
    --min_lr_ratio 0 \
    --weight_decay 0 \
    --gradient_accumulation_steps 1 \
    --output_dir ${OUTPUT_ROOT_DIR} \
    --log_dir ${OUTPUT_ROOT_DIR} \
    --experiment_name ${EXPERIMENT_NAME} \
    --load_action_from_latent 0 \
    --load_action_from_pretrain 1 \
    --use_robot_state 1 \
    --use_tactile_vec 0 \
    --use_tactile_deform 1 \
    --deform_encoder_ckpt ${DEFORM_ENCODER_PATH} \
    --use_pred 0 \
    --run_name ${STAGE1_SUB_NAME}

echo ">>> Stage 1 Finished."
