#!/bin/bash
set -e

cd /mnt/amlfs-01/home/dniu/Project/dex-mot/mot/dex_mot_qwen/scripts

source /mnt/amlfs-01/home/dniu/anaconda3/bin/activate /mnt/amlfs-01/home/dniu/anaconda3/envs/dex_mot
export PATH=/mnt/amlfs-01/home/dniu/anaconda3/envs/dex_mot/bin:$PATH
export HF_HOME=/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/huggingface
export PYTHONPATH=/mnt/amlfs-01/home/dniu/Project/dex-mot/mot/dex_mot_qwen:$PYTHONPATH

export WANDB_MODE=online
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# export NCCL_DEBUG=INFO # only for communication debug
export NCCL_DEBUG_SUBSYS=INIT,NET

BASE_RUN_NAME="qwen3vl_2b_tri_mot_pretrainvlm_flip_book_page_0313view2_tacdeform_wostate_deltabase_eef_stride2_f1s1_res_resize_0331"
EXPERIMENT_NAME="qwen3vl_mot_flow"
OUTPUT_ROOT_DIR="/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/ckpts/dex_mot_qwen/exp"

DATA_JSON="/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/data/bkl_inlab/training_data/three_full_json/flip_book_page_0313_deltabase_axis_eef_newclip_right_stride2_train.json"

# Qwen3vl-2B pretrained model path
ORIGIN_MODEL_PATH="/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/ckpts/Qwen3-VL-2B-Instruct"
DEFORM_ENCODER_PATH="/mnt/amlfs-01/home/dniu/Project/dex-mot/mot/bi-mot/janus/DeformEncoder/ckpt/sharpa_wave_deform_encoder.pth"
# Stage 1 resume checkpoint (set to stage1 model.pt for stage 2 training)
RESUME_CHECKPOINT=""

MASTER_ADDR=10.244.27.42   # run 'ifconfig' to get the ip address of eth0
MASTER_PORT=29500
NUM_MACHINES=1
MACHINE_RANK=0 # remember to modify in different nodes
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
    train_qwen3vl.py \
    --model_path ${ORIGIN_MODEL_PATH} \
    --data_path ${DATA_JSON} \
    --data_root "" \
    --n_epochs 200 \
    --save_freq 50 \
    --action_dim 31 \
    --action_chunk 8 \
    --train_bsz_per_gpu ${TRAIN_BSZ} \
    --learning_rate ${LR} \
    --min_lr_ratio 0 \
    --weight_decay 0 \
    --gradient_accumulation_steps 1 \
    --output_dir ${OUTPUT_ROOT_DIR} \
    --log_dir ${OUTPUT_ROOT_DIR} \
    --experiment_name ${EXPERIMENT_NAME} \
    --run_name ${BASE_RUN_NAME} \
    --use_robot_state 0 \
    --use_tactile_vec 0 \
    --use_tactile_deform 1 \
    --deform_encoder_ckpt ${DEFORM_ENCODER_PATH} \
    --tactile_intermediate_size 1536 \
    --training_stage 2 \
    --tactile_loss_weight 1.0 \
    --resume_checkpoint "${RESUME_CHECKPOINT}" \
    --image_size 384 288 \

# ── Stage 1 pretrain (no tactile):
#    --training_stage 1 --use_tactile_deform 0
#
# ── Stage 2 from Stage 1 checkpoint:
#    --training_stage 2 --resume_checkpoint <stage1_ckpt>/model.pt
# 
# image size 384 288

echo ">>> Training finished."

