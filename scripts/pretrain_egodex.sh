#!/bin/bash
set -e

cd /mnt/amlfs-01/home/dniu/Project/dex-mot/mot/dex_mot_qwen/scripts

source /mnt/amlfs-01/home/dniu/anaconda3/bin/activate /mnt/amlfs-01/home/dniu/anaconda3/envs/dex_mot
export PATH=/mnt/amlfs-01/home/dniu/anaconda3/envs/dex_mot/bin:$PATH
export HF_HOME=/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/huggingface
export PYTHONPATH=/mnt/amlfs-01/home/dniu/Project/dex-mot/mot/dex_mot_qwen:$PYTHONPATH

export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# ─── Multi-node config (override via env vars) ───
MASTER_ADDR=10.244.27.42   # run 'ifconfig' to get the ip address of eth0
MASTER_PORT=29500
NUM_MACHINES=1
MACHINE_RANK=0 # remember to modify in different nodes
NUM_PROCESSES=$((NUM_MACHINES * 8))

BASE_RUN_NAME="qwen3vl_2b_egodex_pretrain_bimanual_62d_stage1_0322_test"
EXPERIMENT_NAME="qwen3vl_egodex_pretrain"
OUTPUT_ROOT_DIR="/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/ckpts/dex_mot_qwen/exp"

DATA_ROOT="/mnt/amlfs-07/shared/datasets/dniu/egodex/cotrain_processed_new"
ORIGIN_MODEL_PATH="/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/ckpts/Qwen3-VL-2B-Instruct"

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
    train_qwen3vl_pretrain_egodex.py \
    --model_path ${ORIGIN_MODEL_PATH} \
    --data_root ${DATA_ROOT} \
    --n_epochs 100 \
    --save_freq 100 \
    --action_dim 62 \
    --action_chunk 16 \
    --train_bsz_per_gpu ${TRAIN_BSZ} \
    --learning_rate ${LR} \
    --min_lr_ratio 0 \
    --warmup_rates 0.03 \
    --weight_decay 0.01 \
    --gradient_accumulation_steps 1 \
    --output_dir ${OUTPUT_ROOT_DIR} \
    --log_dir ${OUTPUT_ROOT_DIR} \
    --experiment_name ${EXPERIMENT_NAME} \
    --run_name ${BASE_RUN_NAME} \
    --use_robot_state 0 \
    --image_size 384 288 \
    --num_workers 8

echo ">>> EgoDex pretraining finished."

