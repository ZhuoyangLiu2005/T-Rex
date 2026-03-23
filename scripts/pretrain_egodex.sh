#!/bin/bash
set -e

cd /mnt/amlfs-01/home/dniu/Project/dex-mot/mot/dex_mot_qwen/scripts

source /mnt/amlfs-01/home/dniu/anaconda3/bin/activate /mnt/amlfs-01/home/dniu/anaconda3/envs/dex_mot
export PATH=/mnt/amlfs-01/home/dniu/anaconda3/envs/dex_mot/bin:$PATH
export HF_HOME=/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/huggingface
export PYTHONPATH=/mnt/amlfs-01/home/dniu/Project/dex-mot/mot/dex_mot_qwen:$PYTHONPATH

export WANDB_MODE=offline

# ─── Multi-node config ───
export NCCL_IB_DISABLE=0          # Enable InfiniBand if available
export NCCL_NET_GDR=1             # GPU Direct RDMA (if hardware supports)
export NCCL_ALGO=Ring             # Ring algorithm often best for 2 nodes
export NCCL_SOCKET_IFNAME=eth0    # Set to your network interface (check with ifconfig)

NUM_MACHINES=1
MACHINE_RANK=0
MASTER_ADDR=10.244.117.13
MASTER_PORT=29500
NUM_PROCESSES_PER_NODE=8
TOTAL_PROCESSES=$((NUM_MACHINES * NUM_PROCESSES_PER_NODE))

BASE_RUN_NAME="qwen3vl_2b_egodex_pretrain_bimanual_62d_stage1_0322_test"
EXPERIMENT_NAME="qwen3vl_egodex_pretrain"
OUTPUT_ROOT_DIR="/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/ckpts/dex_mot_qwen/exp"

MANIFEST_PATH="/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/data/pretrain/example/egodex/processed_part1/fry_egg/pretrain_manifest.json"
ORIGIN_MODEL_PATH="/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/ckpts/Qwen3-VL-2B-Instruct"

TRAIN_BSZ=8
LR=1e-4

accelerate launch \
    --config_file ../config/sft_qwen.yaml \
    --num_processes ${TOTAL_PROCESSES} \
    --num_machines ${NUM_MACHINES} \
    --machine_rank ${MACHINE_RANK} \
    --main_process_ip ${MASTER_ADDR} \
    --main_process_port ${MASTER_PORT} \
    --deepspeed_multinode_launcher standard \
    train_qwen3vl_pretrain_egodex.py \
    --model_path ${ORIGIN_MODEL_PATH} \
    --manifest_path ${MANIFEST_PATH} \
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
    --image_size 384 \
    --num_workers 8

echo ">>> EgoDex pretraining finished."

