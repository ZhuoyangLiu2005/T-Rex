#!/bin/bash
set -e

cd /mnt/amlfs-01/home/dniu/Project/dex-mot/mot/dex_mot_qwen/scripts

source /mnt/amlfs-01/home/dniu/anaconda3/bin/activate /mnt/amlfs-01/home/dniu/anaconda3/envs/dex_mot
export PATH=/mnt/amlfs-01/home/dniu/anaconda3/envs/dex_mot/bin:$PATH
export HF_HOME=/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/huggingface
export PYTHONPATH=/mnt/amlfs-01/home/dniu/Project/dex-mot/mot/dex_mot_qwen:$PYTHONPATH

export WANDB_MODE=online
export WANDB_API_KEY=5bdc90c568050775a6d10650e64857fbbc76742e
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
export NCCL_SOCKET_IFNAME=eth0
# export NCCL_IB_DISABLE=0
export NCCL_TIMEOUT=18000
ulimit -c 0

# ─── Multi-node config (override via env vars) ───
MASTER_ADDR=${MASTER_ADDR:-10.244.27.42}
MASTER_PORT=${MASTER_PORT:-29500}
NUM_MACHINES=${NUM_MACHINES:-4}
MACHINE_RANK=${MACHINE_RANK:-3}
NUM_PROCESSES=$((NUM_MACHINES * 8))

BASE_RUN_NAME="qwen3vl_2b_egodex_pretrain_bimanual_62d_stage1_handabs_0403"
EXPERIMENT_NAME="qwen3vl_egodex_pretrain"
OUTPUT_ROOT_DIR="/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/ckpts/dex_mot_qwen/exp"

DATA_ROOT="/mnt/amlfs-07/shared/datasets/dniu/egodex/cotrain_processed_new"
ORIGIN_MODEL_PATH="/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/ckpts/Qwen3-VL-2B-Instruct"

# ─── Symlink in-lab data groups as batches under DATA_ROOT ───
BKL_SRC="/mnt/amlfs-02/shared/human_egocentric/dniu/datasets/bkl_inlab/raw/playdata_reorganized/grouped_untilMarch31_reorganized"
for group_dir in "${BKL_SRC}"/*/; do
    group_name=$(basename "${group_dir}")
    link_path="${DATA_ROOT}/bkl_inlab_${group_name}"
    if [ ! -e "${link_path}" ]; then
        ln -s "${group_dir}" "${link_path}"
        echo "Symlinked: ${link_path} -> ${group_dir}"
    fi
done

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
    train_qwen3vl_pretrain_egodex.py \
    --model_path ${ORIGIN_MODEL_PATH} \
    --data_root ${DATA_ROOT} \
    --n_epochs 1 \
    --save_freq 1 \
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
    --use_robot_state 1 \
    --image_size 384 288 \
    --num_workers 6

echo ">>> EgoDex pretraining finished."

