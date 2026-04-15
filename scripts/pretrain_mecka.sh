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
export NCCL_DEBUG_SUBSYS=INIT,NET
export NCCL_SOCKET_IFNAME=eth0
export NCCL_TIMEOUT=18000
ulimit -c 0

# ─── Multi-node config ───────────────────────────────────────────────────────
MASTER_ADDR=${MASTER_ADDR:-10.244.45.52}
MASTER_PORT=${MASTER_PORT:-29500}
NUM_MACHINES=${NUM_MACHINES:-8}
MACHINE_RANK=${MACHINE_RANK:-0}
NUM_PROCESSES=$((NUM_MACHINES * 8))

EXPERIMENT_NAME="qwen3vl_mecka_pretrain_base"
RUN_NAME="qwen3vl_2b_mecka20k_pretrain_bimanual_62d_stage1_base_$(date +%m%d)"
OUTPUT_ROOT_DIR="/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/ckpts/dex_mot_qwen/exp"
ORIGIN_MODEL_PATH="/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/ckpts/Qwen3-VL-2B-Instruct"

# Merged data root where all sub-batches from all 10 sources are visible flat
MERGED_DATA_ROOT="/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/data/mecka_merged"

# ─── Batch sources: (batch_id  done_dir  manifest_base_dir) ─────────────────
#
# manifest_base_dir: the base path hardcoded in each batch's pretrain_manifest.json
# for episode_dir. We create batch-XX → batch-XX-done symlinks there so the
# manifests resolve correctly.
#
#  batch  |  actual -done location                                         |  manifest base
#  -------+-----------------------------------------------------------------+-------------------------------------
#  01     |  /mnt/amlfs-03/…/mecka_dexmot/batch-01-done                   |  /mnt/amlfs-07/…/mecka_dexmot
#  02     |  /mnt/amlfs-03/…/mecka_dexmot/batch-02-done                   |  /mnt/amlfs-03/…/mecka_dexmot
#  03     |  /mnt/amlfs-07/…/mecka_dexmot/batch-03-done                   |  /mnt/amlfs-07/…/mecka_dexmot
#  04     |  /mnt/amlfs-03/…/mecka_dexmot/batch-04-done                   |  /mnt/amlfs-03/…/mecka_dexmot
#  05     |  /mnt/amlfs-03/…/mecka_dexmot/batch-05-done                   |  /mnt/amlfs-02/…/mecka_dexmot
#  06     |  /mnt/amlfs-02/…/mecka_dexmot/batch-06-done                   |  /mnt/amlfs-02/…/mecka_dexmot
#  07     |  /mnt/amlfs-02/…/mecka_dexmot/batch-07-done                   |  /mnt/amlfs-02/…/mecka_dexmot
#  08     |  /mnt/amlfs-07/…/mecka_dexmot/batch-08-done                   |  /mnt/amlfs-07/…/mecka_dexmot
#  09     |  /mnt/amlfs-03/…/mecka_dexmot/batch-09-done                   |  /mnt/amlfs-03/…/mecka_dexmot
#  10     |  /mnt/amlfs-02/…/mecka_dexmot/batch-10-done                   |  /mnt/amlfs-02/…/mecka_dexmot

declare -A DONE_DIR
declare -A MANIFEST_BASE

DONE_DIR[01]="/mnt/amlfs-03/shared/human_egocentric/dniu/mecka_dexmot/batch-01-done"
DONE_DIR[02]="/mnt/amlfs-03/shared/human_egocentric/dniu/mecka_dexmot/batch-02-done"
DONE_DIR[03]="/mnt/amlfs-07/shared/datasets/dniu/mecka_dexmot/batch-03-done"
DONE_DIR[04]="/mnt/amlfs-03/shared/human_egocentric/dniu/mecka_dexmot/batch-04-done"
DONE_DIR[05]="/mnt/amlfs-03/shared/human_egocentric/dniu/mecka_dexmot/batch-05-done"
DONE_DIR[06]="/mnt/amlfs-02/shared/human_egocentric/dniu/mecka_dexmot/batch-06-done"
DONE_DIR[07]="/mnt/amlfs-02/shared/human_egocentric/dniu/mecka_dexmot/batch-07-done"
DONE_DIR[08]="/mnt/amlfs-07/shared/datasets/dniu/mecka_dexmot/batch-08-done"
DONE_DIR[09]="/mnt/amlfs-03/shared/human_egocentric/dniu/mecka_dexmot/batch-09-done"
DONE_DIR[10]="/mnt/amlfs-02/shared/human_egocentric/dniu/mecka_dexmot/batch-10-done"

MANIFEST_BASE[01]="/mnt/amlfs-07/shared/datasets/dniu/mecka_dexmot"
MANIFEST_BASE[02]="/mnt/amlfs-03/shared/human_egocentric/dniu/mecka_dexmot"
MANIFEST_BASE[03]="/mnt/amlfs-07/shared/datasets/dniu/mecka_dexmot"
MANIFEST_BASE[04]="/mnt/amlfs-03/shared/human_egocentric/dniu/mecka_dexmot"
MANIFEST_BASE[05]="/mnt/amlfs-02/shared/human_egocentric/dniu/mecka_dexmot"
MANIFEST_BASE[06]="/mnt/amlfs-02/shared/human_egocentric/dniu/mecka_dexmot"
MANIFEST_BASE[07]="/mnt/amlfs-02/shared/human_egocentric/dniu/mecka_dexmot"
MANIFEST_BASE[08]="/mnt/amlfs-07/shared/datasets/dniu/mecka_dexmot"
MANIFEST_BASE[09]="/mnt/amlfs-03/shared/human_egocentric/dniu/mecka_dexmot"
MANIFEST_BASE[10]="/mnt/amlfs-02/shared/human_egocentric/dniu/mecka_dexmot"

# ─── Step 1: Create batch-XX → batch-XX-done symlinks to fix manifest paths ──
echo ">>> Creating batch-XX symlinks to fix manifest episode_dir paths..."
for batch_id in "${!DONE_DIR[@]}"; do
    done_dir="${DONE_DIR[$batch_id]}"
    base="${MANIFEST_BASE[$batch_id]}"
    link="${base}/batch-${batch_id}"
    if [ ! -e "${link}" ] && [ ! -L "${link}" ]; then
        ln -s "${done_dir}" "${link}"
        echo "  Symlinked: ${link} -> ${done_dir}"
    fi
done

# ─── Step 2: Build flat merged data_root ─────────────────────────────────────
# The training script expects: data_root/*/pretrain_manifest.json
# Sub-batch dirs (batch1, batch2, …) overlap across all 10 sources, so
# we prefix them: b01_batch1, b02_batch1, … to guarantee uniqueness.
echo ">>> Building merged data root at ${MERGED_DATA_ROOT}..."
mkdir -p "${MERGED_DATA_ROOT}"

for batch_id in "${!DONE_DIR[@]}"; do
    done_dir="${DONE_DIR[$batch_id]}"
    created=0
    for sub_batch_dir in "${done_dir}"/batch*/; do
        [ -d "${sub_batch_dir}" ] || continue
        manifest="${sub_batch_dir}/pretrain_manifest.json"
        # Skip sub-batches with missing or empty manifests (incomplete data)
        [ -s "${manifest}" ] || continue
        sub_name=$(basename "${sub_batch_dir}")
        link="${MERGED_DATA_ROOT}/b${batch_id}_${sub_name}"
        if [ ! -e "${link}" ] && [ ! -L "${link}" ]; then
            ln -s "${sub_batch_dir}" "${link}"
            (( created++ )) || true
        fi
    done
    echo "  batch-${batch_id}: ${created} new sub-batch symlinks"
done

total=$(ls "${MERGED_DATA_ROOT}" | wc -l)
echo ">>> Merged data root ready: ${total} sub-batches total"

# Allow data-prep-only mode (used by launch_pretrain_ray.py to avoid race conditions)
if [ "${SKIP_TRAINING:-0}" = "1" ]; then
    echo ">>> SKIP_TRAINING=1: data prep done, exiting before training."
    exit 0
fi

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
    --data_root ${MERGED_DATA_ROOT} \
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
    --run_name ${RUN_NAME} \
    --use_robot_state 1 \
    --image_size 384 288 \
    --num_workers 6 \
    --save_steps 5000 \
    --val_ratio 0.02 \
    --val_freq 1000 \
    --max_val_batches 50

echo ">>> Mecka pretraining finished."
