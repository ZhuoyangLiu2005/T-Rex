#!/bin/bash
set -e
set -o pipefail

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

# ─── NCCL / distributed timeout controls ────────────────────────────────────
# NOTE: NCCL_TIMEOUT is NOT a real PyTorch/NCCL env var — the previous value
# was silently ignored. The actual watchdog/collective timeout is controlled
# by the variables below + the `timeout=` arg to init_process_group.
#
# Symptom we saw at step 53545 in the 0423 run: ALLREDUCE hung for exactly
# 600000 ms (the default 10-min PG timeout) and the watchdog killed the job.
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# Trace dumps on timeout (helps identify which rank stalled)
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
# Watchdog heartbeat — if a rank doesn't make progress in this window, kill.
# 60 min gives us headroom for a slow data read without immediately killing
# the whole 64-GPU job.
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
# IB retry timeout (bumped from default to tolerate brief link blips)
export NCCL_IB_TIMEOUT=22
export NCCL_IB_RETRY_CNT=10

ulimit -c 0

# ─── Multi-node config ───────────────────────────────────────────────────────
MASTER_ADDR=${MASTER_ADDR:-10.244.45.52}
MASTER_PORT=${MASTER_PORT:-29500}
NUM_MACHINES=${NUM_MACHINES:-8}
MACHINE_RANK=${MACHINE_RANK:-0}
NUM_PROCESSES=$((NUM_MACHINES * 8))

EXPERIMENT_NAME="qwen3vl_mecka_pretrain_flare"
RUN_NAME="qwen3vl_2b_mecka20k_pretrain_bimanual_62d_stage1_flare_$(date +%m%d)"
OUTPUT_ROOT_DIR="/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/ckpts/dex_mot_qwen/exp"
ORIGIN_MODEL_PATH="/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/ckpts/Qwen3-VL-2B-Instruct"

# Resume from a prior checkpoint. Accepts either a `model.pt` file (weights
# only) or a checkpoint directory. If the directory contains a `state/`
# subfolder, full optimizer+scheduler+RNG state is restored automatically.
#
#  - Weights-only init: set to a `.../model.pt` path, leave RESUME_STEP=0.
#  - Legacy step resume (no state/ saved): set to the ckpt dir and set
#    RESUME_STEP to the step number. LR scheduler is fast-forwarded;
#    optimizer momentum starts fresh.
#  - Full resume (state/ present): set to the ckpt dir; RESUME_STEP=0
#    (the saved training_state.json is authoritative).
RESUME_CHECKPOINT="/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/ckpts/dex_mot_qwen/exp/qwen3vl_mecka_pretrain_flare/qwen3vl_2b_mecka20k_pretrain_bimanual_62d_stage1_flare_0423/checkpoint-0-190000"
RESUME_STEP=0               # 0 = use training_state.json (this ckpt has state/, full resume)
# Skip past already-seen batches in the resumed epoch. Implemented at the
# *sampler* level (sampler.start_index): skipped indices never reach
# workers, so this is effectively free — no I/O, no GPU idle, no NCCL
# watchdog risk. (Old per-batch `continue` skip was the cause of the slow
# tqdm + 0% GPU you were seeing.)
RESUME_SKIP_DATA=1
if [ -n "${RESUME_CHECKPOINT}" ] && [ ! -e "${RESUME_CHECKPOINT}" ]; then
    echo "ERROR: RESUME_CHECKPOINT does not exist: ${RESUME_CHECKPOINT}" >&2
    exit 1
fi

# Merged data root where all sub-batches from all 10 sources are visible flat
MERGED_DATA_ROOT="/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/data/mecka_merged"

# ─── Batch sources: (batch_id  done_dir  manifest_base_dir) ─────────────────
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

# ─── Extra sources (already-absolute manifest paths, no batch-XX fix-up) ────
# Each entry: src_name -> root dir containing batch*/pretrain_manifest.json
declare -A EXTRA_DONE_DIR
EXTRA_DONE_DIR[egodex_cotrain_new]="/mnt/amlfs-07/shared/datasets/dniu/egodex/cotrain_processed_new"
EXTRA_DONE_DIR[sharpa_r1pro]="/mnt/amlfs-02/shared/human_egocentric/dniu/nv_inlab_robot_data/sharpa_r1pro"
EXTRA_DONE_DIR[sharpa_r1pro_from260129]="/mnt/amlfs-02/shared/human_egocentric/dniu/nv_inlab_robot_data/sharpa_r1pro_from260129"
EXTRA_DONE_DIR[sharpa_human]="/mnt/amlfs-02/shared/human_egocentric/dniu/nv_inlab_human_data/sharpa_human"
EXTRA_DONE_DIR[r1pro_sharpa_from260129]="/mnt/amlfs-07/shared/datasets/dniu/mecka_dexmot/nv_inlab/r1pro_sharpa_data_from260129"
EXTRA_DONE_DIR[maxinsights_530hrs]="/mnt/amlfs-02/shared/human_egocentric/dniu/maxinsights_egoscale_2000hrs_dexmot/530hrs"
EXTRA_DONE_DIR[maxinsights_1530hrs]="/mnt/amlfs-02/shared/human_egocentric/dniu/maxinsights_egoscale_2000hrs_dexmot/1530hrs"

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
echo ">>> Building merged data root at ${MERGED_DATA_ROOT}..."
mkdir -p "${MERGED_DATA_ROOT}"

for batch_id in "${!DONE_DIR[@]}"; do
    done_dir="${DONE_DIR[$batch_id]}"
    created=0
    for sub_batch_dir in "${done_dir}"/batch*/; do
        [ -d "${sub_batch_dir}" ] || continue
        manifest="${sub_batch_dir}/pretrain_manifest.json"
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

for src_name in "${!EXTRA_DONE_DIR[@]}"; do
    src_dir="${EXTRA_DONE_DIR[$src_name]}"
    if [ ! -d "${src_dir}" ]; then
        echo "  [WARN] extra source missing, skipping: ${src_name} -> ${src_dir}"
        continue
    fi
    created=0
    for sub_batch_dir in "${src_dir}"/batch*/; do
        [ -d "${sub_batch_dir}" ] || continue
        manifest="${sub_batch_dir}/pretrain_manifest.json"
        [ -s "${manifest}" ] || continue
        sub_name=$(basename "${sub_batch_dir}")
        link="${MERGED_DATA_ROOT}/${src_name}_${sub_name}"
        if [ ! -e "${link}" ] && [ ! -L "${link}" ]; then
            ln -s "${sub_batch_dir}" "${link}"
            (( created++ )) || true
        fi
    done
    echo "  ${src_name}: ${created} new sub-batch symlinks"
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

# ─── Local log redirect ─────────────────────────────────────────────────────
# Mirror all stdout/stderr to a local file (per machine rank, per invocation)
# while still streaming to the terminal via tee. The checkpoint output
# (OUTPUT_ROOT_DIR) stays on shared storage; only the console log is local.
LOCAL_LOG_DIR="/mnt/amlfs-01/home/dniu/Project/dex-mot/mot/dex_mot_qwen/scripts/logs/${EXPERIMENT_NAME}"
mkdir -p "${LOCAL_LOG_DIR}"
LOCAL_LOG_FILE="${LOCAL_LOG_DIR}/${RUN_NAME}_rank${MACHINE_RANK}_$(date +%Y%m%d_%H%M%S).log"
echo ">>> Mirroring training output to ${LOCAL_LOG_FILE}"

accelerate launch \
    --config_file ../config/sft_qwen.yaml \
    --num_processes ${NUM_PROCESSES} \
    --num_machines ${NUM_MACHINES} \
    --machine_rank ${MACHINE_RANK} \
    --main_process_ip ${MASTER_ADDR} \
    --main_process_port ${MASTER_PORT} \
    --deepspeed_multinode_launcher standard \
    train_qwen3vl_pretrain_egodex_flare.py \
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
    --max_val_batches 50 \
    --use_flare 1 \
    --n_flare_tokens_per_frame 4 \
    --n_flare_steps 8 \
    --flare_loss_weight 0.5 \
    --flare_frame_stride 4 \
    --flare_layer_index -1 \
    --resume_checkpoint "${RESUME_CHECKPOINT}" \
    --resume_step ${RESUME_STEP} \
    --resume_skip_data ${RESUME_SKIP_DATA} \
    --save_full_state 1 \
    --max_ckpts 0 \
    2>&1 | tee -a "${LOCAL_LOG_FILE}"

echo ">>> Mecka pretraining (with flare) finished. Log: ${LOCAL_LOG_FILE}"

