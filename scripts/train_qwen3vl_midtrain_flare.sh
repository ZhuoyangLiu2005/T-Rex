#!/bin/bash
# Mid-training launcher for the Qwen3-VL VLA model on ~100h of mecka data.
# Same model + training loop as train_qwen3vl_flare.sh (tactile expert with
# residual delta_v + FLARE), only the dataloader is swapped to read the
# pretrain-format batches produced by
#   data/midtrain/scripts/gen_pretrain_mecka_parallel.py
#
# Two preprocessed sources are merged via symlinks into a single data_root:
#   • nvidia  — no crop (head video symlinked as ego_view.mp4)
#   • berkeley — cropped re-encode (matches the BKL fine-tune crop)
# 62D bimanual actions, tactile vec + deform on, FLARE on, robot state on.
#
# To enable VQ-VAE tactile-code conditioning on the tactile expert (fast path):
#   1) Train a VQ-VAE: bash tactile_vqvae/scripts/train_vqvae_f6.sh
#      (set GRANULARITY=finger for the 5-codes-per-hand variant.)
#   2) Pre-extract codes per episode (default --alignment historical, which
#      matches post-train and real-world rolling-buffer inference):
#        python -m tactile_vqvae.extract_codes \
#          --checkpoint /path/to/latest.pt --data_root ${MERGED_DATA_ROOT}
#      This drops a tactile_codes.h5 next to each pretrain.hdf5.
#   3) Below, set --use_tactile_code 1 (and --tactile_code_per_finger 1 if
#      using the per-finger VQ-VAE).

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
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export NCCL_IB_TIMEOUT=22
export NCCL_IB_RETRY_CNT=10
ulimit -c 0

ORIGIN_MODEL_PATH="/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/ckpts/Qwen3-VL-2B-Instruct"
OUTPUT_ROOT_DIR="/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/ckpts/dex_mot_qwen/exp"
DEFORM_ENCODER_PATH="/mnt/amlfs-01/home/dniu/Project/dex-mot/mot/bi-mot/janus/DeformEncoder/ckpt/sharpa_wave_deform_encoder.pth"

# Two preprocessed sources (output of gen_pretrain_mecka_parallel.py).
# Each contains batch_NNN/{pretrain_manifest.json, episode_*/{pretrain.hdf5,
# raw.h5, ego_view.mp4, left_wrist.mp4, right_wrist.mp4}}.
NV_PRETRAIN_ROOT="/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/data/midtrain/training_data/nvidia"
BKL_PRETRAIN_ROOT="/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/data/midtrain/training_data/berkeley"
MERGED_DATA_ROOT="/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/data/midtrain/merged"

FROM_VLM_SCRATCH=0
if [ "${FROM_VLM_SCRATCH}" = "1" ]; then
    RESUME_CHECKPOINT=""
    echo ">>> FROM_VLM_SCRATCH=1 — starting from base Qwen3-VL weights, no resume."
else
    RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/ckpts/dex_mot_qwen/exp/qwen3vl_egodex_pretrain_flare/qwen3vl_2b_egodex_pretrain_bimanual_62d_stage1_handabs_flare_0407/checkpoint-0-115665}"
    if [ ! -e "${RESUME_CHECKPOINT}" ]; then
        echo "ERROR: RESUME_CHECKPOINT does not exist: ${RESUME_CHECKPOINT}" >&2
        echo "       Either fix the path, override RESUME_CHECKPOINT=..., or" >&2
        echo "       set FROM_VLM_SCRATCH=1 to train from base VLM weights." >&2
        exit 1
    fi
fi

EXPERIMENT_NAME="qwen3vl_midtrain_flare"
if [ "${FROM_VLM_SCRATCH}" = "1" ]; then
    RUN_NAME="qwen3vl_2b_midtrain_vlmscratch_state[wo]_bimanual_62d_tac[force+deform]_flare_$(date +%m%d)"
else
    RUN_NAME="qwen3vl_2b_midtrain_egodex0407_state[wo]_bimanual_62d_tac[force+deform]_flare_$(date +%m%d)"
fi

MASTER_ADDR=${MASTER_ADDR:-10.244.254.74}
MASTER_PORT=${MASTER_PORT:-29500}
NUM_MACHINES=${NUM_MACHINES:-4}
MACHINE_RANK=${MACHINE_RANK:-0}
NUM_PROCESSES=$((NUM_MACHINES * 8))

echo ">>> Building merged data root at ${MERGED_DATA_ROOT}"
mkdir -p "${MERGED_DATA_ROOT}"

link_source() {
    local src_name="$1"
    local src_root="$2"
    if [ ! -d "${src_root}" ]; then
        echo "  [WARN] missing source '${src_name}' -> ${src_root}; skipping"
        return
    fi
    local created=0
    local seen=0
    for sub in "${src_root}"/batch*/; do
        [ -d "${sub}" ] || continue
        [ -s "${sub}/pretrain_manifest.json" ] || continue
        seen=$((seen + 1))
        local sub_name; sub_name=$(basename "${sub}")
        local link="${MERGED_DATA_ROOT}/${src_name}_${sub_name}"
        if [ ! -e "${link}" ] && [ ! -L "${link}" ]; then
            ln -s "${sub}" "${link}"
            created=$((created + 1))
        fi
    done
    echo "  ${src_name}: ${created} new symlinks (${seen} sub-batches in source)"
}

link_source "nvidia"  "${NV_PRETRAIN_ROOT}"
link_source "berkeley" "${BKL_PRETRAIN_ROOT}"

total=$(ls "${MERGED_DATA_ROOT}" 2>/dev/null | wc -l)
echo ">>> Merged data root ready: ${total} sub-batches total"
if [ "${total}" -eq 0 ]; then
    echo "ERROR: merged data root is empty. Run gen_pretrain_mecka_parallel.py first" >&2
    echo "       on each source, then re-run this script." >&2
    exit 1
fi

if [ "${SKIP_TRAINING:-0}" = "1" ]; then
    echo ">>> SKIP_TRAINING=1: data prep done, exiting before training."
    exit 0
fi

TRAIN_BSZ=16
LR=1e-4

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
    train_qwen3vl_midtrain_flare.py \
    --model_path ${ORIGIN_MODEL_PATH} \
    --data_root ${MERGED_DATA_ROOT} \
    --n_epochs 10 \
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
    --use_robot_state 0 \
    --use_tactile_vec 1 \
    --use_tactile_deform 1 \
    --deform_encoder_ckpt ${DEFORM_ENCODER_PATH} \
    --tactile_intermediate_size 1536 \
    --training_stage 2 \
    --tactile_loss_weight 1.0 \
    --use_tactile_refine_flow 1 \
    --tactile_refine_loss_weight 1.0 \
    --tactile_refine_noise_scale 0.1 \
    --action_flow_train_steps 5 \
    --action_flow_eval_steps 10 \
    --tactile_delay_offsets 0 4 8 12 \
    --tactile_residual_jitter 0.0 \
    --use_tactile_code 0 \
    --vqvae_codebook_size 64 \
    --vqvae_codes_h5_name tactile_codes.h5 \
    --tactile_code_per_finger 0 \
    --resume_checkpoint "${RESUME_CHECKPOINT}" \
    --use_flare 1 \
    --n_flare_tokens_per_frame 4 \
    --n_flare_steps 8 \
    --flare_loss_weight 0.5 \
    --flare_frame_stride 4 \
    --flare_layer_index -1 \
    --image_size 384 288 \
    --num_workers 4 \
    --val_ratio 0.02 \
    --val_freq 100 \
    --max_val_batches 50 \
    2>&1 | tee -a "${LOCAL_LOG_FILE}"

echo ">>> Mid-training (mecka, tactile + flare) finished. Log: ${LOCAL_LOG_FILE}"
