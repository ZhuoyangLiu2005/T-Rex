#!/bin/bash
# Mid-training launcher for the Qwen3-VL VLA on the new in-lab mecka mix
# (NV inlab + BKL play + BKL task).  Cascaded flow matching with VQ-VAE
# tactile-code conditioning.
#
# Three preprocessed sources are merged into a single data_root via
# prepare_midtrain_merged.py (a Python helper that materializes per-episode
# shadow dirs with renamed symlinks — episode_<id>.h5 → raw.h5,
# episode_<id>_head_left_rgb.mp4 → ego_view.mp4, etc. — plus rewritten
# manifests).  Idempotent: safe to re-run; touches only missing entries.
#
#   • nv       — NV in-lab play data (single batch, manifest at root)
#   • bkl_play — BKL in-lab play data, grouped by skill category
#   • bkl_task — BKL in-lab task data, 30 episodes per task
#
# 62D bimanual actions, tactile vec + deform on, FLARE on, robot state off,
# cascaded flow matching (split 6/10).
#
# VQ-VAE tactile-code conditioning:
#   1) Train a VQ-VAE: bash tactile_vqvae/scripts/train_vqvae_f6.sh
#      (set GRANULARITY=finger for the 5-codes-per-hand variant).
#   2) Pre-extract codes per episode (default --alignment historical):
#        python -m tactile_vqvae.extract_codes \
#          --checkpoint /path/to/latest.pt --data_root ${MERGED_DATA_ROOT}
#      This drops a tactile_codes.h5 next to each pretrain.hdf5.
#   3) Below, --use_tactile_code 1 (and --tactile_code_per_finger 1 for
#      the per-finger VQ-VAE) is on by default.

set -e
set -o pipefail

cd /mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/code/dex_mot_final/scripts

source /mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/code/miniconda3/bin/activate /mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/code/miniconda3/envs/dex_mot
export PATH=/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/code/miniconda3/envs/dex_mot/bin:$PATH
export HF_HOME=/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/huggingface
export PYTHONPATH=/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/code/dex_mot_final:$PYTHONPATH

export WANDB_MODE=online
export WANDB_API_KEY=5bdc90c568050775a6d10650e64857fbbc76742e
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export NCCL_DEBUG_SUBSYS=INIT,NET
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export NCCL_IB_TIMEOUT=22
export NCCL_IB_RETRY_CNT=10
ulimit -c 0

ORIGIN_MODEL_PATH="/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/ckpts/Qwen3-VL-2B-Instruct"
OUTPUT_ROOT_DIR="/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/ckpts/dex_mot_expert/exp"
DEFORM_ENCODER_PATH="/mnt/amlfs-01/home/dniu/Project/dex-mot/mot/bi-mot/janus/DeformEncoder/ckpt/sharpa_wave_deform_encoder.pth"

# Three preprocessed sources for the new in-lab mid-training mix on amlfs-07.
# Layout per source:
#   NV  (single batch — manifest at root, episodes flat):
#     <root>/pretrain_manifest.json
#     <root>/episode_NNNN/{pretrain.hdf5, episode_<id>.h5,
#                          episode_<id>_head_left_rgb.mp4,
#                          episode_<id>_left_wrist.mp4,
#                          episode_<id>_right_wrist.mp4, metadata.json}
#   BKL play / BKL task (multi-batch — one manifest per category subdir):
#     <root>/<category>/pretrain_manifest.json
#     <root>/<category>/<demo_name>/{pretrain.hdf5, episode_<id>.h5, *_*.mp4, ...}
#
# prepare_midtrain_merged.py builds shadow dirs under MERGED_DATA_ROOT with
# the canonical names the trainer expects (raw.h5, ego_view.mp4, left_wrist.mp4,
# right_wrist.mp4) plus rewritten manifests pointing at the shadows.
NV_PRETRAIN_ROOT="/mnt/amlfs-07/shared/datasets/dniu/mecka_dexmot/nv_inlab/nv_playdata_filter"
BKL_PLAY_PRETRAIN_ROOT="/mnt/amlfs-07/shared/datasets/dniu/mecka_dexmot/bkl_inlab/tactile_data_collection/grouped_play_data_before060509_reorganized"
BKL_TASK_PRETRAIN_ROOT="/mnt/amlfs-07/shared/datasets/dniu/mecka_dexmot/bkl_inlab/tactile_data_collection/grouped_task_data_260509_30each"
MERGED_DATA_ROOT="/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/data/midtrain/merged_inlab"

# The prep script lives in dex_mot_qwen/scripts; it produces the canonical
# layout that both dex_mot_qwen and dex_mot_expert trainers consume.  No need
# to duplicate it — the .py is self-contained and trainer-agnostic.
PREP_SCRIPT="/mnt/amlfs-01/home/dniu/Project/dex-mot/mot/dex_mot_qwen/scripts/prepare_midtrain_merged.py"

FROM_VLM_SCRATCH=0
if [ "${FROM_VLM_SCRATCH}" = "1" ]; then
    RESUME_CHECKPOINT=""
    echo ">>> FROM_VLM_SCRATCH=1 — starting from base Qwen3-VL weights, no resume."
else
    # mecka20k pretrain — same starting point as the working dex_mot_qwen run.
    RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/ckpts/dex_mot_qwen/exp/qwen3vl_mecka_pretrain_flare/qwen3vl_2b_mecka20k_pretrain_bimanual_62d_stage1_flare_0430/checkpoint-0-610000}"
    if [ ! -e "${RESUME_CHECKPOINT}" ]; then
        echo "ERROR: RESUME_CHECKPOINT does not exist: ${RESUME_CHECKPOINT}" >&2
        echo "       Either fix the path, override RESUME_CHECKPOINT=..., or" >&2
        echo "       set FROM_VLM_SCRATCH=1 to train from base VLM weights." >&2
        exit 1
    fi
fi

EXPERIMENT_NAME="qwen3vl_midtrain_flare"
if [ "${FROM_VLM_SCRATCH}" = "1" ]; then
    RUN_NAME="qwen3vl_2b_midtrain_vlmscratch_state[wo]_bimanual_62d_tac[force+deform]_flare_cascaded_fix_$(date +%m%d)"
else
    RUN_NAME="qwen3vl_2b_midtrain_mecka0507_state[wo]_bimanual_62d_tac[force+deform]_flare_cascaded_fix_$(date +%m%d)"
fi

MASTER_ADDR=${MASTER_ADDR:-10.244.16.162}
MASTER_PORT=${MASTER_PORT:-29500}
NUM_MACHINES=${NUM_MACHINES:-1}
MACHINE_RANK=${MACHINE_RANK:-0}
NUM_PROCESSES=$((NUM_MACHINES * 8))

mkdir -p "${MERGED_DATA_ROOT}"

# Only rank 0 builds the merged data root.  Other ranks would race on
# pretrain_manifest.json.tmp os.replace.  Idempotent: skips already-merged
# batches.  Re-running here is a no-op if the dex_mot_qwen run already
# materialized the layout.  Set SKIP_PREP=1 to bypass entirely (useful when
# the merged root + tactile_f6-stats-injected manifests are already in place).
if [ "${SKIP_PREP:-0}" = "1" ]; then
    echo ">>> SKIP_PREP=1: bypassing data prep entirely."
elif [ "${MACHINE_RANK}" = "0" ]; then
    echo ">>> [rank 0] Building merged data root at ${MERGED_DATA_ROOT}"
    python3 "${PREP_SCRIPT}" \
        --merged_root "${MERGED_DATA_ROOT}" \
        --source "nv=${NV_PRETRAIN_ROOT}" \
        --source "bkl_play=${BKL_PLAY_PRETRAIN_ROOT}" \
        --source "bkl_task=${BKL_TASK_PRETRAIN_ROOT}"
else
    echo ">>> [rank ${MACHINE_RANK}] Skipping data prep (rank 0 owns it)."
fi

# Multi-batch layout: each <category>/ contains its own pretrain_manifest.json.
total=$(find "${MERGED_DATA_ROOT}" -mindepth 2 -maxdepth 2 -name pretrain_manifest.json 2>/dev/null | wc -l)
echo ">>> Merged data root ready: ${total} batch manifests total"
if [ "${total}" -eq 0 ]; then
    echo "ERROR: merged data root has no batch manifests. Check source paths." >&2
    exit 1
fi

if [ "${SKIP_TRAINING:-0}" = "1" ]; then
    echo ">>> SKIP_TRAINING=1: data prep done, exiting before training."
    exit 0
fi

TRAIN_BSZ=16
LR=1e-4

LOCAL_LOG_DIR="/mnt/amlfs-01/home/dniu/Project/dex-mot/mot/dex_mot_expert/scripts/logs/${EXPERIMENT_NAME}"
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
    --n_epochs 6 \
    --save_freq 2 \
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
    --tactile_delay_offsets 0 4 8 12 \
    --use_tactile_code 1 \
    --vqvae_codebook_size 64 \
    --vqvae_codes_h5_name tactile_codes.h5 \
    --tactile_code_per_finger 1 \
    --tactile_f6_stats_ckpt /mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/ckpts/dex_mot_expert/tactile_vqvae/vqvae_f6_w16_k64_finger_0507_0939/latest.pt \
    --cascaded_total_steps 10 \
    --cascaded_split_step 6 \
    --cascaded_tactile_dropout 0.1 \
    --cascaded_loss_weight 1.0 \
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
    --val_freq 500 \
    --max_val_batches 50 \
    2>&1 | tee -a "${LOCAL_LOG_FILE}"

echo ">>> Mid-training (mecka inlab, tactile + flare, residual) finished. Log: ${LOCAL_LOG_FILE}"
