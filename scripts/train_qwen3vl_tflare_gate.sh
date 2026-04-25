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

ORIGIN_MODEL_PATH="/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/ckpts/Qwen3-VL-2B-Instruct"
OUTPUT_ROOT_DIR="/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/ckpts/dex_mot_qwen/exp"
DATA_JSON="/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/data/bkl_inlab/training_data/three_full_json/erase_whiteboard_0416_deltabase_axis_eef_lr_bimanual_crop_stride1_train.json"
DEFORM_ENCODER_PATH="/mnt/amlfs-01/home/dniu/Project/dex-mot/mot/bi-mot/janus/DeformEncoder/ckpt/sharpa_wave_deform_encoder.pth"

EXPERIMENT_NAME="qwen3vl_mot_tflare_gate"
RUN_NAME="qwen3vl_2b_tri_mot_erase_whiteboard_0416_traj[100]_view2_tac[force+deform]_gate[perdim]_tflare[tpf4step4stride2]_solo_flare[tpf4step8stride4]_resize_lr_$(date +%m%d)"
RESUME_CHECKPOINT="/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/ckpts/dex_mot_qwen/exp/qwen3vl_egodex_pretrain_flare/qwen3vl_2b_egodex_pretrain_bimanual_62d_stage1_handabs_flare_0407/checkpoint-0-115665"

MASTER_ADDR=${MASTER_ADDR:-10.244.165.78}
MASTER_PORT=${MASTER_PORT:-29500}
NUM_MACHINES=${NUM_MACHINES:-2}
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
    train_qwen3vl_tflare_gate.py \
    --model_path ${ORIGIN_MODEL_PATH} \
    --data_path ${DATA_JSON} \
    --n_epochs 200 \
    --save_freq 50 \
    --action_dim 62 \
    --action_chunk 16 \
    --train_bsz_per_gpu ${TRAIN_BSZ} \
    --learning_rate ${LR} \
    --min_lr_ratio 0 \
    --weight_decay 0 \
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
    --resume_checkpoint "${RESUME_CHECKPOINT}" \
    --use_flare 1 \
    --n_flare_tokens_per_frame 4 \
    --n_flare_steps 8 \
    --flare_loss_weight 0.5 \
    --flare_frame_stride 4 \
    --flare_layer_index -1 \
    --image_size 384 288 \
    --use_tactile_flare 1 \
    --n_tfl_tokens_per_step 4 \
    --n_tfl_steps 8 \
    --tactile_flare_stride 2 \
    --tflare_loss_weight 0.5 \
    --gate_init_bias -2.0 \
    --tac_solo_weight 1.0

# ── Notes ─────────────────────────────────────────────────────────────────
# Fusion:        v_final = (1-g)*v_act + g*v_tac, g ∈ [0,1] per (chunk, action_dim)
# Gate init:     zero weights + bias=-2 → g≈0.12 at step 0 (start stable on v_act)
# Gate target:   NONE — gate learns where v_tac beats v_act purely from the
#                fusion MSE gradient (∂L/∂g = 2·(v_final−target)·(v_tac−v_act)).
# Solo loss:     MSE(v_tac, target) on all frames (unweighted) — anchors
#                v_tac as a full action predictor so the gate has something
#                meaningful to gate.
# Tactile-FLARE: 16 query tokens (4 per step × 4 steps) appended to the
#                tactile block; cosine loss vs future tactile embeddings
#                encoded through frozen-init snapshots of tacf6_embedder
#                and deform_proj (taken after resume-load, before stage-2
#                xavier reinit). The deform_encoder itself is pretrained.
# Stage-2 init:  final_layer_tactile is copied from final_layer (Stage-1
#                pretrained action head) rather than zero-inited. Removes
#                the v_tac≈0 race condition that caused gate collapse on
#                some tasks (see diagnosis notes).
# Health check:  `mean_gate` should drift above 0.12 as training proceeds;
#                if it stays stuck the tactile expert is not helping.
# ──────────────────────────────────────────────────────────────────────────

echo ">>> Training (gated fusion + tactile-FLARE) finished."
