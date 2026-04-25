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
DATA_JSON="/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/data/bkl_inlab/training_data/three_full_json/flip_book_page_0405_deltabase_axis_eef_newclip_right_stride2_train.json"
DEFORM_ENCODER_PATH="/mnt/amlfs-01/home/dniu/Project/dex-mot/mot/bi-mot/janus/DeformEncoder/ckpt/sharpa_wave_deform_encoder.pth"

EXPERIMENT_NAME="qwen3vl_mot_flare_cfg"
RUN_NAME="qwen3vl_2b_tri_mot_pretrain0407_flip_book_page_0405view2_tac[force+deform]_wostate_deltabase_eef_stride1_f1s1_res_flare[tpf4step8stride4]_cfg[split+null]_mem[T4s1]_resize_lr_fix_$(date +%m%d)"
RESUME_CHECKPOINT="/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/ckpts/dex_mot_qwen/exp/qwen3vl_egodex_pretrain_flare/qwen3vl_2b_egodex_pretrain_bimanual_62d_stage1_handabs_flare_0407/checkpoint-0-115665"

MASTER_ADDR=${MASTER_ADDR:-10.244.215.27}
MASTER_PORT=${MASTER_PORT:-29500}
NUM_MACHINES=${NUM_MACHINES:-2}
MACHINE_RANK=${MACHINE_RANK:-1}
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
    train_qwen3vl_flare_cfg.py \
    --model_path ${ORIGIN_MODEL_PATH} \
    --data_path ${DATA_JSON} \
    --n_epochs 200 \
    --save_freq 50 \
    --action_dim 31 \
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
    --cfg_drop_force 0.15 \
    --cfg_drop_deform 0.15 \
    --use_learnable_null 1 \
    --tactile_cfg_scale 1.0 \
    --tactile_history_len 4 \
    --tactile_history_stride 1

# Notes:
#   Per-modality CFG: force and deform are dropped independently; each has its
#   own learnable null embedding (tac_null_f6 / tac_null_deform) registered on
#   the model. Set --use_learnable_null 0 to fall back to zero-masking.
#
#   Memory: --tactile_history_len 4 loads the current + 3 past tactile frames
#   per sample (fetched from adjacent hf_dataset rows that share the same
#   episode directory; out-of-episode steps pad with current). A learnable
#   temporal position embedding (tac_time_embed) is added per-timestep. With
#   T=4 and ~n_fingers=10, each modality contributes 40 tokens to the tactile
#   stream (vs. 10 without memory).

echo ">>> Training (flare + per-modality tactile CFG + memory) finished."

