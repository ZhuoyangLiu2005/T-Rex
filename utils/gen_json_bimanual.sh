#!/bin/bash
# Raw in-lab bimanual episodes (success/episode_*/ : .h5 + 3 mp4) -> training JSON
# (eef-62 delta-base) + sibling _statistics.json.
set -e

PROJECT_ROOT=/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/code/T-Rex
cd ${PROJECT_ROOT}

source /mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/code/miniconda3/bin/activate /mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/code/miniconda3/envs/dex_mot
export PYTHONPATH=${PROJECT_ROOT}:$PYTHONPATH

DATA_ROOTS="/path/to/raw/task_a /path/to/raw/task_b"
IMG_SAVE_ROOT="/path/to/training_data/three_dense_fastslow_full"
JSON_SAVE_ROOT="/path/to/training_data/three_full_json"
TASK_NAME="place_card_lr_bimanual_crop_stride1"
JSON_NAME_BASE="place_card_deltabase_axis_eef_lr_bimanual_crop_stride1_train"
INSTRUCTION="I am T-Rex."

ACTION_CHUNK=16
FRAME_STRIDE=1
CROP_BOX="0 300 140 540" 
NUM_TRAJ=0
SEED=42
DATASET_NAME=rlbench

python utils/gen_json_tac_deltabase_eef_bimanual_parallel.py \
    --data_roots ${DATA_ROOTS} \
    --img_save_root ${IMG_SAVE_ROOT} \
    --json_save_root ${JSON_SAVE_ROOT} \
    --task_name ${TASK_NAME} \
    --json_name_base ${JSON_NAME_BASE} \
    --instruction "${INSTRUCTION}" \
    --action_chunk ${ACTION_CHUNK} \
    --frame_stride ${FRAME_STRIDE} \
    --crop_box ${CROP_BOX} \
    --num_trajectories ${NUM_TRAJ} \
    --seed ${SEED} \
    --dataset_name ${DATASET_NAME}

echo ">>> raw -> JSON done: ${JSON_SAVE_ROOT}/${JSON_NAME_BASE}.json"
