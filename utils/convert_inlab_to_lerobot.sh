#!/bin/bash
# Raw in-lab episodes (success/episode_*/ : .h5 + 3 mp4) -> LeRobot v3.0 dataset
# (eef-62 delta-base; training twin of gen_json_bimanual.sh).
set -e

PROJECT_ROOT=/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/code/T-Rex
LEROBOT_SRC=/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/code/lerobot/src
cd ${PROJECT_ROOT}

source /mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/code/miniconda3/bin/activate /mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/code/miniconda3/envs/dex_mot
export PYTHONPATH=${PROJECT_ROOT}:${LEROBOT_SRC}:$PYTHONPATH

DATA_ROOTS="/path/to/raw/task_a /path/to/raw/task_b"
OUTPUT_ROOT="/path/to/lerobot/place_card"
REPO_ID="trex/place_card"
INSTRUCTION="I am T-Rex."
FPS=30
CROP_BOX="0 300 140 540"
NUM_TRAJ=0
SEED=42

python utils/convert_inlab_to_lerobot.py \
    --data_roots ${DATA_ROOTS} \
    --output_root ${OUTPUT_ROOT} \
    --repo_id ${REPO_ID} \
    --instruction "${INSTRUCTION}" \
    --fps ${FPS} \
    --crop_box ${CROP_BOX} \
    --num_trajectories ${NUM_TRAJ} \
    --seed ${SEED}

echo ">>> raw -> LeRobot (eef-62) done: ${OUTPUT_ROOT}"
