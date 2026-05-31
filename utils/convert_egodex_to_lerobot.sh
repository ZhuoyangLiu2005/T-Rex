#!/bin/bash
# Egodex/mecka pretrain episodes (pretrain.hdf5 + ego_view.mp4) -> LeRobot v3.0
# (head + state + baked action; no tactile).
set -e

PROJECT_ROOT=/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/code/T-Rex
LEROBOT_SRC=/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/code/lerobot/src
cd ${PROJECT_ROOT}

source /mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/code/miniconda3/bin/activate /mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/code/miniconda3/envs/dex_mot
export PYTHONPATH=${PROJECT_ROOT}:${LEROBOT_SRC}:$PYTHONPATH

DATA_ROOTS="/path/to/egodex_merged"
OUTPUT_ROOT="/path/to/lerobot/egodex"
REPO_ID="trex/egodex"
FPS=30

python utils/convert_egodex_to_lerobot.py \
    --data_roots ${DATA_ROOTS} \
    --output_root ${OUTPUT_ROOT} \
    --repo_id ${REPO_ID} \
    --fps ${FPS}

echo ">>> egodex -> LeRobot done: ${OUTPUT_ROOT}"
