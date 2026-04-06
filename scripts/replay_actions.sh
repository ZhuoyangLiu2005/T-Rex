#!/bin/bash
set -e

cd /mnt/amlfs-01/home/dniu/Project/dex-mot/mot/dex_mot_qwen/scripts
source /mnt/amlfs-01/home/dniu/anaconda3/bin/activate /mnt/amlfs-01/home/dniu/anaconda3/envs/dex_mot
export PATH=/mnt/amlfs-01/home/dniu/anaconda3/envs/dex_mot/bin:$PATH

HDF5_PATH="${1:-/mnt/amlfs-07/shared/datasets/dniu/egodex/cotrain_processed_new/batch1/extra_assemble_disassemble_jigsaw_puzzle_40/pretrain.hdf5}"
PORT=5678
STRIDE=1

python replay_actions.py \
    --hdf5_path "${HDF5_PATH}" \
    --port ${PORT} \
    --stride ${STRIDE}
