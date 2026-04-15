#!/bin/bash
set -e

cd /mnt/amlfs-01/home/dniu/Project/dex-mot/mot/dex_mot_qwen/scripts
source /mnt/amlfs-01/home/dniu/anaconda3/bin/activate /mnt/amlfs-01/home/dniu/anaconda3/envs/dex_mot

DATA_ROOT="/mnt/amlfs-07/shared/datasets/dniu/egodex/cotrain_processed_new"

python precompute_fingertip_coords.py \
    --data_root ${DATA_ROOT} \
    --workers 32

echo ">>> Precompute finished."
