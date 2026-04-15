#!/bin/bash
# Replay pre-recorded actions from a training .json to the robot via ZMQ.
#
# This script is a drop-in replacement for the VLA model server:
# run it instead of test_qwen3vl_flare_real.sh to send ground-truth
# actions to the robot client for data validation.
#
# Usage examples:
#   bash scripts/replay_json_client.sh
#   STRIDE=1 PORT=5678 bash scripts/replay_json_client.sh

set -e

cd /mnt/amlfs-01/home/dniu/Project/dex-mot/mot/dex_mot_qwen/scripts

source /mnt/amlfs-01/home/dniu/anaconda3/bin/activate /mnt/amlfs-01/home/dniu/anaconda3/envs/dex_mot
export PATH=/mnt/amlfs-01/home/dniu/anaconda3/envs/dex_mot/bin:$PATH
export PYTHONPATH=/mnt/amlfs-01/home/dniu/Project/dex-mot/mot/dex_mot_qwen:$PYTHONPATH

# ── Config ───────────────────────────────────────────────────────────────────
JSON_PATH="${JSON_PATH:-/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/data/bkl_inlab/training_data/three_full_json/pour_sugar_0411_deltabase_axis_eef_lr_bimanual_stride2_train.json}"
PORT="${PORT:-5678}"

# Step through samples non-overlapping (stride = action_chunk)
STRIDE="${STRIDE:-16}"

# ── Run ──────────────────────────────────────────────────────────────────────
python replay_json_client.py \
    --json_path "${JSON_PATH}" \
    --port      "${PORT}" \
    --stride    "${STRIDE}" \
    "$@"
