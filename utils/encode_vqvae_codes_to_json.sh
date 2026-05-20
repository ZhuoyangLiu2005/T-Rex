#!/bin/bash
# Pre-bake VQ-VAE tactile codes into a post-training JSON.
#
# Usage:
#   bash encode_vqvae_codes_to_json.sh
#
# Override defaults via env vars, e.g.:
#   INPUT_JSON=/path/other.json bash encode_vqvae_codes_to_json.sh
#   VQVAE_CKPT=/path/other_ckpt/latest.pt bash encode_vqvae_codes_to_json.sh

set -e
set -o pipefail

PARENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PARENT_DIR}"

source /mnt/amlfs-01/home/dniu/anaconda3/bin/activate /mnt/amlfs-01/home/dniu/anaconda3/envs/dex_mot
export PATH=/mnt/amlfs-01/home/dniu/anaconda3/envs/dex_mot/bin:$PATH
export PYTHONPATH=${PARENT_DIR}:${PYTHONPATH}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

INPUT_JSON="/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/data/bkl_inlab/training_data/three_full_json/insert_battery_0504+0506_deltabase_axis_eef_lr_bimanual_crop_stride1_train.json"
VQVAE_CKPT="/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/ckpts/dex_mot_expert/tactile_vqvae/vqvae_f6_w16_k64_finger_0507_0939/latest.pt"

# Default output: alongside input with _vqvae_k<size> suffix derived from ckpt dir.
CKPT_TAG="$(basename "$(dirname "${VQVAE_CKPT}")" | sed -E 's/.*_(k[0-9]+)_.*/\1/')"
DEFAULT_OUT="${INPUT_JSON%.json}_vqvae_${CKPT_TAG}.json"
OUTPUT_JSON="${OUTPUT_JSON:-${DEFAULT_OUT}}"

BATCH_SIZE="${BATCH_SIZE:-512}"

echo ">>> input  = ${INPUT_JSON}"
echo ">>> output = ${OUTPUT_JSON}"
echo ">>> ckpt   = ${VQVAE_CKPT}"

python -m utils.encode_vqvae_codes_to_json \
    --input_json  "${INPUT_JSON}" \
    --output_json "${OUTPUT_JSON}" \
    --vqvae_ckpt  "${VQVAE_CKPT}" \
    --batch_size  "${BATCH_SIZE}" \
    --cuda 0

echo ">>> Done. Wrote ${OUTPUT_JSON}"

