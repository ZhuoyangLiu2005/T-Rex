#!/bin/bash
# Extract per-episode tactile codes (sidecar h5) using a trained VQ-VAE.

set -e
set -o pipefail

PARENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PARENT_DIR}"

source /mnt/amlfs-01/home/dniu/anaconda3/bin/activate /mnt/amlfs-01/home/dniu/anaconda3/envs/dex_mot
export PATH=/mnt/amlfs-01/home/dniu/anaconda3/envs/dex_mot/bin:$PATH
export PYTHONPATH=${PARENT_DIR}:${PYTHONPATH}

CHECKPOINT="${CHECKPOINT:?must set CHECKPOINT=/path/to/latest.pt}"
DATA_ROOT="${DATA_ROOT:-/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/data/midtrain/merged}"
NUM_WORKERS=${NUM_WORKERS:-4}

echo ">>> checkpoint = ${CHECKPOINT}"
echo ">>> data_root  = ${DATA_ROOT}"
echo ">>> workers    = ${NUM_WORKERS}"

python -m tactile_vqvae.extract_codes \
    --checkpoint   "${CHECKPOINT}" \
    --data_root    "${DATA_ROOT}" \
    --num_workers  ${NUM_WORKERS} \
    --batch_size   ${BATCH:-512} \
    --overwrite    ${OVERWRITE:-0}

echo ">>> Done. Each episode_*/ now has tactile_codes.h5"
