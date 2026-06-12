#!/bin/bash
# ---------------------------------------------------------------------------
# Extract per-episode tactile codes (sidecar tactile_codes.h5) using a
# trained tactile VQ-VAE checkpoint.
#
# Required env vars:
#   CHECKPOINT      path to the VQ-VAE checkpoint (latest.pt)
#   DATA_ROOT       merged midtrain data root
#
# Optional env vars:
#   NUM_WORKERS=4
#   BATCH=512
#   OVERWRITE=0     set to 1 to recompute even if tactile_codes.h5 exists
# ---------------------------------------------------------------------------
set -euo pipefail

PARENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PARENT_DIR}"

: "${CHECKPOINT:?set CHECKPOINT=/path/to/latest.pt}"
: "${DATA_ROOT:?set DATA_ROOT=/path/to/merged midtrain root}"

export PYTHONPATH="${PARENT_DIR}:${PYTHONPATH:-}"

: "${NUM_WORKERS:=4}"
: "${BATCH:=512}"
: "${OVERWRITE:=0}"

echo ">>> checkpoint = ${CHECKPOINT}"
echo ">>> data_root  = ${DATA_ROOT}"
echo ">>> workers    = ${NUM_WORKERS}"

python -m tactile_vqvae.extract_codes \
    --checkpoint   "${CHECKPOINT}" \
    --data_root    "${DATA_ROOT}" \
    --num_workers  "${NUM_WORKERS}" \
    --batch_size   "${BATCH}" \
    --overwrite    "${OVERWRITE}"

echo ">>> Done. Each episode_*/ now has tactile_codes.h5"
