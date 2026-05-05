#!/bin/bash
# Evaluate a trained Tactile VQ-VAE checkpoint.
#
# Usage (defaults to latest run):
#   bash eval_vqvae_f6.sh
#
# Override checkpoint or data root:
#   CHECKPOINT=/path/to/latest.pt bash eval_vqvae_f6.sh
#   DATA_ROOT=/other/root CHECKPOINT=... bash eval_vqvae_f6.sh

set -e
set -o pipefail

PARENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PARENT_DIR}"

source /mnt/amlfs-01/home/dniu/anaconda3/bin/activate /mnt/amlfs-01/home/dniu/anaconda3/envs/dex_mot
export PATH=/mnt/amlfs-01/home/dniu/anaconda3/envs/dex_mot/bin:$PATH
export PYTHONPATH=${PARENT_DIR}:${PYTHONPATH}

CKPT_ROOT="/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/ckpts/dex_mot_expert/tactile_vqvae"
DATA_ROOT="${DATA_ROOT:-/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/data/midtrain/merged}"

# Auto-pick the most recently modified run dir if CHECKPOINT is not set.
if [ -z "${CHECKPOINT}" ]; then
    RUN_DIR=$(ls -td "${CKPT_ROOT}"/vqvae_f6_w16_k256_* 2>/dev/null | head -1)
    if [ -z "${RUN_DIR}" ]; then
        echo "ERROR: no run dir found under ${CKPT_ROOT}" >&2; exit 1
    fi
    CHECKPOINT="${RUN_DIR}/latest.pt"
fi

if [ ! -f "${CHECKPOINT}" ]; then
    echo "ERROR: checkpoint not found: ${CHECKPOINT}" >&2; exit 1
fi

RUN_DIR="$(dirname "${CHECKPOINT}")"
OUTPUT="${RUN_DIR}/eval.json"
EXEMPLARS="${RUN_DIR}/exemplars.npz"

echo ">>> checkpoint = ${CHECKPOINT}"
echo ">>> data_root  = ${DATA_ROOT}"
echo ">>> output     = ${OUTPUT}"

python -m tactile_vqvae.eval \
    --checkpoint   "${CHECKPOINT}" \
    --data_root    "${DATA_ROOT}" \
    --output       "${OUTPUT}" \
    --exemplars    "${EXEMPLARS}" \
    --top_k        20 \
    --batch_size   ${BATCH:-512} \
    --num_workers  ${NUM_WORKERS:-4}

echo ">>> Wrote eval.json  → ${OUTPUT}"
echo ">>> Wrote exemplars  → ${EXEMPLARS}"

# Print a compact summary from the JSON.
python3 - <<'PY'
import json, sys, os
p = os.environ.get("OUTPUT", "")
# find latest eval.json
import glob
hits = sorted(glob.glob("/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/ckpts/dex_mot_expert/tactile_vqvae/*/eval.json"), key=os.path.getmtime)
if not hits: sys.exit(0)
d = json.load(open(hits[-1]))
print("\n─── Eval Summary ────────────────────────────────────")
print(f"  checkpoint      : {os.path.basename(os.path.dirname(d['checkpoint']))}")
print(f"  n_samples       : {d['n_samples']:,}")
print(f"  overall recon   : {d['overall_recon_mse']:.5f}")
print(f"  perplexity      : {d['perplexity']:.1f}  (max={d['codebook_size']})")
print(f"  active codes    : {d['active_codes']}/{d['codebook_size']}  ({d['active_ratio']*100:.1f}%)")
print(f"  max code freq   : {d['max_code_freq']*100:.2f}%")
print("  recon by magnitude quartile:")
for k, v in d["by_magnitude"].items():
    if v.get("count", 0) == 0: continue
    print(f"    {k}: mse={v['recon_mse']:.5f}  mag=[{v['mag_min']:.2f},{v['mag_max']:.2f}]  n={v['count']:,}")
print("─────────────────────────────────────────────────────")
PY
