#!/bin/bash
# ---------------------------------------------------------------------------
# Evaluate a trained tactile VQ-VAE checkpoint.
#
# Usage:
#   CHECKPOINT=/path/to/latest.pt DATA_ROOT=/path/to/merged_root \
#     bash eval_vqvae_f6.sh
#
# Or, with CKPT_ROOT set, auto-pick the most recently modified run dir:
#   CKPT_ROOT=/path/to/tactile_vqvae DATA_ROOT=... bash eval_vqvae_f6.sh
# ---------------------------------------------------------------------------
set -euo pipefail

PARENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PARENT_DIR}"

: "${DATA_ROOT:?set DATA_ROOT to the merged midtrain data root}"

export PYTHONPATH="${PARENT_DIR}:${PYTHONPATH:-}"

: "${CHECKPOINT:=}"
: "${CKPT_ROOT:=}"

if [ -z "${CHECKPOINT}" ]; then
    if [ -z "${CKPT_ROOT}" ]; then
        echo "ERROR: set CHECKPOINT=/path/to/latest.pt (or CKPT_ROOT to auto-pick a run dir)." >&2
        exit 1
    fi
    RUN_DIR=$(ls -td "${CKPT_ROOT}"/vqvae_f6_w16_k* 2>/dev/null | head -1)
    if [ -z "${RUN_DIR}" ]; then
        echo "ERROR: no run dir found under ${CKPT_ROOT}" >&2
        exit 1
    fi
    CHECKPOINT="${RUN_DIR}/latest.pt"
fi

if [ ! -f "${CHECKPOINT}" ]; then
    echo "ERROR: checkpoint not found: ${CHECKPOINT}" >&2
    exit 1
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
    --batch_size   "${BATCH:-512}" \
    --num_workers  "${NUM_WORKERS:-4}"

echo ">>> Wrote eval.json  → ${OUTPUT}"
echo ">>> Wrote exemplars  → ${EXEMPLARS}"

# Print a compact summary from the JSON we just wrote.
OUTPUT="${OUTPUT}" python3 - <<'PY'
import json, os, sys
p = os.environ.get("OUTPUT", "")
if not p or not os.path.isfile(p):
    sys.exit(0)
d = json.load(open(p))
print("\n─── Eval Summary ────────────────────────────────────")
print(f"  checkpoint      : {os.path.basename(os.path.dirname(d['checkpoint']))}")
print(f"  n_samples       : {d['n_samples']:,}")
print(f"  overall recon   : {d['overall_recon_mse']:.5f}")
print(f"  perplexity      : {d['perplexity']:.1f}  (max={d['codebook_size']})")
print(f"  active codes    : {d['active_codes']}/{d['codebook_size']}  ({d['active_ratio']*100:.1f}%)")
print(f"  max code freq   : {d['max_code_freq']*100:.2f}%")
print("  recon by magnitude quartile:")
for k, v in d["by_magnitude"].items():
    if v.get("count", 0) == 0:
        continue
    print(f"    {k}: mse={v['recon_mse']:.5f}  mag=[{v['mag_min']:.2f},{v['mag_max']:.2f}]  n={v['count']:,}")
print("─────────────────────────────────────────────────────")
PY
