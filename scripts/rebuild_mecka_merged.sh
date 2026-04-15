#!/bin/bash
# Diagnostic + rebuild script for Mecka merged data root.
# Checks all 10 batch sources, reports issues, and rebuilds symlinks cleanly.
set -euo pipefail

MERGED_DATA_ROOT="/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/data/mecka_merged"

declare -A DONE_DIR
declare -A MANIFEST_BASE

DONE_DIR[01]="/mnt/amlfs-03/shared/human_egocentric/dniu/mecka_dexmot/batch-01-done"
DONE_DIR[02]="/mnt/amlfs-03/shared/human_egocentric/dniu/mecka_dexmot/batch-02-done"
DONE_DIR[03]="/mnt/amlfs-07/shared/datasets/dniu/mecka_dexmot/batch-03-done"
DONE_DIR[04]="/mnt/amlfs-03/shared/human_egocentric/dniu/mecka_dexmot/batch-04-done"
DONE_DIR[05]="/mnt/amlfs-02/shared/human_egocentric/dniu/mecka_dexmot/batch-05-done"
DONE_DIR[06]="/mnt/amlfs-02/shared/human_egocentric/dniu/mecka_dexmot/batch-06-done"
DONE_DIR[07]="/mnt/amlfs-02/shared/human_egocentric/dniu/mecka_dexmot/batch-07-done"
DONE_DIR[08]="/mnt/amlfs-07/shared/datasets/dniu/mecka_dexmot/batch-08-done"
DONE_DIR[09]="/mnt/amlfs-03/shared/human_egocentric/dniu/mecka_dexmot/batch-09-done"
DONE_DIR[10]="/mnt/amlfs-02/shared/human_egocentric/dniu/mecka_dexmot/batch-10-done"

MANIFEST_BASE[01]="/mnt/amlfs-07/shared/datasets/dniu/mecka_dexmot"
MANIFEST_BASE[02]="/mnt/amlfs-03/shared/human_egocentric/dniu/mecka_dexmot"
MANIFEST_BASE[03]="/mnt/amlfs-07/shared/datasets/dniu/mecka_dexmot"
MANIFEST_BASE[04]="/mnt/amlfs-03/shared/human_egocentric/dniu/mecka_dexmot"
MANIFEST_BASE[05]="/mnt/amlfs-02/shared/human_egocentric/dniu/mecka_dexmot"
MANIFEST_BASE[06]="/mnt/amlfs-02/shared/human_egocentric/dniu/mecka_dexmot"
MANIFEST_BASE[07]="/mnt/amlfs-02/shared/human_egocentric/dniu/mecka_dexmot"
MANIFEST_BASE[08]="/mnt/amlfs-07/shared/datasets/dniu/mecka_dexmot"
MANIFEST_BASE[09]="/mnt/amlfs-03/shared/human_egocentric/dniu/mecka_dexmot"
MANIFEST_BASE[10]="/mnt/amlfs-02/shared/human_egocentric/dniu/mecka_dexmot"

echo "============================================================"
echo "  PHASE 1: Diagnose all 10 batch sources"
echo "============================================================"

total_ok=0
total_missing=0
total_no_manifest=0

for batch_id in $(echo "${!DONE_DIR[@]}" | tr ' ' '\n' | sort); do
    done_dir="${DONE_DIR[$batch_id]}"
    base="${MANIFEST_BASE[$batch_id]}"
    echo ""
    echo "--- Batch ${batch_id} ---"
    echo "  done_dir:      ${done_dir}"
    echo "  manifest_base: ${base}"

    # Check done_dir exists
    if [ ! -d "${done_dir}" ]; then
        echo "  [FAIL] done_dir does NOT exist or is not accessible!"
        total_missing=$((total_missing + 1))
        continue
    fi
    echo "  [OK] done_dir exists"

    # Check manifest base symlink: batch-XX → batch-XX-done
    link="${base}/batch-${batch_id}"
    if [ -L "${link}" ]; then
        target=$(readlink "${link}")
        if [ -d "${link}" ]; then
            echo "  [OK] symlink ${link} -> ${target} (resolves)"
        else
            echo "  [WARN] symlink ${link} -> ${target} (DANGLING - target unreachable)"
        fi
    elif [ -d "${link}" ]; then
        echo "  [OK] ${link} is a real directory (no symlink needed)"
    else
        echo "  [MISSING] symlink ${link} does not exist"
    fi

    # Count sub-batches with valid manifests
    n_sub=0
    n_manifest=0
    for sub_batch_dir in "${done_dir}"/batch*/; do
        [ -d "${sub_batch_dir}" ] || continue
        n_sub=$((n_sub + 1))
        manifest="${sub_batch_dir}/pretrain_manifest.json"
        if [ -s "${manifest}" ]; then
            n_manifest=$((n_manifest + 1))
        fi
    done
    echo "  sub-batch dirs: ${n_sub}, with valid manifest: ${n_manifest}"
    total_ok=$((total_ok + n_manifest))
    if [ "${n_manifest}" -eq 0 ]; then
        total_no_manifest=$((total_no_manifest + 1))
        echo "  [WARN] No valid manifests found in any sub-batch!"
    fi
done

echo ""
echo "============================================================"
echo "  PHASE 1 SUMMARY"
echo "============================================================"
echo "  Total sub-batches with valid manifests: ${total_ok}"
echo "  Batch sources with missing done_dir:    ${total_missing}"
echo "  Batch sources with zero manifests:      ${total_no_manifest}"

echo ""
echo "============================================================"
echo "  PHASE 2: Clean and rebuild Step 1 symlinks (batch-XX)"
echo "============================================================"

for batch_id in $(echo "${!DONE_DIR[@]}" | tr ' ' '\n' | sort); do
    done_dir="${DONE_DIR[$batch_id]}"
    base="${MANIFEST_BASE[$batch_id]}"
    link="${base}/batch-${batch_id}"

    # Remove old/stale symlink
    if [ -L "${link}" ]; then
        rm "${link}"
        echo "  Removed stale symlink: ${link}"
    fi

    # Only create if done_dir actually exists
    if [ -d "${done_dir}" ]; then
        if [ ! -e "${link}" ]; then
            ln -s "${done_dir}" "${link}"
            echo "  Created: ${link} -> ${done_dir}"
        else
            echo "  Skipped (real dir exists): ${link}"
        fi
    else
        echo "  [SKIP] done_dir missing: ${done_dir}"
    fi
done

echo ""
echo "============================================================"
echo "  PHASE 3: Clean and rebuild merged data root"
echo "============================================================"
echo "  Merged root: ${MERGED_DATA_ROOT}"

# Remove all existing symlinks in the merged root (only symlinks, not real dirs)
if [ -d "${MERGED_DATA_ROOT}" ]; then
    n_removed=$(find "${MERGED_DATA_ROOT}" -maxdepth 1 -type l | wc -l)
    find "${MERGED_DATA_ROOT}" -maxdepth 1 -type l -delete
    echo "  Removed ${n_removed} old symlinks from merged root"
else
    mkdir -p "${MERGED_DATA_ROOT}"
    echo "  Created merged root directory"
fi

# Rebuild
total_linked=0
for batch_id in $(echo "${!DONE_DIR[@]}" | tr ' ' '\n' | sort); do
    done_dir="${DONE_DIR[$batch_id]}"
    [ -d "${done_dir}" ] || continue

    created=0
    for sub_batch_dir in "${done_dir}"/batch*/; do
        [ -d "${sub_batch_dir}" ] || continue
        manifest="${sub_batch_dir}/pretrain_manifest.json"
        [ -s "${manifest}" ] || continue
        sub_name=$(basename "${sub_batch_dir}")
        link="${MERGED_DATA_ROOT}/b${batch_id}_${sub_name}"
        ln -s "${sub_batch_dir}" "${link}"
        created=$((created + 1))
    done
    echo "  batch-${batch_id}: ${created} sub-batches linked"
    total_linked=$((total_linked + created))
done

echo ""
echo "============================================================"
echo "  FINAL RESULT"
echo "============================================================"
echo "  Total sub-batches in merged root: ${total_linked}"
ls "${MERGED_DATA_ROOT}" | head -5
echo "  ..."
echo ""
echo "Done. You can now run pretrain_mecka.sh."
