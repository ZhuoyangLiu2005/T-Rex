#!/usr/bin/env python3
"""Materialize a merged data_root for train_qwen3vl_midtrain_flare.py.

The trainer expects, under <data_root>/<batch>/:
  pretrain_manifest.json
  <demo_name>/pretrain.hdf5
  <demo_name>/raw.h5            (the original episode_<id>.h5)
  <demo_name>/ego_view.mp4      (head/slow camera)
  <demo_name>/left_wrist.mp4
  <demo_name>/right_wrist.mp4

The new datasets store these as:
  <category>/pretrain_manifest.json
  <category>/<demo_name>/pretrain.hdf5
  <category>/<demo_name>/episode_<id>.h5
  <category>/<demo_name>/episode_<id>_head_left_rgb.mp4
  <category>/<demo_name>/episode_<id>_left_wrist.mp4
  <category>/<demo_name>/episode_<id>_right_wrist.mp4

For each (prefix, src_root) we build a shadow under <merged>/<prefix>[_<category>]/
that contains a rewritten manifest plus per-episode dirs of renamed symlinks.
"""

import argparse
import glob
import json
import os
import sys

import h5py
import numpy as np


def _safe_link(src: str, dst: str) -> None:
    """Create a symlink dst -> src, replacing any stale link."""
    if os.path.islink(dst):
        try:
            if os.readlink(dst) == src:
                return
        except OSError:
            pass
        os.unlink(dst)
    elif os.path.exists(dst):
        return
    os.symlink(src, dst)


def _find_episode_files(orig_ep_dir: str):
    """Return (h5_name, prefix) for the canonical episode_<id>.h5 in orig_ep_dir.

    Returns (None, None) if no unique match is found.
    """
    candidates = sorted(
        n for n in os.listdir(orig_ep_dir)
        if n.startswith("episode_") and n.endswith(".h5") and not n.endswith(".bak")
    )
    if len(candidates) != 1:
        return None, None
    h5_name = candidates[0]
    return h5_name, h5_name[:-3]  # strip ".h5"


def _read_episode_tacf6(orig_ep_dir: str):
    """Return concatenated [left, right] tactile_f6 as (N, 60) float32, or None.

    Reads only the f6 datasets (small — ~N*60*4 bytes each), not the full
    raw HDF5, so this is cheap.
    """
    h5_name, _ = _find_episode_files(orig_ep_dir)
    if h5_name is None:
        return None
    rh5 = os.path.join(orig_ep_dir, h5_name)
    try:
        with h5py.File(rh5, "r") as f:
            if ("left_hand_tactile_f6" not in f
                    or "right_hand_tactile_f6" not in f):
                return None
            l = f["left_hand_tactile_f6"][:]   # (N, 5, 6) float32
            r = f["right_hand_tactile_f6"][:]
    except Exception:
        return None
    v = np.concatenate([l, r], axis=1).reshape(-1, 60).astype(np.float32)
    return v


def _materialize_episode(orig_ep_dir: str, shadow_ep_dir: str) -> bool:
    """Return True iff at least the pretrain.hdf5 + raw.h5 symlinks exist after."""
    pretrain_src = os.path.join(orig_ep_dir, "pretrain.hdf5")
    if not os.path.isfile(pretrain_src):
        return False

    h5_name, prefix = _find_episode_files(orig_ep_dir)
    if prefix is None:
        return False

    os.makedirs(shadow_ep_dir, exist_ok=True)
    _safe_link(pretrain_src, os.path.join(shadow_ep_dir, "pretrain.hdf5"))
    _safe_link(os.path.join(orig_ep_dir, h5_name),
               os.path.join(shadow_ep_dir, "raw.h5"))

    for canonical, suffix in [
        ("ego_view.mp4",     "_head_left_rgb.mp4"),
        ("left_wrist.mp4",   "_left_wrist.mp4"),
        ("right_wrist.mp4",  "_right_wrist.mp4"),
    ]:
        src = os.path.join(orig_ep_dir, prefix + suffix)
        if os.path.isfile(src):
            _safe_link(src, os.path.join(shadow_ep_dir, canonical))
    return True


def _iter_batches(prefix: str, src_root: str):
    """Yield (batch_name, batch_root_with_manifest)."""
    if not os.path.isdir(src_root):
        print(f"  [WARN] missing source: {src_root}", file=sys.stderr)
        return
    if os.path.isfile(os.path.join(src_root, "pretrain_manifest.json")):
        yield prefix, src_root
        return
    found = 0
    for name in sorted(os.listdir(src_root)):
        sub = os.path.join(src_root, name)
        if os.path.isfile(os.path.join(sub, "pretrain_manifest.json")):
            yield f"{prefix}_{name}", sub
            found += 1
    if found == 0:
        print(f"  [WARN] no pretrain_manifest.json under {src_root} (or its subdirs)",
              file=sys.stderr)


def materialize_source(prefix: str, src_root: str, merged_root: str) -> dict:
    """Build shadow batches for one source. Returns simple counters."""
    n_batches = 0
    n_eps_ok = 0
    n_eps_skip = 0

    for batch_name, batch_dir in _iter_batches(prefix, src_root):
        with open(os.path.join(batch_dir, "pretrain_manifest.json")) as f:
            manifest = json.load(f)

        episodes = manifest.get("episodes") or []
        if not episodes:
            print(f"  [WARN] empty manifest in {batch_dir}", file=sys.stderr)
            continue

        shadow_batch_dir = os.path.join(merged_root, batch_name)
        os.makedirs(shadow_batch_dir, exist_ok=True)

        new_eps = []
        f6_samples = []   # for batch-level q01/q99 over all frames
        for ep in episodes:
            orig_ep_dir = ep["episode_dir"]
            demo_name = ep.get("demo_name") or os.path.basename(orig_ep_dir)
            shadow_ep_dir = os.path.join(shadow_batch_dir, demo_name)

            if _materialize_episode(orig_ep_dir, shadow_ep_dir):
                ep_out = dict(ep)
                ep_out["episode_dir"] = shadow_ep_dir
                new_eps.append(ep_out)
                n_eps_ok += 1
                f6 = _read_episode_tacf6(orig_ep_dir)
                if f6 is not None:
                    f6_samples.append(f6)
            else:
                n_eps_skip += 1

        new_manifest = dict(manifest)
        new_manifest["batch_root"] = shadow_batch_dir
        new_manifest["episodes"] = new_eps
        new_manifest["num_episodes"] = len(new_eps)

        # Inject tactile_f6 q01/q99 stats so the trainer doesn't fall back to
        # the [-1, +1] default that clips real readings (which range up to
        # roughly ±17 on the force channels).
        if f6_samples:
            f6_all = np.concatenate(f6_samples, axis=0)   # (Nframes_total, 60)
            stats = dict(new_manifest.get("statistics") or {})
            stats["tactile_f6"] = {
                "q01": np.quantile(f6_all, 0.01, axis=0).astype(float).tolist(),
                "q99": np.quantile(f6_all, 0.99, axis=0).astype(float).tolist(),
            }
            new_manifest["statistics"] = stats

        out_path = os.path.join(shadow_batch_dir, "pretrain_manifest.json")
        tmp_path = f"{out_path}.tmp.{os.getpid()}"
        with open(tmp_path, "w") as f:
            json.dump(new_manifest, f)
        os.replace(tmp_path, out_path)
        n_batches += 1
        print(f"  {batch_name}: {len(new_eps)} episodes "
              f"(skipped {len(episodes) - len(new_eps)})")

    return {"batches": n_batches, "ok": n_eps_ok, "skip": n_eps_skip}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged_root", required=True)
    ap.add_argument("--source", action="append", default=[],
                    metavar="PREFIX=ROOT",
                    help="repeat per source, e.g. --source nv=/path/to/nv")
    args = ap.parse_args()

    os.makedirs(args.merged_root, exist_ok=True)

    sources = []
    for spec in args.source:
        if "=" not in spec:
            print(f"bad --source spec: {spec!r}", file=sys.stderr)
            return 2
        prefix, root = spec.split("=", 1)
        sources.append((prefix.strip(), root.strip()))

    totals = {"batches": 0, "ok": 0, "skip": 0}
    for prefix, root in sources:
        print(f">>> source '{prefix}' -> {root}")
        c = materialize_source(prefix, root, args.merged_root)
        for k in totals:
            totals[k] += c[k]

    n_dirs = len(glob.glob(os.path.join(args.merged_root,
                                         "*", "pretrain_manifest.json")))
    print(f">>> merged ready: {totals['batches']} batches, "
          f"{totals['ok']} episodes ok, {totals['skip']} skipped, "
          f"{n_dirs} manifests visible at {args.merged_root}/*/")
    return 0 if n_dirs > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
