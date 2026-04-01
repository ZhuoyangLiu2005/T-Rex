"""Quick script to inspect egodex data structure. Run with: python check_egodex_data.py"""
import h5py, cv2, json, os

ep_dir = "/mnt/amlfs-07/shared/datasets/dniu/egodex/cotrain_processed_new/batch1/extra_assemble_disassemble_jigsaw_puzzle_40"

# 1. HDF5 structure
h5_path = os.path.join(ep_dir, "pretrain.hdf5")
print("=== pretrain.hdf5 ===")
with h5py.File(h5_path, "r") as f:
    for k in f.keys():
        print(f"  {k}: shape={f[k].shape}, dtype={f[k].dtype}")
    print("  attrs:")
    for k, v in f.attrs.items():
        print(f"    {k}: {v}")
    if not f.attrs.items():
        print("    (none)")

# 2. Video info
vid_path = os.path.join(ep_dir, "ego_view.mp4")
print(f"\n=== ego_view.mp4 ===")
cap = cv2.VideoCapture(vid_path)
print(f"  opened: {cap.isOpened()}")
print(f"  frames: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")
print(f"  fps:    {cap.get(cv2.CAP_PROP_FPS)}")
print(f"  width:  {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}")
print(f"  height: {int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
cap.release()

# 3. Metadata
meta_path = os.path.join(ep_dir, "metadata.json")
with open(meta_path) as f:
    meta = json.load(f)
print(f"\n=== metadata.json ===")
print(f"  frame_count: {meta.get('frame_count')}")
print(f"  description: {meta.get('session', {}).get('description', '')}")

# 4. Check manifest if exists
base = os.path.dirname(ep_dir)
for name in ["pretrain_manifest.json", "manifest.json"]:
    p = os.path.join(base, name)
    if os.path.exists(p):
        print(f"\n=== {name} (in {base}) ===")
        with open(p) as f:
            m = json.load(f)
        print(f"  top-level keys: {list(m.keys())}")
        if "episodes" in m:
            print(f"  num episodes: {len(m['episodes'])}")
            print(f"  first episode keys: {list(m['episodes'][0].keys())}")
        if "statistics" in m:
            print(f"  stats keys: {list(m['statistics'].keys())}")
