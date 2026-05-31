import os
import re
import cv2
import h5py
import json
import numpy as np
import sys
import random
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

ACTION_CHUNK = 16
FRAME_STRIDE = 1

IMAGE_VIEWS_SLOW = 'image_primary'
IMAGE_VIEWS_FAST_L = 'image_wrist_left'
IMAGE_VIEWS_FAST_R = 'image_wrist_right'

VIDEO_SLOW_SUFFIX = 'head_left_rgb.mp4'
VIDEO_FAST_L_SUFFIX = 'left_wrist.mp4'
VIDEO_FAST_R_SUFFIX = 'right_wrist.mp4'

CROP_BOX_SLOW = (0, 300, 140, 540)  # (y_min, y_max, x_min, x_max) or None to disable

DEFAULT_INSTRUCTION = "I am T-Rex."

# Bimanual: Left Arm(9D) + Left Hand(22D) + Right Arm(9D) + Right Hand(22D) = 62D
ACTION_MASK = [True] * 62
STATE_MASK = [True] * 62
TACTILE_F6_MASK = [True] * 60  # 10 fingers x 6 channels

def pose_matrix_to_9d(pose_matrices):
    trans = pose_matrices[:, :3, 3]
    rot_col1 = pose_matrices[:, :3, 0]
    rot_col2 = pose_matrices[:, :3, 1]
    return np.concatenate([trans, rot_col1, rot_col2], axis=-1)

def get_rot_mat(vec6d):
    col1 = vec6d[:3]
    col2 = vec6d[3:6]
    col3 = np.cross(col1, col2)
    return np.column_stack([col1, col2, col3])

def compute_chunk_delta_pose(curr_pose, target_pose):
    R_curr = curr_pose[:3, :3]
    t_curr = curr_pose[:3, 3]

    R_targ = target_pose[:3, :3]
    t_targ = target_pose[:3, 3]

    delta_xyz = R_curr.T @ (t_targ - t_curr)
    R_delta = R_curr.T @ R_targ

    delta_rot_col1 = R_delta[:3, 0]
    delta_rot_col2 = R_delta[:3, 1]

    return np.concatenate([delta_xyz, delta_rot_col1, delta_rot_col2])


def compute_tracking_error_axis_angle(state_31d, target_31d):
    s_arm = state_31d[:9]
    t_arm = target_31d[:9]

    t_state = s_arm[:3]
    R_state = get_rot_mat(s_arm[3:9])

    t_targ = t_arm[:3]
    R_targ = get_rot_mat(t_arm[3:9])

    delta_xyz = R_state.T @ (t_targ - t_state)
    R_delta = R_state.T @ R_targ

    delta_rot_axis_angle, _ = cv2.Rodrigues(R_delta)
    delta_rot_axis_angle = delta_rot_axis_angle.flatten()

    arm_error_6d = np.concatenate([delta_xyz, delta_rot_axis_angle])
    hand_error_22d = target_31d[9:] - state_31d[9:]

    return np.concatenate([arm_error_6d, hand_error_22d])


def compute_bimanual_tracking_error(state_62d, target_62d):
    """Compute tracking error for both arms. Returns 56D (28D left + 28D right)."""
    left_err = compute_tracking_error_axis_angle(state_62d[:31], target_62d[:31])
    right_err = compute_tracking_error_axis_angle(state_62d[31:], target_62d[31:])
    return np.concatenate([left_err, right_err])


def is_video_readable(video_path):
    """Return True only if the video exists, opens successfully, and yields at least one frame."""
    if not os.path.exists(video_path):
        return False
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return False
    ret, _ = cap.read()
    cap.release()
    return ret


def extract_frames_from_video(video_path, save_dir, view_name, crop_box=None):
    if not os.path.exists(video_path):
        print(f"warning: video not exist {video_path}")
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"warning: cannot open video {video_path}")
        return []

    frame_idx = 0
    img_paths = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if crop_box is not None:
            y_min, y_max, x_min, x_max = crop_box
            frame = frame[y_min:y_max, x_min:x_max]

        img_filename = f'image{frame_idx}_{view_name}.png'
        img_path = os.path.join(save_dir, img_filename)
        if not os.path.exists(img_path):
            cv2.imwrite(img_path, frame)

        img_paths.append(img_path)
        frame_idx += 1

    cap.release()
    return img_paths


def process_single_episode_worker(args):
    """
    Worker function for parallel processing.
    Returns (episode_name, list_of_episode_data_dicts) instead of writing directly.
    """
    episode_dir, save_dir, task_name, source_prefix = args
    episode_name = os.path.basename(episode_dir)
    episode_id = f"{source_prefix}/{episode_name}"
    print(f"processing: {episode_id} ...", end=" ", flush=True)

    h5_path = os.path.join(episode_dir, f"{episode_name}.h5")
    video_slow_path = os.path.join(episode_dir, f"{episode_name}_{VIDEO_SLOW_SUFFIX}")
    video_fast_l_path = os.path.join(episode_dir, f"{episode_name}_{VIDEO_FAST_L_SUFFIX}")
    video_fast_r_path = os.path.join(episode_dir, f"{episode_name}_{VIDEO_FAST_R_SUFFIX}")

    if not os.path.exists(h5_path):
        print(f"skip: H5 file not found")
        return episode_name, []

    # Validate all videos before creating directories or reading the H5 file.
    # "moov atom not found" means the recording was interrupted before the
    # container was finalized — cap.isOpened() returns False in that case.
    for vpath in [video_slow_path, video_fast_l_path, video_fast_r_path]:
        if not is_video_readable(vpath):
            print(f"skip: corrupted or missing video {os.path.basename(vpath)}", flush=True)
            return episode_name, []

    episode_img_save_dir = os.path.join(save_dir, task_name, source_prefix, episode_name)
    os.makedirs(episode_img_save_dir, exist_ok=True)

    slow_img_paths = extract_frames_from_video(video_slow_path, episode_img_save_dir, IMAGE_VIEWS_SLOW, CROP_BOX_SLOW)
    fast_l_img_paths = extract_frames_from_video(video_fast_l_path, episode_img_save_dir, IMAGE_VIEWS_FAST_L)
    fast_r_img_paths = extract_frames_from_video(video_fast_r_path, episode_img_save_dir, IMAGE_VIEWS_FAST_R)

    with h5py.File(h5_path, 'r') as f:
        # ----- Left arm state: current pose (Absolute 9D) -----
        s_l_arm_pose = f['left_arm_current_pose'][:]         # (N, 4, 4)
        s_l_arm_9d = pose_matrix_to_9d(s_l_arm_pose)        # (N, 9)
        s_l_hnd = f['left_hand_joint_positions'][:]          # (N, 22)
        
        # ----- Right arm state: current pose (Absolute 9D) -----
        s_r_arm_pose = f['right_arm_current_pose'][:]       # (N, 4, 4)
        s_r_arm_9d = pose_matrix_to_9d(s_r_arm_pose)        # (N, 9)
        s_r_hnd = f['right_hand_joint_positions'][:]         # (N, 22)

        # State: left(31D) + right(31D) = 62D
        states = np.concatenate([s_l_arm_9d, s_l_hnd, s_r_arm_9d, s_r_hnd], axis=1)

        # ----- Right arm target: target pose (for delta actions) -----
        a_r_arm_pose = f['right_arm_target_pose'][:]         # (N, 4, 4)
        a_r_arm_9d = pose_matrix_to_9d(a_r_arm_pose)        # (N, 9)
        a_r_hnd = f['right_hand_target_joint_positions'][:]  # (N, 22)

        # ----- Left arm target: target pose (for delta actions) -----
        a_l_arm_pose = f['left_arm_target_pose'][:]          # (N, 4, 4)
        a_l_arm_9d = pose_matrix_to_9d(a_l_arm_pose)        # (N, 9)
        a_l_hnd = f['left_hand_target_joint_positions'][:]   # (N, 22)

        # Absolute targets: left(31D) + right(31D) = 62D
        absolute_targets = np.concatenate([a_l_arm_9d, a_l_hnd, a_r_arm_9d, a_r_hnd], axis=1)

        # ----- Tactile -----
        t_l_f6 = f['left_hand_tactile_f6'][:]                # (N, 5, 6)
        t_r_f6 = f['right_hand_tactile_f6'][:]               # (N, 5, 6)
        t_l_deform = f['left_hand_tactile_deform'][:]        # (N, 5, H, W)
        t_r_deform = f['right_hand_tactile_deform'][:]       # (N, 5, H, W)

    episode_length = len(states)
    min_len = min(episode_length, len(slow_img_paths), len(fast_r_img_paths), len(fast_l_img_paths))
    print(f"min length: {min_len}", flush=True)

    records = []
    for i in range(0, min_len):
        img_slow = slow_img_paths[i]
        img_fast_list = [fast_r_img_paths[i], fast_l_img_paths[i]]

        # Tactile deform images: 5 right + 5 left = 10 fingers
        tactile_img_paths_current = []
        for finger_idx in range(5):
            img_arr = t_l_deform[i, finger_idx]
            path = os.path.join(episode_img_save_dir, f'image{i}_tactile_left_deform_{finger_idx}.png')
            if not os.path.exists(path):
                cv2.imwrite(path, img_arr)
            tactile_img_paths_current.append(path)
        
        for finger_idx in range(5):
            img_arr = t_r_deform[i, finger_idx]
            path = os.path.join(episode_img_save_dir, f'image{i}_tactile_right_deform_{finger_idx}.png')
            if not os.path.exists(path):
                cv2.imwrite(path, img_arr)
            tactile_img_paths_current.append(path)

        # ----- Action chunk: delta-base for both arms -----
        chunk_base_l_arm_pose = s_l_arm_pose[i]
        chunk_base_r_arm_pose = s_r_arm_pose[i]

        action_chunk_list = []
        for k in range(ACTION_CHUNK):
            base_future_idx = min(i + k * FRAME_STRIDE, min_len - 1)
            actual_future_idx = min(base_future_idx + FRAME_STRIDE - 1, min_len - 1)
            
            # Left arm delta
            target_l_arm_pose = a_l_arm_pose[actual_future_idx]
            target_l_hand = a_l_hnd[actual_future_idx]
            delta_l_9d = compute_chunk_delta_pose(chunk_base_l_arm_pose, target_l_arm_pose)

            # Right arm delta
            target_r_arm_pose = a_r_arm_pose[actual_future_idx]
            target_r_hand = a_r_hnd[actual_future_idx]
            delta_r_9d = compute_chunk_delta_pose(chunk_base_r_arm_pose, target_r_arm_pose)

            # 62D: left(9+22) + right(9+22)
            act = np.concatenate([delta_l_9d, target_l_hand, delta_r_9d, target_r_hand]).tolist()
            action_chunk_list.append(act)

        current_state = states[i].tolist()

        # Tactile f6: concatenate left(5,6) + right(5,6) -> flatten to 60D
        tactile_f6_combined = np.concatenate([t_l_f6[i], t_r_f6[i]], axis=0).tolist()

        target_idx_next = min(i + FRAME_STRIDE - 1, min_len - 1)
        current_abs_action = absolute_targets[target_idx_next].tolist()

        episode_data = {
            'episode_id': episode_id,
            'image_old_slow': img_slow,
            'image_old_fast': img_fast_list,
            'action': action_chunk_list,
            'absolute_target_action': current_abs_action,
            'state': current_state,
            'tactile_f6': tactile_f6_combined,
            'tactile_image_deform': tactile_img_paths_current,
            'language_instruction': DEFAULT_INSTRUCTION,
        }
        records.append(episode_data)

    return episode_name, records

def jsonl_2_json(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    output_data = []
    for line in lines:
        item = json.loads(line)
        new_item = {
            "input_prompt": item["language_instruction"],
            "input_image_slow": [item["image_old_slow"]],
            "input_image_fast": item["image_old_fast"],
            "input_image_resolution": [384, 384],
            "action": item["action"],
            "state_slow": item["state"],
            "state_fast": item["state"],
            "tactile_f6": item["tactile_f6"],
            "tactile_image_deform": item["tactile_image_deform"]
        }
        output_data.append(new_item)

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

def cal_stats(jsonl_filename, dataset_name="rlbench"):
    actions = []
    states = []
    tactiles = []
    episode_numbers = set()
    tracking_errors = []

    with open(jsonl_filename, 'r') as f:
        current_ep_data = []
        current_ep_id = None

        for line in f:
            data = json.loads(line)
            # Prefer explicit episode_id; fall back to regex on image path for old files
            if 'episode_id' in data:
                ep_id = data['episode_id']
            else:
                match = re.search(r'episode_?(\d+)', data['image_old_slow'])
                ep_id = match.group(0) if match else 'unknown'
            episode_numbers.add(ep_id)

            if ep_id != current_ep_id and len(current_ep_data) > 0:
                for t in range(1, len(current_ep_data)):
                    state_t = np.array(current_ep_data[t]['state'])
                    action_t_minus_1 = np.array(current_ep_data[t-1]['absolute_target_action'])
                    err = compute_bimanual_tracking_error(state_t, action_t_minus_1)
                    tracking_errors.append(err)
                current_ep_data = []

            current_ep_id = ep_id
            current_ep_data.append(data)
            actions.append(data['action'])
            states.append(data['state'])
            tactiles.append(data['tactile_f6'])

        if len(current_ep_data) > 0:
            for t in range(1, len(current_ep_data)):
                state_t = np.array(current_ep_data[t]['state'])
                action_t_minus_1 = np.array(current_ep_data[t-1]['absolute_target_action'])
                err = compute_bimanual_tracking_error(state_t, action_t_minus_1)
                tracking_errors.append(err)

    actions = np.array(actions)
    states = np.array(states)
    tactiles_arr = np.array(tactiles)
    tactiles_flat = tactiles_arr.reshape(tactiles_arr.shape[0], -1)
    tracking_errors = np.array(tracking_errors)

    def calculate_stats(data, mask=None):
        if mask is None:
            mask = [True] * data.shape[-1]
        stats = {
            'mean': np.mean(data, axis=0).tolist(),
            'std': np.std(data, axis=0).tolist(),
            'max': np.max(data, axis=0).tolist(),
            'min': np.min(data, axis=0).tolist(),
            'q01': np.quantile(data, 0.01, axis=0).tolist(),
            'q99': np.quantile(data, 0.99, axis=0).tolist(),
            'mask': mask,
        }
        return stats

    action_stats = calculate_stats(actions, ACTION_MASK)
    state_stats = calculate_stats(states, STATE_MASK)
    tactile_f6_stats = calculate_stats(tactiles_flat, TACTILE_F6_MASK)

    # Tracking error: 28D left + 28D right = 56D
    TRACKING_ERROR_MASK = [True] * 56

    if len(tracking_errors) > 0:
        tracking_error_stats = {
            'mean': np.mean(tracking_errors, axis=0).tolist(),
            'std': np.std(tracking_errors, axis=0).tolist(),
            'mean_abs': np.mean(np.abs(tracking_errors), axis=0).tolist(),
            'mask': TRACKING_ERROR_MASK
        }
    else:
        tracking_error_stats = {}

    result = {
        dataset_name: {
            "action": action_stats,
            "state": state_stats,
            "tactile_f6": tactile_f6_stats,
            "tracking_error": tracking_error_stats,
            "num_transitions": len(actions),
            "num_trajectories": len(episode_numbers)
        }
    }

    output_path = jsonl_filename.replace(".jsonl", "_statistics.json")
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"stat saved to: {output_path}")


def process_dataset(data_roots, img_save_root, json_save_root, json_name_base, task_name,
                    num_workers=None, num_trajectories=None, seed=42, dataset_name="rlbench"):
    """
    data_roots: a single path string or a list of path strings.
    Each root must contain a 'success/' subdirectory with episode_* folders.
    Episodes across roots are merged; image dirs are namespaced by the root's
    basename to avoid collisions when the same episode numbers appear in multiple roots.
    """
    if isinstance(data_roots, str):
        data_roots = [data_roots]

    if num_workers is None:
        num_workers = min(multiprocessing.cpu_count(), 32)

    os.makedirs(img_save_root, exist_ok=True)
    os.makedirs(json_save_root, exist_ok=True)

    # Collect (source_prefix, episode_dir) from all roots, sorted for determinism
    all_episode_entries = []
    for data_root in data_roots:
        source_prefix = os.path.basename(data_root.rstrip('/'))
        episodes_root = os.path.join(data_root, 'success')
        if not os.path.isdir(episodes_root):
            print(f"Warning: {episodes_root} not found, skipping root {data_root}")
            continue
        ep_dirs = sorted([
            os.path.join(episodes_root, item)
            for item in sorted(os.listdir(episodes_root))
            if os.path.isdir(os.path.join(episodes_root, item)) and item.startswith("episode_")
        ])
        for ep_dir in ep_dirs:
            all_episode_entries.append((source_prefix, ep_dir))
        print(f"  {source_prefix}: {len(ep_dirs)} episodes")

    total_episodes = len(all_episode_entries)

    # --- Few-shot subsetting ---
    if num_trajectories is not None and num_trajectories < total_episodes:
        rng = random.Random(seed)
        all_episode_entries = sorted(rng.sample(all_episode_entries, num_trajectories),
                                     key=lambda x: (x[0], x[1]))
        print(f"Few-shot mode: randomly selected {num_trajectories}/{total_episodes} episodes "
              f"(seed={seed})")
        name_suffix = f"_traj{num_trajectories}_seed{seed}"
        json_name_base = json_name_base + name_suffix
    else:
        print(f"Using all {total_episodes} episodes from {len(data_roots)} root(s).")

    jsonl_filename = os.path.join(json_save_root, f'{json_name_base}.jsonl')
    json_filename = os.path.join(json_save_root, f'{json_name_base}.json')

    print(f"Output base: {json_name_base}")
    print(f"Launching {num_workers} workers...")

    # Build args with unique (source_prefix, episode_name) keys
    args_list = [(ep_dir, img_save_root, task_name, source_prefix)
                 for source_prefix, ep_dir in all_episode_entries]
    index_map = {(source_prefix, os.path.basename(ep_dir)): idx
                 for idx, (source_prefix, ep_dir) in enumerate(all_episode_entries)}

    results = [None] * len(all_episode_entries)

    executor = ProcessPoolExecutor(max_workers=num_workers)
    try:
        future_to_idx = {
            executor.submit(process_single_episode_worker, args):
                index_map[(args[3], os.path.basename(args[0]))]
            for args in args_list
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                _, records = future.result()
                results[idx] = records
            except Exception as e:
                print(f"Error processing episode at index {idx}: {e}")
                results[idx] = []
    except KeyboardInterrupt:
        print("\nInterrupted! Terminating all worker processes...")
        for proc in executor._processes.values():
            proc.terminate()
        executor.shutdown(wait=False, cancel_futures=True)
        sys.exit(1)
    else:
        executor.shutdown(wait=False)

    # Write JSONL in sorted (source, episode) order
    with open(jsonl_filename, 'w') as f_jsonl:
        for records in results:
            if records:
                for episode_data in records:
                    f_jsonl.write(json.dumps(episode_data) + '\n')

    print(f"JSONL written to: {jsonl_filename}")
    cal_stats(jsonl_filename, dataset_name=dataset_name)
    jsonl_2_json(jsonl_filename, json_filename)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_roots", nargs='+', required=True, help="One or more data root paths to merge. Each must contain a 'success/' subdirectory with episode_* folders.")
    parser.add_argument("--img_save_root", type=str, required=True, help="Directory where extracted frame images are saved.")
    parser.add_argument("--json_save_root", type=str, required=True, help="Directory where JSONL/JSON output files are saved.")
    parser.add_argument("--task_name", type=str, required=True, help="Task name subfolder used inside img_save_root.")
    parser.add_argument("--json_name_base", type=str, required=True, help="Base name for the output .jsonl and .json files.")
    parser.add_argument("--num_trajectories", type=int, default=0, help="Number of trajectories to randomly select for few-shot data. Omit or set to 0 to use all available episodes.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for trajectory selection (default: 42).")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of parallel worker processes (default: auto).")
    # ── format knobs (override the module defaults; workers inherit via fork) ──
    parser.add_argument("--action_chunk", type=int, default=ACTION_CHUNK, help="Action chunk length.")
    parser.add_argument("--frame_stride", type=int, default=FRAME_STRIDE, help="Frame stride between chunk steps.")
    parser.add_argument("--instruction", type=str, default=DEFAULT_INSTRUCTION, help="Language instruction written into every sample.")
    parser.add_argument("--crop_box", type=int, nargs=4, default=list(CROP_BOX_SLOW),
                        metavar=("Y0", "Y1", "X0", "X1"), help="Head-cam crop (y_min y_max x_min x_max).")
    parser.add_argument("--no_crop", action="store_true", help="Disable head-cam cropping.")
    parser.add_argument("--dataset_name", type=str, default="rlbench", help="Top-level key in the _statistics.json (loader reads the first key).")
    args = parser.parse_args()

    # Apply overrides to the module globals so the fork-spawned workers (which
    # read these as module-level constants) pick them up.
    ACTION_CHUNK = args.action_chunk
    FRAME_STRIDE = args.frame_stride
    DEFAULT_INSTRUCTION = args.instruction
    CROP_BOX_SLOW = None if args.no_crop else tuple(args.crop_box)

    data_roots = args.data_roots
    num_traj = args.num_trajectories if args.num_trajectories and args.num_trajectories > 0 else None

    print(f"Data roots ({len(data_roots)}):")
    for r in data_roots:
        print(f"  {r}")
    print(f"action_chunk={ACTION_CHUNK} frame_stride={FRAME_STRIDE} "
          f"crop={CROP_BOX_SLOW} dataset_name={args.dataset_name}")

    process_dataset(data_roots, args.img_save_root, args.json_save_root,
                    args.json_name_base, args.task_name,
                    num_workers=args.num_workers,
                    num_trajectories=num_traj,
                    seed=args.seed,
                    dataset_name=args.dataset_name)


