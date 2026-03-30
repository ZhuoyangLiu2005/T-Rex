import os
import re
import cv2
import h5py
import json
import numpy as np
import sys
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

ACTION_CHUNK = 16
FRAME_STRIDE = 2

IMAGE_VIEWS_SLOW = 'image_primary'
IMAGE_VIEWS_FAST_R = 'image_wrist_right'
IMAGE_VIEWS_FAST_L = 'image_wrist_left'

VIDEO_SLOW_SUFFIX = 'head_left_rgb.mp4'
VIDEO_FAST_R_SUFFIX = 'right_wrist.mp4'
VIDEO_FAST_L_SUFFIX = 'left_wrist.mp4'

DEFAULT_INSTRUCTION = "Pick up the AirPods case from the desk with the right hand and handover it to the left; then, carefully open its lid using the right thumb. Next, using the right index finger and thumb in succession, remove the two wireless earbuds and place them on the desk. Subsequently, close the lid of the case with the right hand finger, and finally, place the case itself onto the desk using the left hand."

# Bimanual: Right Arm(9D) + Right Hand(22D) + Left Arm(9D) + Left Hand(22D) = 62D
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
    """Compute tracking error for both arms. Returns 56D (28D right + 28D left)."""
    right_err = compute_tracking_error_axis_angle(state_62d[:31], target_62d[:31])
    left_err = compute_tracking_error_axis_angle(state_62d[31:], target_62d[31:])
    return np.concatenate([right_err, left_err])


def extract_frames_from_video(video_path, save_dir, view_name):
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
    episode_dir, save_dir, task_name = args
    episode_name = os.path.basename(episode_dir)
    print(f"processing: {episode_name} ...", end=" ", flush=True)

    h5_path = os.path.join(episode_dir, f"{episode_name}.h5")
    video_slow_path = os.path.join(episode_dir, f"{episode_name}_{VIDEO_SLOW_SUFFIX}")
    video_fast_r_path = os.path.join(episode_dir, f"{episode_name}_{VIDEO_FAST_R_SUFFIX}")
    video_fast_l_path = os.path.join(episode_dir, f"{episode_name}_{VIDEO_FAST_L_SUFFIX}")

    if not os.path.exists(h5_path):
        print(f"skip: H5 file not found")
        return episode_name, []

    episode_img_save_dir = os.path.join(save_dir, task_name, episode_name)
    os.makedirs(episode_img_save_dir, exist_ok=True)

    # Extract frames from all three views (no cropping)
    slow_img_paths = extract_frames_from_video(video_slow_path, episode_img_save_dir, IMAGE_VIEWS_SLOW)
    fast_r_img_paths = extract_frames_from_video(video_fast_r_path, episode_img_save_dir, IMAGE_VIEWS_FAST_R)
    fast_l_img_paths = extract_frames_from_video(video_fast_l_path, episode_img_save_dir, IMAGE_VIEWS_FAST_L)

    with h5py.File(h5_path, 'r') as f:
        # ----- Right arm state: current pose (Absolute 9D) -----
        s_r_arm_pose = f['right_arm_current_pose'][:]       # (N, 4, 4)
        s_r_arm_9d = pose_matrix_to_9d(s_r_arm_pose)        # (N, 9)
        s_r_hnd = f['right_hand_joint_positions'][:]         # (N, 22)

        # ----- Left arm state: current pose (Absolute 9D) -----
        s_l_arm_pose = f['left_arm_current_pose'][:]         # (N, 4, 4)
        s_l_arm_9d = pose_matrix_to_9d(s_l_arm_pose)        # (N, 9)
        s_l_hnd = f['left_hand_joint_positions'][:]          # (N, 22)

        # State: right(31D) + left(31D) = 62D
        states = np.concatenate([s_r_arm_9d, s_r_hnd, s_l_arm_9d, s_l_hnd], axis=1)

        # ----- Right arm target: target pose (for delta actions) -----
        a_r_arm_pose = f['right_arm_target_pose'][:]         # (N, 4, 4)
        a_r_arm_9d = pose_matrix_to_9d(a_r_arm_pose)        # (N, 9)
        a_r_hnd = f['right_hand_target_joint_positions'][:]  # (N, 22)

        # ----- Left arm target: target pose (for delta actions) -----
        a_l_arm_pose = f['left_arm_target_pose'][:]          # (N, 4, 4)
        a_l_arm_9d = pose_matrix_to_9d(a_l_arm_pose)        # (N, 9)
        a_l_hnd = f['left_hand_target_joint_positions'][:]   # (N, 22)

        # Absolute targets: right(31D) + left(31D) = 62D
        absolute_targets = np.concatenate([a_r_arm_9d, a_r_hnd, a_l_arm_9d, a_l_hnd], axis=1)

        # ----- Tactile -----
        t_r_f6 = f['right_hand_tactile_f6'][:]               # (N, 5, 6)
        t_l_f6 = f['left_hand_tactile_f6'][:]                # (N, 5, 6)
        t_r_deform = f['right_hand_tactile_deform'][:]       # (N, 5, H, W)
        t_l_deform = f['left_hand_tactile_deform'][:]        # (N, 5, H, W)

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
            img_arr = t_r_deform[i, finger_idx]
            path = os.path.join(episode_img_save_dir, f'image{i}_tactile_right_deform_{finger_idx}.png')
            if not os.path.exists(path):
                cv2.imwrite(path, img_arr)
            tactile_img_paths_current.append(path)

        for finger_idx in range(5):
            img_arr = t_l_deform[i, finger_idx]
            path = os.path.join(episode_img_save_dir, f'image{i}_tactile_left_deform_{finger_idx}.png')
            if not os.path.exists(path):
                cv2.imwrite(path, img_arr)
            tactile_img_paths_current.append(path)

        # ----- Action chunk: delta-base for both arms -----
        chunk_base_r_arm_pose = s_r_arm_pose[i]
        chunk_base_l_arm_pose = s_l_arm_pose[i]

        action_chunk_list = []
        for k in range(ACTION_CHUNK):
            base_future_idx = min(i + k * FRAME_STRIDE, min_len - 1)
            actual_future_idx = min(base_future_idx + FRAME_STRIDE - 1, min_len - 1)

            # Right arm delta
            target_r_arm_pose = a_r_arm_pose[actual_future_idx]
            target_r_hand = a_r_hnd[actual_future_idx]
            delta_r_9d = compute_chunk_delta_pose(chunk_base_r_arm_pose, target_r_arm_pose)

            # Left arm delta
            target_l_arm_pose = a_l_arm_pose[actual_future_idx]
            target_l_hand = a_l_hnd[actual_future_idx]
            delta_l_9d = compute_chunk_delta_pose(chunk_base_l_arm_pose, target_l_arm_pose)

            # 62D: right(9+22) + left(9+22)
            act = np.concatenate([delta_r_9d, target_r_hand, delta_l_9d, target_l_hand]).tolist()
            action_chunk_list.append(act)

        current_state = states[i].tolist()

        # Tactile f6: concatenate right(5,6) + left(5,6) -> flatten to 60D
        tactile_f6_combined = np.concatenate([t_r_f6[i], t_l_f6[i]], axis=0).tolist()

        target_idx_next = min(i + FRAME_STRIDE - 1, min_len - 1)
        current_abs_action = absolute_targets[target_idx_next].tolist()

        episode_data = {
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

def cal_stats(jsonl_filename):
    actions = []
    states = []
    tactiles = []
    episode_numbers = set()
    tracking_errors = []

    with open(jsonl_filename, 'r') as f:
        current_ep_data = []
        current_ep_num = -1

        for line in f:
            data = json.loads(line)
            match = re.search(r'episode_?(\d+)', data['image_old_slow'])
            ep_num = int(match.group(1)) if match else -1
            episode_numbers.add(ep_num)

            if ep_num != current_ep_num and len(current_ep_data) > 0:
                for t in range(1, len(current_ep_data)):
                    state_t = np.array(current_ep_data[t]['state'])
                    action_t_minus_1 = np.array(current_ep_data[t-1]['absolute_target_action'])
                    err = compute_bimanual_tracking_error(state_t, action_t_minus_1)
                    tracking_errors.append(err)
                current_ep_data = []

            current_ep_num = ep_num
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

    # Tracking error: 28D right + 28D left = 56D
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
        "rlbench": {
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


def process_dataset(data_root, img_save_root, json_save_root, json_name_base, task_name,
                    num_workers=None):
    if num_workers is None:
        num_workers = min(multiprocessing.cpu_count(), 32)

    os.makedirs(img_save_root, exist_ok=True)
    os.makedirs(json_save_root, exist_ok=True)

    jsonl_filename = os.path.join(json_save_root, f'{json_name_base}.jsonl')
    json_filename = os.path.join(json_save_root, f'{json_name_base}.json')
    episodes_root = os.path.join(data_root, 'success')

    # Collect all episode dirs in sorted order so the JSONL output is deterministic
    episode_dirs = sorted([
        os.path.join(episodes_root, item)
        for item in sorted(os.listdir(episodes_root))
        if os.path.isdir(os.path.join(episodes_root, item)) and item.startswith("episode_")
    ])

    print(f"Found {len(episode_dirs)} episodes. Launching {num_workers} workers...")

    # Map each episode_dir to its sorted index so we can reassemble in order
    args_list = [(ep_dir, img_save_root, task_name) for ep_dir in episode_dirs]
    index_map = {os.path.basename(ep_dir): idx for idx, ep_dir in enumerate(episode_dirs)}

    # results[i] will hold the list of records for episode_dirs[i]
    results = [None] * len(episode_dirs)

    executor = ProcessPoolExecutor(max_workers=num_workers)
    try:
        future_to_idx = {
            executor.submit(process_single_episode_worker, args): index_map[os.path.basename(args[0])]
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

    # Write JSONL in sorted episode order
    with open(jsonl_filename, 'w') as f_jsonl:
        for records in results:
            if records:
                for episode_data in records:
                    f_jsonl.write(json.dumps(episode_data) + '\n')

    print(f"JSONL written to: {jsonl_filename}")
    cal_stats(jsonl_filename)
    jsonl_2_json(jsonl_filename, json_filename)


if __name__ == "__main__":
    DATA_ROOT = "/mnt/amlfs-02/shared/human_egocentric/dniu/datasets/bkl_inlab/raw/task_data/open_airpods_03-25-2026_100"
    IMG_SAVE_ROOT = "/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/data/bkl_inlab/training_data/three_dense_fastslow_full"
    JSON_SAVE_ROOT = "/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/data/bkl_inlab/training_data/three_full_json"
    TASK_NAME = "open_airpods_0325_bimanual_stride2"
    JSON_NAME_BASE = "open_airpods_0325_deltabase_axis_eef_bimanual_stride2_train"

    # Tune num_workers to fit your machine. None = auto (up to 32).
    NUM_WORKERS = None

    process_dataset(DATA_ROOT, IMG_SAVE_ROOT, JSON_SAVE_ROOT, JSON_NAME_BASE, TASK_NAME,
                    num_workers=NUM_WORKERS)
