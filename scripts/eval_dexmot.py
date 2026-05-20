import sys
import os
# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import threading
import time
from pathlib import Path
from collections import deque
from typing import Optional
import pickle

import cv2
import numpy as np
import torch
import tyro
from loguru import logger
from loop_rate_limiters import RateLimiter
import yaml
from PIL import Image
import zmq

import select
import tty
import termios

# Robot and hand imports
from dexcontrol.robot import Robot
from sharpa import SharpaWave, SharpaWaveManager, ControlMode, ControlSource, HandSide

# Wrist camera imports
from camera.wrist_camera_receiver import WristCameraReceiver
from camera.head_camera_receiver import HeadCameraReceiver 

# IK and robot utilities
from teleop.ik_utils import PinkLocalIK
from teleop.robot_descriptions import build_full_robot, add_env_obstacles, DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES
from teleop.arm_hand_control import (
    ArmIKManager,
    SmoothingAndSafetyManager,
    InitializationCollisionPlanner,
    full_robot_action_loop,
    move_robot_to_position_safe,
)
import pinocchio as pin

# Configuration
COMMAND_HZ = 30.0
ARM_ACTION_HZ = 300.0  # High-frequency thread for sending arm commands
IMAGE_HEIGHT = 360
IMAGE_WIDTH = 640
RESET_DOF_ERR_TOL = 0.2  # Rad

# Hand serial numbers (must match your hardware)
LEFT_HAND_SERIAL = 'C5549736C554'
RIGHT_HAND_SERIAL = 'C1519737C151'
HAND_JOINT_COUNT = 22
HAND_INTERPOLATE = True

# Default joint positions for initialization
LEFT_ARM_DEFAULT_JOINT_POS = [0.84, 0.51, 0.37, -1.30, -0.65, -0.29, -0.03]
RIGHT_ARM_DEFAULT_JOINT_POS = [-0.84, -0.51, -0.37, -1.30, 0.65, 0.29, 0.03]
TORSO_DEFAULT_JOINT_POS = [0.9, 1.57, 0.1]
HEAD_DEFAULT_JOINT_POS = [0.28, 0., 0.]
DEFAULT_JOINT_POS = {
    "left_arm": np.array(LEFT_ARM_DEFAULT_JOINT_POS),
    "right_arm": np.array(RIGHT_ARM_DEFAULT_JOINT_POS),
    "head": np.array(HEAD_DEFAULT_JOINT_POS),
    "torso": np.array(TORSO_DEFAULT_JOINT_POS),
}

RIGHT_ARM_DEFAULT_JOINT_POS_SAFE = [-0.84, -0.51, -0.37, -1.60, 0.65, 0.29, 0.03]

# 16 Benchmark Grid Points (X, Y)
EVAL_GRID_POINTS = [
    [0.499607, -0.060863], [0.499607, -0.155273], [0.499607, -0.249683], [0.499607, -0.344092],
    [0.413190, -0.060863], [0.413190, -0.155273], [0.413190, -0.249683], [0.413190, -0.344092],
    [0.326774, -0.060863], [0.326774, -0.155273], [0.326774, -0.249683], [0.326774, -0.344092],
    [0.240357, -0.060863], [0.240357, -0.155273], [0.240357, -0.249683], [0.240357, -0.344092]
]

# Wrist camera configuration
WRIST_SENDER_IP = "192.168.50.25"
WRIST_CAM_PORTS = {
    "LEFT_WRIST": 5555,
    "RIGHT_WRIST": 5556,
}

HEAD_SENDER_IP = "192.168.50.20"
HEAD_CAM_PORTS = {
    "HEAD_CAM": 5555,
}


# Tactile configuration
TACTILE_TYPES_TO_CAPTURE = ["F6", "DEFORM"]
TACTILE_BUFFER_SHAPES = {
    "f6": ((5, 6), np.float32),
    "deform": ((5, 240, 240), np.uint8),
}
TACTILE_FETCH_HZ = 30.0

# Length of the per-hand rolling F6 history sent to the server.  Must match
# the VQ-VAE training window so the server can run encode() directly without
# rebuilding its own buffer.  Sampled at TACTILE_FETCH_HZ regardless of
# inference cadence — closes the train/test temporal-density mismatch.
F6_HISTORY_LEN = 16

CHUNK_SIZE = 16
MAX_STEPS = 10000
INFERENCE_HZ = 5
EXECUTE_FIRST_FEW_IN_CHUNK = 16  # Number of steps to execute from each predicted chunk before fetching a new action chunk

# ── Async tactile refinement (cascaded slow/fast) ──
# When True, the slow VLA call sends mode='slow_and_fast' so the server caches
# its (latent + action) KV state at τ_split.  Then at each REFINE_OFFSETS
# within the chunk we send a small mode='fast' payload (tactile only, no
# images) and the server replies with a refined chunk that we splice onto
# the remaining waypoints.  Set False for single chunk-rate inference.
USE_TACTILE_REFINE = True
REFINE_OFFSETS = (4, 8, 12)      # within-chunk action_idx values at which to refine

# ── ACT-style temporal aggregation ──
# When True, instead of splicing refined[action_idx:] into the in-flight chunk,
# we keep every chunk we receive (slow + each fast) in a small buffer and, for
# every robot step, exponentially-weight-average all chunk predictions that
# cover that step.  This smooths the boundary between successive predictions
# and cancels independent random noise across chunks (helps with residual
# jitter pollution at inference).  Reference: ACT, imitate_episodes.py:218.
USE_TEMPORAL_AGGREGATION = True
TEMPORAL_AGG_K = 0.1             # exp decay; 0 = uniform avg, larger = newer dominates


def aggregate_chunks(chunk_buffer, current_global_step, k=TEMPORAL_AGG_K):
    """Exp-weighted average of all chunk predictions covering current_global_step.

    chunk_buffer : list of (start_step, np.ndarray[CHUNK, action_dim]) tuples,
                   ordered oldest → newest by append time.
    Newest entries get the highest weight; weight decays as exp(-k * age).
    Returns a 1-D action vector or None if no chunk covers this step.
    """
    preds = []
    for start, chunk in chunk_buffer:
        rel = current_global_step - start
        if 0 <= rel < len(chunk):
            preds.append(chunk[rel])
    if not preds:
        return None
    if len(preds) == 1:
        return preds[0]
    preds = np.stack(preds)                                  # [N, action_dim]
    # Newest at the end of `preds`; reverse arange so newest gets highest weight.
    weights = np.exp(-k * np.arange(len(preds))[::-1])
    weights = weights / weights.sum()
    return (preds * weights[:, None]).sum(axis=0)


# CROP_BOX_SLOW = (100, 280, 250, 510)
# CROP_BOX_SLOW = (20, 280, 150, 510) # for airpods and new flip book
# CROP_BOX_SLOW = (90, 280, 260, 550) # zhuoyang, spetial test
CROP_BOX_SLOW = (0, 300, 140, 540) # zhuoyang, for dual arm

DUAL_ARM = True  # False means only right arm

# =============================================================================
# Background Controller Class
# =============================================================================
class KeyController:
    """Listens to keyboard inputs in the background via stdin (works on Wayland/SSH).

    Call suspend() before any input() call and resume() after, so that the
    listener does not race with normal line-buffered stdin reads.
    """
    def __init__(self):
        self.paused = False
        self.reset_requested = False
        self.quit_requested = False
        self._stop = threading.Event()
        self._suspended = threading.Event()   # set = listener should sleep
        self._ack_suspend = threading.Event()  # set = listener has released stdin
        self._old_settings = None
        self._thread = threading.Thread(target=self._listen, daemon=True)
        self._thread.start()

    def _listen(self):
        if not sys.stdin.isatty():
            logger.warning("stdin is not a TTY, keyboard hotkeys disabled.")
            return
        self._old_settings = termios.tcgetattr(sys.stdin)
        try:
            while not self._stop.is_set():
                # If suspended, restore terminal and wait
                if self._suspended.is_set():
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)
                    self._ack_suspend.set()
                    while self._suspended.is_set() and not self._stop.is_set():
                        time.sleep(0.05)
                    if self._stop.is_set():
                        break
                    tty.setcbreak(sys.stdin.fileno())

                tty.setcbreak(sys.stdin.fileno())
                ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                if ready:
                    ch = sys.stdin.read(1)
                    if ch == 'p':
                        self.paused = not self.paused
                        state_str = "PAUSED" if self.paused else "RESUMED"
                        logger.info(f"\n[HOTKEY] Trajectory {state_str}")
                    elif ch == 'r':
                        self.reset_requested = True
                        logger.info("\n[HOTKEY] RESET requested. Aborting trajectory...")
                    elif ch == 'q':
                        self.quit_requested = True
                        logger.info("\n[HOTKEY] QUIT requested.")
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)

    def suspend(self):
        """Pause listening and restore normal terminal mode for input() calls."""
        self._ack_suspend.clear()
        self._suspended.set()
        self._ack_suspend.wait(timeout=1)

    def resume(self):
        """Re-enable cbreak listening after input() is done."""
        self._suspended.clear()

    def stop(self):
        self._stop.set()
        self._suspended.clear()  # unblock if suspended
        self._thread.join(timeout=1)

FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

def visualize_inference_inputs(
    step: int,
    head_img: np.ndarray,
    right_wrist_img: np.ndarray,
    left_wrist_img: Optional[np.ndarray],
    tactile_deform: np.ndarray,
    tactile_f6: np.ndarray,
    dual_arm: bool,
):
    """Show a live OpenCV dashboard of all inference inputs.

    Layout (top row = cameras, bottom row = tactile deformation per finger):
      ┌──────────┬──────────────┬─────────────┐
      │   Head   │  Right Wrist │  Left Wrist  │
      ├────┬────┬┴───┬────┬─────┤              │
      │ T  │ I  │ M  │ R  │ P   │  (right)     │
      ├────┼────┼────┼────┼─────┤              │
      │ T  │ I  │ M  │ R  │ P   │  (left)      │
      └────┴────┴────┴────┴─────┘              │
    """
    cam_h = 240
    deform_thumb_size = 120  # display size per finger deform image

    # --- Resize camera images to uniform height ---
    def resize_to_height(img, h):
        scale = h / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * scale), h))

    head_bgr = cv2.cvtColor(resize_to_height(head_img, cam_h), cv2.COLOR_RGB2BGR)
    rw_bgr = cv2.cvtColor(resize_to_height(right_wrist_img, cam_h), cv2.COLOR_RGB2BGR)

    cam_row_parts = [head_bgr, rw_bgr]
    if left_wrist_img is not None:
        lw_bgr = cv2.cvtColor(resize_to_height(left_wrist_img, cam_h), cv2.COLOR_RGB2BGR)
        cam_row_parts.append(lw_bgr)
    cam_row = np.hstack(cam_row_parts)

    # --- Build tactile deform rows ---
    n_fingers = tactile_deform.shape[0]
    right_count = 5
    left_count = n_fingers - 5 if dual_arm else 0

    def build_finger_strip(deform_slice, f6_slice):
        panels = []
        for i in range(deform_slice.shape[0]):
            d = deform_slice[i]  # (240, 240) uint8
            d_resized = cv2.resize(d, (deform_thumb_size, deform_thumb_size))
            d_color = cv2.applyColorMap(d_resized, cv2.COLORMAP_JET)
            # Label with finger name and f6 norm
            f6_norm = np.linalg.norm(f6_slice[i])
            label = f"{FINGER_NAMES[i]} {f6_norm:.1f}"
            cv2.putText(d_color, label, (2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            panels.append(d_color)
        return np.hstack(panels)

    tactile_rows = []
    right_strip = build_finger_strip(tactile_deform[:right_count], tactile_f6[:right_count])
    # Add "R" label on the left
    tag_r = np.zeros((deform_thumb_size, 30, 3), dtype=np.uint8)
    cv2.putText(tag_r, "R", (4, deform_thumb_size // 2 + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    tactile_rows.append(np.hstack([tag_r, right_strip]))

    if dual_arm and left_count > 0:
        left_strip = build_finger_strip(tactile_deform[right_count:], tactile_f6[right_count:])
        tag_l = np.zeros((deform_thumb_size, 30, 3), dtype=np.uint8)
        cv2.putText(tag_l, "L", (4, deform_thumb_size // 2 + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        tactile_rows.append(np.hstack([tag_l, left_strip]))

    tactile_block = np.vstack(tactile_rows)

    # --- Combine camera row and tactile block, padding to same width ---
    cam_w = cam_row.shape[1]
    tac_w = tactile_block.shape[1]
    target_w = max(cam_w, tac_w)

    def pad_to_width(img, w):
        if img.shape[1] < w:
            pad = np.zeros((img.shape[0], w - img.shape[1], 3), dtype=np.uint8)
            return np.hstack([img, pad])
        return img[:, :w]

    cam_row = pad_to_width(cam_row, target_w)
    tactile_block = pad_to_width(tactile_block, target_w)

    # Step label bar
    label_bar = np.zeros((24, target_w, 3), dtype=np.uint8)
    cv2.putText(label_bar, f"Step {step}", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    dashboard = np.vstack([label_bar, cam_row, tactile_block])

    cv2.imshow("Eval Inputs", dashboard)
    cv2.waitKey(1)


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    v1 = rot6d[0:3]
    v2 = rot6d[3:6]

    e1 = v1 / np.linalg.norm(v1)
    u2 = v2 - np.dot(e1, v2) * e1
    e2 = u2 / np.linalg.norm(u2)
    e3 = np.cross(e1, e2)

    return np.column_stack((e1, e2, e3))

def matrix_to_rot6d(matrix: np.ndarray) -> np.ndarray:
    return np.concatenate([matrix[:, 0], matrix[:, 1]])

# =============================================================================
# Hand Connection and Initialization
# =============================================================================

def connect_hands():
    """Connect to left and right hands via SharpaWaveSDK."""
    manager = SharpaWaveManager.get_instance()
    time.sleep(1)  # Wait for device discovery

    devices = manager.get_all_device_sn()
    logger.info(f"Available hand devices: {devices}")

    if not devices:
        raise RuntimeError("No hand devices found!")

    if LEFT_HAND_SERIAL not in devices:
        raise RuntimeError(f"Left hand serial {LEFT_HAND_SERIAL} not found in devices: {devices}")

    if RIGHT_HAND_SERIAL not in devices:
        raise RuntimeError(f"Right hand serial {RIGHT_HAND_SERIAL} not found in devices: {devices}")

    left_hand = manager.connect(LEFT_HAND_SERIAL)
    logger.info(f"Connected left hand: {LEFT_HAND_SERIAL}")

    right_hand = manager.connect(RIGHT_HAND_SERIAL)
    logger.info(f"Connected right hand: {RIGHT_HAND_SERIAL}")

    return left_hand, right_hand


def initialize_hand(hand: SharpaWave):
    """Initialize hand with proper control settings. Raises on failure."""
    error = hand.set_control_mode(ControlMode.POSITION)
    if error.code != 0:
        raise RuntimeError(f"Failed to set control mode: {error.message}")

    error = hand.set_speed_coeff(0.3)
    if error.code != 0:
        raise RuntimeError(f"Failed to set speed coeff: {error.message}")

    error = hand.set_current_coeff(0.6)
    if error.code != 0:
        raise RuntimeError(f"Failed to set current coeff: {error.message}")

    error = hand.set_control_source(ControlSource.SDK)
    if error.code != 0:
        raise RuntimeError(f"Failed to set control source: {error.message}")


# TODO (Zekai): Flip the order of tactile channel to match main_teleop.py
def _fetch_tactile_f6_arrays(left_hand: SharpaWave, right_hand: SharpaWave, hardware_lock: threading.Lock) -> tuple:
    """
    Fetch tactile F6 data from both hands as separate arrays.

    Returns:
        Tuple of (left_f6, right_f6) where each is (5, 6) array
    """
    raise NotImplementedError  # NOTE(Zekai): Check the order of tactile channel...
    left_f6 = np.zeros((5, 6), dtype=np.float32)
    right_f6 = np.zeros((5, 6), dtype=np.float32)

    for ch in range(5):
        try:
            with hardware_lock:
                left_ret = left_hand.fetch_tactile_frame(ch+5, timeout=0.01)
            if left_ret is not None and left_ret["content"] is not None:
                left_f6[ch] = left_ret["content"]["F6"]
        except Exception:
            pass

        try:
            with hardware_lock:
                right_ret = right_hand.fetch_tactile_frame(ch, timeout=0.01)
            if right_ret is not None and right_ret["content"] is not None:
                right_f6[ch] = right_ret["content"]["F6"]
        except Exception:
            pass

    return left_f6, right_f6

def _fetch_tactile_arrays(wave, hand_side, tactile_types):
    """Fetch tactile data for all 5 fingers of a hand as numpy arrays.
    
    Args:
        wave: SharpaWave object for the hand
        hand_side: HandSide.LEFT or HandSide.RIGHT
        tactile_types: List of types to fetch, e.g. ["F6", "DEFORM", "RAW"]
    
    Returns:
        dict with keys "f6", "deform", "raw" containing numpy arrays (or None)
    """
    # Channel mapping: Left hand: Thumb=9..Pinky=5, Right hand: Thumb=4..Pinky=0
    if hand_side == HandSide.LEFT:
        channels = [9, 8, 7, 6, 5]  # Thumb, Index, Middle, Ring, Pinky
    else:
        channels = [4, 3, 2, 1, 0]  # Thumb, Index, Middle, Ring, Pinky
    
    # Pre-allocate numpy arrays
    f6_arr = np.zeros((5, 6), dtype=np.float32) if "F6" in tactile_types else None
    deform_arr = np.zeros((5, 240, 240), dtype=np.uint8) if "DEFORM" in tactile_types else None
    
    for i, ch in enumerate(channels):
        ret = wave.fetch_tactile_frame(ch, timeout=0.01)
        
        if ret is None:
            continue  # Arrays already zeroed
        
        content = ret["content"]
        
        if f6_arr is not None:
            f6 = content.get("F6")
            if f6 is not None:
                f6_arr[i, :] = np.asarray(f6, dtype=np.float32)[:6]
        
        if deform_arr is not None:
            deform = content.get("DEFORM")
            if deform is not None:
                arr = np.asarray(deform, dtype=np.uint8)
                if arr.size == 57600:
                    deform_arr[i, :, :] = arr.reshape(240, 240)
    
    return {
        "f6": f6_arr,
        "deform": deform_arr,
    }


def _tactile_fetch_loop(
    terminate_event: threading.Event,
    buf_lock: threading.Lock,
    hardware_lock: threading.Lock,
    left_tactile_buffers: dict,
    right_tactile_buffers: dict,
    left_hand: SharpaWave,
    right_hand: SharpaWave,
    fetch_hz: float = 30.0,
    left_f6_history: Optional[deque] = None,
    right_f6_history: Optional[deque] = None,
):
    """
    Separate thread to fetch tactile data without blocking main control loop.
    Matches main_teleop.py pattern for timing consistency.

    Runs at specified frequency and updates tactile buffers.  When the optional
    `left_f6_history` / `right_f6_history` deques are provided, also appends
    each fetched F6 frame to them so a dense rolling window can be sent to the
    VLA server (matches VQ-VAE training-time temporal density).
    """
    limiter = RateLimiter(frequency=fetch_hz, name="tactile_fetch_limiter", warn=False)
    logger.info(f"[Tactile] Fetching thread started at {fetch_hz} Hz")

    while not terminate_event.is_set():
        limiter.sleep()

        try:
            # Fetch tactile data from both hands
            left_tactile_data = _fetch_tactile_arrays(left_hand, HandSide.LEFT, TACTILE_TYPES_TO_CAPTURE)
            right_tactile_data = _fetch_tactile_arrays(right_hand, HandSide.RIGHT, TACTILE_TYPES_TO_CAPTURE)

            # Update buffers with lock
            with buf_lock:
                for ttype_lower in ["f6", "deform"]:
                    if left_tactile_data[ttype_lower] is not None:
                        np.copyto(left_tactile_buffers[ttype_lower], left_tactile_data[ttype_lower])
                    if right_tactile_data[ttype_lower] is not None:
                        np.copyto(right_tactile_buffers[ttype_lower], right_tactile_data[ttype_lower])

                # Rolling F6 history for the VQ-VAE window.  Copy() so the
                # consumer sees the values at fetch time, not later overwrites.
                if (left_f6_history is not None
                        and left_tactile_data["f6"] is not None):
                    left_f6_history.append(left_tactile_data["f6"].copy())
                if (right_f6_history is not None
                        and right_tactile_data["f6"] is not None):
                    right_f6_history.append(right_tactile_data["f6"].copy())
        except Exception as e:
            logger.warning(f"[Tactile] Fetch error: {e}")

    logger.info("[Tactile] Fetching thread stopped")


def _snapshot_f6_window(
    buf_lock: threading.Lock,
    left_f6_history: deque,
    right_f6_history: deque,
    dual_arm: bool,
) -> np.ndarray:
    """Snapshot the rolling F6 history into a `[F6_HISTORY_LEN, n_fingers, 6]`
    window left-edge-padded with the oldest available frame.

    Returns
    -------
    np.ndarray
        Dual-arm: shape `[F6_HISTORY_LEN, 10, 6]` (left ⊕ right fingers).
        Single-arm: shape `[F6_HISTORY_LEN, 5, 6]` (right hand only).
    """
    with buf_lock:
        lh = list(left_f6_history)
        rh = list(right_f6_history)

    def _pad(hist: list) -> np.ndarray:
        if not hist:
            return np.zeros((F6_HISTORY_LEN, 5, 6), dtype=np.float32)
        if len(hist) < F6_HISTORY_LEN:
            head = [hist[0]] * (F6_HISTORY_LEN - len(hist))
            hist = head + hist
        return np.stack(hist, axis=0).astype(np.float32, copy=False)

    l_arr = _pad(lh)                                   # [W, 5, 6]
    r_arr = _pad(rh)                                   # [W, 5, 6]
    if dual_arm:
        return np.concatenate([l_arr, r_arr], axis=1)  # [W, 10, 6]
    return r_arr                                       # [W, 5, 6]


# =============================================================================
# State Processing Functions
# =============================================================================

def get_current_proprio(
    dexmate_bimanual_robot: Robot,
    hardware_lock: threading.Lock,
    left_hand: SharpaWave,
    right_hand: SharpaWave,
) -> np.ndarray:
    """
    Get current robot proprioceptive state.

    Format matches the training data format from inference_sharpa_dexmate_no_dataloder.py:
    [pos(3), rot6d(6), hand(22)] for each arm = 31D per arm, 62D total

    Returns:
        (62,) array: [left_trans(3), left_rot6d(6), left_hand(22),
                      right_trans(3), right_rot6d(6), right_hand(22)]
    """
    # Get arm joint positions
    with hardware_lock:
        curr_joint_pos_dict = dexmate_bimanual_robot.get_joint_pos_dict(component=["left_arm", "right_arm"])
    left_arm_joint_positions = np.array([curr_joint_pos_dict[name] for name in DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES["left_arm"]])
    right_arm_joint_positions = np.array([curr_joint_pos_dict[name] for name in DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES["right_arm"]])


    # Get hand joint positions
    with hardware_lock:
        left_hand_joints = np.array(left_hand.get_states().angles, dtype=np.float64)  # (22,)
        right_hand_joints = np.array(right_hand.get_states().angles, dtype=np.float64)  # (22,)

    # Concatenate into proprio array
    if DUAL_ARM:
        proprio = np.concatenate([
            left_arm_joint_positions,   # (7,)
            left_hand_joints,   # (22,)
            right_arm_joint_positions,    # (7,)
            right_hand_joints,    # (22,)
        ])  # Total: 58
    else:
        proprio = np.concatenate([
            right_arm_joint_positions,    # (7,)
            right_hand_joints,    # (22,)
            # left_arm_joint_positions,   # (7,)
            # left_hand_joints,   # (22,)
        ])  # Total: 58

    return proprio


def get_current_pose(
    dexmate_bimanual_robot: Robot,
    ik_solver: PinkLocalIK,
    hardware_lock: threading.Lock,
    left_hand: SharpaWave,
    right_hand: SharpaWave,
) -> np.ndarray:
    with hardware_lock:
        curr_joint_pos_dict = dexmate_bimanual_robot.get_joint_pos_dict(component=["left_arm", "right_arm"])
        right_hand_joints = np.array(right_hand.get_states().angles, dtype=np.float64)  # (22,)
        if DUAL_ARM:
            left_hand_joints = np.array(left_hand.get_states().angles, dtype=np.float64)  # (22,)

    left_arm_joint_positions = np.array([curr_joint_pos_dict[name] for name in DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES["left_arm"]])
    right_arm_joint_positions = np.array([curr_joint_pos_dict[name] for name in DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES["right_arm"]])

    fk_res = ik_solver.fk(
        frames=["L_ee", "R_ee"],
        joint_pos_by_component={
            "left_arm": left_arm_joint_positions,
            "right_arm": right_arm_joint_positions,
        }
    )

    if DUAL_ARM:
        left_ee_pose = fk_res["L_ee"]
        right_ee_pose = fk_res["R_ee"]

        left_pos = left_ee_pose.translation
        left_rot6d = matrix_to_rot6d(left_ee_pose.rotation)
        right_pos = right_ee_pose.translation
        right_rot6d = matrix_to_rot6d(right_ee_pose.rotation)

        pose_62d = np.concatenate([
            left_pos,          # (3,)
            left_rot6d,        # (6,)
            left_hand_joints,  # (22,)
            right_pos,         # (3,)
            right_rot6d,       # (6,)
            right_hand_joints, # (22,)
        ])
        return pose_62d
    else:
        right_ee_pose = fk_res["R_ee"]

        right_pos = right_ee_pose.translation
        right_rot6d = matrix_to_rot6d(right_ee_pose.rotation)

        pose_31d = np.concatenate([
            right_pos,         # (3,)
            right_rot6d,       # (6,)
            right_hand_joints  # (22,)
        ])
        return pose_31d


def get_head_image(
    head_cam_receiver: HeadCameraReceiver, 
    target_size: tuple
) -> np.ndarray:
    # --- MODIFIED: Get image from the receiver ---
    head_frame = head_cam_receiver.get_data("HEAD_CAM")
    if head_frame is None or head_frame.image_rgb is None:
        raise RuntimeError("Failed to get head camera image from ZMQ receiver")
    
    head_image = head_frame.image_rgb

    if head_image.shape[0] != target_size[0] or head_image.shape[1] != target_size[1]:
        head_image = cv2.resize(head_image, (target_size[1], target_size[0]))
    
    assert head_image.shape == (360, 640, 3), f"Unexpected head image shape: {head_image.shape}"
    head_image = head_image[CROP_BOX_SLOW[0]:CROP_BOX_SLOW[1], CROP_BOX_SLOW[2]:CROP_BOX_SLOW[3], :]
    # assert head_image.shape == (180, 260, 3), f"Unexpected cropped head image shape: {head_image.shape}"

    return head_image


def get_all_camera_images(
    dexmate_bimanual_robot: Robot,
    hardware_lock: threading.Lock,
    wrist_cam_receiver: WristCameraReceiver,
    head_cam_receiver: HeadCameraReceiver, 
    target_size: tuple,
) -> list:
    images = {}
    # Left wrist
    left_wrist_frame = wrist_cam_receiver.get_data("LEFT_WRIST")
    left_wrist_img = left_wrist_frame.image_rgb
    if left_wrist_img.shape[0] != target_size[0] or left_wrist_img.shape[1] != target_size[1]:
        left_wrist_img = cv2.resize(left_wrist_img, (target_size[1], target_size[0]))
    images["left_wrist"] = left_wrist_img

    # Head
    images["head"] = get_head_image(head_cam_receiver, target_size) 

    # Right wrist
    right_wrist_frame = wrist_cam_receiver.get_data("RIGHT_WRIST")
    right_wrist_img = right_wrist_frame.image_rgb
    if right_wrist_img.shape[0] != target_size[0] or right_wrist_img.shape[1] != target_size[1]:
        right_wrist_img = cv2.resize(right_wrist_img, (target_size[1], target_size[0]))
    images["right_wrist"] = right_wrist_img

    return images


def test_grid_position(robot: Robot,
                       ik_solver: PinkLocalIK,
                       right_hand: SharpaWave,
                       grid_index: Optional[int],
                       left_hand: SharpaWave = None,
                       initialization_planner: InitializationCollisionPlanner = None,
                       action_buffer: dict = None,
                       action_buf_lock: threading.Lock = None,
                       hardware_lock: threading.Lock = None,
                       object_size: float = 0.05,
                       key_ctrl: 'KeyController' = None):
    """
    Move right arm to the specific grid point for evaluation setup, wait for user input,
    and return to the initial pose.
    """
    if grid_index is None:
        return
        
    if not (0 <= grid_index <= 15):
        logger.error(f"Invalid grid index {grid_index}. Must be between 0 and 15.")
        return

    logger.info(f"--- Setting up for Grid Position {grid_index} ---")

    joint_pos_dict = robot.get_joint_pos_dict(component=["left_arm", "right_arm"])
    left_q = np.array([joint_pos_dict[name] for name in DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES["left_arm"]])
    right_q = np.array([joint_pos_dict[name] for name in DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES["right_arm"]])

    fk_res = ik_solver.fk(
        frames=["L_ee", "R_ee"],
        joint_pos_by_component={
            "left_arm": left_q,
            "right_arm": right_q
        }
    )
    left_ee_pose = fk_res["L_ee"]  # pin.SE3
    right_ee_pose = fk_res["R_ee"] # pin.SE3

    hardcoded_hand_state = [
        0.16, 0.07, 0.601, -0.083, 0.003, 0.302, 0.024, 0.543, 0.714, 0.484, 
        0.016, 0.507, 0.794, 0.423, -0.002, 0.568, 0.697, 0.054, 0.403, 0.014, 0.563, 0.637   
    ]
    hardcoded_rotation = np.array([[-0.07041831, -0.04592435,  0.99645984],         
                                    [-0.03512447,  0.99843434,  0.04353315],         
                                    [-0.99689896, -0.03193459, -0.07192112]     
                                ])     
    
    target_xy = EVAL_GRID_POINTS[grid_index]
    target_right_ee_pose = right_ee_pose.copy()
    target_right_ee_pose.translation[0] = target_xy[0]
    target_right_ee_pose.translation[1] = target_xy[1]
    target_right_ee_pose.translation[2] = 1.25
    logger.info(f"Target Pose XY replaced with: {target_xy[0]:.6f}, {target_xy[1]:.6f}")
    target_right_ee_pose.rotation = hardcoded_rotation

    ik_res = ik_solver.solve_ik(
        ee_target_poses={
            "L_ee": left_ee_pose,    
            "R_ee": target_right_ee_pose 
        },
        arm_initial_joint_pos={
            "left_arm": left_q,
            "right_arm": right_q
        }
    )
    target_right_q = ik_res["right_arm"].copy()
    robot.set_joint_pos(
        joint_pos={"right_arm": target_right_q},
        relative=False,
        wait_time=3.0  
    )
    right_hand.set_joint_position(hardcoded_hand_state, HAND_INTERPOLATE)

    # MOVE down a bit in Z to make it easier for user to place object in the grid
    target_right_ee_pose.translation[2] =0.99

    ik_res = ik_solver.solve_ik(
        ee_target_poses={
            "L_ee": left_ee_pose,
            "R_ee": target_right_ee_pose
        },
        arm_initial_joint_pos={
            "left_arm": left_q,
            "right_arm": target_right_q
        }
    )
    target_right_q = ik_res["right_arm"].copy()

    base_seed = 42
    np.random.seed(base_seed + grid_index)  # Ensure reproducibility per grid point
    offset_min, offset_max = -0.3, 0.3
    random_offset = np.random.uniform(offset_min, offset_max)
    target_right_q[5] += random_offset
    print(f"IK solution for right arm joint positions to reach Grid {grid_index}:")
    print(target_right_q)
    if key_ctrl:
        key_ctrl.suspend()
    input("Press [Enter] to move the right arm to the target grid position...")
    if key_ctrl:
        key_ctrl.resume()

    logger.info("Moving right arm to target grid position...")
    robot.set_joint_pos(
        joint_pos={"right_arm": target_right_q},
        relative=False,
        wait_time=3.0  
    )
    right_hand.set_joint_position(hardcoded_hand_state, HAND_INTERPOLATE)
    if key_ctrl:
        key_ctrl.suspend()
    input(f"\n[!] Right arm reached Grid {grid_index}. Place your object and press [Enter] to return to initial pose...")
    if key_ctrl:
        key_ctrl.resume()

    # lift a little bit up in Z to make it easier for user to remove object from the grid after placement
    target_right_ee_pose.translation[2] = 1.15
    ik_res = ik_solver.solve_ik(
        ee_target_poses={
            "L_ee": left_ee_pose,
            "R_ee": target_right_ee_pose
        },
        arm_initial_joint_pos={
            "left_arm": left_q,
            "right_arm": target_right_q
        }
    )
    target_right_q = ik_res["right_arm"].copy()
    robot.set_joint_pos(
        joint_pos={"right_arm": target_right_q},
        relative=False,
        wait_time=2.0  
    )

    logger.info("Returning to initial right arm position...")
    if initialization_planner is not None:
        # Add a small obstacle at the grid position representing the placed object
        grid_xy = EVAL_GRID_POINTS[grid_index]
        obj_z = 0.85 + object_size / 2 + 0.01  # table surface + half object height + small gap to avoid table collision
        move_robot_to_position_safe(
            dexmate_bimanual_robot=robot,
            sharpa_left_hand=left_hand,
            sharpa_right_hand=right_hand,
            initialization_planner=initialization_planner,
            action_buffer=action_buffer,
            action_buf_lock=action_buf_lock,
            target_joint_pos={
                "left_arm": left_q,
                "right_arm": np.array(RIGHT_ARM_DEFAULT_JOINT_POS),
                "left_hand": np.zeros(HAND_JOINT_COUNT),
                "right_hand": np.zeros(HAND_JOINT_COUNT),
            },
            hardware_lock=hardware_lock,
            command_hz=COMMAND_HZ,
            dof_error_tolerance=RESET_DOF_ERR_TOL,
            hold_time_s=0.5,
            right_arm_collision_boxes=[{
                "name": "placed_object",
                "position": np.array([grid_xy[0], grid_xy[1], obj_z]),
                "full_extents": np.array([object_size, object_size, object_size]),
            }],
        )
    else:
        right_hand.set_joint_position([0.0] * HAND_JOINT_COUNT, HAND_INTERPOLATE)
        robot.set_joint_pos(
            joint_pos={"right_arm": right_q},
            relative=False,
            wait_time=3.0
        )
    logger.info("--- Grid Setup Complete ---\n")



# =============================================================================
# Main Evaluation Loop
# =============================================================================

def main():
    # Initialize robot with head camera
    logger.info("Initializing robot...")
    dexmate_bimanual_robot = Robot()
    logger.info(f"Robot '{dexmate_bimanual_robot.robot_model}' initialized")
    
    # Initialize head camera via ZMQ receiver
    logger.info("Initializing head camera...")
    head_cam_receiver = HeadCameraReceiver(
        im_h=IMAGE_HEIGHT,
        im_w=IMAGE_WIDTH,
        sender_ip=HEAD_SENDER_IP,
        ports=HEAD_CAM_PORTS,
    )
    head_cam_receiver.start_receiving(timeout=10.0)
    logger.info("Head camera started receiving")

    # Initialize wrist cameras
    logger.info("Initializing wrist cameras...")
    wrist_cam_receiver = WristCameraReceiver(
        im_h=IMAGE_HEIGHT,
        im_w=IMAGE_WIDTH,
        sender_ip=WRIST_SENDER_IP,
        ports=WRIST_CAM_PORTS,
    )
    wrist_cam_receiver.start_receiving(timeout=10.0)
    logger.info("Wrist cameras started receiving")

    # Initialize IK solver
    pink_ik_solver = PinkLocalIK(default_joint_by_component=DEFAULT_JOINT_POS)
    logger.info("IK solver initialized")

    # Initialize ArmIKManager and SmoothingAndSafetyManager for processing arm targets and sending commands
    # arm_ik_manager = ArmIKManager(
    #     pink_ik_solver=pink_ik_solver,
    #     warmstart_with_actual=False,
    # )

    # # TMP
    # right_arm_joint_positions = np.array(RIGHT_ARM_DEFAULT_JOINT_POS)
    # right_arm_joint_positions = np.array([-1.15359974, -1.0352211,  -0.57615066, -1.76642501, -0.5535748,  -0.43322912,  -0.47088459])
    # curr_fk_res = pink_ik_solver.fk(
    #     frames=["L_ee", "R_ee"],
    #     joint_pos_by_component={
    #         "left_arm": right_arm_joint_positions,
    #         "right_arm": right_arm_joint_positions,
    #     }
    # )
    # right_arm_current_pose = curr_fk_res["R_ee"].homogeneous
    # logger.info(f"Current right arm end-effector pose:\n{right_arm_current_pose}")
    # input("Press [Enter] to continue with initializing SmoothingAndSafetyManager...")

    # Initialize SmoothingAndSafetyManager for processing arm targets and sending commands
    pin_full_robot_wrapper, assemble_qpos, disassemble_qpos = build_full_robot(default_joint_by_component=DEFAULT_JOINT_POS)
    # TODO: Might need to change this when we move to a new room. 
    pin_full_robot_wrapper = add_env_obstacles(
        robot=pin_full_robot_wrapper,
        default_joint_by_component={
            "left_arm": np.array(LEFT_ARM_DEFAULT_JOINT_POS),
            "right_arm": np.array(RIGHT_ARM_DEFAULT_JOINT_POS),
            "left_hand": np.zeros((HAND_JOINT_COUNT,)),
            "right_hand": np.zeros((HAND_JOINT_COUNT,)),
        },
        assemble_qpos=assemble_qpos,
        back_wall_distance=0.60,  # Measured
        left_wall_distance=None,  # Not present in the current environment
        right_wall_distance=0.75,  # Measured
        # TODO(zekai): Maybe check this before running.
        table_height=0.76,
    )
    smoothing_and_safety_manager = SmoothingAndSafetyManager(
        pin_full_robot_wrapper=pin_full_robot_wrapper,
        assemble_qpos=assemble_qpos,
        disassemble_qpos=disassemble_qpos,
        ruckig_smoothing=False,
        action_hz=ARM_ACTION_HZ
    )
    logger.info("SmoothingAndSafetyManager initialized")

    # Initialize hands
    logger.info("Connecting to hands...")
    left_hand, right_hand = connect_hands()

    logger.info("Initializing hands...")
    initialize_hand(left_hand)
    left_hand.start()
    logger.info("Left hand initialized and started")

    initialize_hand(right_hand)
    right_hand.start()
    logger.info("Right hand initialized and started")

    # Reset hands to initial position
    logger.info("Resetting hands to initial position...")
    left_hand.set_joint_position([0.0] * HAND_JOINT_COUNT, HAND_INTERPOLATE)
    right_hand.set_joint_position([0.0] * HAND_JOINT_COUNT, HAND_INTERPOLATE)
    time.sleep(1)
    logger.info("Hands ready")

    # Initialize full robot action buffer and thread
    logger.info("Initializing full robot action thread...")
    action_buf_lock = threading.Lock()
    action_buffer = {
        "left_arm": None,
        "right_arm": None,
        "left_hand": None,
        "right_hand": None,
    }  # None means buffer is empty/stopped
    hardware_lock = threading.Lock()    
    full_robot_action_terminate_event = threading.Event()
    full_robot_action_thread = threading.Thread(
        target=full_robot_action_loop,
        kwargs={
            "terminate_event": full_robot_action_terminate_event,
            "action_buf_lock": action_buf_lock,
            "action_buffer": action_buffer,
            "hardware_lock": hardware_lock,
            "dexmate_bimanual_robot": dexmate_bimanual_robot,
            "sharpa_left_hand": left_hand,
            "sharpa_right_hand": right_hand,
            "smoothing_and_safety_manager": smoothing_and_safety_manager,
            "action_hz": ARM_ACTION_HZ,
            "hand_interpolate": HAND_INTERPOLATE,
        },
        daemon=True
    )
    full_robot_action_thread.start()
    logger.info(f"Full robot action thread started at {ARM_ACTION_HZ} Hz")

    # Move robot to initial position - CRITICAL for teleop_pub retargeting to work correctly
    assert 0.3/COMMAND_HZ < 0.05  # Sanity check to ensure the robot won't move too fast
    initialization_planner = InitializationCollisionPlanner(
        pin_full_robot_wrapper=pin_full_robot_wrapper,
        assemble_qpos=assemble_qpos,
        disassemble_qpos=disassemble_qpos,
        max_edge_joint_step=0.3/COMMAND_HZ,  # 0.3 rad/s max speed
        plan_timeout_s=10.0,
        solve_step_s=0.1,
    )

    move_robot_to_position_safe(
        dexmate_bimanual_robot=dexmate_bimanual_robot,
        sharpa_left_hand=left_hand,
        sharpa_right_hand=right_hand,
        initialization_planner=initialization_planner,
        action_buffer=action_buffer,
        action_buf_lock=action_buf_lock,
        target_joint_pos={
            "left_arm": np.array(LEFT_ARM_DEFAULT_JOINT_POS),
            "right_arm": np.array(RIGHT_ARM_DEFAULT_JOINT_POS),
            "left_hand": np.zeros((HAND_JOINT_COUNT,)),
            "right_hand": np.zeros((HAND_JOINT_COUNT,)),
            "head": np.array(HEAD_DEFAULT_JOINT_POS),
            "torso": np.array(TORSO_DEFAULT_JOINT_POS),
        },
        hardware_lock=hardware_lock,
        command_hz=COMMAND_HZ,
        dof_error_tolerance=RESET_DOF_ERR_TOL,
        hold_time_s=0.5,
    )
    logger.info("Robot moved to initial position")

    # Initialize tactile buffers and thread (matches main_teleop.py pattern)
    logger.info("Initializing tactile buffers...")
    tactile_buf_lock = threading.Lock()
    left_tactile_buffers = {}
    right_tactile_buffers = {}
    
    for ttype_lower, (shape, dtype) in TACTILE_BUFFER_SHAPES.items():
        left_tactile_buffers[ttype_lower] = np.zeros(shape, dtype=dtype)
        right_tactile_buffers[ttype_lower] = np.zeros(shape, dtype=dtype)

    # Rolling F6 history (one entry per tactile fetch tick) so the VLA server
    # receives a dense [F6_HISTORY_LEN, n_fingers, 6] window matching the
    # VQ-VAE training-time temporal density.
    left_f6_history: deque = deque(maxlen=F6_HISTORY_LEN)
    right_f6_history: deque = deque(maxlen=F6_HISTORY_LEN)

    tactile_terminate_event = threading.Event()
    tactile_thread = threading.Thread(
        target=_tactile_fetch_loop,
        args=(
            tactile_terminate_event,
            tactile_buf_lock,
            hardware_lock,
            left_tactile_buffers,
            right_tactile_buffers,
            left_hand,
            right_hand,
            TACTILE_FETCH_HZ,
            left_f6_history,
            right_f6_history,
        ),
        daemon=True
    )
    tactile_thread.start()
    logger.info("Tactile fetching thread started")

    vla_context = zmq.Context()
    vla_socket = vla_context.socket(zmq.REQ)
    vla_socket.connect(f"tcp://0.0.0.0:{5678}")

    # Target image size for camera capture (before vision_transform resizes)
    # We capture at IMAGE_WIDTH x IMAGE_HEIGHT, then vision_transform resizes to model resolution
    target_capture_size = (IMAGE_HEIGHT, IMAGE_WIDTH)
    logger.info(f"Camera capture size: {target_capture_size}")

    # Wait for user to start
    input("\nPress [Enter] to start policy evaluation...")

    logger.info("\n" + "="*50)
    logger.info("BACKGROUND CONTROLS ACTIVE:")
    logger.info("  Press 'p' anytime to PAUSE/RESUME execution")
    logger.info("  Press 'r' anytime to RESET current trajectory")
    logger.info("  Press 'q' anytime to QUIT the entire program")
    logger.info("="*50 + "\n")
    key_ctrl = KeyController()

    # Main control loop
    logger.info("=" * 50)
    logger.info("POLICY EVALUATION STARTED")
    logger.info("=" * 50)
    # Synchronize sending new 
    limiter = RateLimiter(frequency=COMMAND_HZ, name="eval_policy_control_loop", warn=True)

    try:
        while True:
            if key_ctrl.quit_requested:
                break

            # Clear hotkey flags for the new run
            key_ctrl.reset_requested = False
            key_ctrl.paused = False

            # Halt any lingering actions from the previous trajectory
            with action_buf_lock:
                action_buffer["left_arm"] = None
                action_buffer["right_arm"] = None
                action_buffer["left_hand"] = None
                action_buffer["right_hand"] = None

            logger.info("\n--- PREPARING ROBOT FOR NEW EVALUATION ---")
            move_robot_to_position_safe(
                dexmate_bimanual_robot=dexmate_bimanual_robot,
                sharpa_left_hand=left_hand,
                sharpa_right_hand=right_hand,
                initialization_planner=initialization_planner,
                action_buffer=action_buffer,
                action_buf_lock=action_buf_lock,
                target_joint_pos={
                    "left_arm": np.array(LEFT_ARM_DEFAULT_JOINT_POS),
                    "right_arm": np.array(RIGHT_ARM_DEFAULT_JOINT_POS),
                    "left_hand": np.zeros((HAND_JOINT_COUNT,)),
                    "right_hand": np.zeros((HAND_JOINT_COUNT,)),
                    "head": np.array(HEAD_DEFAULT_JOINT_POS),
                    "torso": np.array(TORSO_DEFAULT_JOINT_POS),
                },
                hardware_lock=hardware_lock,
                command_hz=COMMAND_HZ,
                dof_error_tolerance=RESET_DOF_ERR_TOL,
                hold_time_s=0.5,
            )
            time.sleep(1)

            # Interactive Grid Prompt
            _quit = False
            while True:
                key_ctrl.suspend()
                grid_input = input("\nEnter grid index (0-15) to test, 's' to skip grid, or 'q' to quit: ").strip().lower()
                key_ctrl.resume()
                print(f"Received input: '{grid_input}'")
                if grid_input == 'q' or key_ctrl.quit_requested:
                    _quit = True
                    break
                elif grid_input.isdigit() and 0 <= int(grid_input) <= 15:
                    test_grid_position(
                        robot=dexmate_bimanual_robot,
                        ik_solver=pink_ik_solver,
                        right_hand=right_hand,
                        grid_index=int(grid_input),
                        left_hand=left_hand,
                        initialization_planner=initialization_planner,
                        action_buffer=action_buffer,
                        action_buf_lock=action_buf_lock,
                        hardware_lock=hardware_lock,
                        key_ctrl=key_ctrl,
                    )
                    break
                elif grid_input == 's':
                    logger.info("Skipping grid test and proceeding to policy evaluation...")
                    break
            if _quit:
                logger.info("Quit requested. Exiting program.")
                break


            # Wait for user to start
            key_ctrl.suspend()
            start_cmd = input("\nPress [Enter] to start policy evaluation (or 'q' to quit)...").strip().lower()
            key_ctrl.resume()
            if start_cmd == 'q' or key_ctrl.quit_requested:
                break

            logger.info("=" * 50)
            logger.info("POLICY EVALUATION STARTED")
            logger.info("=" * 50)
            limiter = RateLimiter(frequency=COMMAND_HZ, name="eval_policy_control_loop", warn=True)

            # --- Inner execution loop ---
            for step in range(MAX_STEPS):
                # 1. Hotkey Checks
                if key_ctrl.quit_requested:
                    logger.info("Quit requested during execution.")
                    break
                
                if key_ctrl.reset_requested:
                    logger.info("Reset requested! Breaking out of current trajectory...")
                    break
                
                if key_ctrl.paused:
                    # Clear action buffer so robot safety manager halts movement
                    with action_buf_lock:
                        action_buffer["left_arm"] = None
                        action_buffer["right_arm"] = None
                        action_buffer["left_hand"] = None
                        action_buffer["right_hand"] = None
                    time.sleep(0.1)
                    continue

                # 2. Get current state
                # proprio = get_current_proprio(dexmate_bimanual_robot, hardware_lock, left_hand, right_hand)
                proprio = get_current_pose(dexmate_bimanual_robot, pink_ik_solver, hardware_lock, left_hand, right_hand) 


                images = get_all_camera_images(dexmate_bimanual_robot, hardware_lock, wrist_cam_receiver, head_cam_receiver, target_capture_size)
                Image.fromarray(images["head"]).save(f"head_{step}.jpg")
                Image.fromarray(images["right_wrist"]).save(f"right_wrist_{step}.jpg")
                Image.fromarray(images["left_wrist"]).save(f"left_wrist_{step}.jpg")
                with tactile_buf_lock:
                    if DUAL_ARM:
                        tactile_f6 = np.concatenate([left_tactile_buffers['f6'], right_tactile_buffers['f6']], axis=0)
                        tactile_deform = np.concatenate([left_tactile_buffers['deform'], right_tactile_buffers['deform']], axis=0)
                    else:
                        tactile_f6 = right_tactile_buffers['f6'].copy()
                        tactile_deform = right_tactile_buffers['deform'].copy()
                # Dense temporal F6 window for the server (matches VQ-VAE
                # training window length).  Server takes the last frame for
                # the single-frame tacf6_embedder and uses the full window for
                # VQ-VAE encoding.
                tactile_f6_window = _snapshot_f6_window(
                    tactile_buf_lock, left_f6_history, right_f6_history,
                    dual_arm=DUAL_ARM)

                # Visualize inputs before JPEG encoding overwrites the images dict
                visualize_inference_inputs(
                    step=step,
                    head_img=images["head"],
                    right_wrist_img=images["right_wrist"],
                    left_wrist_img=images.get("left_wrist") if DUAL_ARM else None,
                    tactile_deform=tactile_deform,
                    tactile_f6=tactile_f6,
                    dual_arm=DUAL_ARM,
                )

                for key, rgb_img in images.items():
                    bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
                    frame = cv2.imencode('.jpg', bgr_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])[1]
                    images[key] = frame.tobytes()

                # 3. VLA Inference (slow tick — full payload).
                # When USE_TACTILE_REFINE is on, mode='slow_and_fast' tells the
                # server to (a) run forward_flow_action_partial → cache KV at
                # τ_split, and (b) immediately run the first tactile_flow_continue
                # to produce a clean action chunk.  Subsequent in-chunk fast
                # ticks (steps 4/8/12) reuse that cached KV.
                payload = {
                    "mode": "slow_and_fast" if USE_TACTILE_REFINE else "slow",
                    "task_description": "A pair of gray shorts lay on the tabletop. Grasping them from underneath with both hands, fold them upward; then, using your right hand to hold down the right side, use your left hand to grasp the left side and fold them upward a second time.",
                    # "task_description": "Pour the sugar from the filled cup to the empty cup.",
                    # "task_description": "Next to the cube on the table lies a card case containing two cards. Pick up the case with the left hand, then use the right thumb to slide the cards out through the central opening; subsequently, use the right thumb and index finger to slide out the first card, taking care not to pull out the second one.",
                    #"Turn a page of the book from right to left using your right index finger.",
                    # "Turn a page of the book from right to left using your right index finger."
                    # Pick up the airpods case, open it, take out the airpods and place them on the table. Then close the case and place it back on the table.
                    # Pick up the egg and place it in the carton.
                    # "Using the right thumb and index finger, pick up the egg from the green egg tray and place   it into the only empty spot in the box."
                    "image_head": images["head"],
                    "image_wrist_right": images["right_wrist"],
                    "state_fast": proprio,
                    "state_slow": proprio,
                    "tactile_f6": tactile_f6_window,    # dense [W, 10, 6] window
                    "tactile_deform": tactile_deform,
                }
                if DUAL_ARM:
                    payload["image_wrist_left"] = images["left_wrist"]
                vla_socket.send(pickle.dumps(payload))
                # Poll with timeout so hotkey checks aren't blocked
                response_bytes = None
                while response_bytes is None:
                    if key_ctrl.quit_requested or key_ctrl.reset_requested:
                        logger.info("Hotkey detected while waiting for VLA response.")
                        break
                    if vla_socket.poll(timeout=500):  # 500ms poll
                        response_bytes = vla_socket.recv()
                if response_bytes is None:
                    break
                response = pickle.loads(response_bytes)
                assert response.get('status') == 'success'
                action = np.array(response.get('actions'))
                # print(f"Received raw action from VLA: {action.shape}")
                # input("Press [Enter] to execute this action chunk...")
                if DUAL_ARM:
                    assert action.shape in [(CHUNK_SIZE, 58), (CHUNK_SIZE, 62)], f"Unexpected action shape: {action.shape}"
                else:
                    assert action.shape in [(CHUNK_SIZE, 29), (CHUNK_SIZE, 31)], f"Unexpected action shape: {action.shape}"
                print(f"Received action chunk of shape {action.shape} from VLA")
                print(f"Step {step}")

                # ── ACT temporal aggregation: seed buffer with this slow chunk.
                # Each outer iteration starts a fresh window covering global
                # steps [chunk_start_step, chunk_start_step + CHUNK_SIZE).
                chunk_start_step = step * EXECUTE_FIRST_FEW_IN_CHUNK
                chunk_buffer = []
                if USE_TEMPORAL_AGGREGATION:
                    chunk_buffer.append((chunk_start_step, action.copy()))

                with hardware_lock:
                    initial_arm_joint_pos_dict = dexmate_bimanual_robot.get_joint_pos_dict(component=["left_arm", "right_arm"])
                    initial_left_q = np.array([initial_arm_joint_pos_dict[name] for name in DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES["left_arm"]])
                    initial_right_q = np.array([initial_arm_joint_pos_dict[name] for name in DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES["right_arm"]])
                    initial_right_hand_q = np.array(right_hand.get_states().angles, dtype=np.float64)
                    if DUAL_ARM:
                        initial_left_hand_q = np.array(left_hand.get_states().angles, dtype=np.float64)

                fk_res_initial = pink_ik_solver.fk(
                    frames=["L_ee", "R_ee"],
                    joint_pos_by_component={
                        "left_arm": initial_left_q,
                        "right_arm": initial_right_q
                    }
                )
                initial_left_ee_pose = fk_res_initial["L_ee"]
                initial_right_ee_pose = fk_res_initial["R_ee"]

                ik_warmstart_right_q = initial_right_q.copy()
                if DUAL_ARM:
                    ik_warmstart_left_q = initial_left_q.copy()

                # ==========================================
                # Mode 1: Delta Joint (based on current state at each action step)
                # action_shape: (CHUNK_SIZE, 29)
                # ==========================================
                # for action_idx in range(EXECUTE_FIRST_FEW_IN_CHUNK):
                #     with action_buf_lock:
                #         action_buffer["left_arm"] = np.array(LEFT_ARM_DEFAULT_JOINT_POS)
                #         action_buffer["right_arm"] = initial_right_q + action[action_idx, 0:7]
                #         action_buffer["left_hand"] = np.zeros((22,))
                #         action_buffer["right_hand"] = initial_right_hand_q + action[action_idx, 7:29]

                #     limiter.sleep()

                # ==========================================
                # Mode 2: Absolute EEF
                # action_shape: (CHUNK_SIZE, 31) -> 3(xyz) + 6(rot) + 22(hand)
                # ==========================================
                # for action_idx in range(EXECUTE_FIRST_FEW_IN_CHUNK):
                #     target_pos = action[action_idx, 0:3]
                #     target_rot6d = action[action_idx, 3:9]
                #     target_hand = action[action_idx, 9:31]  # abs hand joint

                #     target_R = rot6d_to_matrix(target_rot6d)
                #     target_right_ee_pose = pin.SE3(target_R, target_pos)

                #     ik_res = pink_ik_solver.solve_ik(
                #         ee_target_poses={
                #             "L_ee": initial_left_ee_pose, 
                #             "R_ee": target_right_ee_pose
                #         },
                #         arm_initial_joint_pos={
                #             "left_arm": initial_left_q,
                #             "right_arm": ik_warmstart_right_q  
                #         }
                #     )
                #     target_right_q = ik_res["right_arm"].copy()
                #     ik_warmstart_right_q = target_right_q  
                #     # print(f"Action idx {action_idx}: Target EEF pos {target_right_q}, hand {target_hand[0:5]}...")
                #     # input("Press [Enter] to execute this action step...")

                #     with action_buf_lock:
                #         action_buffer["left_arm"] = np.array(LEFT_ARM_DEFAULT_JOINT_POS)
                #         action_buffer["right_arm"] = target_right_q
                #         action_buffer["left_hand"] = np.zeros((22,))
                #         action_buffer["right_hand"] = target_hand

                #     limiter.sleep()

                # ==========================================
                # Mode 3: Delta EEF (基于 chunk 第一帧的 Local Delta)
                # action_shape: (CHUNK_SIZE, 31) -> 3(local_delta_pos) + 6(local_delta_rot) + 22(abs_hand)
                # ==========================================
                initial_pos = initial_right_ee_pose.translation
                initial_R = initial_right_ee_pose.rotation
                if DUAL_ARM:
                    initial_left_pos = initial_left_ee_pose.translation
                    initial_left_R = initial_left_ee_pose.rotation

                for action_idx in range(EXECUTE_FIRST_FEW_IN_CHUNK):
                    if key_ctrl.quit_requested or key_ctrl.reset_requested:
                        logger.info("Hotkey detected mid-chunk, aborting action execution.")
                        break
                    if key_ctrl.paused:
                        with action_buf_lock:
                            action_buffer["left_arm"] = None
                            action_buffer["right_arm"] = None
                            action_buffer["left_hand"] = None
                            action_buffer["right_hand"] = None
                        while key_ctrl.paused and not key_ctrl.quit_requested and not key_ctrl.reset_requested:
                            time.sleep(0.1)
                        if key_ctrl.quit_requested or key_ctrl.reset_requested:
                            break

                    # ── Async tactile refinement at REFINE_OFFSETS within the chunk ──
                    # Read the most recent tactile (background thread keeps it
                    # fresh at TACTILE_FETCH_HZ), send a `mode='fast'` payload,
                    # and append the refined chunk to chunk_buffer for ACT
                    # temporal aggregation (or splice into `action` if that
                    # mode is off).  ZMQ REP is single-threaded server-side so
                    # this call blocks until the previous slow inference is
                    # done — the "wait until refinement finishes" guarantee.
                    if USE_TACTILE_REFINE and action_idx in REFINE_OFFSETS:
                        with tactile_buf_lock:
                            if DUAL_ARM:
                                _tac_def = np.concatenate(
                                    [left_tactile_buffers['deform'],
                                     right_tactile_buffers['deform']], axis=0)
                            else:
                                _tac_def = right_tactile_buffers['deform'].copy()
                        # Dense F6 window — server uses last frame for the
                        # tacf6_embedder and the full window for VQ-VAE.
                        _tac_f6 = _snapshot_f6_window(
                            tactile_buf_lock, left_f6_history, right_f6_history,
                            dual_arm=DUAL_ARM)
                        fast_payload = {
                            "mode": "fast",
                            "tactile_f6": _tac_f6,
                            "tactile_deform": _tac_def,
                        }
                        vla_socket.send(pickle.dumps(fast_payload))
                        fast_resp_bytes = None
                        while fast_resp_bytes is None:
                            if key_ctrl.quit_requested or key_ctrl.reset_requested:
                                break
                            if vla_socket.poll(timeout=500):
                                fast_resp_bytes = vla_socket.recv()
                        if fast_resp_bytes is not None:
                            fast_resp = pickle.loads(fast_resp_bytes)
                            if fast_resp.get("status") == "success":
                                refined = np.array(fast_resp["actions"])
                                if refined.shape == action.shape:
                                    if USE_TEMPORAL_AGGREGATION:
                                        # Append to buffer; aggregation will
                                        # blend with previous chunks below.
                                        chunk_buffer.append(
                                            (chunk_start_step, refined.copy()))
                                    else:
                                        # Legacy splice — replaces remainder.
                                        action[action_idx:] = refined[action_idx:]
                                    print(f"  [refine @{action_idx}] chunk_id="
                                          f"{fast_resp.get('chunk_id')}, "
                                          f"latency={fast_resp.get('latency_ms', 0):.1f} ms")
                                else:
                                    logger.warning(
                                        f"refine response shape mismatch: "
                                        f"{refined.shape} vs {action.shape}; skipping")

                    # ── Pick the action to command this step ──
                    # Under USE_TEMPORAL_AGGREGATION we exp-weight-average all
                    # chunks in chunk_buffer that cover this global step.  All
                    # chunks share chunk_start_step so the average is over
                    # action_idx of {slow chunk, refine@4 chunk, refine@8, …}
                    # restricted to entries where action_idx ≥ their refine
                    # point (which always holds here since we're past it).
                    global_step = chunk_start_step + action_idx
                    if USE_TEMPORAL_AGGREGATION:
                        agg = aggregate_chunks(chunk_buffer, global_step,
                                               k=TEMPORAL_AGG_K)
                        current_action = agg if agg is not None else action[action_idx]
                    else:
                        current_action = action[action_idx]

                    if DUAL_ARM:
                        delta_pos_local_left = current_action[0:3]
                        delta_rot6d_local_left = current_action[3:9]
                        target_hand_left = current_action[9:31]

                        delta_pos_local_right = current_action[31:34]
                        delta_rot6d_local_right = current_action[34:40]
                        target_hand_right = current_action[40:62]

                        target_left_pos = initial_left_pos + initial_left_R @ delta_pos_local_left
                        delta_R_local_left = rot6d_to_matrix(delta_rot6d_local_left)
                        target_left_R = initial_left_R @ delta_R_local_left
                        target_left_ee_pose = pin.SE3(target_left_R, target_left_pos)

                        target_right_pos = initial_pos + initial_R @ delta_pos_local_right
                        delta_R_local_right = rot6d_to_matrix(delta_rot6d_local_right)
                        target_right_R = initial_R @ delta_R_local_right
                        target_right_ee_pose = pin.SE3(target_right_R, target_right_pos)

                        ik_res = pink_ik_solver.solve_ik(
                            ee_target_poses={
                                "L_ee": target_left_ee_pose,
                                "R_ee": target_right_ee_pose
                            },
                            arm_initial_joint_pos={
                                "left_arm": ik_warmstart_left_q,
                                "right_arm": ik_warmstart_right_q
                            }
                        )
                        target_left_q = ik_res["left_arm"].copy()
                        target_right_q = ik_res["right_arm"].copy()
                        ik_warmstart_left_q = target_left_q 
                        ik_warmstart_right_q = target_right_q 

                        with action_buf_lock:
                            action_buffer["left_arm"] = target_left_q
                            action_buffer["right_arm"] = target_right_q
                            action_buffer["left_hand"] = target_hand_left
                            action_buffer["right_hand"] = target_hand_right

                    else:
                        delta_pos_local = current_action[0:3]
                        delta_rot6d_local = current_action[3:9]

                        target_hand = current_action[9:31]

                        target_pos = initial_pos + initial_R @ delta_pos_local

                        delta_R_local = rot6d_to_matrix(delta_rot6d_local)
                        target_R = initial_R @ delta_R_local

                        target_right_ee_pose = pin.SE3(target_R, target_pos)

                        ik_res = pink_ik_solver.solve_ik(
                            ee_target_poses={
                                "L_ee": initial_left_ee_pose,
                                "R_ee": target_right_ee_pose
                            },
                            arm_initial_joint_pos={
                                "left_arm": initial_left_q,
                                "right_arm": ik_warmstart_right_q
                            }
                        )
                        target_right_q = ik_res["right_arm"].copy()
                        ik_warmstart_right_q = target_right_q 

                        with action_buf_lock:
                            action_buffer["left_arm"] = np.array(LEFT_ARM_DEFAULT_JOINT_POS)
                            action_buffer["right_arm"] = target_right_q
                            action_buffer["left_hand"] = np.zeros((22,))
                            action_buffer["right_hand"] = target_hand

                    limiter.sleep()

                # # ==========================================
                # # Mode 4_Alt: 严格闭环 Delta EEF (与你未修改的数据 100% 对齐)
                # # ==========================================
                # for action_idx in range(EXECUTE_FIRST_FEW_IN_CHUNK):
                #     with hardware_lock:
                #         curr_joint_pos_dict = dexmate_bimanual_robot.get_joint_pos_dict(component=["left_arm", "right_arm"])

                #     curr_left_q = np.array([curr_joint_pos_dict[name] for name in DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES["left_arm"]])
                #     curr_right_q = np.array([curr_joint_pos_dict[name] for name in DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES["right_arm"]])

                #     curr_fk_res = pink_ik_solver.fk(
                #         frames=["L_ee", "R_ee"],
                #         joint_pos_by_component={"left_arm": curr_left_q, "right_arm": curr_right_q}
                #     )
                #     curr_R = curr_fk_res["R_ee"].rotation
                #     curr_pos = curr_fk_res["R_ee"].translation

                #     delta_pos_local = action[action_idx, 0:3]
                #     delta_rot6d_local = action[action_idx, 3:9]
                #     target_hand = action[action_idx, 9:31] # 手部依然是绝对值

                #     # 3. 计算 Target (将增量作用于刚刚读到的【真实位姿】上)
                #     target_pos = curr_pos + curr_R @ delta_pos_local

                #     delta_R_local = rot6d_to_matrix(delta_rot6d_local)
                #     target_R = curr_R @ delta_R_local

                #     target_right_ee_pose = pin.SE3(target_R, target_pos)

                #     # 4. IK 求解
                #     ik_res = pink_ik_solver.solve_ik(
                #         ee_target_poses={
                #             "L_ee": initial_left_ee_pose,
                #             "R_ee": target_right_ee_pose
                #         },
                #         arm_initial_joint_pos={
                #             "left_arm": curr_left_q,
                #             "right_arm": curr_right_q
                #         }
                #     )
                #     target_right_q = ik_res["right_arm"].copy()

                #     # 5. 发送指令
                #     with action_buf_lock:
                #         action_buffer["left_arm"] = np.array(LEFT_ARM_DEFAULT_JOINT_POS)
                #         action_buffer["right_arm"] = target_right_q
                #         action_buffer["left_hand"] = np.zeros((22,))
                #         action_buffer["right_hand"] = target_hand

                #     limiter.sleep()
                
                # # Delta joint mode, no base
                # for action_idx in range(EXECUTE_FIRST_FEW_IN_CHUNK): 
                #     # input("Press [Enter] to execute the next action chunk...")
                #     # NOTE: If we compute delta action chunk relative to the first state, we do not need to get the robot
                #     # state at each time we execute a new action from the action chunk.
                #     with hardware_lock:
                #         arm_joint_pos_dict = dexmate_bimanual_robot.get_joint_pos_dict(component=["left_arm", "right_arm"])
                #         left_hand_joint_positions = np.array(left_hand.get_states().angles, dtype=np.float64)
                #         right_hand_joint_positions = np.array(right_hand.get_states().angles, dtype=np.float64)
                #     with action_buf_lock:
                #         # Delta Action Mode
                #         # Only have right hand and arm in action
                #         action_buffer["left_arm"] = np.array(LEFT_ARM_DEFAULT_JOINT_POS)
                #         action_buffer["right_arm"] = action[action_idx, 0:7] + np.array([arm_joint_pos_dict[name] for name in DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES["right_arm"]])
                #         action_buffer["left_hand"] = np.zeros((22,))
                #         action_buffer["right_hand"] = action[action_idx, 7:29] + right_hand_joint_positions
                #     # Wait according to COMMAND_HZ to align with data collection.
                #     limiter.sleep()

                # # Abs joint mode
                # for action_idx in range(EXECUTE_FIRST_FEW_IN_CHUNK): 
                #     # Extra mid-chunk hotkey check in case chunk sizes get larger
                #     if key_ctrl.reset_requested or key_ctrl.quit_requested or key_ctrl.paused:
                #         break 
                #     with hardware_lock:
                #         arm_joint_pos_dict = dexmate_bimanual_robot.get_joint_pos_dict(component=["left_arm", "right_arm"])
                #         left_hand_joint_positions = np.array(left_hand.get_states().angles, dtype=np.float64)
                #         right_hand_joint_positions = np.array(right_hand.get_states().angles, dtype=np.float64)
                #     with action_buf_lock:
                #         action_buffer["left_arm"] = np.array(LEFT_ARM_DEFAULT_JOINT_POS)
                #         action_buffer["right_arm"] = action[action_idx, 0:7]
                #         action_buffer["left_hand"] = np.zeros((22,))
                #         action_buffer["right_hand"] = action[action_idx, 7:29]
                #     limiter.sleep()
            
            logger.info("--- Trajectory Execution Loop Exited ---")

    except KeyboardInterrupt:
        logger.info("Received interrupt signal, stopping evaluation...")
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
    finally:
        logger.info("Cleaning up...")
        cv2.destroyAllWindows()

        # Stop arm action thread
        with action_buf_lock:
            action_buffer["left_arm"] = None
            action_buffer["right_arm"] = None
            action_buffer["left_hand"] = None
            action_buffer["right_hand"] = None
        full_robot_action_terminate_event.set()
        full_robot_action_thread.join(timeout=2.0)
        
        # Stop tactile thread
        tactile_terminate_event.set()
        tactile_thread.join(timeout=2.0)

        # Stop wrist cameras
        wrist_cam_receiver.stop()

        # Stop hands
        left_hand.stop()
        right_hand.stop()
        SharpaWaveManager.get_instance().disconnect_all()
        logger.info("Disconnected all hands")

        # Shutdown robot
        dexmate_bimanual_robot.shutdown()
        logger.info("Robot shutdown complete")

        logger.info("=" * 50)
        logger.info("POLICY EVALUATION COMPLETE")
        logger.info("=" * 50)



            


    # try:
    #     for step in range(MAX_STEPS):
    #         # Get current state - IMPORTANT: Match teleop data collection order!
    #         # In main_teleop.py, the order is: proprio → images → tactile (from buffer)
    #         # This ensures timing consistency between training data and inference
    #         proprio = get_current_proprio(dexmate_bimanual_robot, hardware_lock, left_hand, right_hand)
    #         images = get_all_camera_images(dexmate_bimanual_robot, hardware_lock, wrist_cam_receiver, target_capture_size)
    #         for key, rgb_img in images.items():
    #             bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    #             frame = cv2.imencode('.jpg', bgr_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])[1]
    #             images[key] = frame.tobytes()
    #         # Read tactile from buffer (updated by separate thread)
    #         with tactile_buf_lock:
    #             # tactile = np.concatenate([
    #             #     right_tactile_buffer.copy(),
    #             #     left_tactile_buffer.copy()
    #             # ])
    #             tactile = right_tactile_buffer.copy()  # Only use right hand tactile for now

    #         # VLA Inference
    #         # TODO: Right now the inference latency is not being considered. In the long run we probably want
    #         # to define a new function in teleop.arm_hand_control that executes an action chunk, but for now
    #         # this version works.
    #         payload = {
    #             "task_description": "Move down to pick the orange cube with the right robotic hand and lift it up.",
    #             "image_head": images["head"],
    #             "image_wrist_right": images["right_wrist"],
    #             # "image_wrist_left": images["left_wrist"],
    #             "state_fast": proprio,
    #             "state_slow": proprio,
    #             "tactile_f6": tactile,
    #             # "delay_steps": 4
    #         }
    #         vla_socket.send(pickle.dumps(payload))
    #         response_bytes = vla_socket.recv()
    #         response = pickle.loads(response_bytes)
    #         assert response.get('status') == 'success'
    #         action = np.array(response.get('actions'))
    #         assert action.shape == (CHUNK_SIZE, 29)
    #         print(f"Received action chunk of shape {action.shape} from VLA")
    #         print(action)
    #         print(f"Step {step}")
            
    #         # # Execute action, the first few steps in the chunk with delta action mode, then fetch a new chunk from VLA.
    #         # with hardware_lock:
    #         #     arm_joint_pos_dict = dexmate_bimanual_robot.get_joint_pos_dict(component=["left_arm", "right_arm"])
    #         #     left_hand_joint_positions = np.array(left_hand.get_states().angles, dtype=np.float64)
    #         #     right_hand_joint_positions = np.array(right_hand.get_states().angles, dtype=np.float64)

    #         # initial_right_arm = np.array([arm_joint_pos_dict[name] for name in DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES["right_arm"]])
    #         # initial_right_hand = right_hand_joint_positions
    #         # initial_state = np.concatenate([initial_right_arm, initial_right_hand])
    #         # cumulative_deltas = np.cumsum(action, axis=0)
    #         # abs_action_trajectory = cumulative_deltas + initial_state
    #         # print("Initial right arm joint positions:", initial_right_arm)
    #         # print("Initial right hand joint positions:", initial_right_hand)
    #         # print("Cumulative deltas for the action chunk:")
    #         # print(cumulative_deltas)
    #         # print("Absolute action trajectory for the first few steps:")
    #         # print(abs_action_trajectory[:EXECUTE_FIRST_FEW_IN_CHUNK, :])
    #         # for action_idx in range(EXECUTE_FIRST_FEW_IN_CHUNK):
    #         #     # input("Press [Enter] to execute the next action chunk...")
    #         #     time.sleep(0.5)  # Short pause to observe the robot state before executing the action
    #         #     with hardware_lock:
    #         #         action_buffer["left_arm"] = np.array(LEFT_ARM_DEFAULT_JOINT_POS)
    #         #         action_buffer["right_arm"] = abs_action_trajectory[action_idx, 0:7]
    #         #         action_buffer["left_hand"] = np.zeros((22,))
    #         #         action_buffer["right_hand"] = abs_action_trajectory[action_idx, 7:29]
    #         #     limiter.sleep()

    #         # Execute action, absolute action mode relative to current state
    #         for action_idx in range(EXECUTE_FIRST_FEW_IN_CHUNK): 
    #             # input("Press [Enter] to execute the next action chunk...")
    #             # NOTE: If we compute absolute action chunk relative to the first state, we do not need to get the robot
    #             # state at each time we execute a new action from the action chunk.
    #             with hardware_lock:
    #                 arm_joint_pos_dict = dexmate_bimanual_robot.get_joint_pos_dict(component=["left_arm", "right_arm"])
    #                 left_hand_joint_positions = np.array(left_hand.get_states().angles, dtype=np.float64)
    #                 right_hand_joint_positions = np.array(right_hand.get_states().angles, dtype=np.float64)
    #             with action_buf_lock:
    #                 # Absolute Action Mode
    #                 # Only have right hand and arm in action
    #                 action_buffer["left_arm"] = np.array(LEFT_ARM_DEFAULT_JOINT_POS)
    #                 action_buffer["right_arm"] = action[action_idx, 0:7]
    #                 action_buffer["left_hand"] = np.zeros((22,))
    #                 action_buffer["right_hand"] = action[action_idx, 7:29]
    #             # Wait according to COMMAND_HZ to align with data collection.
    #             limiter.sleep()

    #         # # Execute action, delta action mode relative to current state
    #         # for action_idx in range(EXECUTE_FIRST_FEW_IN_CHUNK): 
    #         #     # input("Press [Enter] to execute the next action chunk...")
    #         #     # NOTE: If we compute delta action chunk relative to the first state, we do not need to get the robot
    #         #     # state at each time we execute a new action from the action chunk.
    #         #     with hardware_lock:
    #         #         arm_joint_pos_dict = dexmate_bimanual_robot.get_joint_pos_dict(component=["left_arm", "right_arm"])
    #         #         left_hand_joint_positions = np.array(left_hand.get_states().angles, dtype=np.float64)
    #         #         right_hand_joint_positions = np.array(right_hand.get_states().angles, dtype=np.float64)
    #         #     with action_buf_lock:
    #         #         # Delta Action Mode
    #         #         # Only have right hand and arm in action
    #         #         action_buffer["left_arm"] = np.array(LEFT_ARM_DEFAULT_JOINT_POS)
    #         #         action_buffer["right_arm"] = action[action_idx, 0:7] + np.array([arm_joint_pos_dict[name] for name in DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES["right_arm"]])
    #         #         action_buffer["left_hand"] = np.zeros((22,))
    #         #         action_buffer["right_hand"] = action[action_idx, 7:29] + right_hand_joint_positions
    #         #     # Wait according to COMMAND_HZ to align with data collection.
    #         #     limiter.sleep()


    # except KeyboardInterrupt:
    #     logger.info("Received interrupt signal, stopping evaluation...")
    # except Exception as e:
    #     logger.error(f"Error during evaluation: {e}")
    # finally:
    #     logger.info("Cleaning up...")
        
    #     # Stop arm action thread
    #     logger.info("Stopping arm action thread...")
    #     with action_buf_lock:
    #         action_buffer["left_arm"] = None
    #         action_buffer["right_arm"] = None
    #         action_buffer["left_hand"] = None
    #         action_buffer["right_hand"] = None
    #     full_robot_action_terminate_event.set()
    #     full_robot_action_thread.join(timeout=2.0)
        
    #     # Stop tactile thread
    #     logger.info("Stopping tactile thread...")
    #     tactile_terminate_event.set()
    #     tactile_thread.join(timeout=2.0)

    #     # Stop wrist cameras
    #     logger.info("Stopping wrist cameras...")
    #     wrist_cam_receiver.stop()

    #     # Stop hands
    #     logger.info("Stopping hands...")
    #     left_hand.stop()
    #     right_hand.stop()
    #     SharpaWaveManager.get_instance().disconnect_all()
    #     logger.info("Disconnected all hands")

    #     # Shutdown robot
    #     dexmate_bimanual_robot.shutdown()
    #     logger.info("Robot shutdown complete")

    #     logger.info("=" * 50)
    #     logger.info("POLICY EVALUATION COMPLETE")
    #     logger.info("=" * 50)


if __name__ == "__main__":
    input("Make sure language instruction in payload, crop box and num_exec are correct")
    tyro.cli(main)