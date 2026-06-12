"""Policy inference client for the T-Rex ZMQ inference server.

Connects to the slow/fast cascaded inference server (T-Rex/scripts/test.py)
over ZMQ REQ and executes the returned action chunks on the robot.

Control mode (the only one supported): the policy outputs, per action step,
a delta end-effector pose relative to the chunk-start EEF pose (3 local xyz +
6 rot6d per arm) plus absolute hand joint targets (22 per hand); deltas are
resolved to joint targets via differential IK (PinkLocalIK). Other control
modes (absolute EEF, delta/absolute joint space) are straightforward
variations on the chunk-execution loop if your policy head differs.

All site/deployment settings come from the YAML config (see
config/default.yaml, including the `inference:` section); run with
    python eval_trex_async.py --config ../config/default.yaml \
        --task-description "..."
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import select
import termios
import threading
import time
import tty
from typing import Optional

import cv2
import numpy as np
import pinocchio as pin
import tyro
import zmq

# Robot and hand imports
from dexcontrol.robot import Robot
from loguru import logger
from loop_rate_limiters import RateLimiter
from PIL import Image
from sharpa import ControlMode, ControlSource, HandSide, SharpaWave, SharpaWaveManager

from camera.head_camera_receiver import HeadCameraReceiver

# Wrist camera imports
from camera.wrist_camera_receiver import WristCameraReceiver
from teleop.arm_hand_control import (
    InitializationCollisionPlanner,
    SmoothingAndSafetyManager,
    full_robot_action_loop,
    move_robot_to_position_safe,
)

# IK and robot utilities
from teleop.config import DEFAULT_CONFIG_PATH, TeleopConfig, load_config
from teleop.ik_utils import PinkLocalIK
from teleop.robot_descriptions import (
    DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES,
    add_env_obstacles,
    build_full_robot,
)

# Tactile configuration (coupled to the model input format, not site-specific)
TACTILE_TYPES_TO_CAPTURE = ["F6", "DEFORM"]
TACTILE_BUFFER_SHAPES = {
    "f6": ((5, 6), np.float32),
    "deform": ((5, 240, 240), np.uint8),
}
HAND_JOINT_COUNT = 22


def aggregate_chunks(chunk_buffer, current_global_step, k):
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
    preds = np.stack(preds)  # [N, action_dim]
    # Newest at the end of `preds`; reverse arange so newest gets highest weight.
    weights = np.exp(-k * np.arange(len(preds))[::-1])
    weights = weights / weights.sum()
    return (preds * weights[:, None]).sum(axis=0)


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
        self._suspended = threading.Event()  # set = listener should sleep
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
                    if ch == "p":
                        self.paused = not self.paused
                        state_str = "PAUSED" if self.paused else "RESUMED"
                        logger.info(f"\n[HOTKEY] Trajectory {state_str}")
                    elif ch == "r":
                        self.reset_requested = True
                        logger.info("\n[HOTKEY] RESET requested. Aborting trajectory...")
                    elif ch == "q":
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
    cv2.putText(
        tag_r,
        "R",
        (4, deform_thumb_size // 2 + 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
    tactile_rows.append(np.hstack([tag_r, right_strip]))

    if dual_arm and left_count > 0:
        left_strip = build_finger_strip(tactile_deform[right_count:], tactile_f6[right_count:])
        tag_l = np.zeros((deform_thumb_size, 30, 3), dtype=np.uint8)
        cv2.putText(
            tag_l,
            "L",
            (4, deform_thumb_size // 2 + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
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


def connect_hands(left_serial: str, right_serial: str):
    """Connect to left and right hands via SharpaWaveSDK."""
    manager = SharpaWaveManager.get_instance()
    time.sleep(1)  # Wait for device discovery

    devices = manager.get_all_device_sn()
    logger.info(f"Available hand devices: {devices}")

    if not devices:
        raise RuntimeError("No hand devices found!")

    if left_serial not in devices:
        raise RuntimeError(f"Left hand serial {left_serial} not found in devices: {devices}")

    if right_serial not in devices:
        raise RuntimeError(f"Right hand serial {right_serial} not found in devices: {devices}")

    left_hand = manager.connect(left_serial)
    logger.info(f"Connected left hand: {left_serial}")

    right_hand = manager.connect(right_serial)
    logger.info(f"Connected right hand: {right_serial}")

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
):
    """Separate thread to fetch tactile data without blocking main control loop.
    Matches main_teleop.py pattern for timing consistency.

    Runs at specified frequency and updates tactile buffers.
    """
    limiter = RateLimiter(frequency=fetch_hz, name="tactile_fetch_limiter", warn=False)
    logger.info(f"[Tactile] Fetching thread started at {fetch_hz} Hz")

    while not terminate_event.is_set():
        limiter.sleep()

        try:
            # Fetch tactile data from both hands
            left_tactile_data = _fetch_tactile_arrays(
                left_hand, HandSide.LEFT, TACTILE_TYPES_TO_CAPTURE
            )
            right_tactile_data = _fetch_tactile_arrays(
                right_hand, HandSide.RIGHT, TACTILE_TYPES_TO_CAPTURE
            )

            # Update buffers with lock
            with buf_lock:
                for ttype_lower in ["f6", "deform"]:
                    if left_tactile_data[ttype_lower] is not None:
                        np.copyto(left_tactile_buffers[ttype_lower], left_tactile_data[ttype_lower])
                    if right_tactile_data[ttype_lower] is not None:
                        np.copyto(
                            right_tactile_buffers[ttype_lower], right_tactile_data[ttype_lower]
                        )
        except Exception as e:
            logger.warning(f"[Tactile] Fetch error: {e}")

    logger.info("[Tactile] Fetching thread stopped")


# =============================================================================
# State Processing Functions
# =============================================================================


def get_current_pose(
    dexmate_bimanual_robot: Robot,
    ik_solver: PinkLocalIK,
    hardware_lock: threading.Lock,
    left_hand: SharpaWave,
    right_hand: SharpaWave,
    dual_arm: bool,
) -> np.ndarray:
    with hardware_lock:
        curr_joint_pos_dict = dexmate_bimanual_robot.get_joint_pos_dict(
            component=["left_arm", "right_arm"]
        )
        right_hand_joints = np.array(right_hand.get_states().angles, dtype=np.float64)  # (22,)
        if dual_arm:
            left_hand_joints = np.array(left_hand.get_states().angles, dtype=np.float64)  # (22,)

    left_arm_joint_positions = np.array(
        [curr_joint_pos_dict[name] for name in DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES["left_arm"]]
    )
    right_arm_joint_positions = np.array(
        [curr_joint_pos_dict[name] for name in DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES["right_arm"]]
    )

    fk_res = ik_solver.fk(
        frames=["L_ee", "R_ee"],
        joint_pos_by_component={
            "left_arm": left_arm_joint_positions,
            "right_arm": right_arm_joint_positions,
        },
    )

    if dual_arm:
        left_ee_pose = fk_res["L_ee"]
        right_ee_pose = fk_res["R_ee"]

        left_pos = left_ee_pose.translation
        left_rot6d = matrix_to_rot6d(left_ee_pose.rotation)
        right_pos = right_ee_pose.translation
        right_rot6d = matrix_to_rot6d(right_ee_pose.rotation)

        pose_62d = np.concatenate(
            [
                left_pos,  # (3,)
                left_rot6d,  # (6,)
                left_hand_joints,  # (22,)
                right_pos,  # (3,)
                right_rot6d,  # (6,)
                right_hand_joints,  # (22,)
            ]
        )
        return pose_62d
    else:
        right_ee_pose = fk_res["R_ee"]

        right_pos = right_ee_pose.translation
        right_rot6d = matrix_to_rot6d(right_ee_pose.rotation)

        pose_31d = np.concatenate(
            [
                right_pos,  # (3,)
                right_rot6d,  # (6,)
                right_hand_joints,  # (22,)
            ]
        )
        return pose_31d


def get_head_image(
    head_cam_receiver: HeadCameraReceiver, target_size: tuple, crop_box: "list[int] | None"
) -> np.ndarray:
    # --- MODIFIED: Get image from the receiver ---
    head_frame = head_cam_receiver.get_data("HEAD_CAM")
    if head_frame is None or head_frame.image_rgb is None:
        raise RuntimeError("Failed to get head camera image from ZMQ receiver")

    head_image = head_frame.image_rgb

    if head_image.shape[0] != target_size[0] or head_image.shape[1] != target_size[1]:
        head_image = cv2.resize(head_image, (target_size[1], target_size[0]))

    assert head_image.shape == (360, 640, 3), f"Unexpected head image shape: {head_image.shape}"
    if crop_box is not None:
        head_image = head_image[crop_box[0] : crop_box[1], crop_box[2] : crop_box[3], :]

    return head_image


def get_all_camera_images(
    dexmate_bimanual_robot: Robot,
    hardware_lock: threading.Lock,
    wrist_cam_receiver: WristCameraReceiver,
    head_cam_receiver: HeadCameraReceiver,
    target_size: tuple,
    crop_box: "list[int] | None",
) -> list:
    images = {}
    # Left wrist
    left_wrist_frame = wrist_cam_receiver.get_data("LEFT_WRIST")
    left_wrist_img = left_wrist_frame.image_rgb
    if left_wrist_img.shape[0] != target_size[0] or left_wrist_img.shape[1] != target_size[1]:
        left_wrist_img = cv2.resize(left_wrist_img, (target_size[1], target_size[0]))
    images["left_wrist"] = left_wrist_img

    # Head
    images["head"] = get_head_image(head_cam_receiver, target_size, crop_box)

    # Right wrist
    right_wrist_frame = wrist_cam_receiver.get_data("RIGHT_WRIST")
    right_wrist_img = right_wrist_frame.image_rgb
    if right_wrist_img.shape[0] != target_size[0] or right_wrist_img.shape[1] != target_size[1]:
        right_wrist_img = cv2.resize(right_wrist_img, (target_size[1], target_size[0]))
    images["right_wrist"] = right_wrist_img

    return images


def main(config: str = str(DEFAULT_CONFIG_PATH), task_description: Optional[str] = None):
    """Run policy inference on the robot against the T-Rex inference server.

    Args:
        config: Path to the YAML config file (see config/default.yaml; the
            `inference:` section holds the client settings).
        task_description: Language instruction sent to the policy. Overrides
            config's inference.task_description.
    """
    cfg: TeleopConfig = load_config(config)
    logger.info(f"Loaded config from {config}")
    inf = cfg.inference
    task_description = task_description or inf.task_description
    if not task_description:
        raise ValueError(
            "No task description: pass --task-description or set inference.task_description "
            "in the config."
        )
    logger.info(f"Task: {task_description}")
    default_joint_pos = cfg.robot.default_joint_pos
    dual_arm = inf.dual_arm

    # Initialize robot with head camera
    logger.info("Initializing robot...")
    dexmate_bimanual_robot = Robot()
    logger.info(f"Robot '{dexmate_bimanual_robot.robot_model}' initialized")

    # Initialize head camera via ZMQ receiver
    logger.info("Initializing head camera...")
    head_cam_receiver = HeadCameraReceiver(
        im_h=cfg.cameras.image_height,
        im_w=cfg.cameras.image_width,
        sender_ip=cfg.cameras.head.sender_ip,
        ports=cfg.cameras.head.ports,
    )
    head_cam_receiver.start_receiving(timeout=10.0)
    logger.info("Head camera started receiving")

    # Initialize wrist cameras
    logger.info("Initializing wrist cameras...")
    wrist_cam_receiver = WristCameraReceiver(
        im_h=cfg.cameras.image_height,
        im_w=cfg.cameras.image_width,
        sender_ip=cfg.cameras.wrist.sender_ip,
        ports=cfg.cameras.wrist.ports,
    )
    wrist_cam_receiver.start_receiving(timeout=10.0)
    logger.info("Wrist cameras started receiving")

    # Initialize IK solver
    pink_ik_solver = PinkLocalIK(default_joint_by_component=default_joint_pos)
    logger.info("IK solver initialized")

    # Initialize ArmIKManager and SmoothingAndSafetyManager for processing arm targets and sending commands
    # Initialize SmoothingAndSafetyManager for processing arm targets and sending commands
    pin_full_robot_wrapper, assemble_qpos, disassemble_qpos = build_full_robot(
        default_joint_by_component=default_joint_pos
    )
    pin_full_robot_wrapper = add_env_obstacles(
        robot=pin_full_robot_wrapper,
        default_joint_by_component={
            "left_arm": default_joint_pos["left_arm"],
            "right_arm": default_joint_pos["right_arm"],
            "left_hand": np.zeros((HAND_JOINT_COUNT,)),
            "right_hand": np.zeros((HAND_JOINT_COUNT,)),
        },
        assemble_qpos=assemble_qpos,
        back_wall_distance=cfg.environment.back_wall_distance,
        left_wall_distance=cfg.environment.left_wall_distance,
        right_wall_distance=cfg.environment.right_wall_distance,
        table_height=cfg.environment.table_height,
    )
    smoothing_and_safety_manager = SmoothingAndSafetyManager(
        pin_full_robot_wrapper=pin_full_robot_wrapper,
        assemble_qpos=assemble_qpos,
        disassemble_qpos=disassemble_qpos,
        ruckig_smoothing=False,
        action_hz=cfg.control.arm_action_hz,
    )
    logger.info("SmoothingAndSafetyManager initialized")

    # Initialize hands
    logger.info("Connecting to hands...")
    left_hand, right_hand = connect_hands(cfg.hands.left_serial, cfg.hands.right_serial)

    logger.info("Initializing hands...")
    initialize_hand(left_hand)
    left_hand.start()
    logger.info("Left hand initialized and started")

    initialize_hand(right_hand)
    right_hand.start()
    logger.info("Right hand initialized and started")

    # Reset hands to initial position
    logger.info("Resetting hands to initial position...")
    left_hand.set_joint_position([0.0] * HAND_JOINT_COUNT, cfg.hands.interpolate)
    right_hand.set_joint_position([0.0] * HAND_JOINT_COUNT, cfg.hands.interpolate)
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
            "action_hz": cfg.control.arm_action_hz,
            "hand_interpolate": cfg.hands.interpolate,
        },
        daemon=True,
    )
    full_robot_action_thread.start()
    logger.info(f"Full robot action thread started at {cfg.control.arm_action_hz} Hz")

    # Move robot to initial position - CRITICAL for teleop_pub retargeting to work correctly
    assert (
        0.3 / cfg.control.command_hz < 0.05
    )  # Sanity check to ensure the robot won't move too fast
    initialization_planner = InitializationCollisionPlanner(
        pin_full_robot_wrapper=pin_full_robot_wrapper,
        assemble_qpos=assemble_qpos,
        disassemble_qpos=disassemble_qpos,
        max_edge_joint_step=0.3 / cfg.control.command_hz,  # 0.3 rad/s max speed
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
            "left_arm": default_joint_pos["left_arm"],
            "right_arm": default_joint_pos["right_arm"],
            "left_hand": np.zeros((HAND_JOINT_COUNT,)),
            "right_hand": np.zeros((HAND_JOINT_COUNT,)),
            "head": default_joint_pos["head"],
            "torso": default_joint_pos["torso"],
        },
        hardware_lock=hardware_lock,
        command_hz=cfg.control.command_hz,
        dof_error_tolerance=cfg.control.reset_dof_err_tol,
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
            cfg.tactile.fetch_hz,
        ),
        daemon=True,
    )
    tactile_thread.start()
    logger.info("Tactile fetching thread started")

    vla_context = zmq.Context()
    vla_socket = vla_context.socket(zmq.REQ)
    vla_socket.connect(inf.server_address)
    logger.info(f"Connected to inference server at {inf.server_address}")

    # Target image size for camera capture (before vision_transform resizes)
    # We capture at the configured image size; vision_transform resizes to model resolution
    target_capture_size = (cfg.cameras.image_height, cfg.cameras.image_width)
    logger.info(f"Camera capture size: {target_capture_size}")

    # Wait for user to start
    input("\nPress [Enter] to start policy evaluation...")

    logger.info("\n" + "=" * 50)
    logger.info("BACKGROUND CONTROLS ACTIVE:")
    logger.info("  Press 'p' anytime to PAUSE/RESUME execution")
    logger.info("  Press 'r' anytime to RESET current trajectory")
    logger.info("  Press 'q' anytime to QUIT the entire program")
    logger.info("=" * 50 + "\n")
    key_ctrl = KeyController()

    # Main control loop
    logger.info("=" * 50)
    logger.info("POLICY EVALUATION STARTED")
    logger.info("=" * 50)
    # Synchronize sending new
    limiter = RateLimiter(
        frequency=cfg.control.command_hz, name="eval_policy_control_loop", warn=True
    )

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
                    "left_arm": default_joint_pos["left_arm"],
                    "right_arm": default_joint_pos["right_arm"],
                    "left_hand": np.zeros((HAND_JOINT_COUNT,)),
                    "right_hand": np.zeros((HAND_JOINT_COUNT,)),
                    "head": default_joint_pos["head"],
                    "torso": default_joint_pos["torso"],
                },
                hardware_lock=hardware_lock,
                command_hz=cfg.control.command_hz,
                dof_error_tolerance=cfg.control.reset_dof_err_tol,
                hold_time_s=0.5,
            )
            time.sleep(1)

            # Wait for user to start
            key_ctrl.suspend()
            start_cmd = (
                input("\nPress [Enter] to start policy evaluation (or 'q' to quit)...")
                .strip()
                .lower()
            )
            key_ctrl.resume()
            if start_cmd == "q" or key_ctrl.quit_requested:
                break

            logger.info("=" * 50)
            logger.info("POLICY EVALUATION STARTED")
            logger.info("=" * 50)
            limiter = RateLimiter(
                frequency=cfg.control.command_hz, name="eval_policy_control_loop", warn=True
            )

            # --- Inner execution loop ---
            for step in range(inf.max_steps):
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
                proprio = get_current_pose(
                    dexmate_bimanual_robot,
                    pink_ik_solver,
                    hardware_lock,
                    left_hand,
                    right_hand,
                    dual_arm,
                )

                images = get_all_camera_images(
                    dexmate_bimanual_robot,
                    hardware_lock,
                    wrist_cam_receiver,
                    head_cam_receiver,
                    target_capture_size,
                    inf.head_crop_box,
                )
                if inf.save_debug_images:
                    Image.fromarray(images["head"]).save(f"head_{step}.jpg")
                    Image.fromarray(images["right_wrist"]).save(f"right_wrist_{step}.jpg")
                    Image.fromarray(images["left_wrist"]).save(f"left_wrist_{step}.jpg")
                with tactile_buf_lock:
                    if dual_arm:
                        tactile_f6 = np.concatenate(
                            [left_tactile_buffers["f6"], right_tactile_buffers["f6"]], axis=0
                        )
                        tactile_deform = np.concatenate(
                            [left_tactile_buffers["deform"], right_tactile_buffers["deform"]],
                            axis=0,
                        )
                    else:
                        tactile_f6 = right_tactile_buffers["f6"].copy()
                        tactile_deform = right_tactile_buffers["deform"].copy()

                # Visualize inputs before JPEG encoding overwrites the images dict
                if inf.show_live_viz:
                    visualize_inference_inputs(
                        step=step,
                        head_img=images["head"],
                        right_wrist_img=images["right_wrist"],
                        left_wrist_img=images.get("left_wrist") if dual_arm else None,
                        tactile_deform=tactile_deform,
                        tactile_f6=tactile_f6,
                        dual_arm=dual_arm,
                    )

                for key, rgb_img in images.items():
                    bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
                    frame = cv2.imencode(".jpg", bgr_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])[1]
                    images[key] = frame.tobytes()

                # 3. VLA Inference (slow tick — full payload).
                # When inf.use_tactile_refine is on, mode='slow_and_fast' tells the
                # server to (a) run forward_flow_action_only → cache KV, and
                # (b) immediately run the first tactile residual refinement.
                # Subsequent in-chunk refinements (steps 4/8/12) reuse that KV.
                payload = {
                    "mode": "slow_and_fast" if inf.use_tactile_refine else "slow",
                    "task_description": task_description,
                    "image_head": images["head"],
                    "image_wrist_right": images["right_wrist"],
                    "state_fast": proprio,
                    "state_slow": proprio,
                    "tactile_f6": tactile_f6,
                    "tactile_deform": tactile_deform,
                }
                if dual_arm:
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
                assert response.get("status") == "success"
                action = np.array(response.get("actions"))
                # print(f"Received raw action from VLA: {action.shape}")
                # input("Press [Enter] to execute this action chunk...")
                if dual_arm:
                    assert action.shape in [(inf.chunk_size, 58), (inf.chunk_size, 62)], (
                        f"Unexpected action shape: {action.shape}"
                    )
                else:
                    assert action.shape in [(inf.chunk_size, 29), (inf.chunk_size, 31)], (
                        f"Unexpected action shape: {action.shape}"
                    )
                print(f"Received action chunk of shape {action.shape} from VLA")
                print(f"Step {step}")

                # ── ACT temporal aggregation: seed buffer with this slow chunk.
                # Each outer iteration starts a fresh window covering global
                # steps [chunk_start_step, chunk_start_step + inf.chunk_size).
                chunk_start_step = step * inf.execute_steps_per_chunk
                chunk_buffer = []
                if inf.use_temporal_aggregation:
                    chunk_buffer.append((chunk_start_step, action.copy()))

                with hardware_lock:
                    initial_arm_joint_pos_dict = dexmate_bimanual_robot.get_joint_pos_dict(
                        component=["left_arm", "right_arm"]
                    )
                    initial_left_q = np.array(
                        [
                            initial_arm_joint_pos_dict[name]
                            for name in DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES["left_arm"]
                        ]
                    )
                    initial_right_q = np.array(
                        [
                            initial_arm_joint_pos_dict[name]
                            for name in DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES["right_arm"]
                        ]
                    )
                    initial_right_hand_q = np.array(
                        right_hand.get_states().angles, dtype=np.float64
                    )
                    if dual_arm:
                        initial_left_hand_q = np.array(
                            left_hand.get_states().angles, dtype=np.float64
                        )

                fk_res_initial = pink_ik_solver.fk(
                    frames=["L_ee", "R_ee"],
                    joint_pos_by_component={
                        "left_arm": initial_left_q,
                        "right_arm": initial_right_q,
                    },
                )
                initial_left_ee_pose = fk_res_initial["L_ee"]
                initial_right_ee_pose = fk_res_initial["R_ee"]

                ik_warmstart_right_q = initial_right_q.copy()
                if dual_arm:
                    ik_warmstart_left_q = initial_left_q.copy()

                # ── Chunk execution: delta-EEF control mode (the only supported one) ──
                # Each action step holds, per arm, a delta EEF pose relative to the
                # EEF pose at chunk start (3 local xyz + 6 rot6d) plus 22 absolute
                # hand joints: (chunk_size, 62) dual-arm, (chunk_size, 31) single-arm.
                # Deltas resolve to joint targets via differential IK below. Other
                # control modes (absolute EEF / joint space) are straightforward
                # variations of this loop.
                initial_pos = initial_right_ee_pose.translation
                initial_R = initial_right_ee_pose.rotation
                if dual_arm:
                    initial_left_pos = initial_left_ee_pose.translation
                    initial_left_R = initial_left_ee_pose.rotation

                for action_idx in range(inf.execute_steps_per_chunk):
                    if key_ctrl.quit_requested or key_ctrl.reset_requested:
                        logger.info("Hotkey detected mid-chunk, aborting action execution.")
                        break
                    if key_ctrl.paused:
                        with action_buf_lock:
                            action_buffer["left_arm"] = None
                            action_buffer["right_arm"] = None
                            action_buffer["left_hand"] = None
                            action_buffer["right_hand"] = None
                        while (
                            key_ctrl.paused
                            and not key_ctrl.quit_requested
                            and not key_ctrl.reset_requested
                        ):
                            time.sleep(0.1)
                        if key_ctrl.quit_requested or key_ctrl.reset_requested:
                            break

                    # ── Async tactile refinement at tuple(inf.refine_offsets) within the chunk ──
                    # Read the most recent tactile (background thread keeps it
                    # fresh at cfg.tactile.fetch_hz), send a `mode='fast'` payload,
                    # and append the refined chunk to chunk_buffer for ACT
                    # temporal aggregation (or splice into `action` if that
                    # mode is off).  ZMQ REP is single-threaded server-side so
                    # this call blocks until the previous slow inference is
                    # done — the "wait until refinement finishes" guarantee.
                    if inf.use_tactile_refine and action_idx in tuple(inf.refine_offsets):
                        with tactile_buf_lock:
                            if dual_arm:
                                _tac_f6 = np.concatenate(
                                    [left_tactile_buffers["f6"], right_tactile_buffers["f6"]],
                                    axis=0,
                                )
                                _tac_def = np.concatenate(
                                    [
                                        left_tactile_buffers["deform"],
                                        right_tactile_buffers["deform"],
                                    ],
                                    axis=0,
                                )
                            else:
                                _tac_f6 = right_tactile_buffers["f6"].copy()
                                _tac_def = right_tactile_buffers["deform"].copy()
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
                                    if inf.use_temporal_aggregation:
                                        # Append to buffer; aggregation will
                                        # blend with previous chunks below.
                                        chunk_buffer.append((chunk_start_step, refined.copy()))
                                    else:
                                        # Legacy splice — replaces remainder.
                                        action[action_idx:] = refined[action_idx:]
                                    print(
                                        f"  [refine @{action_idx}] chunk_id="
                                        f"{fast_resp.get('chunk_id')}, "
                                        f"latency={fast_resp.get('latency_ms', 0):.1f} ms"
                                    )
                                else:
                                    logger.warning(
                                        f"refine response shape mismatch: "
                                        f"{refined.shape} vs {action.shape}; skipping"
                                    )

                    # ── Pick the action to command this step ──
                    # Under inf.use_temporal_aggregation we exp-weight-average all
                    # chunks in chunk_buffer that cover this global step.  All
                    # chunks share chunk_start_step so the average is over
                    # action_idx of {slow chunk, refine@4 chunk, refine@8, …}
                    # restricted to entries where action_idx ≥ their refine
                    # point (which always holds here since we're past it).
                    global_step = chunk_start_step + action_idx
                    if inf.use_temporal_aggregation:
                        agg = aggregate_chunks(chunk_buffer, global_step, k=inf.temporal_agg_k)
                        current_action = agg if agg is not None else action[action_idx]
                    else:
                        current_action = action[action_idx]

                    if dual_arm:
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
                                "R_ee": target_right_ee_pose,
                            },
                            arm_initial_joint_pos={
                                "left_arm": ik_warmstart_left_q,
                                "right_arm": ik_warmstart_right_q,
                            },
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
                                "R_ee": target_right_ee_pose,
                            },
                            arm_initial_joint_pos={
                                "left_arm": initial_left_q,
                                "right_arm": ik_warmstart_right_q,
                            },
                        )
                        target_right_q = ik_res["right_arm"].copy()
                        ik_warmstart_right_q = target_right_q

                        with action_buf_lock:
                            action_buffer["left_arm"] = default_joint_pos["left_arm"]
                            action_buffer["right_arm"] = target_right_q
                            action_buffer["left_hand"] = np.zeros((22,))
                            action_buffer["right_hand"] = target_hand

                    limiter.sleep()

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


if __name__ == "__main__":
    tyro.cli(main)
