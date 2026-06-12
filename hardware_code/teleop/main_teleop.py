"""main_teleop.py: Main teleop control loop with episode workflow.

- Reads targets from an in-process TeleopTargetSource (Vive + Manus retargeting)
- Connects to robot and hands hardware
- Reads actual state from hardware
- Sends commands to robot arms and hands
- Saves data to HDF5 file with episode-based workflow.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import select
import termios
import threading
import time
import tty
from enum import Enum, auto
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import tyro
from data_writer import DataWriter, find_last_episode_index, move_episode_files
from dexcontrol.robot import Robot
from loguru import logger
from loop_rate_limiters import RateLimiter

# SharpaWaveSDK imports for hand control
from sharpa import (
    ControlMode,
    ControlSource,
    HandSide,
    SharpaWave,
    SharpaWaveManager,
    setup_cpp_logging,
)

from camera.head_camera_receiver import HeadCameraReceiver
from camera.wrist_camera_receiver import WristCameraReceiver
from teleop.arm_hand_control import (
    ArmIKManager,
    InitializationCollisionPlanner,
    SmoothingAndSafetyManager,
    full_robot_action_loop,
    move_robot_to_position_safe,
)
from teleop.robot_descriptions import (
    DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES,
    add_env_obstacles,
    build_full_robot,
)

# NOTE: SDK console logging is disabled because loose hand connections cause frequent
# disconnect/reconnect info messages that block user input prompts in the terminal;
# logs go to /tmp/sharpa_wave.log instead.
setup_cpp_logging("/tmp/sharpa_wave.log", console_log=False)

from teleop_targets import HAND_JOINT_COUNT, TeleopTargetSource

from teleop.config import DEFAULT_CONFIG_PATH, TeleopConfig, load_config
from teleop.ik_utils import PinkLocalIK

# Tactile configuration (coupled to the recorded data format, not site-specific)
TACTILE_TYPES_TO_CAPTURE = ["F6", "DEFORM", "RAW"]

TACTILE_BUFFER_SHAPES = {
    "f6": ((5, 6), np.float32),
    "deform": ((5, 240, 240), np.uint8),
    "raw": ((5, 240, 320), np.uint8),
}

# Tactile finger ordering in the (5, ...) tactile arrays (see _fetch_tactile_arrays)
FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

# =============================================================================
# Live teleop visualization configuration
# =============================================================================
# When True, show a live OpenCV window during episodes with the right wrist
# camera, plus the right-hand DEFORM map and estimated F6 (as bars) for the
# fingers listed in VIZ_FINGER_NAMES.
SHOW_LIVE_VIZ = True
VIZ_WINDOW_NAME = "Right Wrist + Tactile (live)"
# Which right-hand fingers to display. Extensible: add/remove names from
# FINGER_NAMES, e.g. ["Thumb", "Index", "Middle"].
VIZ_FINGER_NAMES = ["Thumb", "Index"]
# Also show the RAW tactile image (grayscale) under each finger's DEFORM map.
VIZ_SHOW_RAW = True
# Rendering/display runs in its own thread at this rate so it never slows the
# 30 Hz control loop. Keep this low on slow machines.
VIZ_HZ = 15.0
VIZ_DEFORM_SIZE = 160  # px, per-finger DEFORM map display size (square)
VIZ_F6_PANEL_HEIGHT = 130  # px, height of the F6 bar chart under each finger
VIZ_F6_RANGE = 5.0  # F6 bars are clamped to +/- this range
# Labels for the 6 F6 components (assumed 3-axis force + 3-axis torque).
VIZ_F6_COMPONENT_LABELS = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]
VIZ_CAM_HEIGHT = 360  # px, display height for the right wrist camera
# Default window magnification: the window opens at this multiple of the
# natural frame size (the window manager scales the small frame up, so this is
# cheap). You can still freely drag-resize the window afterwards.
VIZ_DISPLAY_SCALE = 2.5


def _render_f6_bars(
    f6_vec: np.ndarray,
    width: int,
    height: int,
    value_range: float = VIZ_F6_RANGE,
    labels: list = VIZ_F6_COMPONENT_LABELS,
) -> np.ndarray:
    """Render a 6-component F6 vector as a vertical bar chart (BGR image).

    Bars grow up (positive) or down (negative) from a central zero line and are
    clamped to +/- value_range.
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)
    n = len(f6_vec)
    if n == 0:
        return img
    mid_y = height // 2
    cv2.line(img, (0, mid_y), (width, mid_y), (80, 80, 80), 1)  # zero line
    slot_w = width / n
    bar_w = max(2, int(slot_w * 0.6))
    max_bar = mid_y - 14  # leave room for component labels at the bottom
    for i in range(n):
        frac = float(np.clip(f6_vec[i] / value_range, -1.0, 1.0))
        bar_px = int(abs(frac) * max_bar)
        cx = int((i + 0.5) * slot_w)
        x0, x1 = cx - bar_w // 2, cx + bar_w // 2
        if frac >= 0:
            y0, y1, color = mid_y - bar_px, mid_y, (0, 200, 0)  # green up
        else:
            y0, y1, color = mid_y, mid_y + bar_px, (0, 140, 255)  # orange down
        cv2.rectangle(img, (x0, y0), (x1, y1), color, -1)
        if i < len(labels):
            cv2.putText(
                img,
                labels[i],
                (cx - 9, height - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (180, 180, 180),
                1,
            )
    return img


def _build_finger_panel(
    deform_img: np.ndarray,
    f6_vec: np.ndarray,
    finger_name: str,
    raw_img: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build the vertical panel for one finger.

    DEFORM colormap on top, optional RAW grayscale in the middle, F6 bars at
    the bottom.
    """
    d_resized = cv2.resize(deform_img, (VIZ_DEFORM_SIZE, VIZ_DEFORM_SIZE))
    d_color = cv2.applyColorMap(d_resized, cv2.COLORMAP_JET)
    cv2.putText(d_color, finger_name, (3, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    f6_norm = float(np.linalg.norm(f6_vec))
    cv2.putText(
        d_color,
        f"|F6| {f6_norm:.1f}",
        (3, VIZ_DEFORM_SIZE - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 0),
        1,
    )

    sections = [d_color]
    if raw_img is not None:
        r_resized = cv2.resize(raw_img, (VIZ_DEFORM_SIZE, VIZ_DEFORM_SIZE))
        r_color = cv2.cvtColor(r_resized, cv2.COLOR_GRAY2BGR)
        cv2.putText(r_color, "raw", (3, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        sections.append(r_color)

    sections.append(_render_f6_bars(f6_vec, VIZ_DEFORM_SIZE, VIZ_F6_PANEL_HEIGHT))
    return np.vstack(sections)


def _build_live_viz(
    right_wrist_image: Optional[np.ndarray],
    right_tactile: dict,
    finger_indices: list,
    freeze_right_hand: bool = False,
) -> Optional[np.ndarray]:
    """Compose the live teleop window.

    Right wrist camera on top, per-finger DEFORM (+ optional RAW) + F6 bars
    below. Returns a BGR image, or None if nothing to show. Draws a banner
    when the right hand is frozen.
    """
    parts = []

    if right_wrist_image is not None:
        cam = cv2.cvtColor(right_wrist_image, cv2.COLOR_RGB2BGR)
        scale = VIZ_CAM_HEIGHT / cam.shape[0]
        cam = cv2.resize(cam, (int(cam.shape[1] * scale), VIZ_CAM_HEIGHT))
        parts.append(cam)

    deform = right_tactile.get("deform")  # (5, 240, 240) uint8
    f6 = right_tactile.get("f6")  # (5, 6) float32
    raw = right_tactile.get("raw")  # (5, 240, 320) uint8 or None
    if deform is not None and f6 is not None and finger_indices:
        panels = [
            _build_finger_panel(
                deform[idx],
                f6[idx],
                FINGER_NAMES[idx],
                raw_img=raw[idx] if (raw is not None and VIZ_SHOW_RAW) else None,
            )
            for idx in finger_indices
        ]
        parts.append(np.hstack(panels))

    if not parts:
        return None

    target_w = max(p.shape[1] for p in parts)
    padded = []
    for p in parts:
        if p.shape[1] < target_w:
            pad = np.zeros((p.shape[0], target_w - p.shape[1], 3), dtype=np.uint8)
            p = np.hstack([p, pad])
        padded.append(p)
    frame = np.vstack(padded)

    if freeze_right_hand:
        banner_h = 36
        banner = np.zeros((banner_h, frame.shape[1], 3), dtype=np.uint8)
        banner[:] = (0, 0, 160)  # dark red
        cv2.putText(
            banner,
            "RIGHT HAND FROZEN (press 'a' to release)",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        frame = np.vstack([banner, frame])
    return frame


def _default_viz_size(n_fingers: int, image_width: int, image_height: int) -> tuple:
    """Natural (height, width) of the composed live frame.

    Used to size the window and the idle placeholder before any real frame
    exists.
    """
    panel_h = VIZ_DEFORM_SIZE + (VIZ_DEFORM_SIZE if VIZ_SHOW_RAW else 0) + VIZ_F6_PANEL_HEIGHT
    panel_w = VIZ_DEFORM_SIZE * max(1, n_fingers)
    cam_w = int(round(VIZ_CAM_HEIGHT * image_width / image_height))
    return (VIZ_CAM_HEIGHT + panel_h, max(cam_w, panel_w))


def _make_idle_frame(height: int, width: int) -> np.ndarray:
    """Black placeholder frame shown while not in an episode.

    Keeps the cv2 window rendering and draggable/resizable between episodes.
    """
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(
        frame, "idle - not in episode", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 1
    )
    return frame


def _input_live(prompt: str, viz_enabled: bool, idle_frame: Optional[np.ndarray]) -> str:
    """Drop-in replacement for input() that keeps the cv2 window alive.

    While waiting for a line on stdin it pumps imshow/waitKey with ``idle_frame``
    (a black placeholder) at ~VIZ_HZ so the window stays responsive and the user
    can resize/drag it between episodes. Falls back to plain input() when viz is
    disabled. Returns the entered line without the trailing newline (like input).
    """
    if not viz_enabled or idle_frame is None:
        return input(prompt)
    sys.stdout.write(prompt)
    sys.stdout.flush()
    poll_timeout = max(0.001, 1.0 / VIZ_HZ if VIZ_HZ > 0 else 0.03)
    while True:
        cv2.imshow(VIZ_WINDOW_NAME, idle_frame)
        cv2.waitKey(1)
        readable, _, _ = select.select([sys.stdin], [], [], poll_timeout)
        if readable:
            return sys.stdin.readline().rstrip("\n")


def _live_viz_loop(
    terminate_event: threading.Event,
    viz_lock: threading.Lock,
    viz_shared: dict,
    finger_indices: list,
    hz: float = VIZ_HZ,
):
    """Render the live teleop frame in a dedicated thread.

    The heavy cv2 rendering (resize/colormap/cvtColor, which release the GIL)
    happens here so the 30 Hz control loop is never blocked. NO GUI calls are
    made here: OpenCV's HighGUI is not thread-safe, so imshow/waitKey must run
    on the main thread. This thread only publishes the latest rendered BGR frame
    into ``viz_shared["frame"]`` for the main thread to display.
    """
    limiter = RateLimiter(frequency=hz, name="live_viz_limiter", warn=False)
    logger.info(f"[Viz] Render thread started at {hz} Hz")
    while not terminate_event.is_set():
        limiter.sleep()
        with viz_lock:
            wrist = viz_shared.get("wrist")
            tactile = viz_shared.get("tactile")
            freeze_right_hand = viz_shared.get("freeze_right_hand", False)
        if tactile is None and wrist is None:
            continue
        try:
            frame = _build_live_viz(
                wrist, tactile or {}, finger_indices, freeze_right_hand=freeze_right_hand
            )
        except Exception as e:
            logger.warning(f"[Viz] Render error: {e}")
            continue
        if frame is not None:
            with viz_lock:
                viz_shared["frame"] = frame
    logger.info("[Viz] Render thread stopped")


class EpisodeKeyListener:
    """Per-episode keyboard listener.

    One instance is created and started at the start of each episode, then
    stopped when the episode ends.

    Hotkeys (read one character at a time in cbreak mode, so no Enter needed for
    the toggles):
      - Enter      -> stop the current episode (sets ``episode_done``)
      - 'a' / 'A'  -> toggle freezing the RIGHT-hand target joint positions

    When ``freeze_right_hand`` is True, the control loop holds the right-hand
    target constant (see usage in the IN_EPISODE state) so the right hand stays
    put while the rest of the body keeps tracking.

    Only runs during an episode (when there are no line-buffered input()/_input_live
    prompts competing for stdin), and restores the terminal on stop. Falls back
    to plain Enter-to-stop if stdin is not a TTY.
    """

    def __init__(self, episode_done: threading.Event):
        self._episode_done = episode_done
        self._stop = threading.Event()
        self.freeze_right_hand = False
        self._thread = threading.Thread(target=self._listen, daemon=True)

    def start(self):
        self._thread.start()

    def _listen(self):
        if not sys.stdin.isatty():
            logger.warning(
                "[KeyListener] stdin is not a TTY; only Enter-to-stop "
                "is available (right-hand freeze hotkey disabled)."
            )
            sys.stdin.readline()
            self._episode_done.set()
            return

        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            while not self._stop.is_set():
                ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                if not ready:
                    continue
                ch = sys.stdin.read(1)
                if ch in ("\n", "\r"):
                    logger.info("[KeyListener] Enter -> stopping episode")
                    self._episode_done.set()
                    break
                elif ch in ("a", "A"):
                    self.freeze_right_hand = not self.freeze_right_hand
                    logger.info(
                        f"[KeyListener] Right-hand freeze "
                        f"{'ON' if self.freeze_right_hand else 'OFF'}"
                    )
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=1.0)


class TeleopState(Enum):
    INIT = auto()  # Initial state, before any episode starts, starts with self.episode_done.is_set() == True, self.episode_start.is_set() == False and ends after self._prepare_for_new_episode()
    WAITING_FOR_START = auto()  # Waiting for EPISODE_START signal, starts with self.episode_done.is_set() == False, self.episode_start.is_set() == False; ends when self.episode_start.is_set() == True
    IN_EPISODE = auto()  # In an episode, starts with self.episode_done.is_set() == False, self.episode_start.is_set() == True; ends when self.episode_done.is_set() == True and also manually set self.episode_start.is_set() == False
    TERMINATE = auto()  # Termination state indicating the teleoperation system is shutting down.


# =============================================================================
# Hand Connection and Initialization
# =============================================================================


def connect_hands(left_serial: str, right_serial: str):
    """Connect to left and right hands via SharpaWaveSDK."""
    manager = SharpaWaveManager.get_instance()
    time.sleep(3)  # Wait for device discovery

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
    for i in range(10):
        if hand.is_hand_ready():
            break
        logger.warning(f"Hand not ready, attempt {i + 1}/10...")
        time.sleep(0.5)
    else:
        raise RuntimeError("Hand never became ready")

    for attempt in range(5):
        error = hand.set_control_mode(ControlMode.POSITION)
        if error.code == 0:
            break
        logger.warning(f"set_control_mode failed (attempt {attempt + 1}/5): {error.message}")
        time.sleep(2)
    else:
        raise RuntimeError(f"Failed to set control mode after 5 attempts: {error.message}")

    error = hand.set_speed_coeff(0.3)
    if error.code != 0:
        raise RuntimeError(f"Failed to set speed coeff: {error.message}")

    error = hand.set_current_coeff(0.6)
    if error.code != 0:
        raise RuntimeError(f"Failed to set current coeff: {error.message}")

    error = hand.set_control_source(ControlSource.SDK)
    if error.code != 0:
        raise RuntimeError(f"Failed to set control source: {error.message}")


# =============================================================================
# Tactile Data Fetching
# =============================================================================


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
    raw_arr = np.zeros((5, 240, 320), dtype=np.uint8) if "RAW" in tactile_types else None

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

        if raw_arr is not None:
            raw = content.get("RAW")
            if raw is not None:
                arr = np.asarray(raw, dtype=np.uint8)
                if arr.size == 76800:
                    raw_arr[i, :, :] = arr.reshape(240, 320)

    return {
        "f6": f6_arr,
        "deform": deform_arr,
        "raw": raw_arr,
    }


def _tactile_fetch_loop(
    terminate_event: threading.Event,
    buf_lock: threading.Lock,
    left_tactile_buffers: dict,
    right_tactile_buffers: dict,
    left_hand: SharpaWave,
    right_hand: SharpaWave,
    fetch_hz: float = 30.0,
):
    """Separate thread to fetch tactile data without blocking hand control.

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
                for ttype_lower in ["f6", "deform", "raw"]:
                    if left_tactile_data[ttype_lower] is not None:
                        np.copyto(left_tactile_buffers[ttype_lower], left_tactile_data[ttype_lower])
                    if right_tactile_data[ttype_lower] is not None:
                        np.copyto(
                            right_tactile_buffers[ttype_lower], right_tactile_data[ttype_lower]
                        )
        except Exception as e:
            # Don't crash the thread on errors, just skip this frame
            logger.warning(f"[Tactile] Fetch error: {e}")

    logger.info("[Tactile] Fetching thread stopped")


def main(
    config: str = str(DEFAULT_CONFIG_PATH),
    data_dir: Optional[str] = None,
    table_height: Optional[float] = None,
    save_data: bool = True,
    no_wrist_cam: bool = False,
    no_head_cam: bool = False,
):
    """Main teleop control loop with episode workflow.

    Ensure the Vive server and Manus glove client + retargeting publisher are running first.

    Args:
        config: Path to the YAML config file (see config/default.yaml and
            teleop/config.py for the schema). All site-specific settings
            (endpoints, serials, default poses, rates, environment) live there.
        data_dir: Override config's data.data_dir. Episodes will be saved here
            initially, then moved to success/ or failure/ subdirectories.
        table_height: Override config's environment.table_height (meters; use
            the actual measured height - 0.04 m to account for the table top
            thickness).
        save_data: Whether to save episode data
        no_wrist_cam: If True, disable wrist cameras (no recording or storing)
        no_head_cam: If True, disable head camera (no recording or storing)
    """
    logger.info("Starting main_teleop...")

    # Load config and apply CLI overrides
    cfg: TeleopConfig = load_config(config)
    logger.info(f"Loaded config from {config}")
    if data_dir is not None:
        cfg.data.data_dir = data_dir
    if table_height is not None:
        cfg.environment.table_height = table_height
    data_dir = cfg.data.data_dir
    default_joint_pos = cfg.robot.default_joint_pos

    # Resolve which right-hand fingers to show in the live viz window
    viz_finger_indices = []
    if SHOW_LIVE_VIZ:
        for name in VIZ_FINGER_NAMES:
            if name not in FINGER_NAMES:
                logger.warning(f"[Viz] Unknown finger name '{name}' in VIZ_FINGER_NAMES, skipping")
                continue
            viz_finger_indices.append(FINGER_NAMES.index(name))
        logger.info(
            f"[Viz] Live window enabled for right-hand fingers: "
            f"{[FINGER_NAMES[i] for i in viz_finger_indices]}"
        )

    # Convert to Path and ensure it exists
    data_dir_path = Path(data_dir)
    data_dir_path.mkdir(parents=True, exist_ok=True)

    # Create success and failure subdirectories
    (data_dir_path / "success").mkdir(parents=True, exist_ok=True)
    (data_dir_path / "failure").mkdir(parents=True, exist_ok=True)

    # Find last episode index
    last_episode_idx = find_last_episode_index(data_dir_path)
    next_episode_idx = last_episode_idx + 1
    logger.info(f"Last episode index: {last_episode_idx}, starting from: {next_episode_idx}")

    # Initialize the in-process teleop target source (Vive + Manus retargeting)
    target_source = TeleopTargetSource(
        vive_ip=cfg.vive.ip,
        vive_port=cfg.vive.port,
        vive_left_tracker_name=cfg.vive.left_tracker_name,
        vive_right_tracker_name=cfg.vive.right_tracker_name,
        vive_update_hz=cfg.vive.update_hz,
        vive_timeout_tol_s=cfg.vive.timeout_tol_s,
        hand_action_address=cfg.hand_action.address,
        update_hz=cfg.targets.update_hz,
        default_joint_pos=default_joint_pos,
    )
    target_source.start()

    dexmate_bimanual_robot = Robot()
    logger.info(f"Dexmate Bimanual Robot '{dexmate_bimanual_robot.robot_model}' initialized")

    # Initialize IK solver for computing current arm poses (FK) and for ArmCmdHandler
    pink_ik_solver = PinkLocalIK(default_joint_by_component=default_joint_pos)
    logger.info("IK solver initialized for FK computations")

    # Initialize ArmIKManager and SmoothingAndSafetyManager for processing arm targets and sending commands
    arm_ik_manager = ArmIKManager(
        pink_ik_solver=pink_ik_solver,
        warmstart_with_actual=False,
    )
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
    logger.info("ArmIKManager and SmoothingAndSafetyManager initialized")

    # Initialize hands via SharpaWaveSDK
    logger.info("Connecting to hands...")
    left_hand, right_hand = connect_hands(cfg.hands.left_serial, cfg.hands.right_serial)

    logger.info("Initializing hands...")
    initialize_hand(left_hand)
    left_hand.start()
    logger.info("Left hand initialized and started")

    initialize_hand(right_hand)
    right_hand.start()
    logger.info("Right hand initialized and started")

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

    # Move robot to initial position - CRITICAL for retargeting initialization to work correctly
    assert (
        0.6 / cfg.control.command_hz < 0.05
    )  # Sanity check to ensure the robot won't move too fast
    initialization_planner = InitializationCollisionPlanner(
        pin_full_robot_wrapper=pin_full_robot_wrapper,
        assemble_qpos=assemble_qpos,
        disassemble_qpos=disassemble_qpos,
        max_edge_joint_step=0.6 / cfg.control.command_hz,  # 0.6 rad/s max speed
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

    # Initialize wrist cameras (if enabled)
    wrist_cam_receiver = None
    if not no_wrist_cam:
        logger.info("Initializing wrist cameras...")
        wrist_cam_receiver = WristCameraReceiver(
            im_h=cfg.cameras.image_height,
            im_w=cfg.cameras.image_width,
            sender_ip=cfg.cameras.wrist.sender_ip,
            ports=cfg.cameras.wrist.ports,
        )
        wrist_cam_receiver.start_receiving(timeout=10.0)
        logger.info("Wrist cameras started receiving")
    else:
        logger.info("Wrist cameras disabled (--no-wrist-cam)")

    # Initialize head camera (if enabled)
    head_cam_receiver = None
    if not no_head_cam:
        logger.info("Initializing head camera...")
        head_cam_receiver = HeadCameraReceiver(
            im_h=cfg.cameras.image_height,
            im_w=cfg.cameras.image_width,
            sender_ip=cfg.cameras.head.sender_ip,
            ports=cfg.cameras.head.ports,
        )
        head_cam_receiver.start_receiving(timeout=10.0)
        logger.info("Head camera started receiving")
    else:
        logger.info("Head camera disabled (--no-head-cam)")

    # Initialize tactile buffers
    logger.info("Initializing tactile buffers...")
    tactile_buf_lock = threading.Lock()
    left_tactile_buffers = {}
    right_tactile_buffers = {}

    for ttype_lower, (shape, dtype) in TACTILE_BUFFER_SHAPES.items():
        left_tactile_buffers[ttype_lower] = np.zeros(shape, dtype=dtype)
        right_tactile_buffers[ttype_lower] = np.zeros(shape, dtype=dtype)

    # Start tactile fetch thread
    tactile_terminate_event = threading.Event()
    tactile_thread = threading.Thread(
        target=_tactile_fetch_loop,
        args=(
            tactile_terminate_event,
            tactile_buf_lock,
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

    # Start live visualization thread (renders off the control loop; the main
    # thread does the actual imshow/waitKey since OpenCV GUI is not thread-safe)
    viz_lock = threading.Lock()
    viz_shared: dict = {"wrist": None, "tactile": None, "frame": None}
    viz_terminate_event = threading.Event()
    viz_thread: Optional[threading.Thread] = None
    last_viz_show_t = 0.0
    viz_show_interval = 1.0 / VIZ_HZ if VIZ_HZ > 0 else 0.0
    viz_idle_frame: Optional[np.ndarray] = None
    if SHOW_LIVE_VIZ and viz_finger_indices:
        viz_thread = threading.Thread(
            target=_live_viz_loop,
            args=(
                viz_terminate_event,
                viz_lock,
                viz_shared,
                viz_finger_indices,
                VIZ_HZ,
            ),
            daemon=True,
        )
        viz_thread.start()
        # Create the display window ONCE here on the main thread so it lives for
        # the whole session (not re-created per episode) and doesn't pop up
        # mid-episode to steal terminal keyboard focus. Click back to the
        # terminal once after it appears; subsequent imshow updates reuse this
        # window without re-grabbing focus.
        default_h, default_w = _default_viz_size(
            len(viz_finger_indices), cfg.cameras.image_width, cfg.cameras.image_height
        )
        viz_idle_frame = _make_idle_frame(default_h, default_w)
        cv2.namedWindow(VIZ_WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(
            VIZ_WINDOW_NAME, int(default_w * VIZ_DISPLAY_SCALE), int(default_h * VIZ_DISPLAY_SCALE)
        )
        cv2.imshow(VIZ_WINDOW_NAME, viz_idle_frame)
        cv2.waitKey(1)
        logger.info(
            f"Live viz thread started at {VIZ_HZ} Hz "
            f"(window {int(default_w * VIZ_DISPLAY_SCALE)}x"
            f"{int(default_h * VIZ_DISPLAY_SCALE)})"
        )

    # State machine
    state = TeleopState.INIT

    # Episode management
    episode_name: Optional[str] = None
    episode_idx: int = next_episode_idx
    data_writer: Optional[DataWriter] = None
    episode_done = threading.Event()
    episode_start_time: Optional[float] = None
    key_listener: Optional[EpisodeKeyListener] = None
    # Holds the right-hand target captured when freeze is toggled on (None when
    # not frozen). Reset each episode.
    frozen_right_hand_target: Optional[np.ndarray] = None

    # Main control loop
    limiter = RateLimiter(
        frequency=cfg.control.command_hz, name="main_teleop_control_loop", warn=True
    )

    try:
        while True:
            limiter.sleep()

            if state == TeleopState.INIT:
                logger.info("Initialization complete. Ready to start episodes.")
                state = TeleopState.WAITING_FOR_START

            elif state == TeleopState.WAITING_FOR_START:
                # Step 1: Ask user to press enter to reset to initial pose
                _input_live(
                    "\nPress [Enter] to reset robot to initial pose...",
                    viz_thread is not None,
                    viz_idle_frame,
                )

                with action_buf_lock:
                    action_buffer["left_arm"] = None
                    action_buffer["right_arm"] = None
                    action_buffer["left_hand"] = None
                    action_buffer["right_hand"] = None
                logger.info("Arm action buffer cleared (moving to initial position)")

                # Reset robot arms/hands to initial position
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

                time.sleep(0.5)
                # Step 1: Ask user to press enter to save to initial pose
                _input_live(
                    "\nPress [Enter] to save robot to initial pose...",
                    viz_thread is not None,
                    viz_idle_frame,
                )

                # Set in_episode flag to True, which enables data stream check
                target_source.set_in_episode()

                target_source.request_retargeting_reinit()
                time.sleep(0.5)

                # Reset ArmIKManager and SmoothingAndSafetyManager internal state
                arm_ik_manager.reset()
                smoothing_and_safety_manager.reset()
                logger.info("ArmIKManager and SmoothingAndSafetyManager state reset")

                logger.info("Arms and hands reset to initial position")

                # Step 2: Ask user to press enter to start recording
                _input_live(
                    "Press [Enter] to start recording new episode...",
                    viz_thread is not None,
                    viz_idle_frame,
                )

                # Generate episode name with zero-padded index
                episode_name = f"episode_{episode_idx:04d}"
                logger.info(f"Starting episode: {episode_name}")

                if save_data:
                    data_writer = DataWriter(
                        episode_name=episode_name,
                        save_dir=data_dir_path,
                        command_hz=cfg.control.command_hz,
                        no_wrist_cam=no_wrist_cam,
                        no_head_cam=no_head_cam,
                        tactile_maps_as_video=cfg.data.tactile_maps_as_video,
                        tactile_video_codec=cfg.data.tactile_video_codec,
                    )
                    data_writer.start()
                    logger.info(
                        f"Episode data will be saved to: {data_dir_path / episode_name / f'{episode_name}.h5'}"
                    )

                episode_done.clear()
                frozen_right_hand_target = None  # start each episode unfrozen
                key_listener = EpisodeKeyListener(episode_done)
                key_listener.start()
                logger.info(
                    "[KeyListener] Episode hotkeys active: [Enter]=stop, "
                    "[a]=toggle right-hand freeze"
                )

                episode_start_time = time.perf_counter()

                logger.info("=" * 50)
                logger.info(f"EPISODE {episode_name} STARTED - Press [Enter] to stop recording")
                logger.info("=" * 50)
                state = TeleopState.IN_EPISODE

            elif state == TeleopState.IN_EPISODE:
                # Read the latest targets from the in-process source
                assert target_source.in_episode, "in_episode should be True during episode"
                control_timestamp = time.perf_counter()

                if target_source.invalid:
                    logger.error("Teleop target source became invalid (data dropout)")
                    break  # Exit teleop immediately

                target_data = target_source.get_targets()
                assert target_data is not None, (
                    "No targets available during episode (retargeting not initialized?)"
                )

                # Get actual arm state from robot
                with hardware_lock:
                    curr_joint_pos_dict = dexmate_bimanual_robot.get_joint_pos_dict(
                        component=["left_arm", "right_arm"]
                    )
                left_arm_joint_positions = np.array(
                    [
                        curr_joint_pos_dict[name]
                        for name in DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES["left_arm"]
                    ]
                )
                right_arm_joint_positions = np.array(
                    [
                        curr_joint_pos_dict[name]
                        for name in DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES["right_arm"]
                    ]
                )

                # Compute current arm poses using FK
                curr_fk_res = pink_ik_solver.fk(
                    frames=["L_ee", "R_ee"],
                    joint_pos_by_component={
                        "left_arm": left_arm_joint_positions,
                        "right_arm": right_arm_joint_positions,
                    },
                )
                left_arm_current_pose = curr_fk_res["L_ee"].homogeneous.copy()
                right_arm_current_pose = curr_fk_res["R_ee"].homogeneous.copy()

                # Get actual hand state from hands
                with hardware_lock:
                    left_hand_joint_positions = np.array(
                        left_hand.get_states().angles, dtype=np.float64
                    )
                    right_hand_joint_positions = np.array(
                        right_hand.get_states().angles, dtype=np.float64
                    )

                # Get camera images
                # Head camera (via ZMQ receiver)
                head_image = None
                if head_cam_receiver is not None:
                    head_image = head_cam_receiver.get_data("HEAD_CAM").image_rgb

                # Wrist cameras (already capturing in background threads)
                left_wrist_image = None
                right_wrist_image = None
                if wrist_cam_receiver is not None:
                    left_wrist_image = wrist_cam_receiver.get_data("LEFT_WRIST").image_rgb
                    right_wrist_image = wrist_cam_receiver.get_data("RIGHT_WRIST").image_rgb

                # Get tactile data (copy from buffers with lock)
                left_tactile = None
                right_tactile = None
                with tactile_buf_lock:
                    left_tactile = {
                        "f6": left_tactile_buffers["f6"].copy(),
                        "deform": left_tactile_buffers["deform"].copy(),
                        "raw": left_tactile_buffers["raw"].copy(),
                    }
                    right_tactile = {
                        "f6": right_tactile_buffers["f6"].copy(),
                        "deform": right_tactile_buffers["deform"].copy(),
                        "raw": right_tactile_buffers["raw"].copy(),
                    }

                # Live visualization:
                #  1. Hand off latest frame data to the render thread (cheap
                #     reference swap; heavy rendering happens off this loop).
                #  2. Display the latest rendered frame here on the main thread
                #     (imshow/waitKey must be on the main thread), throttled to
                #     VIZ_HZ so it adds minimal load to the 30 Hz control loop.
                if viz_thread is not None:
                    with viz_lock:
                        viz_shared["wrist"] = right_wrist_image
                        viz_shared["tactile"] = {
                            "deform": right_tactile["deform"],
                            "f6": right_tactile["f6"],
                            "raw": right_tactile["raw"] if VIZ_SHOW_RAW else None,
                        }
                        viz_shared["freeze_right_hand"] = (
                            key_listener is not None and key_listener.freeze_right_hand
                        )
                    if control_timestamp - last_viz_show_t >= viz_show_interval:
                        with viz_lock:
                            viz_frame = viz_shared.get("frame")
                        if viz_frame is not None:
                            cv2.imshow(VIZ_WINDOW_NAME, viz_frame)
                            cv2.waitKey(1)
                        last_viz_show_t = control_timestamp

                # Process arm targets using ArmIKManager to get raw IK solutions
                absolute_target_poses = {
                    "L_ee": target_data.left_arm_target_pose,
                    "R_ee": target_data.right_arm_target_pose,
                }
                # Get raw IK solution (smoothing done in high-frequency action loop)
                ik_targets = arm_ik_manager.get_arm_action(
                    arm_absolute_target_poses=absolute_target_poses,
                    arm_current_joint_positions={
                        "left_arm": left_arm_joint_positions,
                        "right_arm": right_arm_joint_positions,
                    },
                )

                # Right-hand target: when freeze is toggled on (hotkey 'a'),
                # hold the target captured at the moment of freezing so the right
                # hand stays put while the rest of the body keeps tracking.
                right_hand_target = target_data.right_hand_target_joint_positions.copy()
                if key_listener is not None and key_listener.freeze_right_hand:
                    if frozen_right_hand_target is None:
                        frozen_right_hand_target = right_hand_target.copy()
                    right_hand_target = frozen_right_hand_target.copy()
                else:
                    frozen_right_hand_target = None

                # Write IK solution to buffer (high-frequency thread will smooth and send commands)
                with action_buf_lock:
                    action_buffer["left_arm"] = ik_targets["left_arm"].copy()
                    action_buffer["right_arm"] = ik_targets["right_arm"].copy()
                    action_buffer["left_hand"] = target_data.left_hand_target_joint_positions.copy()
                    action_buffer["right_hand"] = right_hand_target.copy()

                # Record data
                if data_writer is not None:
                    frame_data = {
                        # Timestamps
                        "timestamp": control_timestamp,
                        "vive_timestamp": target_data.vive_timestamp,
                        "hand_timestamp": target_data.hand_timestamp,
                        # Vive poses
                        "left_vive_pose": target_data.left_vive_pose,
                        "right_vive_pose": target_data.right_vive_pose,
                        # Arm targets (absolute poses from TeleopTargetSource, raw IK solution from ArmIKManager)
                        "left_arm_target_dofs": ik_targets["left_arm"],
                        "right_arm_target_dofs": ik_targets["right_arm"],
                        "left_arm_target_pose": target_data.left_arm_target_pose,
                        "right_arm_target_pose": target_data.right_arm_target_pose,
                        # Arm actual state
                        "left_arm_joint_positions": left_arm_joint_positions,
                        "right_arm_joint_positions": right_arm_joint_positions,
                        "left_arm_current_pose": left_arm_current_pose,
                        "right_arm_current_pose": right_arm_current_pose,
                        # Hand targets (right hand reflects the value actually
                        # sent, i.e. the frozen target while freeze is active)
                        "left_hand_target_joint_positions": target_data.left_hand_target_joint_positions,
                        "right_hand_target_joint_positions": right_hand_target,
                        # Hand actual state
                        "left_hand_joint_positions": left_hand_joint_positions,
                        "right_hand_joint_positions": right_hand_joint_positions,
                        # Camera images
                        "head_image": head_image,
                        "left_wrist_image": left_wrist_image,
                        "right_wrist_image": right_wrist_image,
                        # Tactile data
                        "left_tactile": left_tactile,
                        "right_tactile": right_tactile,
                    }
                    data_writer.queue_frame(frame_data)

                # Check if episode should end
                if episode_done.is_set():
                    episode_end_time = time.perf_counter()
                    episode_duration = episode_end_time - episode_start_time

                    # Clear arm action buffer to stop high-frequency commands
                    with action_buf_lock:
                        action_buffer["left_arm"] = None
                        action_buffer["right_arm"] = None
                        action_buffer["left_hand"] = None
                        action_buffer["right_hand"] = None
                    logger.info("Arm action buffer cleared (episode ending)")

                    # Stop the keyboard listener (restores the terminal to line
                    # mode before the success/failure prompts below)
                    if key_listener is not None:
                        key_listener.stop()
                    frozen_right_hand_target = None
                    episode_done.clear()

                    # Mark episode end so the target source disables the data dropout check
                    target_source.set_not_in_episode()

                    if data_writer is not None:
                        steps = data_writer.stop(episode_duration)
                        actual_fps = steps / episode_duration if episode_duration > 0 else 0

                        # Log episode summary
                        logger.info("=" * 60)
                        logger.success(f"Episode complete: {episode_name}")
                        logger.info("=" * 60)
                        logger.info(
                            f"Duration: {episode_duration:.2f}s | Frames: {steps} | FPS: {actual_fps:.1f} (target: {cfg.control.command_hz})"
                        )

                        if actual_fps < cfg.control.command_hz * 0.9:
                            logger.warning(
                                f"Actual FPS ({actual_fps:.1f}) is lower than target ({cfg.control.command_hz})."
                            )

                        # Log saved files
                        logger.info("Saved files:")
                        episode_dir = data_dir_path / episode_name
                        logger.info(f"  Episode directory: {episode_dir}")
                        logger.info(f"  HDF5:        {episode_dir / f'{episode_name}.h5'}")
                        if not no_head_cam:
                            logger.info(
                                f"  Head video:  {episode_dir / f'{episode_name}_head_left_rgb.mp4'}"
                            )
                        if not no_wrist_cam:
                            logger.info(
                                f"  Left wrist:  {episode_dir / f'{episode_name}_left_wrist.mp4'}"
                            )
                            logger.info(
                                f"  Right wrist: {episode_dir / f'{episode_name}_right_wrist.mp4'}"
                            )
                        logger.info("=" * 60)

                        # Step 3: Prompt user for success/failure
                        # change_dir = False
                        while True:
                            user_input = (
                                _input_live(
                                    "\nWas this episode a success or failure? (S/F): ",
                                    viz_thread is not None,
                                    viz_idle_frame,
                                )
                                .strip()
                                .upper()
                            )
                            if user_input in {"S", "F"}:
                                is_success = user_input == "S"
                                # if user_input == 'S':
                                #    next_object_tmp = input(f"Suggest the next object: ").strip().lower()
                                #    next_object = '_'.join(next_object_tmp.split())
                                #    original_path = data_dir_path
                                #    suggested = Path(original_path).parent / next_object
                                #    change_dir = True
                                break
                            elif user_input in {"A", "AA"}:
                                logger.info(
                                    "Input 'A' or 'AA' detected, assuming stepping on pedal: step once for success, twice for failure."
                                )
                                is_success = user_input == "A"
                                logger.info(
                                    f"Interpreted input as {'success' if is_success else 'failure'} based on pedal step count"
                                )
                                # if user_input == 'A':
                                #    next_object_tmp = input(f"Suggest the next object: ").strip().lower()
                                #    next_object = '_'.join(next_object_tmp.split())
                                #    original_path = data_dir_path
                                #    suggested = Path(original_path).parent / next_object
                                #    change_dir = True
                                break
                            # elif user_input in {'AAA'}:
                            #     is_success = True  # Default to success for the current episode, but suggest switching data directory for the next episode
                            #     if 'right_hand' in str(data_dir_path):
                            #         suggested = str(data_dir_path).replace('right_hand', 'left_hand')
                            #     elif 'left_hand' in str(data_dir_path):
                            #         suggested = str(data_dir_path).replace('left_hand', 'right_hand')
                            #     else:
                            #         suggested = data_dir_path

                            #     confirm = input(f"Input 'AAA' detected as success, suggesting data directory for the other hand: {suggested}. Press [Enter] to switch to this directory for the next episode or any other key to continue with the current directory: ").strip().upper()
                            #     if confirm == '':
                            #         change_dir = True
                            #         break
                            #     else:
                            #         logger.info("Continuing with the current data directory.")
                            #         break

                            # elif user_input in {'AAAA'}:
                            #     is_success = True  # Default to success for the current episode, but suggest switching data directory for the next episode
                            #     if 'insert' in str(data_dir_path):
                            #         suggested = str(data_dir_path).replace('insert', 'extract')
                            #     elif 'extract' in str(data_dir_path):
                            #         suggested = str(data_dir_path).replace('extract', 'insert')
                            #     else:
                            #         suggested = data_dir_path

                            #     if 'right_hand' in str(suggested):
                            #         suggested = str(suggested).replace('right_hand', 'left_hand')
                            #     elif 'left_hand' in str(suggested):
                            #         suggested = str(suggested).replace('left_hand', 'right_hand')
                            #     else:
                            #         suggested = suggested

                            #     confirm = input(f"Input 'AAAA' detected as success, suggesting data directory for the other task: {suggested}. Press [Enter] to switch to this directory for the next episode or any other key to continue with the current directory: ").strip().upper()
                            #     if confirm == '':
                            #         change_dir = True
                            #         break
                            #     else:
                            #         logger.info("Continuing with the current data directory.")
                            #         break

                        # Step 4: Move episode files to success/failure directory
                        if move_episode_files(data_dir_path, episode_name, is_success):
                            logger.info(
                                f"Episode {episode_name} moved to {'success' if is_success else 'failure'} directory"
                            )
                        else:
                            logger.error(f"Failed to move episode {episode_name} files")

                        # if change_dir:
                        #     data_dir_path = Path(suggested)
                        #     data_dir_path.mkdir(parents=True, exist_ok=True)
                        #     (data_dir_path / 'success').mkdir(parents=True, exist_ok=True)
                        #     (data_dir_path / 'failure').mkdir(parents=True, exist_ok=True)
                        #     logger.info(f"Switched to new data directory: {data_dir_path}")
                        #     episode_idx = find_last_episode_index(data_dir_path)

                        # Step 5: Ask user whether to change the data_dir for the next episode
                        user_input = _input_live(
                            "\nDo you want to change the base data directory for the next episode? If so please provide the new path and press [Enter], if not please press [Enter]: ",
                            viz_thread is not None,
                            viz_idle_frame,
                        ).strip()
                        if user_input:
                            logger.info(f"Base data directory changed to: {user_input}")

                            # Convert to Path and ensure it exists
                            data_dir_path = Path(user_input)
                            data_dir_path.mkdir(parents=True, exist_ok=True)

                            # Create success and failure subdirectories
                            (data_dir_path / "success").mkdir(parents=True, exist_ok=True)
                            (data_dir_path / "failure").mkdir(parents=True, exist_ok=True)

                            # Find last episode index
                            last_episode_idx = find_last_episode_index(data_dir_path)
                            next_episode_idx = last_episode_idx + 1
                            logger.info(
                                f"Last episode index: {last_episode_idx}, starting from: {next_episode_idx}"
                            )

                            episode_idx = last_episode_idx
                        else:
                            logger.info("Base data directory remains unchanged.")

                        data_writer = None
                    else:
                        logger.info(f"Episode ended (no data saved): {episode_name}")

                    # Increment episode index for next episode
                    episode_idx += 1
                    episode_name = None
                    episode_start_time = None
                    key_listener = None
                    state = TeleopState.WAITING_FOR_START

            elif state == TeleopState.TERMINATE:
                logger.info("Terminating...")
                break

            else:
                raise RuntimeError(f"Invalid state: {state}")

    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
    finally:
        logger.info("Cleaning up...")

        # Stop the keyboard listener first so the terminal is restored to normal
        # (line) mode before any further teardown / logging
        if key_listener is not None:
            key_listener.stop()

        if data_writer is not None:
            if episode_start_time is not None:
                episode_duration = time.perf_counter() - episode_start_time
                data_writer.stop(episode_duration)

        # Stop arm action thread
        logger.info("Stopping arm action thread...")
        with action_buf_lock:
            action_buffer["left_arm"] = None
            action_buffer["right_arm"] = None
            action_buffer["left_hand"] = None
            action_buffer["right_hand"] = None
        full_robot_action_terminate_event.set()
        full_robot_action_thread.join(timeout=2.0)

        # Stop tactile thread
        logger.info("Stopping tactile thread...")
        tactile_terminate_event.set()
        tactile_thread.join(timeout=2.0)

        # Stop live viz thread
        if viz_thread is not None:
            logger.info("Stopping live viz thread...")
            viz_terminate_event.set()
            viz_thread.join(timeout=2.0)

        # Stop wrist cameras (if enabled)
        if wrist_cam_receiver is not None:
            logger.info("Stopping wrist camera receiving...")
            wrist_cam_receiver.stop()
            logger.info("Wrist camera receiving stopped")

        # Stop head camera (if enabled)
        if head_cam_receiver is not None:
            logger.info("Stopping head camera receiving...")
            head_cam_receiver.stop()
            logger.info("Head camera receiving stopped")

        # Close CV2 windows
        cv2.destroyAllWindows()

        target_source.stop()
        dexmate_bimanual_robot.shutdown()

        logger.info("Stopping hands...")
        left_hand.stop()
        right_hand.stop()
        SharpaWaveManager.get_instance().disconnect_all()
        logger.info("Disconnected all hands")

        logger.info("main_teleop shutdown complete")


if __name__ == "__main__":
    tyro.cli(main)
