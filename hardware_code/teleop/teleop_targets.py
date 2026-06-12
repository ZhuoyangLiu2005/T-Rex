"""teleop_targets.py: In-process source of teleop targets.

TeleopTargetSource replaces the former standalone teleop_pub.py process:
- Receives Vive tracker poses (ViveReceiver thread)
- Receives HandAction messages from the Manus retargeting system via ZMQ
- Retargets Vive wrist poses to absolute end-effector target poses in the
  robot base frame (no IK here; IK runs in the consumer)
- Exposes the latest targets through get_targets() (thread-safe snapshot)

It is constructed and started by main_teleop.py. Call
request_retargeting_reinit() after moving the robot to its default pose so
the Vive-to-robot frame alignment is computed against the pose the robot is
actually in.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pinocchio as pin
import zmq
from loguru import logger
from loop_rate_limiters import RateLimiter

from teleop.ik_utils import PinkLocalIK
from vive_tracker.vive_streamer import ViveStreamer

# Add path for protobuf imports. The sharpa_hand_pb2 bindings ship with the
# Sharpa Manus SDK (https://github.com/sharpa-robotics/sharpa-manus-sdk); see
# third_party/README.md. Set SHARPA_MANUS_PROTO_DIR if you cloned it elsewhere.
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_proto_dir_candidates = [
    os.environ.get("SHARPA_MANUS_PROTO_DIR"),
    *sorted(
        glob.glob(
            os.path.join(
                _project_root,
                "third_party",
                "sharpa-manus-sdk",
                "retargeting_alg_release_*",
                "include",
                "proto_hand",
            )
        )
    ),
    os.path.join(_project_root, "manus", "retargeting", "include", "proto_hand"),  # legacy layout
]
_proto_dir = next(
    (
        p
        for p in _proto_dir_candidates
        if p and os.path.isfile(os.path.join(p, "sharpa_hand_pb2.py"))
    ),
    None,
)
if _proto_dir is None:
    raise ImportError(
        "Could not locate sharpa_hand_pb2.py. Clone the Sharpa Manus SDK into "
        "third_party/sharpa-manus-sdk (see third_party/README.md) or set the "
        "SHARPA_MANUS_PROTO_DIR environment variable to its include/proto_hand directory."
    )
sys.path.append(_proto_dir)
import sharpa_hand_pb2

# =============================================================================
# Configuration
# =============================================================================

# Hand joint count (Sharpa Wave has 22 joints) — a hardware property, not site
# config. All site/deployment settings (Vive endpoints, rates, default joint
# poses) live in the YAML config: see config/default.yaml and teleop/config.py.
HAND_JOINT_COUNT = 22


# =============================================================================
# Vive Receiver
# =============================================================================


class ViveReceiver:
    """Thread that receives Vive tracker data and writes to a shared buffer."""

    def __init__(
        self, ip: str, port: str, left_name: str, right_name: str, update_hz: float = 60.0
    ):
        self.ip = ip
        self.port = port
        self.left_name = left_name
        self.right_name = right_name
        self.update_hz = update_hz

        self._lock = threading.Lock()
        self._left_vive_pose: Optional[np.ndarray] = None
        self._right_vive_pose: Optional[np.ndarray] = None
        self._vive_timestamp: Optional[float] = None
        self._data_ready = False

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._streamer: Optional[ViveStreamer] = None

    def start(self):
        """Start the receiver thread."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("ViveReceiver thread already running")
            return

        self._stop_event.clear()
        self._streamer = ViveStreamer(
            vive_names=[self.left_name, self.right_name],
            ip=self.ip,
            port=int(self.port),
            fps=None,
        )
        self._streamer.start_streaming()

        self._thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._thread.start()
        logger.info(f"ViveReceiver started at {self.update_hz} Hz")

    def stop(self):
        """Stop the receiver thread gracefully."""
        if self._thread is None:
            return

        self._stop_event.set()
        if self._streamer is not None:
            self._streamer.stop_streaming()
        self._thread.join(timeout=2.0)
        logger.info("ViveReceiver stopped")

    def _receive_loop(self):
        """Main loop for receiving Vive data."""
        limiter = RateLimiter(name="vive_receive_loop_limiter", frequency=self.update_hz, warn=True)

        while not self._stop_event.is_set():
            limiter.sleep()

            out = self._streamer.get()
            if (
                out is None
                or self.left_name not in out.vive_data
                or self.right_name not in out.vive_data
            ):
                logger.warning("Invalid data received from ViveStreamer")
                with self._lock:
                    self._left_vive_pose = None
                    self._right_vive_pose = None
                    self._vive_timestamp = None
                    self._data_ready = False
            else:
                T_L_vive = pin.SE3(out.vive_data[self.left_name].copy())
                T_R_vive = pin.SE3(out.vive_data[self.right_name].copy())
                timestamp = time.perf_counter()

                with self._lock:
                    self._left_vive_pose = T_L_vive.homogeneous.copy()
                    self._right_vive_pose = T_R_vive.homogeneous.copy()
                    self._vive_timestamp = timestamp
                    self._data_ready = True

    def get_latest(self) -> Optional[dict]:
        """Get latest Vive data from buffer (thread-safe)."""
        with self._lock:
            if not self._data_ready:
                return None

            return {
                "left_vive_pose": self._left_vive_pose.copy(),
                "right_vive_pose": self._right_vive_pose.copy(),
                "vive_timestamp": self._vive_timestamp,
            }


# =============================================================================
# HandAction Receiver
# =============================================================================


class HandActionReceiver:
    """HandAction message receiver."""

    def __init__(self, context, address="tcp://localhost:6668"):
        self.address = address

        self.socket = context.socket(zmq.SUB)

        self.socket.setsockopt(zmq.RCVHWM, 1)
        self.socket.setsockopt(zmq.LINGER, 0)

        self.socket.connect(self.address)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        logger.info(f"[HandAction Receiver] Connected to {address}, waiting for messages...")

    def receive_hand_action(self):
        """Receive HandAction message (non-blocking)."""
        try:
            # Check if buffer is full and drop old messages
            if self.socket.getsockopt(zmq.RCVBUF) > 8:
                # Drop old messages silently
                while self.socket.getsockopt(zmq.RCVBUF) > 0:
                    try:
                        self.socket.recv(flags=zmq.NOBLOCK)
                    except zmq.Again:
                        break

            payload = self.socket.recv(flags=zmq.NOBLOCK)

            msg = sharpa_hand_pb2.HandAction()
            msg.ParseFromString(payload)
            return msg

        except zmq.Again:
            # No message available
            return None
        except Exception as e:
            logger.warning(f"[HandAction Receiver] Receive failed: {e}")
            return None

    def close(self):
        self.socket.close()


# =============================================================================
# Retargeting Functions
# =============================================================================


def _compute_retargeting_rotation(T_L_vive: pin.SE3, T_R_vive: pin.SE3) -> np.ndarray:
    """Compute Rot_viveworld_alignedworld rotation matrix."""
    Rot_z = np.array([0.0, 1.0, 0.0])  # Assume vive world frame's y-axis is up!
    Rot_y = T_L_vive.translation - T_R_vive.translation

    if np.linalg.norm(Rot_y) < 5e-2:
        raise ValueError("Left and Right tracker positions are too close")

    Rot_y /= np.linalg.norm(Rot_y)
    Rot_y -= np.dot(Rot_y, Rot_z) * Rot_z

    if np.linalg.norm(Rot_y) < 5e-1:
        raise ValueError("Left and Right tracker positions are vertically aligned")

    Rot_y /= np.linalg.norm(Rot_y)
    Rot_x = np.cross(Rot_y, Rot_z)
    Rot_x /= np.linalg.norm(Rot_x)
    Rot_y = np.cross(Rot_z, Rot_x)
    Rot_y /= np.linalg.norm(Rot_y)

    return np.column_stack((Rot_x, Rot_y, Rot_z))


def _compute_absolute_target_poses(
    vive_curr_poses: dict[str, pin.SE3],
    vive_robot_init_poses: dict,
) -> dict[str, pin.SE3]:
    """Compute absolute target poses in robot base frame from Vive poses."""
    R_viveworld_alignedworld = vive_robot_init_poses["Rot_viveworld_alignedworld"]
    absolute_target_poses = {}

    for name in ["L", "R"]:
        ee_name = f"{name}_ee"
        vive_name = f"{name}_vive"

        vive_relative_translation = (
            vive_curr_poses[vive_name].translation - vive_robot_init_poses[vive_name].translation
        )

        absolute_translation = (
            R_viveworld_alignedworld.T @ vive_relative_translation
            + vive_robot_init_poses[ee_name].translation
        )

        absolute_rotation = (
            R_viveworld_alignedworld.T
            @ vive_curr_poses[vive_name].rotation
            @ vive_robot_init_poses[vive_name].rotation.T
            @ R_viveworld_alignedworld
            @ vive_robot_init_poses[ee_name].rotation
        )

        absolute_target_poses[ee_name] = pin.SE3(
            translation=absolute_translation,
            rotation=absolute_rotation,
        )

    return absolute_target_poses


def _initialize_retargeting(
    pink_ik_solver: PinkLocalIK,
    T_L_vive: pin.SE3,
    T_R_vive: pin.SE3,
    initial_arm_joint_pos: dict[str, np.ndarray],
) -> dict:
    """Initialize retargeting using default joint positions."""
    logger.info("Initializing retargeting...")

    # Compute FK from initial joint positions
    fk_res = pink_ik_solver.fk(
        frames=["L_ee", "R_ee"], joint_pos_by_component=initial_arm_joint_pos
    )

    Rot_viveworld_alignedworld = _compute_retargeting_rotation(T_L_vive, T_R_vive)

    vive_robot_init_poses = {
        "L_ee": fk_res["L_ee"].copy(),
        "R_ee": fk_res["R_ee"].copy(),
        "L_vive": T_L_vive.copy(),
        "R_vive": T_R_vive.copy(),
        "Rot_viveworld_alignedworld": Rot_viveworld_alignedworld,
    }

    logger.info("Retargeting initialization complete")
    return vive_robot_init_poses


# =============================================================================
# Teleop target source
# =============================================================================


@dataclass
class TeleopTargets:
    """Snapshot of the latest teleop targets (all arrays are owned copies)."""

    # Timestamps (time.perf_counter clock)
    vive_timestamp: float
    hand_timestamp: float
    # Raw Vive tracker poses (4x4 homogeneous, vive world frame)
    left_vive_pose: np.ndarray
    right_vive_pose: np.ndarray
    # Absolute end-effector target poses (4x4 homogeneous, robot base frame)
    left_arm_target_pose: np.ndarray
    right_arm_target_pose: np.ndarray
    # Hand joint targets (HAND_JOINT_COUNT,)
    left_hand_target_joint_positions: np.ndarray
    right_hand_target_joint_positions: np.ndarray


class TeleopTargetSource:
    """Self-contained source of teleop targets, used in-process by main_teleop.py.

    Runs a background thread (at update_hz) that fuses the latest Vive wrist
    poses with the latest HandAction message and retargets them to absolute
    end-effector target poses. Consumers poll get_targets() for the latest
    snapshot.

    Episode workflow:
    - set_in_episode()/set_not_in_episode() bracket each episode; while in an
      episode, a data dropout longer than vive_timeout_tol_s marks the source
      invalid (check the `invalid` property and stop teleop).
    - request_retargeting_reinit() recomputes the Vive-to-robot alignment
      assuming the robot is at default_joint_pos; call it right after moving
      the robot to that pose.
    """

    def __init__(
        self,
        vive_ip: str,
        vive_port: str,
        vive_left_tracker_name: str,
        vive_right_tracker_name: str,
        vive_update_hz: float,
        vive_timeout_tol_s: float,
        hand_action_address: str,
        update_hz: float,
        default_joint_pos: dict[str, np.ndarray],
    ):
        self._default_joint_pos = {k: np.asarray(v).copy() for k, v in default_joint_pos.items()}
        self._update_hz = update_hz
        self._vive_timeout_tol_s = vive_timeout_tol_s

        # Own IK solver instance for FK during retargeting initialization.
        # Not shared with the consumer's solver: pinocchio data objects are not
        # thread-safe and this one is used from the update thread.
        self._pink_ik_solver = PinkLocalIK(default_joint_by_component=self._default_joint_pos)

        self._vive_receiver = ViveReceiver(
            ip=vive_ip,
            port=vive_port,
            left_name=vive_left_tracker_name,
            right_name=vive_right_tracker_name,
            update_hz=vive_update_hz,
        )
        self._zmq_context = zmq.Context.instance()
        self._hand_action_receiver = HandActionReceiver(
            context=self._zmq_context, address=hand_action_address
        )

        # Latest snapshot + state flags (guarded by _lock)
        self._lock = threading.Lock()
        self._targets: Optional[TeleopTargets] = None
        self._invalid = False
        self._in_episode = False
        self._reinit_requested = False

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def start(self):
        """Start the receivers and the target update thread."""
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("TeleopTargetSource already started")
        self._vive_receiver.start()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._update_loop, daemon=True)
        self._thread.start()
        logger.info(f"TeleopTargetSource started at {self._update_hz} Hz")

    def stop(self):
        """Stop the update thread and receivers."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._vive_receiver.stop()
        self._hand_action_receiver.close()
        logger.info("TeleopTargetSource stopped")

    # ------------------------------------------------------------------ #
    # Consumer API
    # ------------------------------------------------------------------ #

    def get_targets(self) -> Optional[TeleopTargets]:
        """Return the latest targets snapshot (None until first valid data)."""
        with self._lock:
            return self._targets

    @property
    def invalid(self) -> bool:
        """True if target data dropped out during an episode (teleop must stop)."""
        with self._lock:
            return self._invalid

    @property
    def in_episode(self) -> bool:
        with self._lock:
            return self._in_episode

    def request_retargeting_reinit(self):
        """Recompute Vive-to-robot alignment on the next valid data.

        Call after moving the robot to default_joint_pos, before starting a
        new episode.
        """
        with self._lock:
            self._reinit_requested = True
        logger.info("Requested retargeting reinitialization")

    def set_in_episode(self):
        """Mark episode start (enables the data dropout check)."""
        with self._lock:
            if self._in_episode:
                raise RuntimeError(
                    "in_episode flag is already set, cannot set again without clearing. "
                    "This indicates a logic error in episode workflow management."
                )
            self._in_episode = True
        logger.info("in_episode set to True")

    def set_not_in_episode(self):
        """Mark episode end (disables the data dropout check)."""
        with self._lock:
            if not self._in_episode:
                raise RuntimeError(
                    "in_episode flag is already cleared, cannot clear again. "
                    "This indicates a logic error in episode workflow management."
                )
            self._in_episode = False
        logger.info("in_episode set to False")

    # ------------------------------------------------------------------ #
    # Internal update loop
    # ------------------------------------------------------------------ #

    def _update_loop(self):
        vive_robot_init_poses: Optional[dict] = None
        arm_initialized = False

        last_left_hand_target = np.zeros(HAND_JOINT_COUNT, dtype=np.float64)
        last_right_hand_target = np.zeros(HAND_JOINT_COUNT, dtype=np.float64)

        # Start with current time to avoid false invalid at startup
        last_valid_time = time.perf_counter()

        limiter = RateLimiter(
            frequency=self._update_hz, name="teleop_targets_update_limiter", warn=True
        )
        try:
            logger.info("TeleopTargetSource update loop entered")
            while not self._stop_event.is_set():
                limiter.sleep()

                vive_data = self._vive_receiver.get_latest()
                hand_msg = self._hand_action_receiver.receive_hand_action()
                curr_timestamp = time.perf_counter()

                with self._lock:
                    in_episode_flag = self._in_episode
                    reinit_requested = self._reinit_requested

                if vive_data is not None and hand_msg is not None:
                    # Update last valid time on successful data reception
                    last_valid_time = curr_timestamp

                    # === Process Vive data ===
                    T_L_vive = pin.SE3(vive_data["left_vive_pose"])
                    T_R_vive = pin.SE3(vive_data["right_vive_pose"])
                    vive_timestamp = vive_data["vive_timestamp"]

                    # (Re)initialize retargeting on request or on first valid data
                    if reinit_requested or not arm_initialized:
                        if reinit_requested:
                            logger.info("Reinitialization requested - reinitializing retargeting")
                        initial_arm_joint_pos = {
                            "left_arm": self._default_joint_pos["left_arm"].copy(),
                            "right_arm": self._default_joint_pos["right_arm"].copy(),
                        }
                        vive_robot_init_poses = _initialize_retargeting(
                            pink_ik_solver=self._pink_ik_solver,
                            T_L_vive=T_L_vive,
                            T_R_vive=T_R_vive,
                            initial_arm_joint_pos=initial_arm_joint_pos,
                        )
                        arm_initialized = True
                        with self._lock:
                            self._reinit_requested = False

                    # Compute retargeting to robot coordinates (no IK solving)
                    vive_curr_poses = {"L_vive": T_L_vive, "R_vive": T_R_vive}
                    absolute_target_poses = _compute_absolute_target_poses(
                        vive_curr_poses=vive_curr_poses,
                        vive_robot_init_poses=vive_robot_init_poses,
                    )

                    # === Process HandAction data ===
                    assert len(hand_msg.joint_left.position) == HAND_JOINT_COUNT
                    assert len(hand_msg.joint_right.position) == HAND_JOINT_COUNT
                    last_left_hand_target = np.array(hand_msg.joint_left.position, dtype=np.float64)
                    last_right_hand_target = np.array(
                        hand_msg.joint_right.position, dtype=np.float64
                    )

                    # === Publish snapshot ===
                    targets = TeleopTargets(
                        vive_timestamp=vive_timestamp,
                        hand_timestamp=curr_timestamp,
                        left_vive_pose=vive_data["left_vive_pose"],
                        right_vive_pose=vive_data["right_vive_pose"],
                        left_arm_target_pose=absolute_target_poses["L_ee"].homogeneous.copy(),
                        right_arm_target_pose=absolute_target_poses["R_ee"].homogeneous.copy(),
                        left_hand_target_joint_positions=last_left_hand_target,
                        right_hand_target_joint_positions=last_right_hand_target,
                    )
                    with self._lock:
                        self._targets = targets
                elif (
                    curr_timestamp - last_valid_time < self._vive_timeout_tol_s
                    or not in_episode_flag
                ):
                    # Allow short grace period for temporary data issues;
                    # outside an episode missing data is expected (keep last snapshot).
                    continue
                else:
                    # Data dropout during an episode: mark invalid so the consumer stops.
                    error_string = "Invalid teleop target data: "
                    if vive_data is None:
                        error_string += "No Vive Data; "
                    if hand_msg is None:
                        error_string += "No HandAction Data; "
                    logger.error(error_string)
                    with self._lock:
                        self._invalid = True
                    break
        except Exception as e:
            logger.exception(f"Exception in TeleopTargetSource update loop: {e}")
            with self._lock:
                self._invalid = True
        logger.info("TeleopTargetSource update loop exited")


if __name__ == "__main__":
    # Debug runner: print retargeted poses without a robot connection.
    # Requires the Vive server and the Manus retargeting publisher to be running.
    # Endpoints come from the YAML config (same file main_teleop.py uses).
    import argparse

    from teleop.config import DEFAULT_CONFIG_PATH, load_config

    parser = argparse.ArgumentParser(description="Print retargeted teleop targets (no robot).")
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="YAML config providing the Vive/HandAction endpoints (see config/default.yaml)",
    )
    cli_args = parser.parse_args()
    cfg = load_config(cli_args.config)
    logger.info(f"Loaded config from {cli_args.config}")

    source = TeleopTargetSource(
        vive_ip=cfg.vive.ip,
        vive_port=cfg.vive.port,
        vive_left_tracker_name=cfg.vive.left_tracker_name,
        vive_right_tracker_name=cfg.vive.right_tracker_name,
        vive_update_hz=cfg.vive.update_hz,
        vive_timeout_tol_s=cfg.vive.timeout_tol_s,
        hand_action_address=cfg.hand_action.address,
        update_hz=cfg.targets.update_hz,
        default_joint_pos=cfg.robot.default_joint_pos,
    )
    source.start()
    try:
        while True:
            time.sleep(1.0)
            targets = source.get_targets()
            if targets is None:
                logger.info("No targets yet...")
                continue
            logger.info(
                f"L_ee target t: {targets.left_arm_target_pose[:3, 3]} | "
                f"R_ee target t: {targets.right_arm_target_pose[:3, 3]} | "
                f"hand L[0:3]: {targets.left_hand_target_joint_positions[:3]}"
            )
            if source.invalid:
                logger.error("Source became invalid, exiting")
                break
    except KeyboardInterrupt:
        pass
    finally:
        source.stop()
