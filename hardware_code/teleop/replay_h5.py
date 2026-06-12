"""replay_h5.py: Replay recorded actions from an HDF5 file on the robot.

Reads target poses and hand targets from an h5 file and replays them through
the same control pipeline used during recording (IK solving + smoothing +
collision checking via full_robot_action_loop).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import threading
import time

import h5py
import numpy as np
import tyro
from dexcontrol.robot import Robot
from loguru import logger
from loop_rate_limiters import RateLimiter
from sharpa import ControlMode, ControlSource, SharpaWave, SharpaWaveManager

from teleop.arm_hand_control import (
    ArmIKManager,
    InitializationCollisionPlanner,
    SmoothingAndSafetyManager,
    full_robot_action_loop,
    move_robot_to_position_safe,
)
from teleop.ik_utils import PinkLocalIK
from teleop.robot_descriptions import (
    DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES,
    add_env_obstacles,
    build_full_robot,
)

# Configuration (must match recording settings)
COMMAND_HZ = 30.0
ARM_ACTION_HZ = 300.0

LEFT_HAND_SERIAL = "C35F913DC35F"
RIGHT_HAND_SERIAL = "C55F913BC55F"
HAND_JOINT_COUNT = 22
HAND_INTERPOLATE = True

LEFT_ARM_DEFAULT_JOINT_POS = [0.84, 0.51, 0.37, -1.30, -0.65, -0.29, -0.03]
RIGHT_ARM_DEFAULT_JOINT_POS = [-0.84, -0.51, -0.37, -1.30, 0.65, 0.29, 0.03]
TORSO_DEFAULT_JOINT_POS = [0.9, 1.57, 0.1]
HEAD_DEFAULT_JOINT_POS = [0.28, 0.0, 0.0]
DEFAULT_JOINT_POS = {
    "left_arm": np.array(LEFT_ARM_DEFAULT_JOINT_POS),
    "right_arm": np.array(RIGHT_ARM_DEFAULT_JOINT_POS),
    "head": np.array(HEAD_DEFAULT_JOINT_POS),
    "torso": np.array(TORSO_DEFAULT_JOINT_POS),
}
RESET_DOF_ERR_TOL = 0.2


def connect_hands():
    """Connect to left and right hands via SharpaWaveSDK."""
    manager = SharpaWaveManager.get_instance()
    time.sleep(3)  # Wait for device discovery

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


def main(
    h5_path: str,
    start_idx: int = 0,
    end_idx: int = -1,
):
    """Replay recorded actions from an HDF5 file on the robot.

    Args:
        h5_path: Path to the HDF5 file containing recorded episode data.
        start_idx: Start index in the data (0-indexed timestep).
        end_idx: End index in the data (-1 for end of file).
    """
    # =========================================================================
    # Load data from h5
    # =========================================================================
    logger.info(f"Loading data from {h5_path}...")
    with h5py.File(h5_path, "r") as f:
        # Actions: target poses (4x4) and hand targets (22,)
        left_arm_target_pose_all = f["left_arm_target_pose"][:]
        right_arm_target_pose_all = f["right_arm_target_pose"][:]
        left_hand_target_all = f["left_hand_target_joint_positions"][:]
        right_hand_target_all = f["right_hand_target_joint_positions"][:]

        # States: actual joint positions (for initial pose)
        left_arm_state_all = f["left_arm_joint_positions"][:]
        right_arm_state_all = f["right_arm_joint_positions"][:]
        left_hand_state_all = f["left_hand_joint_positions"][:]
        right_hand_state_all = f["right_hand_joint_positions"][:]

        command_hz = float(f.attrs["command_hz"])
        total_steps = int(f.attrs["total_steps"])

    logger.info(f"Loaded {total_steps} total steps, recorded at {command_hz} Hz")

    # Resolve end_idx
    if end_idx == -1:
        end_idx = total_steps

    # Validate indices
    if start_idx < 0 or start_idx >= total_steps:
        raise ValueError(f"start_idx {start_idx} out of range [0, {total_steps})")
    if end_idx <= start_idx or end_idx > total_steps:
        raise ValueError(f"end_idx {end_idx} out of range ({start_idx}, {total_steps}]")

    # Slice data to replay range
    left_arm_target_pose = left_arm_target_pose_all[start_idx:end_idx]
    right_arm_target_pose = right_arm_target_pose_all[start_idx:end_idx]
    left_hand_target = left_hand_target_all[start_idx:end_idx]
    right_hand_target = right_hand_target_all[start_idx:end_idx]

    num_steps = end_idx - start_idx
    logger.info(f"Will replay [{start_idx}:{end_idx}] ({num_steps} steps)")

    # Initial state from data (for moving robot to starting pose)
    initial_left_arm = left_arm_state_all[start_idx]
    initial_right_arm = right_arm_state_all[start_idx]
    initial_left_hand = left_hand_state_all[start_idx]
    initial_right_hand = right_hand_state_all[start_idx]

    # =========================================================================
    # Initialize robot
    # =========================================================================
    logger.info("Initializing robot...")
    dexmate_bimanual_robot = Robot()
    logger.info(f"Robot '{dexmate_bimanual_robot.robot_model}' initialized")

    # =========================================================================
    # Initialize IK solver and managers
    # =========================================================================
    pink_ik_solver = PinkLocalIK(default_joint_by_component=DEFAULT_JOINT_POS)
    arm_ik_manager = ArmIKManager(
        pink_ik_solver=pink_ik_solver,
        warmstart_with_actual=False,
    )
    pin_full_robot_wrapper, assemble_qpos, disassemble_qpos = build_full_robot(
        default_joint_by_component=DEFAULT_JOINT_POS
    )
    # Site-specific collision environment (this script is not config-wired; adjust here).
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
        table_height=0.64,  # Measured
    )
    smoothing_and_safety_manager = SmoothingAndSafetyManager(
        pin_full_robot_wrapper=pin_full_robot_wrapper,
        assemble_qpos=assemble_qpos,
        disassemble_qpos=disassemble_qpos,
        ruckig_smoothing=False,
        action_hz=ARM_ACTION_HZ,
    )
    logger.info("IK solver and managers initialized")

    # =========================================================================
    # Initialize hands
    # =========================================================================
    logger.info("Connecting to hands...")
    left_hand, right_hand = connect_hands()
    initialize_hand(left_hand)
    initialize_hand(right_hand)
    left_hand.start()
    right_hand.start()
    left_hand.set_joint_position([0.0] * HAND_JOINT_COUNT, HAND_INTERPOLATE)
    right_hand.set_joint_position([0.0] * HAND_JOINT_COUNT, HAND_INTERPOLATE)
    time.sleep(1)
    logger.info("Hands ready")

    # =========================================================================
    # Initialize action buffer and high-frequency action thread
    # =========================================================================
    action_buf_lock = threading.Lock()
    action_buffer = {
        "left_arm": None,
        "right_arm": None,
        "left_hand": None,
        "right_hand": None,
    }
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
        daemon=True,
    )
    full_robot_action_thread.start()
    logger.info(f"Full robot action thread started at {ARM_ACTION_HZ} Hz")

    # Move robot to initial position - CRITICAL for teleop_pub retargeting to work correctly
    assert 0.6 / COMMAND_HZ < 0.05  # Sanity check to ensure the robot won't move too fast
    initialization_planner = InitializationCollisionPlanner(
        pin_full_robot_wrapper=pin_full_robot_wrapper,
        assemble_qpos=assemble_qpos,
        disassemble_qpos=disassemble_qpos,
        max_edge_joint_step=0.6 / COMMAND_HZ,  # 0.6 rad/s max speed
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

    # =========================================================================
    # Move to initial recorded pose, then replay
    # =========================================================================
    input("\nPress [Enter] to move robot to initial recorded pose...")

    # Move arms to the recorded initial state
    logger.info("Moving to initial recorded pose from data...")
    move_robot_to_position_safe(
        dexmate_bimanual_robot=dexmate_bimanual_robot,
        sharpa_left_hand=left_hand,
        sharpa_right_hand=right_hand,
        initialization_planner=initialization_planner,
        action_buffer=action_buffer,
        action_buf_lock=action_buf_lock,
        target_joint_pos={
            "left_arm": initial_left_arm,
            "right_arm": initial_right_arm,
            "left_hand": initial_left_hand,
            "right_hand": initial_right_hand,
            "head": np.array(HEAD_DEFAULT_JOINT_POS),
            "torso": np.array(TORSO_DEFAULT_JOINT_POS),
        },
        hardware_lock=hardware_lock,
        command_hz=COMMAND_HZ,
        dof_error_tolerance=RESET_DOF_ERR_TOL,
        hold_time_s=0.5,
    )
    time.sleep(1)
    logger.info("Robot at initial recorded pose")

    # Reset IK and smoothing managers so warmstart begins from the current state
    arm_ik_manager.reset()
    smoothing_and_safety_manager.reset()

    input("Press [Enter] to start replay...")

    logger.info("=" * 50)
    logger.info(f"REPLAY STARTED - {num_steps} steps at {command_hz} Hz")
    logger.info("=" * 50)

    limiter = RateLimiter(frequency=command_hz, name="replay_control_loop", warn=True)

    try:
        for step in range(num_steps):
            limiter.sleep()

            # Check if action thread terminated due to safety violation
            if full_robot_action_terminate_event.is_set():
                logger.error("Action thread terminated (safety violation). Stopping replay.")
                break

            # Get target poses from recorded data
            left_target_pose = left_arm_target_pose[step]  # (4, 4)
            right_target_pose = right_arm_target_pose[step]  # (4, 4)

            # Get actual arm state from robot (for IK warmstarting)
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

            # Solve IK (same as main_teleop.py)
            ik_targets = arm_ik_manager.get_arm_action(
                arm_absolute_target_poses={
                    "L_ee": left_target_pose,
                    "R_ee": right_target_pose,
                },
                arm_current_joint_positions={
                    "left_arm": left_arm_joint_positions,
                    "right_arm": right_arm_joint_positions,
                },
            )

            # Write to action buffer (consumed by full_robot_action_loop at 300 Hz)
            with action_buf_lock:
                action_buffer["left_arm"] = ik_targets["left_arm"].copy()
                action_buffer["right_arm"] = ik_targets["right_arm"].copy()
                action_buffer["left_hand"] = left_hand_target[step].copy()
                action_buffer["right_hand"] = right_hand_target[step].copy()

            if (step + 1) % 50 == 0:
                logger.info(f"Replayed {step + 1}/{num_steps} steps")

        logger.info(f"Replay complete: {num_steps} steps executed")

    except KeyboardInterrupt:
        logger.info(f"Replay interrupted at step {step + 1}/{num_steps}")

    finally:
        logger.info("Cleaning up...")

        # Clear action buffer to stop sending commands
        with action_buf_lock:
            action_buffer["left_arm"] = None
            action_buffer["right_arm"] = None
            action_buffer["left_hand"] = None
            action_buffer["right_hand"] = None

        # Stop action thread
        full_robot_action_terminate_event.set()
        full_robot_action_thread.join(timeout=2.0)

        # Stop hands
        left_hand.stop()
        right_hand.stop()
        SharpaWaveManager.get_instance().disconnect_all()
        logger.info("Disconnected all hands")

        # Shutdown robot
        dexmate_bimanual_robot.shutdown()
        logger.info("Replay shutdown complete")


if __name__ == "__main__":
    tyro.cli(main)
