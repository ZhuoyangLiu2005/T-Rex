"""Mock teleoperation for testing, only viser visualization."""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import threading
import time
from enum import Enum, auto
from typing import Callable, Literal, Optional

import numpy as np
import pinocchio as pin
import tyro
import viser
from loguru import logger
from loop_rate_limiters import RateLimiter
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.visualize import ViserVisualizer
from teleop_targets import HAND_JOINT_COUNT, TeleopTargetSource

from teleop.arm_hand_control import (
    ArmIKManager,
    InitializationCollisionPlanner,
    SmoothingAndSafetyManager,
    full_robot_action_loop,
    move_robot_to_position_safe,
)
from teleop.config import DEFAULT_CONFIG_PATH, TeleopConfig, load_config
from teleop.ik_utils import PinkLocalIK
from teleop.robot_descriptions import (
    DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES,
    add_env_obstacles,
    build_full_robot,
)

# Mock-specific: move slower than the real robot for easier viser inspection.
MOCK_MAX_RESET_SPEED_RAD_S = 0.3


class TeleopState(Enum):
    INIT = auto()  # Initial state, before any episode starts, starts with self.episode_done.is_set() == True, self.episode_start.is_set() == False and ends after self._prepare_for_new_episode()
    WAITING_FOR_START = auto()  # Waiting for EPISODE_START signal, starts with self.episode_done.is_set() == False, self.episode_start.is_set() == False; ends when self.episode_start.is_set() == True
    IN_EPISODE = auto()  # In an episode, starts with self.episode_done.is_set() == False, self.episode_start.is_set() == True; ends when self.episode_done.is_set() == True and also manually set self.episode_start.is_set() == False
    TERMINATE = auto()  # Termination state indicating the teleoperation system is shutting down.


class MockFullRobot:
    def __init__(
        self,
        pin_full_robot_wrapper: RobotWrapper,
        assemble_qpos: Callable[[dict[str, np.ndarray]], np.ndarray],
        disassemble_qpos: Callable[[np.ndarray], dict[str, np.ndarray]],
    ):
        self.pin_full_robot_wrapper = pin_full_robot_wrapper
        self.assemble_qpos = assemble_qpos
        self.disassemble_qpos = disassemble_qpos

        self.dexmate_bimanual_robot = MockDexmateBimanualRobot(self)
        self.sharpa_left_hand = MockSharpaHand(self, "left")
        self.sharpa_right_hand = MockSharpaHand(self, "right")
        self.qpos = pin.neutral(self.pin_full_robot_wrapper.model)

    def init_viz(self):
        self.server = viser.ViserServer()  # defaults to localhost:8080
        # Add a floor
        self.server.scene.add_box(
            "floor",
            dimensions=(20.0, 20.0, 0.01),
            position=(0, -0.005, 0),
            color=(190, 150, 255),
            opacity=1,
        )
        # Visualize the reduced robot
        self.viz = ViserVisualizer(
            self.pin_full_robot_wrapper.model,
            self.pin_full_robot_wrapper.collision_model,
            self.pin_full_robot_wrapper.visual_model,
            copy_models=True,
        )
        self.viz.initViewer(viewer=self.server, open=False, loadModel=False)
        self.viz.loadViewerModel(rootNodeName="robot")
        self.viz.displayCollisions(False)
        self.viz.displayVisuals(True)
        self.update_viz()

    def update_viz(self):
        self.viz.display(self.qpos)

    def stop_viz(self):
        self.server.stop()


class MockDexmateBimanualRobot:
    def __init__(self, mock_full_robot: MockFullRobot):
        self.mock_full_robot = mock_full_robot

    def set_joint_pos(self, joint_pos: dict[str, np.ndarray], relative: bool, wait_time: float):
        qpos_by_component = self.mock_full_robot.disassemble_qpos(self.mock_full_robot.qpos)
        for component, pos in joint_pos.items():
            if component in qpos_by_component:
                qpos_by_component[component] = np.array(pos).copy()
        self.mock_full_robot.qpos = self.mock_full_robot.assemble_qpos(qpos_by_component)
        time.sleep(wait_time)

    def get_joint_pos_dict(self, component: list[str]) -> dict[str, float]:
        qpos_by_component = self.mock_full_robot.disassemble_qpos(self.mock_full_robot.qpos)
        joint_pos_dict = {}
        for comp in component:
            if comp in qpos_by_component:
                joint_names = DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES[comp]
                for i, joint_name in enumerate(joint_names):
                    joint_pos_dict[joint_name] = qpos_by_component[comp][i]
        return joint_pos_dict


class MockSharpaHand:
    class HandState:
        def __init__(self, angles):
            self.angles = angles

    def __init__(self, mock_full_robot: MockFullRobot, side: Literal["left", "right"]):
        self.mock_full_robot = mock_full_robot
        self.side = side

    def set_joint_position(self, joint_positions: np.ndarray, interpolate: bool):
        del interpolate
        qpos_by_component = self.mock_full_robot.disassemble_qpos(self.mock_full_robot.qpos)
        component_name = f"{self.side}_hand"
        if component_name not in qpos_by_component:
            raise ValueError(f"Invalid component name: {component_name}")
        qpos_by_component[component_name] = np.array(joint_positions).copy()
        self.mock_full_robot.qpos = self.mock_full_robot.assemble_qpos(qpos_by_component)

    def get_states(self):
        return self.HandState(angles=np.zeros(HAND_JOINT_COUNT, dtype=np.float64))


def main(config: str = str(DEFAULT_CONFIG_PATH)):
    """Mock teleop: drives the viser-visualized robot from real Vive + glove input.

    Args:
        config: Path to the YAML config file (see config/default.yaml).
    """
    cfg: TeleopConfig = load_config(config)
    logger.info(f"Loaded config from {config}")
    default_joint_pos = cfg.robot.default_joint_pos

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

    mock_full_robot = MockFullRobot(
        pin_full_robot_wrapper=pin_full_robot_wrapper,
        assemble_qpos=assemble_qpos,
        disassemble_qpos=disassemble_qpos,
    )
    mock_full_robot.dexmate_bimanual_robot.set_joint_pos(
        joint_pos=default_joint_pos,
        relative=False,
        wait_time=0.5,
    )
    mock_full_robot.init_viz()
    input("Please open the visualization. Press [Enter] to continue...")

    pink_ik_solver = PinkLocalIK(default_joint_by_component=default_joint_pos)
    arm_ik_manager = ArmIKManager(
        pink_ik_solver=pink_ik_solver,
        warmstart_with_actual=False,
    )
    smoothing_and_safety_manager = SmoothingAndSafetyManager(
        pin_full_robot_wrapper=pin_full_robot_wrapper,
        assemble_qpos=assemble_qpos,
        disassemble_qpos=disassemble_qpos,
        ruckig_smoothing=False,
        action_hz=cfg.control.arm_action_hz,
    )

    # Initialize full robot action buffer and thread
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
            "dexmate_bimanual_robot": mock_full_robot.dexmate_bimanual_robot,
            "sharpa_left_hand": mock_full_robot.sharpa_left_hand,
            "sharpa_right_hand": mock_full_robot.sharpa_right_hand,
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
        MOCK_MAX_RESET_SPEED_RAD_S / cfg.control.command_hz < 0.05
    )  # Sanity check to ensure the robot won't move too fast
    initialization_planner = InitializationCollisionPlanner(
        pin_full_robot_wrapper=pin_full_robot_wrapper,
        assemble_qpos=assemble_qpos,
        disassemble_qpos=disassemble_qpos,
        max_edge_joint_step=MOCK_MAX_RESET_SPEED_RAD_S / cfg.control.command_hz,
        plan_timeout_s=10.0,
        solve_step_s=0.1,
    )
    move_robot_to_position_safe(
        dexmate_bimanual_robot=mock_full_robot.dexmate_bimanual_robot,
        sharpa_left_hand=mock_full_robot.sharpa_left_hand,
        sharpa_right_hand=mock_full_robot.sharpa_right_hand,
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

    # State machine
    state = TeleopState.INIT

    # Episode management
    episode_done = threading.Event()
    listen_thread: Optional[threading.Thread] = None

    def _listen_for_termination():
        sys.stdin.readline()
        episode_done.set()

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
                input("\nPress [Enter] to reset robot to initial pose...")

                # Set in_episode flag to True, which enables data stream check
                target_source.set_in_episode()

                with action_buf_lock:
                    action_buffer["left_arm"] = None
                    action_buffer["right_arm"] = None
                    action_buffer["left_hand"] = None
                    action_buffer["right_hand"] = None
                logger.info("Arm action buffer cleared (moving to initial position)")

                # Reset robot arms/hands to initial position
                move_robot_to_position_safe(
                    dexmate_bimanual_robot=mock_full_robot.dexmate_bimanual_robot,
                    sharpa_left_hand=mock_full_robot.sharpa_left_hand,
                    sharpa_right_hand=mock_full_robot.sharpa_right_hand,
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

                target_source.request_retargeting_reinit()
                time.sleep(0.5)

                # Reset ArmIKManager and SmoothingAndSafetyManager internal state
                arm_ik_manager.reset()
                smoothing_and_safety_manager.reset()
                logger.info("ArmIKManager and SmoothingAndSafetyManager state reset")

                logger.info("Arms and hands reset to initial position")

                # Step 2: Ask user to press enter to start recording
                input("Press [Enter] to start recording new episode...")

                episode_done.clear()
                listen_thread = threading.Thread(target=_listen_for_termination, daemon=True)
                listen_thread.start()

                logger.info("=" * 50)
                logger.info("EPISODE STARTED - Press [Enter] to stop recording")
                logger.info("=" * 50)
                state = TeleopState.IN_EPISODE

            elif state == TeleopState.IN_EPISODE:
                # Read the latest targets from the in-process source
                assert target_source.in_episode, "in_episode should be True during episode"
                target_data = target_source.get_targets()
                if target_source.invalid:
                    logger.error("Teleop target source became invalid (data dropout)")
                    break  # Exit teleop immediately
                assert target_data is not None, (
                    "No targets available during episode (retargeting not initialized?)"
                )

                # Get actual arm state from robot
                with hardware_lock:
                    curr_joint_pos_dict = mock_full_robot.dexmate_bimanual_robot.get_joint_pos_dict(
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

                # Write IK solution to buffer (high-frequency thread will smooth and send commands)
                with action_buf_lock:
                    action_buffer["left_arm"] = ik_targets["left_arm"].copy()
                    action_buffer["right_arm"] = ik_targets["right_arm"].copy()
                    action_buffer["left_hand"] = target_data.left_hand_target_joint_positions.copy()
                    action_buffer["right_hand"] = (
                        target_data.right_hand_target_joint_positions.copy()
                    )

                mock_full_robot.update_viz()

                # Check if episode should end
                if episode_done.is_set():
                    # Clear arm action buffer to stop high-frequency commands
                    with action_buf_lock:
                        action_buffer["left_arm"] = None
                        action_buffer["right_arm"] = None
                        action_buffer["left_hand"] = None
                        action_buffer["right_hand"] = None
                    logger.info("Arm action buffer cleared (episode ending)")

                    if listen_thread is not None:
                        listen_thread.join(timeout=1.0)
                    episode_done.clear()

                    # Mark episode end so the target source disables the data dropout check
                    target_source.set_not_in_episode()

                    # Increment episode index for next episode
                    listen_thread = None
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
        # Stop arm action thread
        logger.info("Stopping arm action thread...")
        with action_buf_lock:
            action_buffer["left_arm"] = None
            action_buffer["right_arm"] = None
            action_buffer["left_hand"] = None
            action_buffer["right_hand"] = None
        full_robot_action_terminate_event.set()
        full_robot_action_thread.join(timeout=2.0)
        target_source.stop()
        mock_full_robot.stop_viz()


if __name__ == "__main__":
    tyro.cli(main)
