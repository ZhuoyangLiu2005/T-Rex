# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Base teleop class with shared functionality for different teleop modes.

This module provides base classes for robot teleoperation with inverse kinematics
control support for multiple teleoperation modes (joint space, Cartesian space).
"""

import os
import threading

import numpy as np
import pytransform3d.rotations as pr
from dexcomm import RateLimiter
from dexmotion.motion_manager import MotionManager
from dexmotion.utils import robot_utils
from loguru import logger

from dexcontrol.apps.dualsense_teleop_base import DualSenseTeleopBase
from dexcontrol.robot import Robot


def is_running_remote() -> bool:
    """Check if the script is running in a remote environment.

    Returns:
        bool: True if running in a remote environment (no display), False otherwise.
    """
    return "DISPLAY" not in os.environ


class BaseIKController:
    """Unified controller for arm inverse kinematics operations.

    This class supports both joint-space and Cartesian-space control of robot arms
    using the MotionManager for kinematics calculations.

    Attributes:
        bot (Robot): Robot instance to control.
        motion_manager (MotionManager): MotionManager instance for IK solving.
        arm_dof (int): Number of degrees of freedom in the robot arms.
        visualize (bool): Whether to enable visualization.
    """

    def __init__(self, bot: Robot | None = None, visualize: bool = True):
        """Initialize the arm IK controller.

        Args:
            bot (Robot | None): Robot instance to control. If None, a new Robot
                instance will be created.
            visualize (bool): Whether to enable visualization. Defaults to True.

        Raises:
            RuntimeError: If config path can't be found or Motion Manager
                initialization fails.
        """
        # Initialize robot if not provided
        if bot is None:
            self.bot = Robot()
        else:
            self.bot = bot

        self.visualize = visualize

        self.arm_dof = 7  # Number of degrees of freedom in the arms

        # Get initial joint positions
        try:
            if self.bot.has_component("torso"):
                joint_pos_dict = self.bot.get_joint_pos_dict(
                    component=["left_arm", "right_arm", "torso"]
                )
            else:
                joint_pos_dict = self.bot.get_joint_pos_dict(
                    component=["left_arm", "right_arm"]
                )
            logger.info("Initial joint positions retrieved successfully")
        except Exception as e:
            logger.error(f"Error getting joint positions: {e}")
            raise

        # Initialize MotionManager for IK solving

        try:
            self.bot.heartbeat.pause()

            self.motion_manager = MotionManager(
                init_visualizer=self.visualize,
                initial_joint_configuration_dict=joint_pos_dict,
            )

            logger.success("Motion Manager setup complete")
        except Exception as e:
            logger.error(f"Error initializing Motion Manager: {e}")
            raise

    def move_delta_joint_space(
        self, new_joint_pos_dict: dict[str, float]
    ) -> dict[str, float]:
        """Apply a delta movement to specified joints.

        Args:
            new_joint_pos_dict (dict[str, float]): Dictionary of joint names and
                target positions.

        Returns:
            dict[str, float]: Dictionary of updated joint positions.

        Raises:
            RuntimeError: If robot or IK solver is not initialized.
        """
        if self.motion_manager.pin_robot is None:
            raise RuntimeError("Robot is not initialized")
        if self.motion_manager.local_ik_solver is None:
            raise RuntimeError("Local IK solver is not initialized")

        current_qpos_dict = self.motion_manager.get_joint_pos_dict()

        # Create a new dictionary with updated values
        updated_new_joint_pos_dict = dict(current_qpos_dict)
        updated_new_joint_pos_dict.update(new_joint_pos_dict)

        updated_new_joint_pos = robot_utils.get_qpos_from_joint_dict(
            self.motion_manager.pin_robot, updated_new_joint_pos_dict
        )

        qpos_dict, is_in_collision, is_within_joint_limits = (
            self.motion_manager.local_ik_solver.solve_ik(
                target_configuration=updated_new_joint_pos
            )
        )
        self.motion_manager.set_joint_pos(qpos_dict)

        # Update visualizer if available
        if self.motion_manager.visualizer is not None:
            qpos_array = robot_utils.get_qpos_from_joint_dict(
                self.motion_manager.pin_robot, qpos_dict
            )
            self.motion_manager.visualizer.update_motion_plan(
                qpos_array.reshape(1, -1),
                robot_utils.get_joint_names(self.motion_manager.pin_robot),
                duration=0.4,
            )

        return qpos_dict

    def move_delta_cartesian(
        self, delta_xyz: np.ndarray, delta_rpy: np.ndarray, arm_side: str
    ) -> np.ndarray:
        """Move the specified arm by a delta in Cartesian space.

        Args:
            delta_xyz (np.ndarray): Translation delta in x, y, z (meters).
            delta_rpy (np.ndarray): Rotation delta in roll, pitch, yaw (radians).
            arm_side (str): Which arm to move ("left" or "right").

        Returns:
            np.ndarray: Target joint positions for the specified arm.

        Raises:
            ValueError: If an invalid arm side is specified.
            RuntimeError: If robot or IK solver is not initialized.
        """
        if self.motion_manager.pin_robot is None:
            raise RuntimeError("Robot is not initialized")
        if self.motion_manager.local_ik_solver is None:
            raise RuntimeError("Local IK solver is not initialized")

        current_qpos = self.motion_manager.get_joint_pos()
        assert isinstance(current_qpos, np.ndarray)

        ee_pose = self.motion_manager.fk(
            frame_names=self.motion_manager.target_frames,
            qpos=current_qpos,
            update_robot_state=False,
        )

        left_ee_pose = ee_pose["L_ee"]
        right_ee_pose = ee_pose["R_ee"]

        if arm_side == "left":
            ee_pose_np = left_ee_pose.np.copy()  # type: ignore
            ee_pose_np[:3, :3] = (
                pr.matrix_from_euler(delta_rpy, 0, 1, 2, True) @ ee_pose_np[:3, :3]
            )
            ee_pose_np[:3, 3] += delta_xyz
            target_poses_dict = {
                "L_ee": ee_pose_np,
                "R_ee": right_ee_pose.np,
            }  # type: ignore
        elif arm_side == "right":
            ee_pose_np = right_ee_pose.np.copy()  # type: ignore
            ee_pose_np[:3, 3] += delta_xyz
            ee_pose_np[:3, :3] = (
                pr.matrix_from_euler(delta_rpy, 0, 1, 2, True) @ ee_pose_np[:3, :3]
            )
            target_poses_dict = {
                "L_ee": left_ee_pose.np,
                "R_ee": ee_pose_np,
            }  # type: ignore
        else:
            raise ValueError(
                f"Invalid arm side: {arm_side}. Must be 'left' or 'right'."
            )

        qpos_dict, is_in_collision, is_within_joint_limits = (
            self.motion_manager.local_ik_solver.solve_ik(target_poses_dict)
        )
        self.motion_manager.set_joint_pos(qpos_dict)

        # Update visualizer if available
        if self.motion_manager.visualizer is not None:
            qpos_array = robot_utils.get_qpos_from_joint_dict(
                self.motion_manager.pin_robot, qpos_dict
            )
            self.motion_manager.visualizer.update_motion_plan(
                qpos_array.reshape(1, -1),
                robot_utils.get_joint_names(self.motion_manager.pin_robot),
                duration=0.4,
            )

        # Extract joint positions for the requested arm
        arm_prefix = arm_side[0].upper()
        arm_joint_indices = [f"{arm_prefix}_arm_j{i + 1}" for i in range(self.arm_dof)]
        return np.array([qpos_dict[joint_name] for joint_name in arm_joint_indices])


class BaseArmTeleopNode(DualSenseTeleopBase):
    """Base teleop node with shared functionality for arm control.

    Attributes:
        arm_side (str): Currently active arm ("left" or "right").
        arms (dict): Dictionary of robot arm interfaces.
        arm_target_qpos (dict): Target joint positions for each arm.
        arm_motion_lock (threading.Lock): Thread lock for synchronizing arm updates.
        arm_control_thread (threading.Thread): Thread for running the arm control loop.
    """

    def __init__(
        self,
        control_hz: int = 200,
        button_update_hz: int = 20,
        device_index: int = 0,
    ):
        """Initialize the base teleop node.

        Args:
            control_hz (int): Control loop frequency in Hz. Defaults to 200.
            button_update_hz (int): Button update frequency in Hz. Defaults to 20.
            device_index (int): Index of the DualSense controller device. Defaults to 0.
        """
        super().__init__(control_hz, button_update_hz, device_index)
        self.arm_side = "left"
        self.arm_motion_lock = threading.Lock()

        # Real robot interface
        self.arms = {"left": self.bot.left_arm, "right": self.bot.right_arm}

        # Target joint positions for continuous motion
        self.arm_target_qpos = {
            "left": self.arms["left"].get_joint_pos(),
            "right": self.arms["right"].get_joint_pos(),
        }

        # Initialize the arm control thread
        self.arm_control_thread = threading.Thread(target=self.arm_control_loop)
        self.arm_control_thread.daemon = True

        # Set initial controller feedback
        self.update_controller_lightbar()

    def update_controller_lightbar(self):
        """Update the controller lightbar color based on active arm."""
        if self.arm_side == "left":
            # Magenta for left arm
            self.dualsense.lightbar.set_color(255, 0, 255)
        else:
            # Cyan for right arm
            self.dualsense.lightbar.set_color(0, 255, 255)

    def toggle_arm(self):
        """Toggle between left and right arm control."""
        self.arm_side = "left" if self.arm_side == "right" else "right"
        with self.arm_motion_lock:
            self.arm_target_qpos[self.arm_side] = self.arms[
                self.arm_side
            ].get_joint_pos()
        logger.info(f"Active arm: {self.arm_side}")
        self.update_controller_lightbar()

    def arm_control_loop(self):
        """Control loop for smooth arm motion.

        This method runs in a separate thread to continuously apply smoothed
        motion to the robot arms based on the current target positions.
        """

        limiter = RateLimiter(self.control_hz)
        while self.is_running:
            arm = self.arms[self.arm_side]
            if self.safe_pressed:
                with self.arm_motion_lock:
                    target_qpos = self.arm_target_qpos[self.arm_side].copy()
                arm.set_joint_pos(target_qpos)
            limiter.sleep()

    def update_motion(self) -> None:
        """Update the robot's motion based on current controller state.

        This method must be implemented by subclasses to handle specific
        motion control logic for different teleop modes.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def stop_all_motion(self) -> None:
        """Stop all ongoing robot motion and reset motion commands."""
        try:
            # Reset smoothers and target positions
            for arm_side, arm in self.arms.items():
                current_pos = arm.get_joint_pos()
                with self.arm_motion_lock:
                    self.arm_target_qpos[arm_side] = current_pos
            logger.info("All arm motion stopped")
        except Exception as e:
            logger.error(f"Error stopping motion: {e}")

    def _cleanup(self):
        """Clean up resources when shutting down."""
        self.dualsense.lightbar.set_color_white()
        self.dualsense.deactivate()
        logger.info("Exiting teleop node")
