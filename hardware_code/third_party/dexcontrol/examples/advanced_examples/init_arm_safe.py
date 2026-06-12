# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Example script to move the robot arms to a safe pose.

This script demonstrates moving both robot arms from any position to predefined
safe poses while avoiding self-collisions.
"""

import time
from typing import Literal

import numpy as np
import tyro
from dexcomm import RateLimiter
from dexmotion.motion_manager import MotionManager
from dexmotion.tasks.move_out_of_self_collision_task import MoveOutOfSelfCollisionTask
from dexmotion.tasks.move_to_configuration_task import MoveToConfigurationTask
from dexmotion.utils import robot_utils
from loguru import logger

from dexcontrol.core.arm import Arm
from dexcontrol.robot import Robot


class ArmSafeInitializer:
    """Initializes robot arms to safe positions while avoiding self-collisions."""

    def __init__(
        self,
        control_hz: int = 250,
        visualize: bool = True,
        bot: Robot | None = None,
    ) -> None:
        """Initialize the ArmSafeInitializer.

        Args:
            control_hz: Control frequency in Hz.
            visualize: Whether to visualize the trajectory.
        """
        self.control_hz = control_hz
        self.visualize = visualize

        # Initialize robot and components
        self.bot = bot if bot is not None else Robot()
        self.rate_limiter = RateLimiter(self.control_hz)

        # Initialize arms and hands
        self.left_arm, self.right_arm = self._initialize_arms(self.bot)

        # Get torso pitch angle if torso exists, otherwise use default
        if self.bot.has_component("torso"):
            self.torso_pitch_angle = self.bot.torso.pitch_angle
            # Get initial joint configuration including torso
            components = ["left_arm", "right_arm"]
            if self.bot.has_component("head"):
                components.append("head")
            components.append("torso")
            initial_joint_configuration = self.bot.get_joint_pos_dict(components)
        else:
            self.torso_pitch_angle = np.pi / 2  # Default for upper body variants
            components = ["left_arm", "right_arm"]
            if self.bot.has_component("head"):
                components.append("head")
            initial_joint_configuration = self.bot.get_joint_pos_dict(components)

        # Setup motion manager
        self.motion_manager = MotionManager(
            init_visualizer=self.visualize,
            initial_joint_configuration_dict=initial_joint_configuration,
            init_local_ik=False,
        )

        # Initialize self-collision avoidance Task
        self.move_out_of_self_collision_task = MoveOutOfSelfCollisionTask(
            initial_joint_configuration=initial_joint_configuration,
            motion_manager=self.motion_manager,
            range_size=0.3,
            visualize=self.visualize,
        )

        # Will be initialized when needed
        self.move_to_configuration_task = MoveToConfigurationTask(
            motion_manager=self.motion_manager,
            planner_type="ompl",
            visualize=self.visualize,  # Use the same visualization setting
        )

    def _initialize_arms(self, bot: Robot) -> tuple[Arm, Arm]:
        """Initialize and configure robot arms and hands.

        Args:
            bot: Robot instance.

        Returns:
            Tuple of left arm, right arm, left hand, and right hand.
        """
        left_arm = bot.left_arm
        right_arm = bot.right_arm

        return left_arm, right_arm

    def _move_trajectory(
        self,
        left_arm: Arm,
        right_arm: Arm,
        trajectory: list[dict[str, float]],
        rate_limiter: RateLimiter,
    ) -> None:
        """Move both arms along a trajectory.

        Args:
            left_arm: Left arm controller.
            right_arm: Right arm controller.
            trajectory: List of joint configurations.
            rate_limiter: Rate limiter for controlling execution speed.
        """
        total_waypoints = len(trajectory)
        for i, waypoint in enumerate(trajectory):
            left_joints = [v for k, v in waypoint.items() if "L_arm" in k]
            right_joints = [v for k, v in waypoint.items() if "R_arm" in k]

            left_arm.set_joint_pos(np.array(left_joints))
            right_arm.set_joint_pos(np.array(right_joints))

            # Show progress for long trajectories
            if total_waypoints > 10 and i % (total_waypoints // 10) == 0:
                logger.info(f"Trajectory progress: {i}/{total_waypoints}")

            rate_limiter.sleep()

    def run(
        self,
        target: Literal["zero", "L_shape", "lift_up"] = "L_shape",
        shutdown_after_execution: bool = True,
        collision_escape_time: float = 2.0,
    ) -> None:
        """Move robot arms to a preset position and opens the hands.

        Args:
            target: Target preset position.
            shutdown_after_execution: Whether to shutdown the robot after execution.
        """
        assert isinstance(self.motion_manager, MotionManager)
        assert self.motion_manager.pin_robot is not None

        joint_names = robot_utils.get_joint_names(self.motion_manager.pin_robot)

        # Check for and resolve self-collisions
        self._handle_self_collisions(
            joint_names,
            collision_escape_time,
        )

        # Update current position
        current_qpos_dict = self.bot.get_joint_pos_dict(["left_arm", "right_arm"])
        self.motion_manager.set_joint_pos(current_qpos_dict)

        # Get goal configuration dictionary
        goal_configuration_dict = self._create_goal_configuration(target)
        # Initialize move to configuration task
        self.move_to_configuration_task = MoveToConfigurationTask(
            initial_joint_configuration=current_qpos_dict,
            motion_manager=self.motion_manager,
        )

        # Move to target configuration
        ts_sample, qs_sample, qds_sample, qdds_sample, duration = (
            self.move_to_configuration_task.run(
                start_configuration_dict=current_qpos_dict,
                goal_configuration_dict=goal_configuration_dict,
                control_frequency=self.control_hz,
                generate_trajectory=True,
            )
        )

        # Visualize trajectory
        if self.visualize:
            if self.motion_manager.visualizer is None:
                raise RuntimeError("Visualizer is not initialized")
            self.motion_manager.visualizer.update_motion_plan(
                motion_plan=qs_sample,
                joint_names=joint_names,
                duration=1.0,
            )
        user_input = input(
            "Press \033[1mENTER\033[0m to run the trajectory on the real robot (or type 'abort' to cancel): "
        )
        if user_input.lower() == "abort":
            logger.info("Trajectory execution aborted by user")
            return

        # Move the robot arms to the target configuration on the real robot
        list_of_qpos_dicts = [
            {joint_name: qpos for joint_name, qpos in zip(joint_names, qpos_array)}
            for qpos_array in qs_sample
        ]
        logger.info(f"Executing trajectory with {len(list_of_qpos_dicts)} waypoints...")
        self._move_trajectory(
            self.left_arm,
            self.right_arm,
            list_of_qpos_dicts,
            self.rate_limiter,
        )
        logger.success(f"Successfully moved to {target} position")

        if shutdown_after_execution:
            if hasattr(self, "bot") and self.bot:
                self.bot.shutdown()
                logger.info("Robot shutdown complete")

    def _handle_self_collisions(
        self,
        joint_names: list[str],
        collision_escape_time: float = 2.0,
    ) -> None:
        """Handle self-collision detection and resolution.

        Args:
            joint_names: List of joint names.
        """
        is_collision_present, success, collision_free_configuration = (
            self.move_out_of_self_collision_task.run()
        )

        if is_collision_present:
            if not success:
                raise RuntimeError("Failed to resolve self-collision")

            assert isinstance(collision_free_configuration, dict)

            if self.motion_manager.pin_robot is None:
                raise RuntimeError("Robot is not initialized")

            # Get initial configuration
            initial_configuration = (
                self.move_out_of_self_collision_task.initial_joint_configuration
            )
            assert isinstance(initial_configuration, dict)

            # Create linearly interpolated trajectory between initial and
            # collision-free configuration
            num_steps = int(self.control_hz * collision_escape_time)
            interpolated_configuration_list = []

            for i in range(num_steps + 1):
                alpha = i / num_steps
                interpolated_configuration = {}
                for joint in joint_names:
                    if (
                        joint in initial_configuration
                        and joint in collision_free_configuration
                    ):
                        interpolated_configuration[joint] = (
                            1 - alpha
                        ) * initial_configuration[
                            joint
                        ] + alpha * collision_free_configuration[joint]
                interpolated_configuration_list.append(interpolated_configuration)

            # Visualize the interpolated trajectory
            if self.visualize:
                if self.motion_manager.visualizer is None:
                    raise RuntimeError("Visualizer is not initialized")
                self.motion_manager.visualizer.update_motion_plan(
                    motion_plan=np.array(
                        [
                            robot_utils.get_qpos_from_joint_dict(
                                self.motion_manager.pin_robot, config
                            )
                            for config in interpolated_configuration_list
                        ]
                    ),
                    joint_names=joint_names,
                    duration=1.0,
                )

            user_input = input(
                "Press Enter to move to collision-free configuration "
                "(or type 'abort' to cancel): "
            )
            if user_input.lower() == "abort":
                raise RuntimeError("User aborted collision resolution")

            self._move_trajectory(
                self.left_arm,
                self.right_arm,
                interpolated_configuration_list,
                self.rate_limiter,
            )
            logger.info("Waiting for robot to stabilize...")
            time.sleep(3)
            logger.success("Moved to collision-free configuration!")

    def _create_goal_configuration(self, target: str) -> dict[str, float]:
        """Create goal configuration dictionary for the target pose.

        Args:
            target: Target preset position name.

        Returns:
            Goal configuration dictionary.
        """
        goal_configuration_dict = {}

        # Get preset joint positions
        left_arm_joint_pos = self.bot.left_arm.get_predefined_pose(target)
        right_arm_joint_pos = self.bot.right_arm.get_predefined_pose(target)

        # Get joint names
        left_arm_joint_name = self.bot.left_arm.joint_name
        right_arm_joint_name = self.bot.right_arm.joint_name

        # Create configuration dictionary
        goal_configuration_dict.update(
            dict(zip(left_arm_joint_name, left_arm_joint_pos))
        )
        goal_configuration_dict.update(
            dict(zip(right_arm_joint_name, right_arm_joint_pos))
        )

        # Adjust for torso pitch
        goal_configuration_dict["L_arm_j1"] += self.torso_pitch_angle
        goal_configuration_dict["R_arm_j1"] -= self.torso_pitch_angle

        return goal_configuration_dict


def main(
    target: Literal["zero", "L_shape", "lift_up"] = "L_shape",
    control_hz: int = 250,
    visualize: bool = True,
    collision_escape_time: float = 2.0,
) -> None:
    """Main entry point for the script.

    Args:
        target: Target preset position.
        control_hz: Control frequency in Hz.
        visualize: Whether to visualize the trajectory.
        collision_escape_time: Time to move to collision-free
        configuration in seconds.
    """
    logger.info(f"Initializing arm safe movement to '{target}' position")
    arm_safe_initializer = ArmSafeInitializer(
        control_hz=control_hz, visualize=visualize
    )
    arm_safe_initializer.run(
        target,
        collision_escape_time=collision_escape_time,
    )


if __name__ == "__main__":
    tyro.cli(main)
