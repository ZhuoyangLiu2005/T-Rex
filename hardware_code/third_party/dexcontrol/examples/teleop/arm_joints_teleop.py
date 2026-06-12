# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Teleop script for controlling robot arm joints using DualSense controller.

This module provides a teleop interface for controlling the robot's arm joints
using a PlayStation DualSense controller with direct joint-level control.
"""

import numpy as np
import tyro
from base_arm_teleop import BaseArmTeleopNode, BaseIKController
from dexcomm import RateLimiter
from loguru import logger


class JointTeleopNode(BaseArmTeleopNode):
    """Teleop node for controlling robot arm joints directly.

    This class maps DualSense controller inputs to joint-level robot arm
    movements, allowing precise control over individual joints of the robot.

    Attributes:
        current_joint_idx: Index of the currently selected joint (0-6).
        l2_pressed: Flag indicating whether the L2 button is currently pressed.
        MAX_JOINT_NUMBER: Maximum joint index supported (6).
        MIN_JOINT_NUMBER: Minimum joint index supported (0).
        JOINT_STEP_SIZE: Step size for joint movements in radians.
    """

    def __init__(
        self,
        control_hz: int = 200,
        button_update_hz: int = 20,
        device_index: int = 0,
        visualize: bool = False,
    ):
        """Initialize the joint teleop node.

        Args:
            control_hz: Frequency of the control loop in Hz.
            button_update_hz: Frequency of button state updates in Hz.
            device_index: Index of the DualSense controller to use.
            visualize: Whether to enable visualization.
        """
        super().__init__(control_hz, button_update_hz, device_index)

        # Constants
        self.MAX_JOINT_NUMBER = 6
        self.MIN_JOINT_NUMBER = 0
        self.JOINT_STEP_SIZE = 0.3  # radians

        self.arm_ik_controller = BaseIKController(self.bot, visualize=visualize)
        self.current_joint_idx = 0  # Initialize with joint 0 selected
        self.l2_pressed = False  # L2 button state for modifier functionality

        logger.info(
            f"Joint teleop initialized. Selected joint: {self.current_joint_idx}"
        )

    def set_joint_idx(self, idx: int) -> None:
        """Set the current joint index.

        Args:
            idx: Joint index to select (must be between MIN_JOINT_NUMBER and
                MAX_JOINT_NUMBER).
        """
        if self.MIN_JOINT_NUMBER <= idx <= self.MAX_JOINT_NUMBER:
            self.current_joint_idx = idx
            logger.info(
                f"Selected joint: {self.current_joint_idx} for {self.arm_side} arm"
            )

    def all_time_loop(self) -> None:
        """Main loop that sets up controller callbacks and runs continuously.

        This method configures button callbacks and runs the main control loop
        until the program is terminated.
        """
        # Set up controller callbacks
        self.dualsense.btn_r1.on_down(self.toggle_arm)

        # L2 modifier button
        self.dualsense.btn_l2.on_down(lambda: setattr(self, "l2_pressed", True))
        self.dualsense.btn_l2.on_up(lambda: setattr(self, "l2_pressed", False))

        # Joint selection buttons
        self.dualsense.btn_cross.on_down(self.on_joint_selection)
        self.dualsense.btn_circle.on_down(self.on_joint_selection)
        self.dualsense.btn_triangle.on_down(self.on_joint_selection)
        self.dualsense.btn_square.on_down(self.on_joint_selection)

        # Joint position control - continuous motion
        self.dualsense.btn_up.on_down(lambda: self.add_button("dpad_up"))
        self.dualsense.btn_up.on_up(lambda: self.remove_button("dpad_up"))
        self.dualsense.btn_down.on_down(lambda: self.add_button("dpad_down"))
        self.dualsense.btn_down.on_up(lambda: self.remove_button("dpad_down"))

        # L1 safe button only operate while pressed
        self.dualsense.btn_l1.on_down(self.safety_check)
        self.dualsense.btn_l1.on_up(self.safety_check_release)

        # Touchpad for e-stop toggle
        self.dualsense.btn_touchpad.on_down(self.toggle_estop)

        # Start the arm control thread
        self.arm_control_thread.start()

        # Main loop to keep the program running and update motion
        rate_limiter = RateLimiter(self.button_update_hz)
        logger.info("Starting arm teleop loop")
        try:
            while self.is_running:
                self.update_motion()
                rate_limiter.sleep()
        finally:
            self._cleanup()

    def update_motion(self) -> None:
        """Update motion based on active buttons.

        This method calculates the appropriate joint position changes based on
        active buttons, then applies the changes through the IK controller.
        """
        with self.estop_lock:
            estop_on = self.estop_on

        if not self.safe_pressed or estop_on or not self.active_buttons:
            return

        arm = self.arms[self.arm_side]
        current_position = arm.get_joint_pos()
        new_position = current_position.copy()

        # Apply joint changes based on active buttons
        if "dpad_up" in self.active_buttons:
            new_position[self.current_joint_idx] += self.JOINT_STEP_SIZE
        if "dpad_down" in self.active_buttons:
            new_position[self.current_joint_idx] -= self.JOINT_STEP_SIZE

        # Only proceed if there's been a change in the joint position
        if np.array_equal(current_position, new_position):
            return

        # Create joint position dictionary with the arm prefix
        arm_prefix = "L" if self.arm_side == "left" else "R"
        new_joint_pos_dict = {
            f"{arm_prefix}_arm_j{self.current_joint_idx + 1}": (
                new_position[self.current_joint_idx]
            )
        }

        # Apply the change through IK
        new_joint_pos_dict = self.arm_ik_controller.move_delta_joint_space(
            new_joint_pos_dict
        )

        # Extract the updated joint positions for the current arm
        new_joint_pos = [
            new_joint_pos_dict[f"{arm_prefix}_arm_j{i + 1}"] for i in range(7)
        ]

        # Update the target joint position
        with self.arm_motion_lock:
            self.arm_target_qpos[self.arm_side] = np.array(new_joint_pos)

    def on_joint_selection(self) -> None:
        """Handle joint selection based on button presses and L2 modifier.

        This method selects different joints based on which button is pressed
        and whether the L2 modifier is active:
        - Without L2: Buttons select joints 0-3
        - With L2: Buttons select joints 4-6
        """
        if self.l2_pressed:
            # L2 + button combinations for joints 4-6
            if self.dualsense.btn_cross.pressed:
                self.set_joint_idx(4)
            elif self.dualsense.btn_circle.pressed:
                self.set_joint_idx(5)
            elif self.dualsense.btn_triangle.pressed:
                self.set_joint_idx(6)
        else:
            # Regular button presses for joints 0-3
            if self.dualsense.btn_cross.pressed:
                self.set_joint_idx(0)
            elif self.dualsense.btn_circle.pressed:
                self.set_joint_idx(1)
            elif self.dualsense.btn_triangle.pressed:
                self.set_joint_idx(2)
            elif self.dualsense.btn_square.pressed:
                self.set_joint_idx(3)


def main(
    control_hz: int = 200,
    button_update_hz: int = 20,
    device_index: int = 0,
    visualize: bool = False,
) -> None:
    """Main entry point for the teleop application.

    Args:
        control_hz: Frequency of the control loop in Hz.
        button_update_hz: Frequency of button state updates in Hz.
        device_index: Index of the DualSense controller to use.
        visualize: Whether to enable visualization.
    """
    teleop_node = JointTeleopNode(
        control_hz=control_hz,
        button_update_hz=button_update_hz,
        device_index=device_index,
        visualize=visualize,
    )

    # Open robot hands
    logger.info("Opening robot hands...")
    if teleop_node.bot.have_hand("left"):
        teleop_node.bot.left_hand.open_hand()
    if teleop_node.bot.have_hand("right"):
        teleop_node.bot.right_hand.open_hand()

    # Start the teleop loop
    teleop_node.all_time_loop()


if __name__ == "__main__":
    tyro.cli(main)
