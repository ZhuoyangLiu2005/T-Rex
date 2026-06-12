# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Teleop script for controlling robot arms in Cartesian space.

This module provides a teleop interface for controlling robot arms using a DualSense
controller to move the end-effectors in Cartesian space with both position and orientation control.
"""

import numpy as np
import tyro
from base_arm_teleop import BaseArmTeleopNode, BaseIKController
from dexcomm import RateLimiter
from loguru import logger


class CartesianTeleopNode(BaseArmTeleopNode):
    """Teleop node for controlling robot arms in Cartesian space.

    This class maps DualSense controller inputs to Cartesian space movements,
    allowing intuitive control of robot arm end-effectors in both position and
    orientation modes.

    Attributes:
        rpy_mode: Flag to toggle between translation (xyz) and rotation (rpy) modes.
        translation_step: Step size for translations in meters.
        rotation_step: Step size for rotations in radians.
    """

    def __init__(
        self,
        control_hz: int = 200,
        button_update_hz: int = 20,
        device_index: int = 0,
        visualize: bool = False,
    ):
        """Initialize the Cartesian teleop node.

        Args:
            control_hz: Control loop frequency in Hz.
            button_update_hz: Button update frequency in Hz.
            device_index: Index of the DualSense controller device.
        """
        super().__init__(control_hz, button_update_hz, device_index)
        self.rpy_mode = False
        self.arm_ik_controller = BaseIKController(self.bot, visualize=visualize)

        # Motion control parameters
        self.translation_step = 0.01  # x, y, z steps in meters
        self.rotation_step = 0.05  # roll, pitch, yaw steps in radians

        # Update player LEDs for mode indication
        self.update_player_leds()
        logger.info("Cartesian teleop initialized in translation mode")

    def update_player_leds(self) -> None:
        """Update the player LEDs based on the current mode (xyz/rpy)."""
        if self.rpy_mode:
            # Outer LEDs for RPY mode
            self.dualsense.player_leds.set_outer()
        else:
            # Inner LEDs for XYZ mode
            self.dualsense.player_leds.set_inner()
        # Set brightness to high for better visibility
        self.dualsense.player_leds.set_brightness_high()

    def toggle_rpy_mode(self) -> None:
        """Toggle between translation and rotation mode."""
        self.rpy_mode = not self.rpy_mode
        logger.info(f"Mode: {'rotation' if self.rpy_mode else 'translation'}")
        self.update_player_leds()

    def all_time_loop(self) -> None:
        """Main control loop that runs continuously.

        This method sets up button callbacks and runs the main control loop
        until the program is terminated.
        """
        # Map DualSense buttons to motion controls
        button_mappings = {
            "btn_up": "dpad_up",
            "btn_down": "dpad_down",
            "btn_right": "dpad_right",
            "btn_left": "dpad_left",
            "btn_circle": "circle",
            "btn_square": "square",
            "btn_triangle": "triangle",
            "btn_cross": "cross",
        }

        # Register button callbacks
        for btn_name, motion_name in button_mappings.items():
            btn = getattr(self.dualsense, btn_name)
            # Use lambda with default argument to capture the current value
            btn.on_down(lambda m=motion_name: self.add_button(m))
            btn.on_up(lambda m=motion_name: self.remove_button(m))

        # Safety and mode buttons
        self.dualsense.btn_l1.on_down(self.safety_check)
        self.dualsense.btn_l1.on_up(self.safety_check_release)
        self.dualsense.btn_r2.on_down(self.toggle_rpy_mode)
        self.dualsense.btn_touchpad.on_down(self.toggle_estop)
        self.dualsense.btn_r1.on_down(self.toggle_arm)

        # Start the arm control thread
        self.arm_control_thread.start()

        rate_limiter = RateLimiter(self.button_update_hz)
        try:
            while self.is_running:
                self.update_motion()
                rate_limiter.sleep()
        finally:
            self._cleanup()

    def update_motion(self) -> None:
        """Update arm motion based on active buttons.

        This method calculates the appropriate motion commands based on the
        active buttons and current mode (translation or rotation), then
        applies the motion using the IK controller.
        """
        with self.estop_lock:
            estop_on = self.estop_on

        if not self.safe_pressed or estop_on or not self.active_buttons:
            return

        # Calculate motion delta based on active buttons
        delta_arm_motion = np.zeros(6)

        # Map buttons to appropriate motion axes based on mode
        # Format: (xyz_axis, rpy_axis, xyz_dir, rpy_dir)
        button_motion_map = {
            "triangle": (2, 5, 1, 1),  # Z+/Yaw+
            "cross": (2, 5, -1, -1),  # Z-/Yaw-
            "dpad_up": (0, 3, 1, 1),  # X+/Roll+
            "dpad_down": (0, 3, -1, -1),  # X-/Roll-
            "dpad_right": (1, 4, -1, -1),  # Y-/Pitch-
            "dpad_left": (1, 4, 1, 1),  # Y+/Pitch+
        }

        for button, (xyz_axis, rpy_axis, xyz_dir, rpy_dir) in button_motion_map.items():
            if button in self.active_buttons:
                if not self.rpy_mode:
                    delta_arm_motion[xyz_axis] += xyz_dir * self.translation_step
                else:
                    delta_arm_motion[rpy_axis] += rpy_dir * self.rotation_step

        if not np.allclose(delta_arm_motion, np.zeros(6)):
            target_qpos = self.arm_ik_controller.move_delta_cartesian(
                delta_arm_motion[:3],  # Translation (x, y, z)
                delta_arm_motion[3:],  # Rotation (roll, pitch, yaw)
                self.arm_side,
            )
            with self.arm_motion_lock:
                self.arm_target_qpos[self.arm_side] = target_qpos


def main(
    control_hz: int = 400,
    button_update_hz: int = 20,
    device_index: int = 0,
    visualize: bool = False,
) -> None:
    """Main entry point for the teleop script.

    Args:
        robot_model: Name of the robot configuration to use.
        control_hz: Control loop frequency in Hz.
        button_update_hz: Button update frequency in Hz.
        device_index: Index of the DualSense controller device.
    """
    teleop_node = CartesianTeleopNode(
        control_hz=control_hz,
        button_update_hz=button_update_hz,
        device_index=device_index,
        visualize=visualize,
    )
    try:
        teleop_node.all_time_loop()
    except KeyboardInterrupt:
        logger.info("Program terminated by user")
    except Exception as e:
        logger.error(f"Program terminated due to error: {e}")
        raise


if __name__ == "__main__":
    tyro.cli(main)
