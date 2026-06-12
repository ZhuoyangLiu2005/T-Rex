# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Teleop script for controlling robot chassis using DualSense controller.

This module provides a teleop interface for controlling the robot's chassis movements
using a PlayStation DualSense controller with safety-conscious mappings.
"""

import threading
import time

import numpy as np
import tyro
from dexcomm import RateLimiter
from loguru import logger

from dexcontrol.apps.dualsense_teleop_base import DualSenseTeleopBase
from dexcontrol.utils.compat import supported_models


class ChassisVelocityTeleopNode(DualSenseTeleopBase):
    """Teleop node for controlling robot chassis with velocity commands.

    This class implements velocity-based control for a robot chassis using a DualSense
    controller. It supports forward/backward motion, strafing, and rotation with safety
    features to prevent dangerous combinations of movements.

    Control mapping:
        - dpad_up: Forward motion
        - dpad_down: Backward motion
        - dpad_left: Strafe left
        - dpad_right: Strafe right
        - square: Turn clockwise
        - circle: Turn counter-clockwise
        - left_stick: Head control (pitch/yaw)
        - R1: Increase velocity scale
        - R2: Decrease velocity scale

    Safety features:
        - Prevents combined rotation + strafe commands
        - Gradual velocity ramping
        - Maximum velocity limits
        - Sequential steering for sideways motion
        - Emergency stop functionality
    """

    def __init__(
        self,
        control_hz: int = 200,
        button_update_hz: int = 10,
        device_index: int = 0,
    ) -> None:
        """Initializes the teleop node.

        Args:
            control_hz: Frequency of the control loop in Hz.
            button_update_hz: Frequency of button state updates in Hz.
            device_index: Index of the DualSense controller device.
        """
        super().__init__(control_hz, button_update_hz, device_index)

        # Motion control parameters
        self._max_linear_velocity = (
            self.bot.chassis.max_lin_vel
        )  # Maximum linear velocity (m/s)
        self._max_angular_velocity = (
            self.bot.chassis.max_ang_vel
        )  # Maximum angular velocity (rad/s)
        self._velocity_threshold = 0.02  # Threshold for considering velocity non-zero
        self._head_sensitivity = 0.05  # Head movement sensitivity

        # Default velocities
        self._default_lin_vel = 0.3  # Default linear velocity (m/s)
        self._min_lin_vel = 0.1  # Minimum linear velocity (m/s)
        self._default_ang_vel = 0.8  # Default angular velocity (rad/s)
        self._angular_increment = 0.1  # Angular velocity increment (rad/s)
        self._current_vel_scale = 1.0  # Current velocity scale multiplier

        # Initialize motion state
        self._vx = 0.0  # Forward/backward velocity
        self._vy = 0.0  # Left/right velocity
        self._wz = 0.0  # Angular velocity

        # Initialize head position
        home_pose = self.bot.head.get_predefined_pose("home")
        target_pos = self.bot.compensate_torso_pitch(home_pose, "head")
        self.bot.head.set_joint_pos(target_pos, wait_time=2.0)
        self._head_enabled = True
        self._head_qpos = self.bot.head.get_joint_pos()

        # Configure additional button mappings
        self.dualsense.btn_r1.on_down(self._increase_velocity)
        self.dualsense.btn_r2.on_down(self._decrease_velocity)
        self.dualsense.btn_l2.on_down(self._toggle_head_control)

        # Start control thread
        self._control_thread = threading.Thread(target=self._control_loop)
        self._control_thread.daemon = True
        self._control_thread.start()

        logger.info("ChassisVelocityTeleopNode initialized")

    def update_controller_feedback(self) -> None:
        """Updates controller lightbar based on current state."""
        self.dualsense.lightbar.set_color(0, 0, 255)  # Blue for chassis mode

    def update_motion(self) -> None:
        """Updates motion variables based on controller input."""
        # Check safety conditions
        with self.estop_lock:
            if not self.safe_pressed or self.estop_on:
                self.stop_all_motion()
                return

        active_buttons = self.get_active_buttons()

        # Update chassis motion
        self._update_forward_motion(active_buttons)
        self._update_strafe_motion(active_buttons)
        self._update_rotation(active_buttons)

        # Update head position using left analog stick
        self._update_head_position()

        print(f"Steering pos in radians: {self.bot.chassis.steering_angle} ")
        print(f"Wheel encoder pos in m: {self.bot.chassis.wheel_encoder_pos} ")
        print(f"Wheel vel in m/s: {self.bot.chassis.wheel_velocity} ")

    def _toggle_head_control(self) -> None:
        """Toggles the head control mode."""
        if self._head_enabled:
            self.bot.head.set_mode("disable")
            self._head_enabled = False
        else:
            self.bot.head.set_mode("enable")
            self._head_enabled = True
            time.sleep(0.2)
            self._head_qpos = self.bot.head.get_joint_pos()

    def _increase_velocity(self) -> None:
        """Increases the velocity scale when R1 is pressed."""
        if self.safe_pressed:
            self._current_vel_scale = min(
                self._current_vel_scale + 0.2,
                self._max_linear_velocity / self._default_lin_vel,
            )

    def _decrease_velocity(self) -> None:
        """Decreases the velocity scale when R2 is pressed."""
        if self.safe_pressed:
            self._current_vel_scale = max(
                self._current_vel_scale - 0.2, self._min_lin_vel / self._default_lin_vel
            )

    def _update_forward_motion(self, active_buttons: set) -> None:
        """Updates forward/backward velocity based on button state.

        Args:
            active_buttons: Set of currently active buttons.
        """
        if "dpad_up" in active_buttons:
            self._vx = self._default_lin_vel * self._current_vel_scale
        elif "dpad_down" in active_buttons:
            self._vx = -self._default_lin_vel * self._current_vel_scale
        else:
            self._vx = self._reduce_velocity(
                self._vx, velocity_threshold=self._min_lin_vel
            )

    def _update_strafe_motion(self, active_buttons: set) -> None:
        """Updates strafe velocity based on button state.

        Args:
            active_buttons: Set of currently active buttons.
        """
        if {"circle", "square"}.isdisjoint(
            active_buttons
        ):  # No rotation buttons pressed
            if "dpad_left" in active_buttons:
                self._vy = self._default_lin_vel * self._current_vel_scale
            elif "dpad_right" in active_buttons:
                self._vy = -self._default_lin_vel * self._current_vel_scale
            else:
                self._vy = self._reduce_velocity(
                    self._vy, velocity_threshold=self._min_lin_vel
                )
        else:
            self._vy = 0.0

    def _update_rotation(self, active_buttons: set) -> None:
        """Updates rotational velocity based on button state.

        Args:
            active_buttons: Set of currently active buttons.

        Note:
            When moving backwards (vx < 0), rotation direction is reversed
            to maintain intuitive control from driver's perspective.
        """
        if {"dpad_left", "dpad_right"}.isdisjoint(
            active_buttons
        ):  # No strafe buttons pressed
            direction = (
                -1 if self._vx < 0 else 1
            )  # Reverse rotation when moving backwards

            if "square" in active_buttons:
                self._wz = self._ramp_velocity(
                    self._wz,
                    direction,
                    self._angular_increment,
                    max_value=self._default_ang_vel,
                )
            elif "circle" in active_buttons:
                self._wz = self._ramp_velocity(
                    self._wz,
                    -direction,
                    self._angular_increment,
                    max_value=self._default_ang_vel,
                )
            else:
                self._wz = self._reduce_velocity(self._wz, velocity_threshold=0.2)
        else:
            self._wz = 0.0

    def _update_head_position(self) -> None:
        """Updates head position based on left analog stick input."""
        stick_value = self.dualsense.right_stick.value

        # Apply sensitivity factor and invert x-axis for intuitive control
        delta_head_qpos = (
            np.array(
                [
                    stick_value.y,  # Pitch (up/down)
                    -stick_value.x,  # Yaw (left/right)
                    0,  # Roll (fixed)
                ]
            )
            * self._head_sensitivity
        )

        self._head_qpos += delta_head_qpos

        # Clamp head position within safe limits
        head_min_limits = np.array(
            [-np.pi / 3, -np.pi / 6 * 5, -np.pi / 6]
        )  # Lower bounds
        head_max_limits = np.array(
            [np.pi / 3, np.pi / 6 * 5, np.pi / 6]
        )  # Upper bounds
        self._head_qpos = np.clip(self._head_qpos, head_min_limits, head_max_limits)

    def _reduce_velocity(
        self,
        current: float,
        reduction_factor: float = 0.1,
        velocity_threshold: float = 0.01,
    ) -> float:
        """Gradually reduces velocity towards zero.

        Args:
            current: Current velocity value.
            reduction_factor: Factor to reduce velocity by.
            velocity_threshold: Threshold below which velocity is set to zero.

        Returns:
            New velocity value.
        """
        if abs(current) < velocity_threshold:
            return 0.0
        return current * (1 - reduction_factor)

    def _ramp_velocity(
        self, current: float, direction: int, increment: float, max_value: float
    ) -> float:
        """Ramps velocity towards a target value.

        Args:
            current: Current velocity value.
            direction: Direction of the ramp (1 for positive, -1 for negative).
            increment: Velocity increment per step.
            max_value: Maximum allowed velocity value.

        Returns:
            New velocity value.
        """
        target = current + direction * increment
        return min(max(target, -max_value), max_value)

    def _control_loop(self) -> None:
        """Main control loop that sends commands to the robot."""
        limiter = RateLimiter(self.control_hz)

        while self.is_running:
            if self.safe_pressed and not self.estop_on:
                if self._head_enabled:
                    # Update head position
                    self.bot.head.set_joint_pos(self._head_qpos)

                # Send velocity commands to chassis
                self.bot.chassis.set_velocity(
                    vx=self._vx,
                    vy=self._vy,
                    wz=self._wz,
                    sequential_steering=abs(self._vy) > self._velocity_threshold,
                )

            limiter.sleep()

    def stop_all_motion(self) -> None:
        """Stops all robot motion and resets control variables."""
        self._vx = 0.0
        self._vy = 0.0
        self._wz = 0.0
        self._current_vel_scale = 1.0  # Reset velocity scale

        try:
            if hasattr(self, "bot") and self.bot and not self.bot.is_shutdown():
                self.bot.chassis.stop()
        except Exception as e:
            logger.error(f"Error stopping motion: {e}")


@supported_models("vega_1", "vega_1p")
def main(
    control_hz: int = 400,
    button_update_hz: int = 20,
    device_index: int = 0,
) -> None:
    """Main entry point for the teleop application.

    Args:
        control_hz: Control loop frequency in Hz.
        button_update_hz: Frequency of button state updates in Hz.
        device_index: Index of the DualSense controller device.
    """
    teleop_node = ChassisVelocityTeleopNode(
        control_hz=control_hz,
        button_update_hz=button_update_hz,
        device_index=device_index,
    )
    teleop_node.run_forever()


if __name__ == "__main__":
    tyro.cli(main)
