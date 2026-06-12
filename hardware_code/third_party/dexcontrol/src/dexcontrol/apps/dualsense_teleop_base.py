# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""DualSense controller teleoperation base module.

This module provides a base class for controlling robots using a DualSense controller,
with safety features, emergency stop handling, and customizable control mappings.
"""

import threading
import time
from abc import ABC, abstractmethod
from enum import Enum

from dexcomm import RateLimiter
from dualsense_controller import DualSenseController
from loguru import logger

from dexcontrol.robot import Robot


class ControlMode(Enum):
    """Control modes for the teleop interface."""

    POSITION = "position"
    VELOCITY = "velocity"
    TORQUE = "torque"


class ButtonState:
    """Class to track button state with debouncing."""

    def __init__(self, debounce_time: float = 0.05):
        """Initialize button state.

        Args:
            debounce_time: Time in seconds to wait before considering a state change valid.
        """
        self._raw_state = False
        self._debounced_state = False
        self._last_change_time = 0.0
        self._debounce_time = debounce_time
        self._lock = threading.Lock()

    def update(self, new_state: bool) -> None:
        """Update the button state with debouncing.

        Args:
            new_state: New raw button state.
        """
        with self._lock:
            current_time = time.time()
            if new_state != self._raw_state:
                self._last_change_time = current_time
                self._raw_state = new_state

            # Only update debounced state if enough time has passed
            if current_time - self._last_change_time >= self._debounce_time:
                self._debounced_state = self._raw_state

    @property
    def state(self) -> bool:
        """Get the current debounced button state."""
        with self._lock:
            return self._debounced_state


class DualSenseTeleopBase(ABC):
    """Base class for DualSense controller teleoperation.

    This class provides basic functionality for controlling a robot using a DualSense
    controller, including safety features, e-stop handling, and button/stick mappings.

    Attributes:
        is_running: Flag indicating if the control loop is active.
        control_hz: Control loop frequency in Hz.
        button_update_hz: Button update frequency in Hz.
        button_debounce_time: Time in seconds to wait before considering a button state change valid.
        dualsense: Interface to the DualSense controller.
        button_states: Dictionary to track button states with debouncing.
        active_buttons: Currently pressed buttons.
        safe_pressed: Safety button (L1) state.
        estop_on: Emergency stop state.
        estop_lock: Thread lock for e-stop state changes.
        bot: Robot interface instance.
    """

    def __init__(
        self,
        control_hz: int = 200,
        button_update_hz: int = 20,
        device_index: int = 0,
        button_debounce_time: float = 0.05,
    ) -> None:
        """Initialize the DualSense teleoperation base.

        Args:
            control_hz: Control loop frequency in Hz. Defaults to 200.
            button_update_hz: Button update frequency in Hz. Defaults to 20.
            device_index: Index of the DualSense controller device. Defaults to 0.
            button_debounce_time: Time in seconds to wait before considering a button state change valid.
        """
        self.is_running = True
        self.control_hz = control_hz
        self.button_update_hz = button_update_hz
        self.button_debounce_time = button_debounce_time

        # Thread safety
        self.button_lock = threading.Lock()
        self.estop_lock = threading.Lock()

        # Controller setup
        try:
            self.dualsense = DualSenseController(
                device_index_or_device_info=device_index
            )
            self.dualsense.activate()
            logger.info(
                f"DualSense controller activated (device index: {device_index})"
            )
        except Exception as e:
            logger.error(f"Failed to initialize DualSense controller: {e}")
            raise

        # Button state tracking
        self.button_states: dict[str, ButtonState] = {}
        self.active_buttons: set[str] = set()

        # State flags
        self.safe_pressed = False
        self.estop_on = False

        # Control state
        self.control_mode = ControlMode.POSITION
        self.rate_limiter = RateLimiter(self.button_update_hz)

        # Robot interface
        try:
            self.bot = Robot()
            logger.info(f"Robot '{self.bot.robot_model}' initialized")
        except Exception as e:
            logger.error(f"Failed to initialize robot: {e}")
            self.dualsense.deactivate()
            raise

        # Configure button mappings
        self._setup_button_mappings()

    def _setup_button_mappings(self) -> None:
        """Set up controller button mappings and callbacks."""
        # Safety and e-stop buttons
        self.dualsense.btn_l1.on_down(self.safety_check)
        self.dualsense.btn_l1.on_up(self.safety_check_release)
        self.dualsense.btn_touchpad.on_down(self.toggle_estop)

        # Standard button mappings
        button_mapping = {
            "btn_up": "dpad_up",
            "btn_down": "dpad_down",
            "btn_right": "dpad_right",
            "btn_left": "dpad_left",
            "btn_circle": "circle",
            "btn_square": "square",
            "btn_triangle": "triangle",
            "btn_cross": "cross",
            "btn_r1": "r1",
            "btn_r2": "r2",
        }

        for btn_name, motion_name in button_mapping.items():
            btn = getattr(self.dualsense, btn_name)
            # Create button state tracker
            self.button_states[motion_name] = ButtonState(self.button_debounce_time)
            # Use default parameter to capture current value
            btn.on_down(lambda m=motion_name: self._update_button_state(m, True))
            btn.on_up(lambda m=motion_name: self._update_button_state(m, False))

        # Setup additional mappings specific to the subclass
        self._setup_additional_mappings()

    def _setup_additional_mappings(self) -> None:
        """Set up additional controller mappings specific to subclasses.

        This method can be overridden by subclasses to add more button mappings.
        """
        pass

    def _update_button_state(self, button: str, state: bool) -> None:
        """Update the state of a button with debouncing.

        Args:
            button: Name of the button being updated.
            state: New raw button state.
        """
        if button in self.button_states:
            self.button_states[button].update(state)
            if state:
                self.add_button(button)
            else:
                self.remove_button(button)

    def add_button(self, button: str) -> None:
        """Add a button to the set of active buttons if safety is enabled.

        Args:
            button: Name of the button being pressed.
        """
        if self.safe_pressed or button in ["l1", "touchpad"]:
            with self.button_lock:
                self.active_buttons.add(button)
                logger.debug(f"Button added: {button}")

    def remove_button(self, button: str) -> None:
        """Remove a button from the set of active buttons.

        Args:
            button: Name of the button being released.
        """
        with self.button_lock:
            self.active_buttons.discard(button)
            logger.debug(f"Button removed: {button}")

    def toggle_estop(self) -> None:
        """Toggle the emergency stop state of the robot.

        When activated, this will:
        1. Toggle the robot's software e-stop
        2. Stop any ongoing motion
        3. Reset all motion commands to zero
        """
        try:
            with self.estop_lock:
                self.estop_on = not self.estop_on
                if self.estop_on:
                    self.stop_all_motion()
                    self.bot.estop.activate()
                else:
                    self.bot.estop.deactivate()

            logger.info(f"E-stop {'activated' if self.estop_on else 'deactivated'}")

            # Visual feedback on controller
            if self.estop_on:
                self.dualsense.lightbar.set_color(255, 0, 0)  # Red for e-stop
            else:
                self.update_controller_feedback()

        except Exception as e:
            logger.error(f"Error toggling e-stop: {e}")

    def safety_check(self) -> None:
        """Enable safety mode (L1 button pressed).

        Activates haptic feedback and enables safety-gated commands.
        """
        self.dualsense.left_rumble.set(50)
        self.safe_pressed = True
        self.dualsense.left_rumble.set(0)
        logger.debug("Safety mode enabled")

    def safety_check_release(self) -> None:
        """Disable safety mode (L1 button released).

        Deactivates haptic feedback and:
        1. Clears all active buttons
        2. Stops any ongoing motion
        3. Resets all motion commands to zero
        """
        self.safe_pressed = False
        self.dualsense.left_rumble.set(0)

        # Clear all active buttons with lock protection
        with self.button_lock:
            self.active_buttons.clear()

        # Stop any ongoing motion
        self.stop_all_motion()
        logger.debug("Safety mode disabled, motion stopped")

    def get_active_buttons(self) -> set[str]:
        """Get a copy of currently active buttons in a thread-safe way.

        Returns:
            A copy of the set of currently active buttons.
        """
        with self.button_lock:
            return self.active_buttons.copy()

    def update_controller_feedback(self) -> None:
        """Update controller visual feedback based on current state.

        This method can be overridden by subclasses to customize feedback.
        """
        # Default is white lightbar
        self.dualsense.lightbar.set_color_white()

    def run_forever(self) -> None:
        """Run the control loop indefinitely until interrupted.

        The loop runs at the specified button_update_hz frequency.
        Handles KeyboardInterrupt for clean shutdown.
        """
        logger.info("Starting teleop control loop")
        rate_limiter = RateLimiter(self.button_update_hz)
        try:
            while self.is_running and not self.bot.is_shutdown():
                self.update_motion()
                rate_limiter.sleep()
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, shutting down")
        except Exception as e:
            logger.error(f"Error in control loop: {e}")
            raise
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Clean up resources before shutting down."""
        self.is_running = False

        # Reset controller
        if hasattr(self, "dualsense"):
            self.dualsense.lightbar.set_color_white()
            self.dualsense.deactivate()

        logger.info("Teleop node cleaned up and exiting")

    @abstractmethod
    def update_motion(self) -> None:
        """Update the robot's motion based on current controller state.

        This method should be implemented by derived classes to define
        specific motion control behavior.
        """
        pass

    @abstractmethod
    def stop_all_motion(self) -> None:
        """Stop all ongoing robot motion and reset motion commands.

        This method should be implemented by derived classes to properly
        handle motion stopping for their specific use case.
        """
        pass


class DummyDualSenseTeleop(DualSenseTeleopBase):
    """Simple implementation of DualSenseTeleopBase for testing purposes."""

    def update_motion(self) -> None:
        """Dummy implementation of update_motion."""
        if self.active_buttons:
            logger.debug(f"Active buttons: {self.active_buttons}")
        pass

    def stop_all_motion(self) -> None:
        """Dummy implementation of stop_all_motion."""
        logger.info("Stopping all motion (dummy implementation)")
        pass


def test_dummy_dualsense_teleop() -> None:
    """Test function for the DummyDualSenseTeleop class."""
    teleop = DummyDualSenseTeleop()
    teleop.run_forever()


if __name__ == "__main__":
    test_dummy_dualsense_teleop()
