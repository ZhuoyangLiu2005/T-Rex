# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Robot hand control module.

This module provides the Hand class for controlling a robotic hand through Zenoh
communication. It handles joint position control and state monitoring.
"""

from typing import Any, Literal, cast

import numpy as np
from dexbot_utils import RobotInfo
from dexbot_utils.configs.components.vega_1 import (
    DexSGripperConfig,
    F5D6HandV1Config,
    F5D6HandV2Config,
)
from dexcomm.codecs import (
    DictDataCodec,
    FingertipForceCodec,
    JointCmdCodec,
    JointStateCodec,
)
from jaxtyping import Float
from loguru import logger

from dexcontrol.core.component import RobotJointComponent
from dexcontrol.exceptions import ServiceUnavailableError


class Hand(RobotJointComponent):
    """Robot hand control class.

    This class provides methods to control a robotic hand by publishing commands and
    receiving state information through Zenoh communication.
    """

    def __init__(
        self,
        name: str,
        robot_info: RobotInfo,
    ) -> None:
        """Initialize the hand controller.

        Args:
            name: Name of the hand component Node.
            configs: Hand configuration parameters containing communication topics
                and predefined hand positions.
        """
        joint_names = robot_info.get_component_joints(name)
        config = robot_info.get_component_config(name)
        config = cast(F5D6HandV1Config, config)
        super().__init__(
            name=name,
            state_sub_topic=config.state_sub_topic,
            control_pub_topic=config.control_pub_topic,
            state_decoder=JointStateCodec.decode,
            control_encoder=JointCmdCodec.encode,
            joint_name=joint_names,
        )

        # Store predefined hand positions as private attributes
        self._joint_pos_open = np.array(config.pose_pool["open"], dtype=np.float32)
        self._joint_pos_close = np.array(config.pose_pool["close"], dtype=np.float32)

    def _send_position_command(
        self, joint_pos: Float[np.ndarray, " N"] | list[float]
    ) -> None:
        """Send joint position control commands to the hand.

        Args:
            joint_pos: Joint positions as list or numpy array.
        """
        joint_pos_array = self._convert_joint_cmd_to_array(joint_pos)
        data = dict(pos=joint_pos_array)
        self._publish_control(data)

    def open_hand(
        self,
        wait_time: float = 0.0,
        exit_on_reach: bool = False,
        exit_on_reach_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Open the hand to the predefined open position.

        Args:
            wait_time: Time to wait after opening the hand.
            exit_on_reach: If True, the function will exit when the joint positions are reached.
            exit_on_reach_kwargs: Optional parameters for exit when the joint positions are reached.
        """
        self.set_joint_pos(
            self._joint_pos_open,
            wait_time=wait_time,
            exit_on_reach=exit_on_reach,
            exit_on_reach_kwargs=exit_on_reach_kwargs,
        )

    def close_hand(
        self,
        wait_time: float = 0.0,
        exit_on_reach: bool = False,
        exit_on_reach_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Close the hand to the predefined closed position.

        Args:
            wait_time: Time to wait after closing the hand.
            exit_on_reach: If True, the function will exit when the joint positions are reached.
            exit_on_reach_kwargs: Optional parameters for exit when the joint positions are reached.

        """
        self.set_joint_pos(
            self._joint_pos_close,
            wait_time=wait_time,
            exit_on_reach=exit_on_reach,
            exit_on_reach_kwargs=exit_on_reach_kwargs,
        )


class HandF5D6(Hand):
    """Specialized hand class for the F5D6 hand model.

    Extends the basic Hand class with additional control methods specific to
    the F5D6 hand model.
    """

    def __init__(
        self,
        name: str,
        robot_info: RobotInfo,
    ) -> None:
        super().__init__(name, robot_info)
        config = robot_info.get_component_config(name)
        config = cast(F5D6HandV1Config, config)

        # Initialize touch sensor for F5D6_V2 hands

    def close_hand(
        self,
        wait_time: float = 0.0,
        exit_on_reach: bool = False,
        exit_on_reach_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Close the hand fully using a two-step approach for better control."""
        try:
            if self.is_joint_pos_reached(self._joint_pos_close, tolerance=0.1):
                return

            # First step: Move to intermediate position
            intermediate_pos = self._get_intermediate_close_position()
            first_step_wait_time = 0.8
            self.set_joint_pos(
                intermediate_pos,
                wait_time=first_step_wait_time,
                exit_on_reach=exit_on_reach,
                exit_on_reach_kwargs=exit_on_reach_kwargs,
            )

            # Second step: Move to final closed position
            remaining_wait_time = max(0.0, wait_time - first_step_wait_time)
            self.set_joint_pos(
                self._joint_pos_close,
                wait_time=remaining_wait_time,
                exit_on_reach=exit_on_reach,
                exit_on_reach_kwargs=exit_on_reach_kwargs,
            )
        except Exception as e:
            logger.warning(f"Failed to close hand: {e}")

    def open_hand(
        self,
        wait_time: float = 0.0,
        exit_on_reach: bool = False,
        exit_on_reach_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Open the hand fully using a two-step approach for better control."""
        try:
            if self.is_joint_pos_reached(self._joint_pos_open, tolerance=0.1):
                return

            # First step: Move to intermediate position
            intermediate_pos = self._get_intermediate_open_position()
            first_step_wait_time = 0.3
            self.set_joint_pos(
                intermediate_pos,
                wait_time=first_step_wait_time,
                exit_on_reach=exit_on_reach,
                exit_on_reach_kwargs=exit_on_reach_kwargs,
            )

            # Second step: Move to final open position
            remaining_wait_time = max(0.0, wait_time - first_step_wait_time)
            self.set_joint_pos(
                self._joint_pos_open,
                wait_time=remaining_wait_time,
                exit_on_reach=exit_on_reach,
                exit_on_reach_kwargs=exit_on_reach_kwargs,
            )
        except Exception as e:
            logger.warning(f"Failed to open hand: {e}")

    def _get_intermediate_close_position(self) -> np.ndarray:
        """Get intermediate position for closing hand.

        Returns:
            Intermediate joint positions for smooth closing motion.
        """
        intermediate_pos = self._joint_pos_close.copy()
        ratio = 0.2
        # Adjust thumb opposition joint (last joint)
        intermediate_pos[-1] = self._joint_pos_close[-1] * ratio + self._joint_pos_open[
            -1
        ] * (1 - ratio)
        return intermediate_pos

    def _get_intermediate_open_position(self) -> np.ndarray:
        """Get intermediate position for opening hand.

        Returns:
            Intermediate joint positions for smooth opening motion.
        """
        intermediate_pos = self._joint_pos_close.copy()
        ratio = 0.2
        # Adjust thumb opposition joint (last joint)
        intermediate_pos[-1] = self._joint_pos_close[-1] * ratio + self._joint_pos_open[
            -1
        ] * (1 - ratio)
        # Adjust thumb flexion joint (first joint)
        intermediate_pos[0] = self._joint_pos_close[0] * ratio + self._joint_pos_open[
            0
        ] * (1 - ratio)
        return intermediate_pos


class HandF5D6V2(Hand):
    def __init__(self, name: str, robot_info: RobotInfo) -> None:
        super().__init__(name, robot_info)
        config = robot_info.get_component_config(name)
        config = cast(F5D6HandV2Config, config)
        self._touch_sensor_subscriber = self._node.create_subscriber(
            topic=config.touch_sensor_sub_topic,
            callback=None,
            decoder=FingertipForceCodec.decode,
        )

    def get_finger_tip_force(self) -> Float[np.ndarray, "5"] | None:
        """Get the force at the finger tips.

        Returns:
            Array of force values at the finger tips.
        """
        if self._touch_sensor_subscriber is None:
            logger.warning("Touch sensor subscriber not initialized")
            return None
        return self._touch_sensor_subscriber.get_latest()["force"]


class DexGripper(RobotJointComponent):
    """Robot gripper control class.

    This class provides methods to control a robot gripper by publishing commands and
    receiving state information through Zenoh communication.

    Attributes:
        mode_querier: Zenoh querier for setting gripper mode.
        default_vel: Default joint velocities for all joints.
        max_vel: Maximum allowed joint velocities for all joints.
    """

    def __init__(
        self,
        name: str,
        robot_info: RobotInfo,
    ) -> None:
        """Initialize the gripper controller.

        Args:
            name: Component name.
            robot_info: RobotInfo instance.
        """
        config = robot_info.get_component_config(name)
        config = cast(DexSGripperConfig, config)
        joint_names = config.joints
        close_pos = config.pose_pool["close"]
        open_pos = config.pose_pool["open"]
        self._joint_pos_limits = np.array(
            [[close_pos[i], open_pos[i]] for i in range(len(joint_names))]
        )
        self._joint_vel_limits = np.array([1])
        super().__init__(
            name=name,
            state_sub_topic=config.state_sub_topic,
            control_pub_topic=config.control_pub_topic,
            state_decoder=JointStateCodec.decode,
            control_encoder=JointCmdCodec.encode,
            joint_name=joint_names,
            joint_pos_limit=self._joint_pos_limits,
            joint_vel_limit=self._joint_vel_limits,
            pose_pool=config.pose_pool,
        )

        self._side = config.side
        self._joint_pos_open = np.array(config.pose_pool["open"], dtype=np.float32)
        self._joint_pos_close = np.array(config.pose_pool["close"], dtype=np.float32)

        # Mode service client for gripper control mode
        self._mode_querier = self._node.create_service_client(
            service_name=config.set_mode_query,
            request_encoder=DictDataCodec.encode,
            response_decoder=DictDataCodec.decode,
            timeout=5.0,
        )

    def set_mode(self, mode: Literal["mit", "velocity", "force"]) -> dict[str, Any]:
        """Set the gripper control mode.

        This service is only available when the server detects a DexGripper
        as the end effector. It will not be available if a Hand (F5D6) or
        unknown end effector is detected.

        Args:
            mode: Control mode to set:
                - "mit": MIT mode for direct torque control
                - "velocity": Position-velocity control mode
                - "force": Force-position control mode

        Returns:
            Dictionary with 'success' (bool) and 'message' (str).

        Raises:
            ValueError: If an invalid mode is specified.
            ServiceUnavailableError: If the service is not available (e.g., no gripper
                detected) or the operation fails.
        """
        valid_modes = ["mit", "velocity", "force"]
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode: {mode}. Must be one of {valid_modes}")

        if not self._mode_querier.wait_for_service(timeout=5.0):
            raise ServiceUnavailableError(
                f"Gripper mode service not available for {self._side}"
            )

        query_msg = {"mode": mode}
        response = self._mode_querier.call(query_msg)
        if response is None:
            response = {
                "success": False,
                "message": "Response time out from gripper mode service",
            }
        return response

    def set_joint_pos(
        self,
        joint_pos: Float[np.ndarray, "3"] | list[float] | dict[str, float],
        relative: bool = False,
        wait_time: float = 0.0,
        wait_kwargs: dict[str, float] | None = None,
        exit_on_reach: bool = False,
        exit_on_reach_kwargs: dict[str, float] | None = None,
    ) -> None:
        """Send joint position control commands to the head.

        Args:
            joint_pos: Joint positions as either:
                - List of joint values [j1, j2]
                - Numpy array with shape (3,), in radians
                - Dictionary mapping joint names to position values
            relative: If True, the joint positions are relative to the current position.
            wait_time: Time to wait after sending command in seconds. If 0, returns
                immediately after sending command.
            wait_kwargs: Optional parameters for trajectory generation (not used in Head).
            exit_on_reach: If True, the function will exit when the joint positions are reached.
            exit_on_reach_kwargs: Optional parameters for exit when the joint positions are reached.

        Raises:
            ValueError: If joint_pos dictionary contains invalid joint names.
        """
        self.set_joint_pos_vel(
            joint_pos,
            joint_vel=None,
            relative=relative,
            wait_time=wait_time,
            exit_on_reach=exit_on_reach,
            exit_on_reach_kwargs=exit_on_reach_kwargs,
        )

    def set_joint_pos_vel(
        self,
        joint_pos: Float[np.ndarray, "3"] | list[float] | dict[str, float],
        joint_vel: Float[np.ndarray, "3"]
        | list[float]
        | dict[str, float]
        | float
        | None = None,
        relative: bool = False,
        wait_time: float = 0.0,
        exit_on_reach: bool = False,
        exit_on_reach_kwargs: dict[str, float] | None = None,
    ) -> None:
        """Send control commands to the head.

        Args:
            joint_pos: Joint positions as either:
                - List of joint values [j1, j2]
                - Numpy array with shape (3,), in radians
                - Dictionary mapping joint names to position values
            joint_vel: Optional joint velocities as either:
                - List of joint values [v1, v2]
                - Numpy array with shape (3,), in rad/s
                - Dictionary mapping joint names to velocity values
                - Single float value to be applied to all joints
                If None, velocities are calculated based on default velocity setting.
            relative: If True, the joint positions are relative to the current position.
            wait_time: Time to wait after sending command in seconds. If 0, returns
                immediately after sending command.
            exit_on_reach: If True, the function will exit when the joint positions are reached.
            exit_on_reach_kwargs: Optional parameters for exit when the joint positions are reached.

        Raises:
            ValueError: If wait_time is negative or joint_pos dictionary contains
                invalid joint names.
        """
        if wait_time < 0.0:
            raise ValueError("wait_time must be greater than or equal to 0")

        # Handle relative positioning
        if relative:
            joint_pos = self._resolve_relative_joint_cmd(joint_pos)

        # Convert inputs to numpy arrays
        joint_pos = self._convert_joint_cmd_to_array(joint_pos)
        joint_vel = self._process_joint_velocities(joint_vel, joint_pos)

        if self._joint_pos_limit is not None:
            joint_pos = np.clip(
                joint_pos, self._joint_pos_limit[:, 0], self._joint_pos_limit[:, 1]
            )
        if self._joint_vel_limit is not None:
            joint_vel = np.clip(
                joint_vel, -self._joint_vel_limit, self._joint_vel_limit
            )

        # Create and send control message
        data = dict(pos=joint_pos, vel=np.abs(joint_vel), torque=[0.2])
        self._publish_control(control_msg=data)

        # Wait if specified
        self._wait_for_position(
            joint_pos=joint_pos,
            wait_time=wait_time,
            exit_on_reach=exit_on_reach,
            exit_on_reach_kwargs=exit_on_reach_kwargs,
        )

    def stop(self) -> None:
        """Stop the head by setting target position to current position with zero velocity."""
        current_pos = self.get_joint_pos()
        zero_vel = np.zeros(3, dtype=np.float32)
        self.set_joint_pos_vel(current_pos, zero_vel, relative=False, wait_time=0.0)

    def shutdown(self) -> None:
        """Clean up Zenoh resources for the head component."""
        self.stop()
        super().shutdown()
        # No need to undeclare queriers when using DexComm

    def _process_joint_velocities(
        self,
        joint_vel: Float[np.ndarray, "1"]
        | list[float]
        | dict[str, float]
        | float
        | None,
        joint_pos: np.ndarray,
    ) -> np.ndarray:
        """Process and validate joint velocities.

        Args:
            joint_vel: Joint velocities in various formats or None.
            joint_pos: Target joint positions for velocity calculation.

        Returns:
            Processed joint velocities as numpy array.
        """
        if joint_vel is None:
            # Calculate velocities based on motion direction and default velocity
            joint_motion = joint_pos - self.get_joint_pos()
            motion_norm = np.linalg.norm(joint_motion)

            if motion_norm < 1e-6:  # Avoid division by zero
                return np.zeros(3, dtype=np.float32)

            default_vel = (
                2.5 if self._joint_vel_limit is None else np.min(self._joint_vel_limit)
            )
            return (joint_motion / motion_norm) * default_vel

        if isinstance(joint_vel, (int, float)):
            # Single value - apply to all joints
            return np.full(3, joint_vel, dtype=np.float32)

        # Convert to array and clip to velocity limits
        return self._convert_joint_cmd_to_array(
            joint_vel, clip_value=self._joint_vel_limit
        )

    def open_hand(self) -> None:
        """Open the hand."""
        self.set_joint_pos(self._joint_pos_open, wait_time=0.0)

    def close_hand(self) -> None:
        """Close the hand."""
        self.set_joint_pos(self._joint_pos_close, wait_time=0.0)
