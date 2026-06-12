# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Robot torso control module.

This module provides the Torso class for controlling a robot torso through Zenoh
communication. It handles joint position and velocity control and state monitoring.
"""

from typing import cast

import numpy as np
from dexbot_utils import RobotInfo
from dexbot_utils.configs.components.vega_1 import Vega1TorsoConfig
from dexcomm.codecs import JointCmdCodec, JointStateCodec
from jaxtyping import Float

from dexcontrol.core.component import RobotJointComponent


class Torso(RobotJointComponent):
    """Robot torso control class.

    This class provides methods to control a robot torso by publishing commands and
    receiving state information through Zenoh communication.

    Attributes:
        default_vel: Default joint velocity in rad/s when not explicitly specified.
        max_vel: Maximum allowed joint velocity in rad/s.
    """

    def __init__(
        self,
        name: str,
        robot_info: RobotInfo,
    ) -> None:
        """Initialize the torso controller.

        Args:
            robot_info: RobotInfo instance.
                and default velocity settings.
        """
        joint_names = robot_info.get_component_joints(name)
        joint_pos_limits = robot_info.get_joint_pos_limits(joint_names)
        joint_vel_limits = robot_info.get_joint_vel_limits(joint_names)
        config = robot_info.get_component_config(name)
        config = cast(Vega1TorsoConfig, config)
        super().__init__(
            name=name,
            state_sub_topic=config.state_sub_topic,
            control_pub_topic=config.control_pub_topic,
            state_decoder=JointStateCodec.decode,
            control_encoder=JointCmdCodec.encode,
            joint_name=joint_names,
            joint_pos_limit=joint_pos_limits,
            joint_vel_limit=joint_vel_limits,
            pose_pool=config.pose_pool,
        )
        assert self._joint_vel_limit is not None, "joint_vel_limit is not set"

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
        """Send control commands to the torso.

        Args:
            joint_pos: Joint positions as either:
                - List of joint values [j1, j2, j3]
                - Numpy array with shape (3,), in radians
                - Dictionary mapping joint names to position values
            joint_vel: Optional joint velocities as either:
                - List of joint values [v1, v2, v3]
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
        data = dict(pos=joint_pos, vel=joint_vel)
        self._publish_control(control_msg=data)

        # Wait if specified
        self._wait_for_position(
            joint_pos=joint_pos,
            wait_time=wait_time,
            exit_on_reach=exit_on_reach,
            exit_on_reach_kwargs=exit_on_reach_kwargs,
        )

    def set_joint_pos(
        self,
        joint_pos: Float[np.ndarray, "3"] | list[float] | dict[str, float],
        relative: bool = False,
        wait_time: float = 0.0,
        wait_kwargs: dict[str, float] | None = None,
        exit_on_reach: bool = False,
        exit_on_reach_kwargs: dict[str, float] | None = None,
    ) -> None:
        """Send joint position control commands to the torso.

        Args:
            joint_pos: Joint positions as either:
                - List of joint values [j1, j2, j3]
                - Numpy array with shape (3,), in radians
                - Dictionary mapping joint names to position values
            relative: If True, the joint positions are relative to the current position.
            wait_time: Time to wait after sending command in seconds. If 0, returns
                immediately after sending command.
            wait_kwargs: Optional parameters for trajectory generation (not used in Torso).
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

    def stop(self) -> None:
        """Stop the torso by setting target position to current position with zero velocity."""
        current_pos = self.get_joint_pos()
        zero_vel = np.zeros(3, dtype=np.float32)
        self.set_joint_pos_vel(current_pos, zero_vel, relative=False, wait_time=0.0)

    @property
    def pitch_angle(self) -> float:
        """Gets the pitch angle of the torso.

        The pitch angle is defined as the angle between the third link (the link that
        is closest to the arms) and the horizontal plane.

        Examples:
            At zero position: shoulder pitch angle = pi/2
            At (60, 120, -30) degrees: shoulder pitch angle = 0

        Returns:
            The pitch angle in radians.
        """
        joint_pos = self.get_joint_pos()
        return joint_pos[0] + joint_pos[2] + np.pi / 2 - joint_pos[1]

    def shutdown(self) -> None:
        """Clean up Zenoh resources for the torso component."""
        self.stop()
        super().shutdown()

    def _process_joint_velocities(
        self,
        joint_vel: Float[np.ndarray, "3"]
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
            joint_motion = np.abs(joint_pos - self.get_joint_pos())
            motion_norm = np.linalg.norm(joint_motion)

            if motion_norm < 1e-6:  # Avoid division by zero
                return np.zeros(3, dtype=np.float32)

            default_vel = (
                0.6 if self._joint_vel_limit is None else np.min(self._joint_vel_limit)
            )
            return (joint_motion / motion_norm) * default_vel

        if isinstance(joint_vel, (int, float)):
            # Single value - apply to all joints
            return np.full(3, joint_vel, dtype=np.float32)

        # Convert to array and clip to velocity limits
        return self._convert_joint_cmd_to_array(
            joint_vel, clip_value=self._joint_vel_limit
        )
