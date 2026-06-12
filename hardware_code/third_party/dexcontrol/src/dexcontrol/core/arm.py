# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Robot arm control module.

This module provides the Arm class for controlling a robot arm through Zenoh
communication and the ArmWrenchSensor class for reading wrench sensor data.
"""

import threading
import time
from typing import Any, Literal, cast

import numpy as np
from dexbot_utils import RobotInfo
from dexbot_utils.configs.components.vega_1 import Vega1ArmConfig
from dexcomm.codecs import (
    DictDataCodec,
    EEPassThroughCmdCodec,
    JointCmdCodec,
    JointModeCodec,
    JointModeEnum,
    JointStateCodec,
    WrenchStateCodec,
    WristButtonStateCodec,
)
from dexcomm.utils import RateLimiter
from jaxtyping import Float
from loguru import logger
from rich.console import Console

from dexcontrol.core.component import RobotComponent, RobotJointComponent
from dexcontrol.exceptions import ServiceUnavailableError
from dexcontrol.utils.trajectory_utils import generate_linear_trajectory


class Arm(RobotJointComponent):
    """Robot arm control class.

    This class provides methods to control a robot arm by publishing commands and
    receiving state information through Zenoh communication.

    Error handling convention for service calls:
        - **Read-only queries** (``get_pid``, ``get_brake_status``,
          ``get_force_torque_sensor_mode``, ``get_ee_baud_rate``) raise
          ``ServiceUnavailableError`` on timeout because callers depend on the
          returned data and cannot proceed without it.
        - **Write operations** (``set_pid``, ``set_ee_baud_rate``, ``release_brake``,
          ``activate_force_torque_sensor``) return a ``{"success": False, ...}``
          fallback dict on timeout so callers can handle failure gracefully without
          catching exceptions in interactive scripts.

    Attributes:
        mode_querier: Zenoh querier for setting arm mode.
        wrench_sensor: Optional ArmWrenchSensor instance for wrench sensor data.
    """

    def __init__(
        self,
        name: str,
        robot_info: RobotInfo,
    ) -> None:
        """Initialize the arm controller.

        Args:
            name: Component name.
            robot_info: RobotInfo instance (kept for compatibility).
        """
        joint_names = robot_info.get_component_joints(name)
        joint_pos_limits = robot_info.get_joint_pos_limits(joint_names)
        joint_vel_limit = robot_info.get_joint_vel_limits(joint_names)
        config = robot_info.get_component_config(name)
        config = cast(Vega1ArmConfig, config)
        super().__init__(
            name=name,
            state_sub_topic=config.state_sub_topic,
            control_pub_topic=config.control_pub_topic,
            control_encoder=JointCmdCodec.encode,
            state_decoder=JointStateCodec.decode,
            joint_name=joint_names,
            joint_pos_limit=joint_pos_limits,
            joint_vel_limit=joint_vel_limit,
            pose_pool=config.pose_pool,
        )
        self._side = config.side
        # Initialize wrench sensor if configured
        self.wrench_sensor: ArmWrenchSensor | None = None
        if config.wrench_sub_topic:
            wrench_sensor_name = f"{name}_wrench_sensor"
            self.wrench_sensor = ArmWrenchSensor(
                wrench_sensor_name,
                config.wrench_sub_topic,
                config.wrist_button_sub_topic,
            )

        # Initialize end effector pass through publisher and subscriber using DexComm
        self.enable_ee_pass_through = config.enable_ee_pass_through
        self._ee_pass_through_lock = threading.Lock()
        self._latest_ee_pass_through_data: dict[str, Any] | None = None
        if self.enable_ee_pass_through:
            self._ee_pass_through_publisher = self._node.create_publisher(
                topic=config.ee_pass_through_pub_topic,
                encoder=EEPassThroughCmdCodec.encode,
            )
            # Subscribe to EE pass-through response topic
            self._ee_pass_through_subscriber = self._node.create_subscriber(
                topic=config.ee_pass_through_state_sub_topic,
                callback=self._on_ee_pass_through_update,
                decoder=EEPassThroughCmdCodec.decode,
            )

        self._mode_querier = self._node.create_service_client(
            service_name=config.set_mode_query,
            request_encoder=JointModeCodec.encode,
            response_decoder=DictDataCodec.decode,
            timeout=5.0,
        )

        # PID configuration service client
        self._pid_querier = self._node.create_service_client(
            service_name=config.pid_query,
            request_encoder=DictDataCodec.encode,
            response_decoder=DictDataCodec.decode,
            timeout=5.0,
        )

        # Brake release service client
        self._brake_querier = self._node.create_service_client(
            service_name=config.brake_query,
            request_encoder=DictDataCodec.encode,
            response_decoder=DictDataCodec.decode,
            timeout=5.0,
        )

        # Force torque sensor mode service client
        self._force_torque_sensor_querier = self._node.create_service_client(
            service_name=config.force_torque_sensor_query,
            request_encoder=DictDataCodec.encode,
            response_decoder=DictDataCodec.decode,
            timeout=5.0,
        )

        # End-effector baud rate service client
        self._ee_baud_rate_querier = self._node.create_service_client(
            service_name=config.ee_baud_rate_query,
            request_encoder=DictDataCodec.encode,
            response_decoder=DictDataCodec.decode,
            timeout=5.0,
        )

        self._default_control_hz = config.default_control_hz
        if self._joint_vel_limit is not None:
            if np.any(self._joint_vel_limit > 2.8):
                logger.warning(
                    "Joint velocity limit is greater than 2.8. This is not recommended."
                )
                self._joint_vel_limit = np.clip(self._joint_vel_limit, 0, 2.8)
                logger.warning("Joint velocity limit is clamped to 2.8")

    def set_mode(self, mode: Literal["position", "disable"]) -> None:
        """Sets the operating mode of the arm.

        .. deprecated::
            Use set_modes() instead for setting arm modes.

        Args:
            mode: Operating mode for the arm. Must be either "position" or "disable".
                "position": Enable position control
                "disable": Disable control

        Raises:
            ValueError: If an invalid mode is specified.
        """
        logger.warning("arm.set_mode() is deprecated, use set_modes() instead")
        self.set_modes([mode] * 7)

    def set_modes(self, modes: list[Literal["position", "disable"]]) -> None:
        """Sets the operating modes of the arm.

        Args:
            modes: List of operating modes for the arm. Each mode must be either "position", "disable", or "current".

        Raises:
            ValueError: If any mode in the list is invalid.
        """
        mode_map = {
            "position": JointModeEnum.POSITION,
            "disable": JointModeEnum.DISABLE,
        }

        for mode in modes:
            if mode not in mode_map:
                raise ValueError(
                    f"Invalid mode: {mode}. Must be one of {list(mode_map.keys())}"
                )

        if len(modes) != 7:
            raise ValueError("Arm modes length must match arm DoF (7).")

        converted_modes = [mode_map[mode] for mode in modes]
        query_msg = {"mode": converted_modes}

        # Wait for service to be available before calling
        if not self._mode_querier.wait_for_service(timeout=5.0):
            logger.warning(
                f"{self._mode_querier.get_stats()['service_name']}: Mode service not available, command may fail"
            )

        self._mode_querier.call(query_msg)

    def _send_position_command(self, joint_pos: np.ndarray) -> None:
        """Send joint position command.

        Args:
            joint_pos: Joint positions as numpy array.
        """
        control_msg = {"pos": joint_pos}
        self._publish_control(control_msg)

    def set_joint_pos(
        self,
        joint_pos: Float[np.ndarray, " N"] | list[float] | dict[str, float],
        relative: bool = False,
        wait_time: float = 0.0,
        wait_kwargs: dict[str, float] | None = None,
        exit_on_reach: bool = False,
        exit_on_reach_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Controls the arm in joint position mode.

        Args:
            joint_pos: Joint positions as either:
                - List of joint values [j1, j2, ..., j7]
                - Numpy array with shape (7,), in radians
                - Dictionary mapping joint names to position values
            relative: If True, the joint positions are relative to the current position.
            wait_time: Time to wait between movements in seconds. If wait_time is 0,
                the joint positions will be sent, and the function call will return
                immediately. If wait_time is greater than 0, the joint positions will
                be interpolated between the current position and the target position,
                and the function will wait for the specified time between each movement.

                **IMPORTANT: When wait_time is 0, you MUST call this function repeatedly
                in a high-frequency loop (e.g., 100 Hz). DO NOT call it just once!**
                The function is designed for continuous control when wait_time=0.
                The highest frequency that the user can call this function is 500 Hz.


            wait_kwargs: Keyword arguments for the interpolation (only used if
                wait_time > 0). Supported keys:
                - control_hz: Control frequency in Hz (default: 100).
                - max_vel: Maximum velocity in rad/s (default: 0.5).
            exit_on_reach: If True, the function will exit when the joint positions are reached.
            exit_on_reach_kwargs: Optional parameters for exit when the joint positions are reached.
        Raises:
            ValueError: If joint_pos dictionary contains invalid joint names.
        """
        if wait_kwargs is None:
            wait_kwargs = {}

        resolved_joint_pos = (
            self._resolve_relative_joint_cmd(joint_pos) if relative else joint_pos
        )
        resolved_joint_pos = self._convert_joint_cmd_to_array(resolved_joint_pos)
        if self._joint_pos_limit is not None:
            resolved_joint_pos = np.clip(
                resolved_joint_pos,
                self._joint_pos_limit[:, 0],
                self._joint_pos_limit[:, 1],
            )

        if wait_time > 0.0:
            self._execute_trajectory_motion(
                resolved_joint_pos,
                wait_time,
                wait_kwargs,
                exit_on_reach,
                exit_on_reach_kwargs,
            )
        else:
            self._send_position_command(resolved_joint_pos)

    def _execute_trajectory_motion(
        self,
        target_joint_pos: Float[np.ndarray, " N"],
        wait_time: float,
        wait_kwargs: dict[str, float],
        exit_on_reach: bool = False,
        exit_on_reach_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Execute trajectory-based motion to target position.

        Args:
            target_joint_pos: Target joint positions as numpy array.
            wait_time: Total time for the motion.
            wait_kwargs: Parameters for trajectory generation.
            exit_on_reach: If True, the function will exit when the joint positions are reached.
            exit_on_reach_kwargs: Optional parameters for exit when the joint positions are reached.
        """
        # Set default parameters
        control_hz = wait_kwargs.get("control_hz", self._default_control_hz)
        max_vel = wait_kwargs.get("max_vel")
        if max_vel is None:
            max_vel = (
                self._joint_vel_limit if self._joint_vel_limit is not None else 2.8
            )
        exit_on_reach_kwargs = exit_on_reach_kwargs or {}
        exit_on_reach_kwargs.setdefault("tolerance", 0.05)

        # Create rate limiter and get current position
        rate_limiter = RateLimiter(control_hz)
        current_joint_pos = self.get_joint_pos().copy()

        # Generate trajectory using utility function
        trajectory, _ = generate_linear_trajectory(
            current_joint_pos, target_joint_pos, max_vel, control_hz
        )
        # Execute trajectory with time limit
        start_time = time.time()
        for pos in trajectory:
            if time.time() - start_time > wait_time:
                break
            self._send_position_command(pos)
            rate_limiter.sleep()

        # Hold final position for remaining time
        while time.time() - start_time < wait_time:
            self._send_position_command(target_joint_pos)
            rate_limiter.sleep()
            if exit_on_reach and self.is_joint_pos_reached(
                target_joint_pos, **exit_on_reach_kwargs
            ):
                break

    def set_joint_pos_vel(
        self,
        joint_pos: Float[np.ndarray, " N"] | list[float] | dict[str, float],
        joint_vel: Float[np.ndarray, " N"] | list[float] | dict[str, float],
        relative: bool = False,
    ) -> None:
        """Controls the arm in joint position mode with a velocity feedforward term.

        Warning:
            The joint_vel parameter should be well-planned (such as from a trajectory planner).
            Sending poorly planned or inappropriate joint velocity commands can cause the robot
            to behave unexpectedly or potentially get damaged. Ensure velocity commands are
            smooth, within safe limits, and properly coordinated across all joints.

            Additionally, this command MUST be called at high frequency (e.g., 100Hz) to take
            effect properly. DO NOT call this function just once or at low frequency, as this
            can lead to unpredictable robot behavior.

        Args:
            joint_pos: Joint positions as either:
                - List of joint values [j1, j2, ..., j7]
                - Numpy array with shape (7,), in radians
                - Dictionary of joint names and position values
            joint_vel: Joint velocities as either:
                - List of joint values [v1, v2, ..., v7]
                - Numpy array with shape (7,), in radians/sec
                - Dictionary of joint names and velocity values

            relative: If True, the joint positions are relative to the current position.

        Raises:
            ValueError: If joint_pos dictionary contains invalid joint names.
        """
        resolved_joint_pos = (
            self._resolve_relative_joint_cmd(joint_pos) if relative else joint_pos
        )
        resolved_joint_pos = self._convert_joint_cmd_to_array(resolved_joint_pos)
        if self._joint_pos_limit is not None:
            resolved_joint_pos = np.clip(
                resolved_joint_pos,
                self._joint_pos_limit[:, 0],
                self._joint_pos_limit[:, 1],
            )
        target_pos = resolved_joint_pos
        target_vel = self._convert_joint_cmd_to_array(
            joint_vel, clip_value=self._joint_vel_limit
        )

        control_msg = {
            "pos": target_pos,
            "vel": target_vel,
        }
        self._publish_control(control_msg)

    def set_pid(
        self,
        p_multipliers: list[float],
        i_multipliers: list[float] | None = None,
        d_multipliers: list[float] | None = None,
    ) -> dict[str, Any]:
        """Set PID P-gain multipliers for the arm joints.

        This sets the position P-gain multipliers for all 7 arm joints.
        The multipliers are applied to the factory default PID values.
        Currently, only the modification of P value is supported.

        Each multiplier must be in the range [0.1, 4]. Values outside this
        range are not permitted for safety reasons — higher values increase joint
        stiffness, which generates larger forces on collision.

        Args:
            p_multipliers: List of 7 P-gain multipliers, one for each joint.
                Range: [0.1, 4] per joint.

        Returns:
            Dictionary with 'success' (bool), 'p' (list), and 'message' (str).

        Raises:
            ValueError: If multipliers are not exactly 7 values or out of range [0.1, 4].
            ServiceUnavailableError: If the service is not available or the operation fails.
        """
        if i_multipliers is not None or d_multipliers is not None:
            logger.warning("Only the modification of P value is supported for now")
        del i_multipliers, d_multipliers

        if len(p_multipliers) != 7:
            raise ValueError("p_multipliers must have exactly 7 values (one per joint)")

        for i, val in enumerate(p_multipliers):
            if not (0.1 <= val <= 4.0):
                raise ValueError(
                    f"p_multipliers[{i}]={val} is out of range. "
                    "Values must be in [0.1, 4] for safety."
                )

        if not self._pid_querier.wait_for_service(timeout=5.0):
            raise ServiceUnavailableError(
                f"PID service not available for {self._side} arm"
            )

        query_msg = {"p": p_multipliers}

        # Use rich progress indicator since PID setting takes ~40s
        # (factory mode entry/exit + flash save + verification)
        console = Console()
        with console.status(
            f"[bold green]Setting PID for {self._side} arm (~40 seconds)..."
        ):
            response = self._pid_querier.call(query_msg, timeout=45.0)

        if response is None:
            response = {
                "success": False,
                "message": "Response time out from PID service",
            }
        return response

    def get_pid(self) -> dict[str, Any]:
        """Get current PID P-gain multipliers for the arm joints.

        Returns:
            Dictionary with 'success' (bool), 'p' (list of 7 multipliers), and 'message' (str).

        Raises:
            ServiceUnavailableError: If the service is not available or the operation fails.
        """
        if not self._pid_querier.wait_for_service(timeout=5.0):
            raise ServiceUnavailableError(
                f"PID service not available for {self._side} arm"
            )

        response = self._pid_querier.call({})
        if response is None:
            raise ServiceUnavailableError(
                f"Failed to get PID for {self._side} arm: no response"
            )
        return response

    def release_brake(
        self, enable: bool, joints: list[int] | None = None
    ) -> dict[str, Any]:
        """Enable or disable brake release (over-limit drag) for the arm.

        When brake release is enabled, the arm can be manually moved beyond
        normal position limits. This is useful for calibration or recovery.

        Args:
            enable: True to enable brake release, False to disable.
            joints: Optional list of joint indices (0-6) to operate on.
                If None, operates on all joints. Ignored when disabling.

        Returns:
            Dictionary with 'success' (bool) and 'message' (str).

        Raises:
            ServiceUnavailableError: If the service is not available or the operation fails.
        """
        if not self._brake_querier.wait_for_service(timeout=5.0):
            raise ServiceUnavailableError(
                f"Brake service not available for {self._side} arm"
            )

        query_msg: dict[str, Any] = {"enable": enable}
        if joints is not None:
            query_msg["joints"] = joints

        # Use rich progress indicator since brake release takes time
        # (motor mode changes + safety verification)
        action = "Releasing" if enable else "Engaging"
        console = Console()
        with console.status(f"[bold green]{action} brake for {self._side} arm..."):
            response = self._brake_querier.call(query_msg, timeout=45.0)

        if response is None:
            response = {
                "success": False,
                "message": "Response time out from brake service",
            }
        return response

    def get_brake_status(self) -> dict[str, Any]:
        """Get current brake release status for the arm.

        Returns:
            Dictionary with:
                - 'success' (bool): Whether the query succeeded
                - 'enabled' (bool): Whether brake release is currently enabled
                - 'joints' (list): List of joints with brake released
                - 'message' (str): Status message

        Raises:
            ServiceUnavailableError: If the service is not available or the operation fails.
        """
        if not self._brake_querier.wait_for_service(timeout=5.0):
            raise ServiceUnavailableError(
                f"Brake service not available for {self._side} arm"
            )

        response = self._brake_querier.call({})
        if response is None:
            raise ServiceUnavailableError(
                f"Failed to get brake status for {self._side} arm: no response"
            )
        return response

    def activate_force_torque_sensor(self, enable: bool) -> dict[str, Any]:
        """Enable or disable force torque sensor mode for the arm.

        When enabled, all 7 joints are set to mode 0 (non-joint mode, requires
        6-axis force sensor connected). When disabled, all joints are set to
        mode 1 (joint mode, operates without force sensor).

        This operation requires factory mode and takes ~20 seconds.

        Args:
            enable: True to enable force torque sensor (joint mode 0),
                False to disable (joint mode 1).

        Returns:
            Dictionary with 'success' (bool) and 'message' (str).

        Raises:
            ServiceUnavailableError: If the service is not available or the operation fails.
        """
        if not self._force_torque_sensor_querier.wait_for_service(timeout=5.0):
            raise ServiceUnavailableError(
                f"Force torque sensor service not available for {self._side} arm"
            )

        query_msg: dict[str, Any] = {"enable": enable}

        action = "Enabling" if enable else "Disabling"
        console = Console()
        with console.status(
            f"[bold green]{action} force torque sensor for {self._side} arm (~20 seconds)..."
        ):
            response = self._force_torque_sensor_querier.call(query_msg, timeout=45.0)

        if response is None:
            response = {
                "success": False,
                "message": "Response time out from force torque sensor service",
            }
        return response

    def get_force_torque_sensor_mode(self) -> dict[str, Any]:
        """Get current force torque sensor mode for the arm.

        Returns:
            Dictionary with:
                - 'success' (bool): Whether the query succeeded
                - 'enabled' (bool): Whether force torque sensor is enabled (all joints mode 0)
                - 'modes' (list[int]): Current joint modes for all 7 joints
                - 'message' (str): Status message

        Raises:
            ServiceUnavailableError: If the service is not available or the operation fails.
        """
        if not self._force_torque_sensor_querier.wait_for_service(timeout=5.0):
            raise ServiceUnavailableError(
                f"Force torque sensor service not available for {self._side} arm"
            )

        response = self._force_torque_sensor_querier.call({})
        if response is None:
            raise ServiceUnavailableError(
                f"Failed to get force torque sensor mode for {self._side} arm: no response"
            )
        return response

    def set_ee_baud_rate(self, baud_rate: int) -> dict[str, Any]:
        """Set the end-effector RS485 baud rate.

        Args:
            baud_rate: Baud rate value (e.g., 115200, 1000000).

        Returns:
            Dictionary with 'success' (bool), 'baud_rate' (int), and 'message' (str).

        Raises:
            ServiceUnavailableError: If the service is not available or the operation fails.
        """
        if not self._ee_baud_rate_querier.wait_for_service(timeout=5.0):
            raise ServiceUnavailableError(
                f"EE baud rate service not available for {self._side} arm"
            )

        query_msg = {"baud_rate": baud_rate}
        response = self._ee_baud_rate_querier.call(query_msg)
        if response is None:
            response = {
                "success": False,
                "message": "Response time out from EE baud rate service",
            }
        return response

    def get_ee_baud_rate(self) -> dict[str, Any]:
        """Get the current end-effector RS485 baud rate.

        Returns:
            Dictionary with 'success' (bool), 'baud_rate' (int), and 'message' (str).

        Raises:
            ServiceUnavailableError: If the service is not available or the operation fails.
        """
        if not self._ee_baud_rate_querier.wait_for_service(timeout=5.0):
            raise ServiceUnavailableError(
                f"EE baud rate service not available for {self._side} arm"
            )

        response = self._ee_baud_rate_querier.call({})
        if response is None:
            raise ServiceUnavailableError(
                f"Failed to get EE baud rate for {self._side} arm: no response"
            )
        return response

    def shutdown(self) -> None:
        """Cleans up all Zenoh resources."""
        super().shutdown()
        try:
            # Mode querier cleanup no longer needed with DexComm
            pass
        except Exception as e:
            # Don't log "Undeclared querier" errors as warnings - they're expected during shutdown
            error_msg = str(e).lower()
            if not ("undeclared" in error_msg or "closed" in error_msg):
                logger.warning(
                    f"Error undeclaring mode querier for {self.__class__.__name__}: {e}"
                )

        if self.wrench_sensor:
            self.wrench_sensor.shutdown()

    def send_ee_pass_through_message(self, message: bytes) -> None:
        """Send an end effector pass through message to the robot arm.

        This method is only available when the end effector type is UNKNOWN.
        When a known end effector (Hand or DexGripper) is detected, the server
        does not create the EE pass-through subscriber and this method should
        not be called. The `enable_ee_pass_through` config flag controls whether
        this functionality is enabled.

        Args:
            message: The raw bytes message to send to the end effector via RS485.
        """
        self._ee_pass_through_publisher.publish({"data": message})

    def _on_ee_pass_through_update(self, data: dict[str, Any]) -> None:
        """Handle incoming EE pass-through response updates.

        Args:
            data: Decoded EE pass-through response data.
        """
        with self._ee_pass_through_lock:
            self._latest_ee_pass_through_data = data

    def get_ee_pass_through_response(self) -> dict[str, Any] | None:
        """Get the latest end-effector pass-through response data.

        This method is only available when the end effector type is UNKNOWN.
        When a known end effector (Hand or DexGripper) is detected, the server
        does not publish EE pass-through responses and this method will always
        return None. The `enable_ee_pass_through` config flag controls whether
        the subscriber is created.

        Returns:
            Dictionary containing the response data with 'data' key holding
            a list of bytes, or None if no response has been received yet
            or if EE pass-through is not enabled.
        """
        with self._ee_pass_through_lock:
            return self._latest_ee_pass_through_data


class ArmWrenchSensor(RobotComponent):
    """Wrench sensor reader for the robot arm.

    This class provides methods to read wrench sensor data through Zenoh communication.
    """

    def __init__(self, name: str, state_sub_topic: str, button_sub_topic: str) -> None:
        """Initialize the wrench sensor reader.

        Args:
            state_sub_topic: Topic to subscribe to for wrench sensor data.
        """
        super().__init__(
            name=name,
            state_sub_topic=state_sub_topic,
            state_decoder=WrenchStateCodec.decode,
        )
        self._button_lock = threading.Lock()
        self._latest_button_state: Any | None = None
        self._button_subscriber = self._node.create_subscriber(
            topic=button_sub_topic,
            callback=self._on_button_update,
            decoder=WristButtonStateCodec.decode,
        )

    def _on_button_update(self, state: Any) -> None:
        """Handle incoming button updates.

        Args:
            state: Decoded protobuf button state message.
        """
        with self._button_lock:
            self._latest_button_state = state

    def get_wrench_state(self) -> Float[np.ndarray, "6"]:
        """Get the current wrench sensor reading.

        Returns:
            Array of wrench values [fx, fy, fz, tx, ty, tz].
        """
        state = self._get_state()
        return np.array(state["wrench"], dtype=np.float32)

    def get_button_state(self) -> dict[str, bool]:
        """Get the state of the wrench sensor buttons.

        Returns:
            Tuple of (blue_button_state, green_button_state).
        """
        with self._button_lock:
            state = self._latest_button_state
        if state is None:
            return dict(blue_button=False, green_button=False)
        return dict(
            blue_button=state["blue_button"], green_button=state["green_button"]
        )

    def get_state(self) -> dict[str, Any]:
        """Get the complete wrench sensor state.

        Returns:
            Dictionary containing wrench values and button states.
        """
        wrench_state = self.get_wrench_state()
        button_state = self.get_button_state()
        return dict(wrench=wrench_state, **button_state)

    def get_blue_button_state(self) -> bool:
        """Get the state of the blue button.

        Returns:
            True if the blue button is pressed, False otherwise.
        """
        button_state = self.get_button_state()
        return button_state["blue_button"]

    def get_green_button_state(self) -> bool:
        """Get the state of the green button.

        Returns:
            True if the green button is pressed, False otherwise.
        """
        button_state = self.get_button_state()
        return button_state["green_button"]
