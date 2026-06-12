# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Main robot interface module.

This module provides the main Robot class that serves as the primary interface for
controlling and monitoring a robot system. It handles component initialization,
status monitoring, and system-wide operations.

The Robot class manages initialization and coordination of various robot components
including arms, hands, head, chassis, torso, and sensors. It provides methods for
system-wide operations like status monitoring, trajectory execution, and component
control.
"""

import os
import signal
import sys
import time
import weakref
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal

import numpy as np
from dexbot_utils import HandType, RobotInfo
from dexbot_utils.cfg_modifier import runtime_override_robot_config
from dexbot_utils.configs import BaseRobotConfig
from dexcomm import RateLimiter, cleanup_session
from loguru import logger
from rich.console import Console
from rich.table import Table

from dexcontrol.core.component import RobotComponent
from dexcontrol.core.config import get_component_config_map
from dexcontrol.core.robot_query_interface import RobotQueryInterface
from dexcontrol.exceptions import (
    ComponentError,
    ComponentNotAvailableError,
    ConfigurationError,
    DexcontrolError,
)
from dexcontrol.sensors import Sensors
from dexcontrol.utils.constants import (
    COMM_CFG_PATH_ENV_VAR,
    DISABLE_ESTOP_CHECKING_ENV_VAR,
    DISABLE_HEARTBEAT_ENV_VAR,
)
from dexcontrol.utils.os_utils import check_version_compatibility
from dexcontrol.utils.trajectory_utils import generate_linear_trajectory

if TYPE_CHECKING:
    from dexcontrol.core.arm import Arm
    from dexcontrol.core.chassis import Chassis
    from dexcontrol.core.hand import Hand
    from dexcontrol.core.head import Head
    from dexcontrol.core.misc import Battery, EStop, Heartbeat
    from dexcontrol.core.torso import Torso


# Global registry to track active Robot instances for signal handling
_active_robots: weakref.WeakSet["Robot"] = weakref.WeakSet()
_signal_handlers_registered: bool = False


def _register_signal_handlers() -> None:
    """Register signal handlers for graceful shutdown."""
    global _signal_handlers_registered
    if _signal_handlers_registered:
        return

    def signal_handler(signum: int, frame: Any) -> None:
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, signal_handler)

    _signal_handlers_registered = True


class Robot(RobotQueryInterface):
    """Main interface class for robot control and monitoring.

    This class serves as the primary interface for interacting with a robot system.
    It manages initialization and coordination of various robot components including
    arms, hands, head, chassis, torso, and sensors. It provides methods for
    system-wide operations like status monitoring, trajectory execution, and component
    control.

    Example usage:
        # Using context manager (recommended)
        with Robot() as robot:
            robot.set_joint_pos({"left_arm": [0, 0, 0, 0, 0, 0, 0]})
            version_info = robot.get_version_info()

        # Manual usage with explicit shutdown
        robot = Robot()
        try:
            robot.set_joint_pos({"left_arm": [0, 0, 0, 0, 0, 0, 0]})
            hand_types = robot.query_hand_type()
        finally:
            robot.shutdown()

    Attributes:
        left_arm: Left arm component interface (7-DOF manipulator).
        right_arm: Right arm component interface (7-DOF manipulator).
        left_hand: Left hand component interface (conditional, based on hardware).
        right_hand: Right hand component interface (conditional, based on hardware).
        head: Head component interface (3-DOF pan-tilt-roll).
        chassis: Chassis component interface (mobile base).
        torso: Torso component interface (1-DOF pitch).
        battery: Battery monitoring interface.
        estop: Emergency stop interface.
        heartbeat: Heartbeat monitoring interface.
        sensors: Sensor systems interface (cameras, IMU, lidar, etc.).
    """

    # Type annotations for dynamically created attributes
    left_arm: "Arm"
    right_arm: "Arm"
    left_hand: "Hand"
    right_hand: "Hand"
    head: "Head"
    chassis: "Chassis"
    torso: "Torso"
    battery: "Battery"
    estop: "EStop"
    heartbeat: "Heartbeat"
    sensors: Sensors
    _KNOWN_COMPONENTS: Final[set[str]] = {
        "left_arm",
        "right_arm",
        "left_hand",
        "right_hand",
        "head",
        "chassis",
        "torso",
        "battery",
        "estop",
        "heartbeat",
        "sensors",
    }

    def __init__(
        self,
        configs: BaseRobotConfig | None = None,
        auto_shutdown: bool = True,
    ) -> None:
        """Initializes the Robot with the given configuration.

        The robot variant is automatically resolved from the ROBOT_NAME
        environment variable. You can optionally provide a custom config object.

        Args:
            configs: Configuration parameters for all robot components.
                If None, the configuration is resolved from the ROBOT_NAME
                environment variable.
            auto_shutdown: Whether to automatically register signal handlers for
                graceful shutdown on program interruption. Default is True.

        Raises:
            ComponentError: If any critical component fails to become active within timeout.
            ValueError: If configs cannot be loaded.
        """
        self._shutdown_called: bool = False
        self._components: list[RobotComponent] = []

        self._robot_info: RobotInfo = RobotInfo(configs=configs)
        self._robot_name = self._robot_info.robot_name
        self._configs: Final[BaseRobotConfig] = self._robot_info.config

        # Early validation - fail fast with clear errors
        self._validate_configuration()

        super().__init__(configs=self._configs)

        self._pv_components: list[str] = self._robot_info.get_pv_components()
        self._hand_types: dict[str, HandType] = {}

        # Register for automatic shutdown on signals if enabled
        if auto_shutdown:
            _register_signal_handlers()
            _active_robots.add(self)

        self._print_initialization_info()

        # Initialize robot components with safe error handling
        self._safe_initialize_components()

        # Check version compatibility using new JSON interface
        self._check_version_compatibility()

    @property
    def robot_model(self) -> str:
        """Get the base robot model.

        The base model is the core platform type without hand/config suffixes
        (e.g., "vega_1", "vega_1p", "vega_1u").

        Returns:
            The base robot model.
        """
        return self._robot_info.robot_model

    @property
    def robot_name(self) -> str:
        """Get the robot name.

        Returns:
            The robot name.
        """
        return self._robot_name

    def __enter__(self) -> "Robot":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and clean up resources."""
        self.shutdown()

    def __del__(self) -> None:
        """Destructor to ensure cleanup."""
        if not self._shutdown_called:
            # Only log if the logger is still available (process might be terminating)
            try:
                logger.warning(
                    "Robot instance being destroyed without explicit shutdown call"
                )
                self.shutdown()
            except Exception:  # pylint: disable=broad-except
                # During interpreter shutdown, some modules might not be available
                pass

    def _display_server_log(self, log_data: dict[str, str]) -> None:
        """Display server log message.

        Args:
            log_data: Log data dictionary.
        """
        # Extract log information with safe defaults
        timestamp = log_data.get("timestamp", "")
        message = log_data.get("message", "")
        source = log_data.get("source", "unknown")

        # Validate critical fields
        if not message:
            logger.debug("Received log with empty message")
            return

        # Log the server message with clear identification
        logger.info(f"[SERVER_LOG] [{timestamp}] [{source}] {message}")

    def _print_initialization_info(self) -> None:
        """Print initialization information."""
        console = Console()
        table = Table(show_header=False)
        table.add_column(style="cyan", no_wrap=True)
        table.add_column(style="white")

        table.add_row("Robot Name", str(self.robot_name))
        table.add_row("Robot Model", str(self.robot_model))
        table.add_row("Communication Config", os.getenv(COMM_CFG_PATH_ENV_VAR))

        console.print(table)

    def _validate_configuration(self) -> None:
        """Validate configuration before attempting to connect.

        Performs early checks to fail fast with clear error messages:
        1. Check ZENOH_CONFIG env var exists
        2. Check config file exists and is readable

        Raises:
            ConfigurationError: If ZENOH_CONFIG is not set or file doesn't exist.
        """
        # Check ZENOH_CONFIG environment variable
        zenoh_config_path = os.getenv(COMM_CFG_PATH_ENV_VAR)
        if not zenoh_config_path:
            raise ConfigurationError(
                f"{COMM_CFG_PATH_ENV_VAR} environment variable not set.\n"
                f"  Set it with: export {COMM_CFG_PATH_ENV_VAR}="
                f"~/.dexmate/comm/zenoh/<robot>/zenoh_peer_config.json5"
            )

        # Check config file exists and is readable
        config_path = Path(zenoh_config_path).expanduser()
        if not config_path.exists():
            raise ConfigurationError(
                f"Zenoh config file not found: {config_path}\n"
                f"  Verify the path exists or update {COMM_CFG_PATH_ENV_VAR}."
            )

        if not config_path.is_file():
            raise ConfigurationError(
                f"Zenoh config path is not a file: {config_path}\n"
                f"  {COMM_CFG_PATH_ENV_VAR} must point to a valid config file."
            )

        # Attempt to read the file (permission check)
        try:
            config_path.read_text()
        except PermissionError as e:
            raise ConfigurationError(
                f"Cannot read Zenoh config file: {config_path}\n"
                f"  Check file permissions."
            ) from e

    def _safe_initialize_components(self) -> None:
        """Safely initialize all robot components with consolidated error handling.

        This method consolidates the initialization of components, sensors, and
        default modes into a single method with unified error handling.

        Raises:
            ComponentError: If any critical initialization step fails.
        """
        initialization_steps = [
            ("robot components", self._initialize_robot_components),
            ("component activation", self._wait_for_components),
            ("sensors", self._initialize_sensors),
            ("default state", self._set_default_state),
        ]

        for step_name, step_function in initialization_steps:
            step_function()

    def _initialize_sensors(self) -> None:
        """Initialize sensors and wait for activation."""
        # Note: zenoh_session no longer needed as DexComm handles sessions
        self.sensors = Sensors(self._configs.sensors)
        self.sensors.wait_for_all_active()

    def _initialize_robot_components(self) -> None:
        """Initialize robot components from configuration."""
        component_dict = self._configs.components

        initialized_components = []
        component_mapping = get_component_config_map()

        # Check the hand types of the robot, whether the configs and running hardware are compatible
        self._hand_types = self.query_hand_type()
        disable_estop = bool(int(os.getenv(DISABLE_ESTOP_CHECKING_ENV_VAR, 0)))
        disable_heartbeat = bool(int(os.getenv(DISABLE_HEARTBEAT_ENV_VAR, 0)))
        runtime_override_robot_config(
            self._configs,
            hand_types=self._hand_types,
            disable_estop_checking=disable_estop,
            disable_heartbeat=disable_heartbeat,
        )

        # Phase 1: Create all components (subscribers start receiving data immediately)
        component_instances: dict[str, Any] = {}
        for component_name, component_config in component_dict.items():
            if not component_config.enabled:
                logger.debug(f"Skipping disabled component: {component_name}")
                continue

            config_cls = type(component_config)
            if config_cls not in component_mapping:
                print(
                    f"Skipping {component_name} initialization, no known component detected."
                )
                continue

            component_class = component_mapping[config_cls]
            component_instances[component_name] = component_class(
                name=component_name,
                robot_info=self._robot_info,
            )

        # Phase 2: Wait for all components to receive first state update
        for component_name, component_instance in component_instances.items():
            if not component_instance.wait_for_active(timeout=5.0):
                logger.warning(
                    f"Component {component_name} did not become active within 5s"
                )

            setattr(self, str(component_name), component_instance)
            initialized_components.append(component_name)

        # Validate that we initialized expected components
        expected_components = ["left_arm", "right_arm", "head", "torso", "chassis"]
        initialized_critical = [
            c for c in expected_components if c in initialized_components
        ]

        if not initialized_critical:
            raise ComponentError("Failed to initialize any critical components")

        logger.info(f"Initialized: {', '.join(initialized_components)}")

    def _set_default_state(self) -> None:
        """Set default control modes for robot components."""
        for arm in ["left_arm", "right_arm"]:
            if component := getattr(self, arm, None):
                component.set_modes(["position"] * 7)

        if head := getattr(self, "head", None):
            head.set_mode("enable")
            home_pos = head.get_predefined_pose("home")
            home_pos = self.compensate_torso_pitch(home_pos, "head")
            head.set_joint_pos(home_pos)

    def _wait_for_components(self) -> None:
        """Waits for all critical components to become active.

        This method monitors the activation status of essential robot components
        and ensures they are properly initialized before proceeding.

        Raises:
            ComponentError: If any component fails to activate within the timeout period
                        or if shutdown is triggered during activation.
        """
        component_names = self._robot_info.get_component_list()

        console = Console()
        actives: list[bool] = []
        timeout_sec: Final[float] = 5.0

        status = console.status(
            "[bold green]Waiting for components to become active..."
        )
        status.start()

        try:
            for name in component_names:
                # Check if shutdown was triggered
                if self._shutdown_called:
                    raise ComponentError(
                        "Shutdown triggered during component activation"
                    )

                status.update(f"Waiting for {name} to become active...")
                if component := getattr(self, name, None):
                    if component.wait_for_active(timeout=timeout_sec):
                        actives.append(True)
                        self._components.append(component)
                    else:
                        actives.append(False)
        finally:
            status.stop()

        if not any(actives):
            self.shutdown()
            raise ComponentError(f"No components activated within {timeout_sec}s")

        if not all(actives):
            inactive = [
                name for name, active in zip(component_names, actives) if not active
            ]
            logger.error(
                f"Components failed to activate within {timeout_sec}s: {', '.join(inactive)}.\n"
                f"Other components may work, but some features, e.g. collision avoidance, may not work correctly."
                f"Please check the robot status immediately."
            )
        else:
            logger.info("All motor components are active")

    def has_component(self, component: str) -> bool:
        """Check if the robot has an enabled component.

        A component must both exist in the config and be enabled to return True.
        Disabled components are not initialized and have no attribute on Robot.

        Args:
            component: The component to check.

        Returns:
            True if the robot has the component and it is enabled, False otherwise.
        """
        if not self._robot_info.has_component(component):
            return False
        config = self._configs.components.get(component)
        return config is not None and config.enabled

    def has_sensor(self, sensor: str) -> bool:
        """Check if the robot has an initialized sensor.

        A sensor must be enabled in config and successfully initialized to return True.

        Args:
            sensor: The sensor name (e.g., "head_imu", "ultrasonic").

        Returns:
            True if the sensor is initialized and available, False otherwise.
        """
        return self.sensors.has_sensor(sensor)

    def __getattr__(self, name: str) -> Any:
        """Provide clear error messages when accessing unavailable components.

        This method is only called when normal attribute lookup fails, meaning
        the component was not initialized (disabled or not present on this robot).

        Args:
            name: The attribute name being accessed.

        Raises:
            ComponentNotAvailableError: If the name is a known component that
                is not available on this robot.
            AttributeError: If the name is not a known component.
        """
        if name in self._KNOWN_COMPONENTS:
            # Access _robot_info safely to avoid recursion during __init__
            try:
                robot_info = object.__getattribute__(self, "_robot_info")
                model = robot_info.robot_model
            except AttributeError:
                model = "unknown"
            raise ComponentNotAvailableError(name, model)
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def get_controllable_component_map(self) -> dict[str, Any]:
        """Get the component mapping dictionary.

        Returns:
            Dictionary mapping component names to component instances.
        """  # Components that can be commanded (excludes battery, estop, heartbeat)
        controllable_components: Final[list[str]] = [
            "left_arm",
            "right_arm",
            "left_hand",
            "right_hand",
            "head",
            "torso",
            "chassis",
        ]
        return {
            name: getattr(self, name)
            for name in controllable_components
            if self.has_component(name)
        }

    def validate_component_names(self, joint_pos: dict[str, Any]) -> None:
        """Validate that all component names are valid and initialized.

        Args:
            joint_pos: Joint position dictionary to validate.

        Raises:
            ValueError: If invalid component names are found with detailed guidance.
        """
        if not joint_pos:
            raise ValueError("Joint position dictionary cannot be empty")

        component_map = self.get_controllable_component_map()
        valid_components = set(component_map.keys())
        provided_components = set(joint_pos.keys())
        invalid_components = provided_components - valid_components

        if invalid_components:
            available_msg = (
                f"Available components: {', '.join(sorted(valid_components))}"
            )
            invalid_msg = (
                f"Invalid component names: {', '.join(sorted(invalid_components))}"
            )

            # Provide helpful suggestions for common mistakes
            suggestions = []
            for invalid in invalid_components:
                if invalid in ["left_hand", "right_hand"]:
                    suggestions.append(f"'{invalid}' may not be connected or detected")
                elif invalid.replace("_", "") in [
                    c.replace("_", "") for c in valid_components
                ]:
                    close_match = next(
                        (
                            c
                            for c in valid_components
                            if c.replace("_", "") == invalid.replace("_", "")
                        ),
                        None,
                    )
                    if close_match:
                        suggestions.append(
                            f"Did you mean '{close_match}' instead of '{invalid}'?"
                        )

            error_msg = f"{invalid_msg}. {available_msg}."
            if suggestions:
                error_msg += f" Suggestions: {' '.join(suggestions)}"

            raise ValueError(error_msg)

    def _check_version_compatibility(self) -> None:
        """Check version compatibility between client and server.

        This method uses the new JSON-based version interface to:
        1. Compare client library version with server's minimum required version
        2. Check server component versions for compatibility
        3. Provide clear guidance for version mismatches
        """
        try:
            version_info = self.get_version_info(show=False)
            check_version_compatibility(version_info)
        except Exception as e:
            logger.warning(f"Version compatibility check failed: {e}")

    def shutdown(self) -> None:
        """Cleans up and closes all component connections.

        This method ensures proper cleanup of all components and communication
        channels. It is automatically called when using the context manager
        or when the object is garbage collected.
        """
        if self._shutdown_called:
            logger.warning("Shutdown already called, skipping")
            return

        logger.info("Shutting down robot components...")
        self._shutdown_called = True

        try:
            _active_robots.discard(self)
        except Exception:  # pylint: disable=broad-except
            pass

        # First, stop all components that have stop methods to halt ongoing operations
        for component in self._components:
            if component is not None:
                try:
                    if hasattr(component, "stop"):
                        method = getattr(component, "stop")
                        if callable(method):
                            method()
                except Exception as e:  # pylint: disable=broad-except
                    logger.error(
                        f"Error stopping component {component.__class__.__name__}: {e}"
                    )

        # Shutdown sensors first (they may have background threads)
        try:
            if hasattr(self, "sensors") and self.sensors is not None:
                self.sensors.shutdown()
                # Give time for sensor subscribers to undeclare cleanly
                time.sleep(0.2)
        except Exception as e:  # pylint: disable=broad-except
            logger.error(f"Error shutting down sensors: {e}")

        # Shutdown components in reverse order
        for component in reversed(self._components):
            if component is not None:
                try:
                    component.shutdown()
                except Exception as e:  # pylint: disable=broad-except
                    logger.error(
                        f"Error shutting down component {component.__class__.__name__}: {e}"
                    )

        # Brief delay to allow component shutdown to complete
        time.sleep(0.1)

        # Cleanup DexComm shared session
        try:
            cleanup_session()
        except Exception as e:
            logger.debug(f"Session cleanup note: {e}")
        logger.info("Robot shutdown complete")

    def is_shutdown(self) -> bool:
        """Check if the robot has been shutdown.

        Returns:
            True if the robot has been shutdown, False otherwise.
        """
        return self._shutdown_called

    def get_joint_pos_dict(
        self,
        component: Literal[
            "left_arm", "right_arm", "torso", "head", "left_hand", "right_hand"
        ]
        | list[
            Literal["left_arm", "right_arm", "torso", "head", "left_hand", "right_hand"]
        ],
    ) -> dict[str, float]:
        """Get the joint positions of one or more robot components.

        Args:
            component: Component name or list of component names to get joint positions for.
                Valid components are "left_arm", "right_arm", "torso", "head", "left_hand", and "right_hand".

        Returns:
            Dictionary mapping joint names to joint positions.

        Raises:
            ValueError: If component is not a string or list of strings.
            KeyError: If an invalid component name is provided.
        """
        component_map = self.get_controllable_component_map()

        if isinstance(component, str):
            component = [component]
        if isinstance(component, list):
            joint_pos_dict = {}
            for c in component:
                if c not in component_map:
                    raise KeyError(f"Invalid component name: {c}")
                joint_pos_dict.update(component_map[c].get_joint_pos_dict())
            return joint_pos_dict
        else:
            raise ValueError("Component must be a string or list of strings")

    def execute_trajectory(
        self,
        trajectory: dict[str, np.ndarray | dict[str, np.ndarray]],
        control_hz: float = 100,
        relative: bool = False,
    ) -> None:
        """Execute a trajectory on the robot.

        Args:
            trajectory: Dictionary mapping component names to either:
                - numpy arrays of joint positions
                - dictionaries with 'position' and optional 'velocity' keys
            control_hz: Control frequency in Hz.
            relative: Whether positions are relative to current position.

        Raises:
            ValueError: If trajectory is empty or components have different trajectory lengths.
            ValueError: If trajectory format is invalid.
            DexcontrolError: If trajectory execution fails.
        """
        if not trajectory:
            raise ValueError("Trajectory must be a non-empty dictionary")

        try:
            # Process trajectory to standardize format
            processed_trajectory = self._process_trajectory(trajectory)

            # Validate trajectory lengths
            self._validate_trajectory_lengths(processed_trajectory)

            # Execute trajectory
            self._execute_processed_trajectory(
                processed_trajectory, control_hz, relative
            )

        except Exception as e:
            raise DexcontrolError(f"Failed to execute trajectory: {e}") from e

    def set_joint_pos(
        self,
        joint_pos: dict[str, list[float] | np.ndarray],
        relative: bool = False,
        wait_time: float = 0.0,
        wait_kwargs: dict[str, Any] | None = None,
        exit_on_reach: bool = False,
        exit_on_reach_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Set the joint positions of the robot.

        Args:
            joint_pos: Dictionary mapping component names to joint positions.
                Values can be either lists of floats or numpy arrays.
            relative: Whether to set positions relative to current position.
            wait_time: Time to wait for movement completion in seconds.
            wait_kwargs: Additional parameters for trajectory generation.
                control_hz: Control frequency in Hz (default: 100).
                max_vel: Maximum velocity for trajectory (default: 3.).
            exit_on_reach: If True, the function will exit when the joint positions are reached.
            exit_on_reach_kwargs: Optional parameters for exit when the joint positions are reached.

        Raises:
            ValueError: If any component name is invalid.
            DexcontrolError: If joint position setting fails.
        """
        if wait_kwargs is None:
            wait_kwargs = {}

        try:
            start_time = time.time()
            component_map = self.get_controllable_component_map()

            # Validate component names
            self.validate_component_names(joint_pos)

            # Separate position-velocity controlled components from others
            pv_components = [c for c in joint_pos if c in self._pv_components]
            non_pv_components = [c for c in joint_pos if c not in self._pv_components]

            # Set PV components immediately
            self._set_pv_components(pv_components, joint_pos, component_map, relative)

            # Handle non-PV components based on wait_time
            if wait_time <= 0:
                self._set_non_pv_components_immediate(
                    non_pv_components, joint_pos, component_map, relative
                )
            else:
                self._set_non_pv_components_with_trajectory(
                    non_pv_components,
                    joint_pos,
                    component_map,
                    relative,
                    wait_time,
                    wait_kwargs,
                    exit_on_reach=exit_on_reach,
                    exit_on_reach_kwargs=exit_on_reach_kwargs,
                )
            remaining_time = wait_time - (time.time() - start_time)
            if remaining_time <= 0:
                return

            self._wait_for_multi_component_positions(
                component_map,
                pv_components,
                joint_pos,
                start_time,
                wait_time,
                exit_on_reach,
                exit_on_reach_kwargs,
            )

        except Exception as e:
            raise DexcontrolError(f"Failed to set joint positions: {e}") from e

    def _wait_for_multi_component_positions(
        self,
        component_map: dict[str, Any],
        components: list[str],
        joint_pos: dict[str, list[float] | np.ndarray],
        start_time: float,
        wait_time: float,
        exit_on_reach: bool = False,
        exit_on_reach_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Wait for multiple components to reach target positions.

        Args:
            component_map: Mapping of component names to component instances.
            components: List of component names to check.
            joint_pos: Target joint positions for each component.
            start_time: Time when the operation started.
            wait_time: Maximum time to wait.
            exit_on_reach: If True, exit early when all positions are reached.
            exit_on_reach_kwargs: Optional parameters for position checking.
        """
        sleep_interval = 0.01  # Use consistent sleep interval

        if exit_on_reach:
            if components:
                # Set default tolerance if not provided
                exit_on_reach_kwargs = exit_on_reach_kwargs or {}

                # Wait until all positions are reached or timeout
                while time.time() - start_time < wait_time:
                    if all(
                        component_map[c].is_joint_pos_reached(
                            joint_pos[c], **exit_on_reach_kwargs
                        )
                        for c in components
                    ):
                        break
                    time.sleep(sleep_interval)
        else:
            # Simple wait without position checking
            while time.time() - start_time < wait_time:
                time.sleep(sleep_interval)

    def compensate_torso_pitch(self, joint_pos: np.ndarray, part: str) -> np.ndarray:
        """Compensate for torso pitch in joint positions.

        Args:
            joint_pos: Joint positions to compensate.
            robot: Robot instance.
            part: Component name for which joint positions are being compensated.

        Returns:
            Compensated joint positions.
        """
        if self.robot_model == "vega_1u":
            torso_pitch = np.pi / 2
        else:
            torso_pitch = self.torso.pitch_angle

        # Calculate pitch adjustment based on body part
        if part == "right_arm":
            pitch_adjustment = -torso_pitch
        elif part == "left_arm":
            pitch_adjustment = torso_pitch
        elif part == "head":
            pitch_adjustment = torso_pitch - np.pi / 2
        else:
            raise ValueError(
                f"Unsupported body part: {part}. "
                f"Supported parts: left_arm, right_arm, head"
            )

        # Create a copy to avoid modifying the original array
        adjusted_positions = joint_pos.copy()
        adjusted_positions[0] += pitch_adjustment

        return adjusted_positions

    def have_hand(self, side: Literal["left", "right"]) -> bool:
        """Check if the robot has a hand."""
        return self._hand_types.get(side) != HandType.UNKNOWN

    def _process_trajectory(
        self, trajectory: dict[str, np.ndarray | dict[str, np.ndarray]]
    ) -> dict[str, dict[str, np.ndarray]]:
        """Process trajectory to standardize format.

        Args:
            trajectory: Raw trajectory data.

        Returns:
            Processed trajectory with standardized format.

        Raises:
            ValueError: If trajectory format is invalid.
        """
        processed_trajectory: dict[str, dict[str, np.ndarray]] = {}
        for component, data in trajectory.items():
            if isinstance(data, np.ndarray):
                processed_trajectory[component] = {"position": data}
            elif isinstance(data, dict) and "position" in data:
                processed_trajectory[component] = data
            else:
                raise ValueError(f"Invalid trajectory format for component {component}")
        return processed_trajectory

    def _validate_trajectory_lengths(
        self, processed_trajectory: dict[str, dict[str, np.ndarray]]
    ) -> None:
        """Validate that all trajectory components have consistent lengths.

        Args:
            processed_trajectory: Processed trajectory data.

        Raises:
            ValueError: If trajectory lengths are inconsistent.
        """
        first_component = next(iter(processed_trajectory))
        first_length = len(processed_trajectory[first_component]["position"])

        for component, data in processed_trajectory.items():
            if len(data["position"]) != first_length:
                raise ValueError(
                    f"Component {component} has different trajectory length"
                )
            if "velocity" in data and len(data["velocity"]) != first_length:
                raise ValueError(
                    f"Velocity length for {component} doesn't match position length"
                )

    def _execute_processed_trajectory(
        self,
        processed_trajectory: dict[str, dict[str, np.ndarray]],
        control_hz: float,
        relative: bool,
    ) -> None:
        """Execute the processed trajectory.

        Args:
            processed_trajectory: Processed trajectory data.
            control_hz: Control frequency in Hz.
            relative: Whether positions are relative to current position.

        Raises:
            ValueError: If invalid component is specified.
        """
        rate_limiter = RateLimiter(control_hz)
        component_map = self.get_controllable_component_map()

        first_component = next(iter(processed_trajectory))
        trajectory_length = len(processed_trajectory[first_component]["position"])

        for i in range(trajectory_length):
            for c, data in processed_trajectory.items():
                if c not in component_map:
                    raise ValueError(f"Invalid component: {c}")

                position = data["position"][i]
                if "velocity" in data:
                    velocity = data["velocity"][i]
                    component_map[c].set_joint_pos_vel(
                        position, velocity, relative=relative, wait_time=0.0
                    )
                else:
                    component_map[c].set_joint_pos(
                        position, relative=relative, wait_time=0.0
                    )
            rate_limiter.sleep()

    def _set_pv_components(
        self,
        pv_components: list[str],
        joint_pos: dict[str, list[float] | np.ndarray],
        component_map: dict[str, Any],
        relative: bool,
    ) -> None:
        """Set position-velocity controlled components immediately.

        Args:
            pv_components: List of PV component names.
            joint_pos: Joint position dictionary.
            component_map: Component mapping dictionary.
            relative: Whether positions are relative.
        """
        for c in pv_components:
            component_map[c].set_joint_pos(
                joint_pos[c], relative=relative, wait_time=0.0
            )

    def _set_non_pv_components_immediate(
        self,
        non_pv_components: list[str],
        joint_pos: dict[str, list[float] | np.ndarray],
        component_map: dict[str, Any],
        relative: bool,
    ) -> None:
        """Set non-PV components immediately without trajectory.

        Args:
            non_pv_components: List of non-PV component names.
            joint_pos: Joint position dictionary.
            component_map: Component mapping dictionary.
            relative: Whether positions are relative.
        """
        for c in non_pv_components:
            component_map[c].set_joint_pos(joint_pos[c], relative=relative)

    def _set_non_pv_components_with_trajectory(
        self,
        non_pv_components: list[str],
        joint_pos: dict[str, list[float] | np.ndarray],
        component_map: dict[str, Any],
        relative: bool,
        wait_time: float,
        wait_kwargs: dict[str, Any],
        exit_on_reach: bool = False,
        exit_on_reach_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Set non-PV components with smooth trajectory over wait_time.

        Args:
            non_pv_components: List of non-PV component names.
            joint_pos: Joint position dictionary.
            component_map: Component mapping dictionary.
            relative: Whether positions are relative.
            wait_time: Time to wait for movement completion.
            wait_kwargs: Additional trajectory parameters.
            exit_on_reach: If True, the function will exit when the joint positions are reached.
            exit_on_reach_kwargs: Optional parameters for exit when the joint positions are reached.
        """
        control_hz = wait_kwargs.get("control_hz", self.left_arm._default_control_hz)
        max_vel = wait_kwargs.get("max_vel", self.left_arm._joint_vel_limit)

        # Generate trajectories for smooth motion during wait_time
        rate_limiter = RateLimiter(control_hz)
        non_pv_component_traj = {}
        max_traj_steps = 0

        # Calculate trajectories for each component
        for c in non_pv_components:
            current_joint_pos = component_map[c].get_joint_pos().copy()
            target_pos = joint_pos[c]
            # Convert to numpy array if it's a list
            if isinstance(target_pos, list):
                target_pos = np.array(target_pos)
            non_pv_component_traj[c], steps = generate_linear_trajectory(
                current_joint_pos, target_pos, max_vel, control_hz
            )
            max_traj_steps = max(max_traj_steps, steps)

        # Execute trajectories with timing
        start_time = time.time()
        for step in range(max_traj_steps):
            for c in non_pv_components:
                if step < len(non_pv_component_traj[c]):
                    component_map[c].set_joint_pos(
                        non_pv_component_traj[c][step], relative=relative, wait_time=0.0
                    )
            rate_limiter.sleep()
            if time.time() - start_time > wait_time:
                break

        # Wait for any remaining time
        self._wait_for_multi_component_positions(
            component_map,
            non_pv_components,
            joint_pos,
            start_time,
            wait_time,
            exit_on_reach,
            exit_on_reach_kwargs,
        )
