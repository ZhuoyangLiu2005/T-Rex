# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Query utilities for robot communication using DexComm.

This module provides the RobotQueryInterface class that encapsulates all communication
queries with the robot server using DexComm's service pattern. It handles various query
types including hand type detection, version information, status queries, and control
operations.
"""

import time
from typing import Any, Literal

import numpy as np
from dexbot_utils import RobotInfo
from dexbot_utils.configs import BaseRobotConfig
from dexbot_utils.hand import HandType
from dexcomm import Node

# Use DexComm for all communication
from dexcomm.codecs import (
    ClearErrorCodec,
    DictDataCodec,
    RebootComponentCodec,
    RobotComponentEnum,
    RobotComponentStatesCodec,
)
from loguru import logger

from dexcontrol.exceptions import RobotConnectionError, ServiceUnavailableError
from dexcontrol.utils.pb_utils import (
    ComponentStatus,
)
from dexcontrol.utils.viz_utils import show_component_status


class RobotQueryInterface:
    """Base class for zenoh query operations.

    This class provides a clean interface for all zenoh-based queries and
    communication operations. It maintains references to the zenoh session
    and configuration needed for queries.

    Can be used as a context manager for automatic resource cleanup:
        >>> with RobotQueryInterface.create() as interface:
        ...     version_info = interface.get_version_info()
    """

    def __init__(self, configs: BaseRobotConfig):
        """Initialize the RobotQueryInterface.

        Args:
            name: Name of the robot query interface component Node.
            configs: Robot configuration containing query names.
        """
        # Session parameter kept for compatibility but not used
        self._configs = configs
        self._node = Node(name="robot_query_interface")
        self._hand_querier = self._node.create_service_client(
            service_name=configs.querables["hand_info"],
            request_encoder=None,
            response_decoder=DictDataCodec.decode,
            timeout=5.0,
        )
        self._ntp_querier = self._node.create_service_client(
            service_name=configs.querables["soc_ntp"],
            request_encoder=DictDataCodec.encode,
            response_decoder=DictDataCodec.decode,
            timeout=5.0,
        )
        self._version_querier = self._node.create_service_client(
            service_name=configs.querables["version_info"],
            request_encoder=None,
            response_decoder=DictDataCodec.decode,
            timeout=5.0,
        )
        self._component_status_querier = self._node.create_service_client(
            service_name=configs.querables["status_info"],
            request_encoder=None,
            response_decoder=RobotComponentStatesCodec.decode,
            timeout=5.0,
        )
        self._reboot_querier = self._node.create_service_client(
            service_name=configs.querables["reboot"],
            request_encoder=RebootComponentCodec.encode,
            response_decoder=None,
            timeout=5.0,
        )
        self._clear_error_querier = self._node.create_service_client(
            service_name=configs.querables["clear_error"],
            request_encoder=ClearErrorCodec.encode,
            response_decoder=None,
            timeout=5.0,
        )

    @classmethod
    def create(cls) -> "RobotQueryInterface":
        """Create a standalone RobotQueryInterface.

        This class method provides a convenient way to create a RobotQueryInterface
        without requiring the full Robot class. DexComm handles all session
        management internally.

        Returns:
            RobotQueryInterface instance ready for use.

        Example:
            >>> query_interface = RobotQueryInterface.create()
            >>> version_info = query_interface.get_version_info()
            >>> query_interface.close()
        """
        # DexComm handles session internally, we just need config
        config: BaseRobotConfig = RobotInfo().config
        instance = cls(configs=config)

        return instance

    def close(self) -> None:
        """Close the communication session if owned by this instance.

        This method should be called when done using a standalone
        RobotQueryInterface to properly clean up resources.
        """
        self._node.shutdown()

    def __enter__(self) -> "RobotQueryInterface":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager and clean up resources."""
        self.close()

    def query_hand_type(self, max_attempts: int = 5) -> dict[str, HandType]:
        """Query the hand type information from the server.

        Returns:
            Dictionary containing hand type information for left and right hands.
            Format: {"left": hand_type_name, "right": hand_type_name}
            Possible hand types: "UNKNOWN", "HandF5D6_V1", "HandF5D6_V2"
            UNKNOWN means not connected or unknown end effector connected.

        Raises:
            RobotConnectionError: If cannot connect to robot.
            ServiceUnavailableError: If hand type information cannot be retrieved after 3 attempts.
        """
        last_error = None

        # Wait for service to be available - this serves as the connection test
        if not self._hand_querier.wait_for_service(timeout=10.0):  # type: ignore[attr-defined]
            raise RobotConnectionError(
                "Cannot connect to robot (timeout after 10s).\n"
                "  1. Run 'dextop topic list' to verify you can receive topics from the robot.\n"
                "  2. Check that robot is powered on and network is reachable."
            )

        for _ in range(max_attempts):
            response = self._hand_querier.call(None)

            if response:
                hand_info = response

                # Validate the expected format
                if (
                    isinstance(hand_info, dict)
                    and "left" in hand_info
                    and "right" in hand_info
                ):
                    logger.info(f"End effector hand types: {hand_info}")
                    return {
                        "left": HandType(hand_info["left"]),
                        "right": HandType(hand_info["right"]),
                    }
                else:
                    last_error = f"Invalid response format: {hand_info}"
            else:
                time.sleep(0.2)
                last_error = "No response received from server"

        # All attempts failed, raise error
        error_msg = f"Failed to query hand type after {max_attempts} attempts"
        if last_error:
            error_msg += f": {last_error}"
        raise ServiceUnavailableError(error_msg)

    def query_ntp(
        self,
        sample_count: int = 30,
        show: bool = False,
        device: Literal["soc", "jetson"] = "soc",
    ) -> dict[Literal["success", "offset", "rtt"], bool | float]:
        """Query the NTP server via zenoh for time synchronization and compute robust statistics.

        Args:
            sample_count: Number of NTP samples to request (default: 50).
            show: Whether to print summary statistics using a rich table.
            timeout: Timeout for the zenoh querier in seconds (default: 2.0).
            device: Which device to query for NTP ("soc" or "jetson").

        Returns:
            Dictionary with keys:
                - "success": True if any replies were received, False otherwise.
                - "offset": Mean offset (in seconds) after removing RTT outliers.
                - "rtt": Mean round-trip time (in seconds) after removing RTT outliers.
        """
        if device == "jetson":
            raise NotImplementedError("Jetson NTP query is not implemented yet")

        time_offset = []
        time_rtt = []

        reply_count = 0
        for i in range(sample_count):
            request_data = dict(
                client_send_time_ns=time.time_ns(),
                sample_count=sample_count,
                sample_index=i,
            )

            # Use call_service for NTP query
            try:
                response_data = self._ntp_querier.call(request_data)

                if response_data:
                    reply_count += 1
                    client_receive_time_ns = time.time_ns()
                    t0 = request_data["client_send_time_ns"]
                    t1 = response_data["server_receive_time_ns"]
                    t2 = response_data["server_send_time_ns"]
                    t3 = client_receive_time_ns
                    offset = ((t1 - t0) + (t2 - t3)) // 2 / 1e9
                    rtt = (t3 - t0) / 1e9
                    time_offset.append(offset)
                    time_rtt.append(rtt)
            except Exception as e:
                logger.debug(f"NTP query {i} failed: {e}")

            if i < sample_count - 1:
                time.sleep(0.01)
        if reply_count == 0:
            return {"success": False, "offset": 0.0, "rtt": 0.0}

        # Compute simple NTP statistics

        stats = {
            "offset (mean)": float(np.mean(time_offset)) if time_offset else 0.0,
            "round_trip_time (mean)": float(np.mean(time_rtt)) if time_rtt else 0.0,
            "offset (std)": float(np.std(time_offset)) if time_offset else 0.0,
            "round_trip_time (std)": float(np.std(time_rtt)) if time_rtt else 0.0,
        }
        offset = float(stats["offset (mean)"])
        rtt = float(stats["round_trip_time (mean)"])
        if show:
            from dexcontrol.utils.viz_utils import show_ntp_stats

            show_ntp_stats(stats)

        return {"success": True, "offset": offset, "rtt": rtt}

    def get_version_info(self, show: bool = True) -> dict[str, Any]:
        """Retrieve comprehensive version information using JSON interface.

        This method queries the new JSON-based version endpoint that provides:
        - Server component versions (hardware, software, compile_time, hashes)
        - Minimum required client version
        - Version compatibility information

        Args:
            show: Whether to display the version information.

        Returns:
            Dictionary containing comprehensive version information with structure:
            {
                "server": {
                    "component_name": {
                        "hardware_version": int,
                        "software_version": int,
                        "compile_time": str,
                        "main_hash": str,
                        "sub_hash": str
                    }
                },
                "client": {
                    "minimal_version": str
                }
            }

        Raises:
            ServiceUnavailableError: If version information cannot be retrieved.
        """
        # wait_for_service is implemented in Rust, not in Python type stubs
        if not self._version_querier.wait_for_service(timeout=5.0):  # type: ignore[attr-defined]
            raise ServiceUnavailableError("Version info service not responding.\n")

        try:
            version_info = self._version_querier.call(None)
        except Exception as e:
            raise ServiceUnavailableError(
                f"Failed to retrieve version information: {e}"
            ) from e

        if not version_info:
            raise ServiceUnavailableError(
                "No valid version information received from server."
            )

        if show:
            self._show_version_info(version_info)
        return version_info

    def get_component_status(
        self, show: bool = True
    ) -> dict[str, dict[str, bool | ComponentStatus]]:
        """Retrieve status information for all components.

        Args:
            show: Whether to display the status information.

        Returns:
            Dictionary containing status information for all components.

        Raises:
            ServiceUnavailableError: If status information cannot be retrieved.
        """
        try:
            response = self._component_status_querier.call(None)

            if show:
                show_component_status(response)
            return response
        except Exception as e:
            raise ServiceUnavailableError(
                f"Failed to retrieve component status: {e}"
            ) from e

    def reboot_component(
        self,
        part: Literal["arm", "chassis", "torso"],
    ) -> None:
        """Reboot a specific robot component.

        Args:
            part: Component to reboot ("arm", "chassis", or "torso").

        Raises:
            ValueError: If the specified component is invalid.
            ServiceUnavailableError: If the reboot operation fails.
        """
        # wait_for_service is implemented in Rust, not in Python type stubs
        if not self._reboot_querier.wait_for_service(timeout=5.0):  # type: ignore[attr-defined]
            raise ServiceUnavailableError(
                f"Reboot service not responding for {part}.\n"
                "  Robot may be initializing - wait and retry."
            )

        try:
            query_msg = {"component": part}
            self._reboot_querier.call(query_msg)
            logger.info(f"Rebooting component: {part}")
        except Exception as e:
            raise ServiceUnavailableError(
                f"Failed to reboot component {part}: {e}"
            ) from e

    def clear_error(
        self,
        part: Literal["left_arm", "right_arm", "chassis", "head"] | str,
    ) -> None:
        """Clear error state for a specific component.

        Args:
            part: Component to clear error state for.

        Raises:
            ValueError: If the specified component is invalid.
            ServiceUnavailableError: If the error clearing operation fails.
        """
        component_map = {
            "left_arm": RobotComponentEnum.LEFT_ARM,
            "right_arm": RobotComponentEnum.RIGHT_ARM,
            "head": RobotComponentEnum.HEAD,
            "chassis": RobotComponentEnum.CHASSIS,
            "left_hand": RobotComponentEnum.LEFT_HAND,
            "right_hand": RobotComponentEnum.RIGHT_HAND,
        }

        if part not in component_map:
            raise ValueError(f"Invalid component: {part}")

        # wait_for_service is implemented in Rust, not in Python type stubs
        if not self._clear_error_querier.wait_for_service(timeout=5.0):  # type: ignore[attr-defined]
            raise ServiceUnavailableError(
                f"Clear error service not responding for {part}.\n"
                "  Robot may be initializing - wait and retry."
            )

        try:
            query_msg = {"component": component_map[part]}
            self._clear_error_querier.call(query_msg)
            logger.info(f"Cleared error of {part}")
        except Exception as e:
            raise ServiceUnavailableError(
                f"Failed to clear error for component {part}: {e}"
            ) from e

    def _show_version_info(self, version_info: dict[str, Any]) -> None:
        """Display comprehensive version information in a formatted table.

        Args:
            version_info: Dictionary containing server and client version information.
        """
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="🤖 Robot System Version Information")

        table.add_column("Component", justify="left", style="cyan", no_wrap=True)
        table.add_column("Hardware Ver.", justify="center", style="magenta")
        table.add_column("Software Ver.", justify="center", style="green")
        table.add_column("Compile Time", justify="center", style="yellow")
        table.add_column("Main Hash", justify="center", style="blue")
        table.add_column("Sub Hash", justify="center", style="red")

        # Display server components
        server_info = version_info.get("firmware_version", {})
        for component, info in server_info.items():
            if isinstance(info, dict):
                table.add_row(
                    component,
                    str(info.get("hardware_version", "N/A")),
                    str(info.get("software_version", "N/A")),
                    str(info.get("compile_time", "N/A")),
                    str(info.get("main_hash", "N/A")[:8])
                    if info.get("main_hash")
                    else "N/A",
                    str(info.get("sub_hash", "N/A")[:8])
                    if info.get("sub_hash")
                    else "N/A",
                )

        console.print(table)
