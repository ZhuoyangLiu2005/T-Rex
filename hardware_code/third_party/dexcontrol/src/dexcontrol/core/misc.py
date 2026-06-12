# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Miscellaneous robot components module.

This module provides classes for various auxiliary robot components such as Battery,
EStop (emergency stop), ServerLogSubscriber, and UltraSonicSensor.
"""

import os
import threading
import time
from typing import cast

from dexbot_utils import RobotInfo
from dexbot_utils.configs.components.vega_1 import (
    BatteryConfig,
    EStopConfig,
    HeartbeatConfig,
)
from dexcomm import HeartbeatMonitor
from dexcomm.codecs import (
    BMSStateCodec,
    EStopStateCodec,
    SoftwareEstopCodec,
)
from loguru import logger
from rich.console import Console
from rich.table import Table

from dexcontrol.core.component import RobotComponent


class Battery(RobotComponent):
    """Battery component that monitors and displays battery status information.

    This class provides methods to monitor battery state including voltage, current,
    temperature and power consumption. It can display the information in either a
    formatted rich table or plain text format.

    Attributes:
        _console: Rich console instance for formatted output.
        _monitor_thread: Background thread for battery monitoring.
        _shutdown_event: Event to signal thread shutdown.
    """

    def __init__(self, name: str, robot_info: RobotInfo) -> None:
        """Initialize the Battery component.

        Args:
            robot_info: RobotInfo instance.
        """
        config = robot_info.get_component_config(name)
        config = cast(BatteryConfig, config)
        super().__init__(
            name, config.state_sub_topic, state_decoder=BMSStateCodec.decode
        )
        self._console = Console()
        self._shutdown_event = threading.Event()
        self._monitor_thread = threading.Thread(
            target=self._battery_monitor, daemon=True
        )
        self._monitor_thread.start()

    def _battery_monitor(self) -> None:
        """Background thread that periodically checks battery level and warns if low."""
        # Wait for first data to arrive before monitoring to avoid false warnings
        # Battery topic publishes at low frequency, so allow generous timeout
        while not self._shutdown_event.is_set():
            if self._subscriber.get_latest() is not None:
                break
            self._shutdown_event.wait(0.5)

        while not self._shutdown_event.is_set():
            try:
                state = self._subscriber.get_latest()
                if state is not None:
                    battery_level = float(state["percentage"])
                    if battery_level < 20:
                        logger.warning(
                            f"Battery level is low ({battery_level:.1f}%). "
                            "Please charge the battery."
                        )
            except Exception as e:
                logger.debug(f"Battery monitor error: {e}")

            # Check every 30 seconds (low frequency)
            self._shutdown_event.wait(30.0)

    def get_status(self) -> dict[str, float]:
        """Gets the current battery state information.

        Returns:
            Dictionary containing battery metrics including:
                - percentage: Battery charge level (0-100)
                - temperature: Battery temperature in Celsius
                - current: Current draw in Amperes
                - voltage: Battery voltage
                - power: Power consumption in Watts
        """
        state = self._subscriber.get_latest()
        if state is None:
            return {
                "percentage": 0.0,
                "temperature": 0.0,
                "current": 0.0,
                "voltage": 0.0,
                "power": 0.0,
            }
        return {
            "percentage": float(state["percentage"]),
            "temperature": float(state["temperature"]),
            "current": float(state["current"]),
            "voltage": float(state["voltage"]),
            "power": float(state["current"] * state["voltage"]),
        }

    def show(self) -> None:
        """Displays the current battery status as a formatted table with color indicators."""
        state = self._subscriber.get_latest()

        table = Table(title="Battery Status")
        table.add_column("Parameter", style="cyan")
        table.add_column("Value")

        if state is None:
            table.add_row("Status", "[red]No battery data available[/]")
            self._console.print(table)
            return

        battery_style = self._get_battery_level_style(state["percentage"])
        table.add_row(
            "Battery Level", f"[{battery_style}]{state['percentage']:.1f}%[/]"
        )

        temp_style = self._get_temperature_style(state["temperature"])
        table.add_row("Temperature", f"[{temp_style}]{state['temperature']:.1f}°C[/]")

        power = state["current"] * state["voltage"]
        power_style = self._get_power_style(power)
        table.add_row(
            "Power Consumption",
            f"[{power_style}]{power:.2f}W[/] ([blue]{state['current']:.2f}A[/] "
            f"× [blue]{state['voltage']:.2f}V[/])",
        )

        self._console.print(table)

    def shutdown(self) -> None:
        """Shuts down the battery component and stops monitoring thread."""
        self._shutdown_event.set()
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2.0)  # Extended timeout
            if self._monitor_thread.is_alive():
                logger.warning("Battery monitor thread did not terminate cleanly")
        super().shutdown()

    @staticmethod
    def _get_battery_level_style(percentage: float) -> str:
        """Returns the appropriate style based on battery percentage.

        Args:
            percentage: Battery charge level (0-100).

        Returns:
            Rich text style string for color formatting.
        """
        if percentage < 30:
            return "bold red"
        elif percentage < 60:
            return "bold yellow"
        else:
            return "bold dark_green"

    @staticmethod
    def _get_temperature_style(temperature: float) -> str:
        """Returns the appropriate style based on temperature value.

        Args:
            temperature: Battery temperature in Celsius.

        Returns:
            Rich text style string for color formatting.
        """
        if temperature < -1:
            return "bold red"  # Too cold
        elif temperature <= 30:
            return "bold dark_green"  # Normal range
        elif temperature <= 38:
            return "bold orange"  # Getting warm
        else:
            return "bold red"  # Too hot

    @staticmethod
    def _get_power_style(power: float) -> str:
        """Returns the appropriate style based on power consumption.

        Args:
            power: Power consumption in Watts.

        Returns:
            Rich text style string for color formatting.
        """
        if power < 200:
            return "bold dark_green"
        elif power <= 500:
            return "bold orange"
        else:
            return "bold red"


class EStop(RobotComponent):
    """EStop component that monitors and controls emergency stop functionality.

    This class provides methods to monitor EStop state and activate/deactivate
    the software emergency stop.

    Attributes:
        _estop_query_name: Zenoh query name for setting EStop state.
        _monitor_thread: Background thread for EStop monitoring.
        _shutdown_event: Event to signal thread shutdown.
    """

    def __init__(
        self,
        name: str,
        robot_info: RobotInfo,
    ) -> None:
        """Initialize the EStop component.

        Args:
            robot_info: RobotInfo instance.
        """
        config = robot_info.get_component_config(name)
        config = cast(EStopConfig, config)
        self._enabled = config.enabled
        super().__init__(
            name, config.state_sub_topic, state_decoder=EStopStateCodec.decode
        )
        self._estop_querier = self._node.create_service_client(
            service_name=config.estop_query_name,
            request_encoder=SoftwareEstopCodec.encode,
            response_decoder=None,
            timeout=0.05,
        )
        if not self._enabled:
            logger.warning("EStop monitoring is DISABLED via configuration")
            return
        self._shutdown_event = threading.Event()
        self._monitor_thread = threading.Thread(target=self._estop_monitor, daemon=True)
        self._monitor_thread.start()

    def _estop_monitor(self) -> None:
        """Background thread that continuously monitors EStop button state."""
        # Wait for first data to arrive before monitoring
        while not self._shutdown_event.is_set():
            if self._subscriber.get_latest() is not None:
                break
            self._shutdown_event.wait(0.1)

        while not self._shutdown_event.is_set():
            try:
                state = self._subscriber.get_latest()
                if state is not None:
                    button_pressed = (
                        state.get("left_base_estop_enabled", False)
                        or state.get("right_base_estop_enabled", False)
                        or state.get("torso_estop_enabled", False)
                        or state.get("remote_estop_enabled", False)
                    )
                    if button_pressed:
                        logger.critical(
                            "E-STOP BUTTON PRESSED! Exiting program immediately."
                        )
                        os._exit(1)
            except Exception as e:
                logger.debug(f"EStop monitor error: {e}")

            # Check every 100ms for responsive emergency stop
            self._shutdown_event.wait(0.1)

    def _set_estop(self, enable: bool) -> None:
        """Sets the software emergency stop (E-Stop) state of the robot.

        This controls the software E-Stop, which is separate from the physical button
        on the robot. The robot will stop if either the software or hardware E-Stop is
        activated.

        Args:
            enable: If True, activates the software E-Stop. If False, deactivates it.
        """
        # Wait for service to be available
        if not self._estop_querier.wait_for_service(timeout=5.0):
            logger.warning(
                f"{self._node.get_name()}: E-Stop service not available, command may fail"
            )

        query_msg = {"enabled": enable}
        self._estop_querier.call(query_msg)
        logger.info(f"Set E-Stop to {enable}")

    def get_status(self) -> dict[str, bool]:
        """Gets the current EStop state information.

        Returns:
            Dictionary containing EStop metrics including:
                - button_pressed: EStop button pressed
                - software_estop_enabled: Software EStop enabled
        """
        state = self._subscriber.get_latest()
        if state is None:
            return {
                "button_pressed": False,
                "software_estop_enabled": False,
            }
        button_pressed = (
            state.get("left_base_estop_enabled", False)
            or state.get("right_base_estop_enabled", False)
            or state.get("torso_estop_enabled", False)
            or state.get("remote_estop_enabled", False)
        )
        return {
            "button_pressed": button_pressed,
            "software_estop_enabled": state["software_estop_enabled"],
        }

    def is_button_pressed(self) -> bool:
        """Checks if the EStop button is pressed."""
        state = self._subscriber.get_latest()
        if state is None:
            return False
        button_pressed = (
            state.get("left_base_estop_enabled", False)
            or state.get("right_base_estop_enabled", False)
            or state.get("torso_estop_enabled", False)
            or state.get("remote_estop_enabled", False)
        )
        return button_pressed

    def is_software_estop_enabled(self) -> bool:
        """Checks if the software EStop is enabled."""
        state = self._subscriber.get_latest()
        if state is None:
            return False
        return state["software_estop_enabled"]

    def activate(self) -> None:
        """Activates the software emergency stop (E-Stop)."""
        self._set_estop(True)

    def deactivate(self) -> None:
        """Deactivates the software emergency stop (E-Stop)."""
        self._set_estop(False)

    def toggle(self) -> None:
        """Toggles the software emergency stop (E-Stop) state of the robot."""
        self._set_estop(not self.is_software_estop_enabled())

    def shutdown(self) -> None:
        """Shuts down the EStop component and stops monitoring thread."""
        if self._enabled:
            self._shutdown_event.set()
            if self._monitor_thread and self._monitor_thread.is_alive():
                self._monitor_thread.join(timeout=2.0)  # Extended timeout
                if self._monitor_thread.is_alive():
                    logger.warning("EStop monitor thread did not terminate cleanly")
        super().shutdown()

    def show(self) -> None:
        """Displays the current EStop status as a formatted table with color indicators."""
        table = Table(title="E-Stop Status")
        table.add_column("Parameter", style="cyan")
        table.add_column("Value")

        button_pressed = self.is_button_pressed()
        button_style = "bold red" if button_pressed else "bold dark_green"
        table.add_row("Button Pressed", f"[{button_style}]{button_pressed}[/]")

        if_software_estop_enabled = self.is_software_estop_enabled()
        software_style = "bold red" if if_software_estop_enabled else "bold dark_green"
        table.add_row(
            "Software E-Stop Enabled",
            f"[{software_style}]{if_software_estop_enabled}[/]",
        )

        console = Console()
        console.print(table)


class Heartbeat:
    """Heartbeat monitor that ensures the low-level controller is functioning properly.

    This class monitors a heartbeat signal from the low-level controller and exits
    the program immediately if no heartbeat is received within the specified timeout.
    This provides a critical safety mechanism to prevent the robot from operating
    when the low-level controller is not functioning properly.

    The core monitoring runs entirely in Rust (via dexcomm.HeartbeatMonitor),
    making it immune to Python GIL contention. Even under heavy Python workloads
    (e.g., neural network inference, control optimization), the heartbeat subscriber
    and timeout detection operate without GIL involvement.

    Attributes:
        _monitor: Rust-backed HeartbeatMonitor that handles subscription and timeout.
        _enabled: Whether heartbeat monitoring is enabled.
    """

    def __init__(
        self,
        name: str,
        robot_info: RobotInfo,
    ) -> None:
        """Initialize the Heartbeat monitor.

        Args:
            name: Component name.
            robot_info: RobotInfo instance.
        """

        config = robot_info.get_component_config(name)
        config = cast(HeartbeatConfig, config)
        self._enabled = config.enabled

        # Create Rust-backed heartbeat monitor
        # All subscription, decoding, and timeout detection happen in Rust threads
        # with zero GIL involvement. On timeout, it calls os._exit(1) via callback
        # or exits the process directly from Rust if no callback is provided.
        self._monitor = HeartbeatMonitor(
            topic=f"{robot_info.robot_name}/{config.heartbeat_topic}",
            timeout_seconds=config.timeout_seconds,
            enabled=config.enabled,
            on_timeout=self._handle_timeout if config.enabled else None,
        )

    def _handle_timeout(self, time_since_last: float, timeout_seconds: float) -> None:
        """Handle heartbeat timeout - called from Rust when timeout fires.

        This callback is only invoked once when timeout is detected.
        The GIL is only acquired at this moment, not during normal monitoring.

        Args:
            time_since_last: Seconds since last heartbeat.
            timeout_seconds: Configured timeout value.
        """
        logger.critical(
            f"HEARTBEAT TIMEOUT! No fresh heartbeat data received for {time_since_last:.2f}s "
            f"(timeout: {timeout_seconds}s). Low-level controller may have failed. "
            "Exiting program immediately for safety."
        )
        os._exit(1)

    def pause(self) -> None:
        """Pause heartbeat monitoring temporarily.

        When paused, the heartbeat monitor will not check for timeouts or exit
        the program. This is useful for scenarios where you need to temporarily
        disable safety monitoring (e.g., during system maintenance or testing).
        """
        self._monitor.pause()

    def resume(self) -> None:
        """Resume heartbeat monitoring after being paused."""
        self._monitor.resume()

    def is_paused(self) -> bool:
        """Check if heartbeat monitoring is currently paused.

        Returns:
            True if monitoring is paused, False if active or disabled.
        """
        return self._monitor.is_paused()

    def get_status(self) -> dict[str, bool | float | None]:
        """Gets the current heartbeat status information.

        Returns:
            Dictionary containing heartbeat metrics including:
                - is_active: Whether heartbeat signal is being received (bool)
                - last_value: Last received heartbeat value in seconds (float | None)
                - time_since_last: Time since last fresh data in seconds (float | None)
                - timeout_seconds: Configured timeout value (float)
                - enabled: Whether heartbeat monitoring is enabled (bool)
                - paused: Whether heartbeat monitoring is paused (bool)
        """
        rust_status = self._monitor.get_status()
        last_ms = rust_status.get("last_value_ms")
        last_value = (last_ms / 1000.0) if last_ms is not None else None

        return {
            "is_active": rust_status["is_active"],
            "last_value": last_value,
            "time_since_last": rust_status["time_since_last"],
            "timeout_seconds": rust_status["timeout_seconds"],
            "enabled": rust_status["enabled"],
            "paused": rust_status["paused"],
        }

    def is_active(self) -> bool:
        """Check if heartbeat signal is being received.

        Returns:
            True if heartbeat is active, False otherwise.
        """
        return self._monitor.is_active()

    def wait_for_active(self, timeout: float = 5.0) -> bool:
        """Wait for heartbeat signal to become active.

        If heartbeat monitoring is disabled, returns True immediately since
        there's nothing to wait for.

        Args:
            timeout: Maximum time to wait in seconds.

        Returns:
            True if heartbeat becomes active or is disabled, False if timeout is reached.
        """
        # If disabled, nothing to wait for
        if not self._enabled:
            return True

        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_active():
                return True
            time.sleep(0.1)
        return False

    @staticmethod
    def format_uptime(seconds: float) -> str:
        """Convert seconds to human-readable uptime format with high resolution.

        Args:
            seconds: Total seconds of uptime.

        Returns:
            Human-readable string like "1mo 2d 3h 45m 12s 345ms".
        """
        # Calculate months (assuming 30 days per month)
        months = int(seconds // (86400 * 30))
        remaining = seconds % (86400 * 30)

        # Calculate days
        days = int(remaining // 86400)
        remaining = remaining % 86400

        # Calculate hours
        hours = int(remaining // 3600)
        remaining = remaining % 3600

        # Calculate minutes
        minutes = int(remaining // 60)
        remaining = remaining % 60

        # Calculate seconds and milliseconds
        secs = int(remaining)
        milliseconds = int((remaining - secs) * 1000)

        parts = []
        if months > 0:
            parts.append(f"{months}mo")
        if days > 0:
            parts.append(f"{days}d")
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        if secs > 0:
            parts.append(f"{secs}s")
        if milliseconds > 0 or not parts:
            parts.append(f"{milliseconds}ms")

        return " ".join(parts)

    def shutdown(self) -> None:
        """Shuts down the heartbeat monitor and stops monitoring thread."""
        self._monitor.shutdown()

    def show(self) -> None:
        """Displays the current heartbeat status as a formatted table with color indicators."""
        status = self.get_status()

        table = Table(title="Heartbeat Monitor Status")
        table.add_column("Parameter", style="cyan")
        table.add_column("Value")

        # Mode: Enabled/Disabled and Paused state
        mode_parts = []
        if not status["enabled"]:
            mode_parts.append("[yellow]Exit Disabled[/]")
        if status["paused"]:
            mode_parts.append("[yellow]Paused[/]")
        if not mode_parts:
            mode_parts.append("[green]Active[/]")
        table.add_row("Mode", " | ".join(mode_parts))

        # Signal status
        active_style = "green" if status["is_active"] else "red"
        table.add_row(
            "Signal",
            f"[{active_style}]{'Receiving' if status['is_active'] else 'No Signal'}[/]",
        )

        # Server uptime
        if status["last_value"] is not None:
            uptime_str = self.format_uptime(status["last_value"])
            table.add_row("Server Uptime", f"[blue]{uptime_str}[/]")

        # Time since last heartbeat
        if status["time_since_last"] is not None:
            time_since = float(status["time_since_last"])
            timeout = status["timeout_seconds"]
            timeout = float(timeout) if timeout is not None else 1.0
            time_style = (
                "red"
                if time_since > timeout
                else "yellow"
                if time_since > timeout * 0.5
                else "green"
            )
            table.add_row("Last Heartbeat", f"[{time_style}]{time_since:.1f}s ago[/]")

        # Timeout setting
        table.add_row("Timeout", f"[blue]{status['timeout_seconds']}s[/]")

        Console().print(table)
