# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Utility functions for displaying information in a Rich table format."""

from dexcomm.codecs import ConnectionStatusEnum, OperationalStatusEnum
from loguru import logger
from rich.console import Console
from rich.table import Table

from dexcontrol.utils.pb_utils import TYPE_SOFTWARE_VERSION, ComponentStatus


def show_software_version(version_info: dict[str, TYPE_SOFTWARE_VERSION]):
    """Create a Rich table for displaying firmware version information.

    Args:
        version_info: Dictionary containing version info for each component.
    """
    table = Table(title="Firmware Version")
    table.add_column("Component", style="cyan")
    table.add_column("Hardware Version")
    table.add_column("Software Version")
    table.add_column("Main Hash")
    table.add_column("Compile Time")

    for component, version in sorted(version_info.items()):
        table.add_row(
            component,
            str(version["hardware_version"]),
            str(version["software_version"]),
            str(version["main_hash"]),
            str(version["compile_time"]),
        )

    console = Console()
    console.print(table)


def show_component_status(status_info: dict[str, dict]):
    """Create a Rich table for displaying component status information.

    Args:
        status_info: Dictionary containing status info for each component.
                    Expected structure: {'states': {'component_name': {'connection': int, 'operation': int, 'error': {...}}}}
                    or old structure: {'component_name': {'connected': bool, 'enabled': bool, 'error_state': int, 'error_code': int}}
    """
    if "states" not in status_info:
        return
    # Extract states from the new structure, fallback to old structure
    states = status_info.get("states", status_info)

    table = Table(title="Component Status")
    table.add_column("Component", style="cyan")
    table.add_column("Connected", justify="center")
    table.add_column("Enabled", justify="center")
    table.add_column("Error", justify="left")

    status_icons = {
        True: "[green]:white_check_mark:[/green]",
        False: "[red]:x:[/red]",
        ComponentStatus.NORMAL: "[green]:white_check_mark:[/green]",
        ComponentStatus.NA: "[dim]N/A[/dim]",
    }
    # Sort components by name to ensure consistent order
    for component in sorted(states.keys()):
        status = states[component]

        connected = status["connection"] == ConnectionStatusEnum.CONNECTED
        operation = status["operation"]
        error_info = status.get("error", {})
        error_message = error_info.get("error_message", "")

        # Determine enabled status based on operation
        if operation == OperationalStatusEnum.ENABLED:
            enabled = True
        elif operation == OperationalStatusEnum.DISABLED:
            enabled = False
        elif operation == OperationalStatusEnum.NOT_AVAILABLE:
            enabled = ComponentStatus.NA
        elif operation == OperationalStatusEnum.CALIBRATING:
            enabled = "[yellow]CALIBRATING[/yellow]"
        elif operation == OperationalStatusEnum.ERROR:
            enabled = False
        else:
            enabled = False

        # Determine if we should show error details
        has_error = operation == OperationalStatusEnum.ERROR or error_message != ""

        # Format connection status
        connected_icon = status_icons.get(connected, "[red]:x:[/red]")

        # Format enabled status
        if isinstance(enabled, str):
            enabled_icon = enabled  # Already formatted (e.g., CALIBRATING)
        else:
            enabled_icon = status_icons.get(enabled, "[red]:x:[/red]")

        # Format error status
        if not has_error:
            error = "[green]:white_check_mark:[/green]"
        elif (
            "connection" in status and operation == OperationalStatusEnum.NOT_AVAILABLE
        ):
            error = "[dim]N/A[/dim]"
        elif not connected:
            error = "[dim]N/A[/dim]"
        else:
            error = f"[red]{error_message}[/red]"

        table.add_row(
            component,
            connected_icon,
            enabled_icon,
            error,
        )

    console = Console()
    console.print(table)


def show_ntp_stats(stats: dict[str, float]):
    """Display NTP statistics in a Rich table format.

    Args:
        stats: Dictionary containing NTP statistics (e.g., mean_offset, mean_rtt, etc.).
    """
    table = Table()
    table.add_column("Time Statistic", style="cyan")
    table.add_column("Value (Unit: second)", justify="right")

    for key, value in stats.items():
        # Format floats to 6 decimal places, lists as comma-separated, others as str
        if isinstance(value, float):
            value_str = f"{value:.6f}"
        elif isinstance(value, list):
            value_str = ", ".join(
                f"{v:.6f}" if isinstance(v, float) else str(v) for v in value
            )
        else:
            value_str = str(value)
        table.add_row(key, value_str)

    console = Console()
    console.print(table)

    if "offset (mean)" in stats:
        offset = stats["offset (mean)"]
        if offset > 0:
            logger.info(
                f"To synchronize: server_time ≈ local_time + {offset:.3f} second"
            )
        else:
            logger.info(
                f"To synchronize: server_time ≈ local_time - {abs(offset):.3f} second"
            )
