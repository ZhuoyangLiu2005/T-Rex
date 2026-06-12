# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Example script to get/set force torque sensor mode for arm joints.

This script demonstrates how to check or configure the 6-axis force torque
sensor mode on the robot arm. If you uninstall the force torque sensor and
later need to re-enable it, you can use this script to do so.
"""

from typing import Literal

import tyro
from loguru import logger
from rich.console import Console
from rich.table import Table

from dexcontrol.robot import Robot

console = Console()


def _interpret_ft_status(modes: list[int]) -> tuple[str, str]:
    """Interpret force torque sensor modes into a human-readable status.

    Returns:
        Tuple of (status_text, rich_color).
    """
    if all(m == 0 for m in modes):
        return "Enabled", "green"
    if all(m == 1 for m in modes):
        return "Disabled", "yellow"
    return "Error", "red"


def get(side: Literal["left", "right", "both"] = "both") -> None:
    """Get current force torque sensor status.

    Args:
        side: Which arm to query ('left', 'right', or 'both').
    """
    with Robot() as bot:
        table = Table()
        table.add_column("Arm", style="cyan")
        table.add_column("Force Torque Sensor Status")

        has_error = False
        arms: list[tuple[str, dict]] = []

        if side in ("left", "both"):
            arms.append(("Left", bot.left_arm.get_force_torque_sensor_mode()))
        if side in ("right", "both"):
            arms.append(("Right", bot.right_arm.get_force_torque_sensor_mode()))

        for arm_name, result in arms:
            if not result.get("success"):
                msg = result.get("message", "Unknown error")
                table.add_row(arm_name, f"[red]Query Failed: {msg}[/red]")
                has_error = True
                continue
            status, color = _interpret_ft_status(result["modes"])
            table.add_row(arm_name, f"[{color}]{status}[/{color}]")
            if status == "Error":
                has_error = True

        console.print(table)

        if has_error:
            console.print(
                "\n[red]Warning:[/red] Force torque sensor has inconsistent "
                "joint modes. Please contact Dexmate support."
            )


def set(
    side: Literal["left", "right", "both"] = "left",
    enable: bool = True,
) -> None:
    """Enable or disable force torque sensor mode.

    Args:
        side: Which arm to configure ('left', 'right', or 'both').
        enable: True to enable force torque sensor (joint mode 0),
            False to disable (joint mode 1).
    """
    with Robot() as bot:
        if side in ("left", "both"):
            action = "Enabling" if enable else "Disabling"
            logger.info(f"{action} force torque sensor for left arm...")
            result = bot.left_arm.activate_force_torque_sensor(enable)
            logger.info(f"Left arm result: {result}")

        if side in ("right", "both"):
            action = "Enabling" if enable else "Disabling"
            logger.info(f"{action} force torque sensor for right arm...")
            result = bot.right_arm.activate_force_torque_sensor(enable)
            logger.info(f"Right arm result: {result}")


if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict({"get": get, "set": set})
