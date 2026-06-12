# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Example script to display force/torque sensor readings.

This script creates a real-time display of force and torque sensor readings
from a robot arm using Rich console formatting.
"""

import time

import numpy as np
import tyro
from rich.console import Console
from rich.live import Live
from rich.table import Table

from dexcontrol.robot import Robot


def create_wrench_table(bot, arm_side: str) -> Table:
    """Create a table with force/torque sensor values.

    Args:
        bot: Robot instance containing arm sensors.
        arm_side: Side of the arm to read from ('left' or 'right').

    Returns:
        Rich Table object containing formatted sensor data.
    """
    arm = bot.left_arm if arm_side == "left" else bot.right_arm

    table = Table(title=f"{arm_side.upper()} ARM FORCE/TORQUE SENSOR")
    table.add_column("Component", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")
    table.add_column("Unit", style="green")

    if arm.wrench_sensor is None:
        table.add_row("No sensor data", "N/A", "N/A")
        return table

    wrench = arm.wrench_sensor.get_wrench_state()
    components = ["fx", "fy", "fz", "mx", "my", "mz"]
    units = ["N"] * 3 + ["Nm"] * 3

    for val, comp, unit in zip(wrench, components, units):
        table.add_row(comp, f"{val:.4f}", unit)

    return table


def main(arm_side: str = "left") -> None:
    """Display force/torque sensor information in real-time.

    Args:
        arm_side: Side of the arm to monitor ('left' or 'right').

    Raises:
        KeyboardInterrupt: Gracefully handles user interruption and shuts down robot.
    """
    bot = Robot()
    np.set_printoptions(precision=3)
    console = Console()

    try:
        with Live(console=console, refresh_per_second=20) as live:
            while True:
                table = create_wrench_table(bot, arm_side)
                live.update(table)
                time.sleep(0.05)
    except KeyboardInterrupt:
        bot.shutdown()


if __name__ == "__main__":
    tyro.cli(main)
