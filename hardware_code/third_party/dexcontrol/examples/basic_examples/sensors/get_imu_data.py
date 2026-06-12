# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Example script to read chassis and head IMU data live in a nice table.

This script demonstrates how to retrieve inertial measurement data from the robot's
chassis and head IMU sensors and display them live in a formatted table. It continuously
reads IMU data including acceleration, angular velocity, magnetometer, and orientation,
and displays the information in real-time tables.
"""

import time

import numpy as np
import tyro
from dexcomm import RateLimiter
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from dexcontrol.core.config import get_robot_config
from dexcontrol.robot import Robot


def get_imu_data(robot):
    """Simple function to get IMU data from robot sensors.

    This demonstrates how easy it is to get sensor data using our API.

    Args:
        robot: Robot instance

    Returns:
        tuple: (head_imu_data, chassis_imu_data)
    """
    head_imu_data = None
    chassis_imu_data = None
    if robot.has_sensor("head_imu"):
        head_imu_data = robot.sensors.head_imu.get_obs(
            obs_keys=["acc", "gyro", "quat", "mag"]
        )
    if robot.has_sensor("chassis_imu"):
        chassis_imu_data = robot.sensors.chassis_imu.get_obs(
            obs_keys=["acc", "gyro", "quat", "mag"]
        )
    return head_imu_data, chassis_imu_data


def create_imu_table(imu_data, title):
    """Create a formatted table for IMU data."""
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("Parameter", style="cyan", no_wrap=True)
    table.add_column("X", style="green")
    table.add_column("Y", style="green")
    table.add_column("Z", style="green")
    table.add_column("W", style="green")

    if imu_data is not None:
        # Define display names and units for each parameter
        param_info = {
            "acc": ("Acceleration (m/s²)", 3),
            "gyro": ("Angular Vel (rad/s)", 3),
            "quat": ("Orientation (quat)", 4),
            "mag": ("Magnetometer (µT)", 3),
            "ang_vel": ("Angular Vel (rad/s)", 3),  # Backward compat
        }

        for key, value in imu_data.items():
            if key == "timestamp_ns":
                # Display timestamp in seconds
                table.add_row("Timestamp (s)", f"{value / 1e9:.3f}", "-", "-", "-")
            elif key == "sequence":
                # Display sequence number
                table.add_row("Sequence", str(value), "-", "-", "-")
            elif isinstance(value, (np.ndarray, list)):
                # Handle both numpy arrays and lists
                display_name = param_info.get(key, (key, len(value)))[0]
                if len(value) == 3:
                    table.add_row(
                        display_name,
                        f"{value[0]:.4f}",
                        f"{value[1]:.4f}",
                        f"{value[2]:.4f}",
                        "-",
                    )
                elif len(value) == 4:
                    # Quaternion [w, x, y, z] format
                    table.add_row(
                        display_name,
                        f"{value[1]:.4f}",  # x
                        f"{value[2]:.4f}",  # y
                        f"{value[3]:.4f}",  # z
                        f"{value[0]:.4f}",  # w
                    )
                else:
                    table.add_row(display_name, str(value), "-", "-", "-")
            else:
                table.add_row(key, str(value), "-", "-", "-")
    else:
        table.add_row("No data", "N/A", "N/A", "N/A", "N/A")

    return table


def display_live_imu_data(robot, fps: float = 100.0):
    """Handle all visualization and live display of IMU data.

    Args:
        robot: Robot instance
        fps: Display update rate in Hz (default: 100.0)
    """
    console = Console()
    console.print("[bold green]Starting live IMU data display...[/bold green]")
    console.print(f"[yellow]Display rate: {fps} Hz | Press Ctrl+C to exit[/yellow]")

    update_count = 0
    start_time = time.time()

    # Create initial display
    initial_layout = Layout()
    initial_layout.split_column(
        Layout(Panel("Initializing...", title="Status"), size=3),
        Layout(create_imu_table(None, "[bold cyan]Head IMU Data[/bold cyan]")),
        Layout(create_imu_table(None, "[bold yellow]Chassis IMU Data[/bold yellow]")),
    )

    rate_limiter = RateLimiter(fps)

    try:
        with Live(initial_layout, console=console, refresh_per_second=fps) as live:
            while True:
                current_time = time.time()
                update_count += 1
                actual_fps = (
                    update_count / (current_time - start_time)
                    if current_time > start_time
                    else 0
                )

                # Get IMU data using our simple API
                head_imu_data, chassis_imu_data = get_imu_data(robot)

                # Create updated display
                status_panel = Panel(
                    f"Updates: {update_count} | Target: {fps:.1f} Hz | Actual: {actual_fps:.1f} Hz | Press Ctrl+C to exit",
                    title="Status",
                    style="bold blue",
                )

                layout = Layout()
                layout.split_column(
                    Layout(status_panel, size=3),
                    Layout(
                        create_imu_table(
                            head_imu_data, "[bold cyan]Head IMU Data[/bold cyan]"
                        )
                    ),
                    Layout(
                        create_imu_table(
                            chassis_imu_data,
                            "[bold yellow]Chassis IMU Data[/bold yellow]",
                        )
                    ),
                )

                live.update(layout)
                rate_limiter.sleep()

    except KeyboardInterrupt:
        console.print("\n[bold red]Stopping IMU data display...[/bold red]")


def main(fps: float = 100.0):
    """Main function to initialize robot and start live IMU data display.

    Args:
        fps: Display update rate in Hz (default: 100.0)
    """
    # Initialize robot with available IMU sensors enabled
    configs = get_robot_config()
    has_head_imu = configs.has_sensor("head_imu")
    has_chassis_imu = configs.has_sensor("chassis_imu")

    if not has_head_imu and not has_chassis_imu:
        print("No IMU sensors available on this robot configuration.")
        return

    if has_head_imu:
        configs.enable_sensor("head_imu")
    if has_chassis_imu:
        configs.enable_sensor("chassis_imu")

    with Robot(configs=configs) as robot:
        print("Waiting for IMU sensors to become active...")

        if has_head_imu:
            if robot.sensors.head_imu.wait_for_active(timeout=5.0):
                if robot.sensors.head_imu.has_mag():
                    print("Head IMU has magnetometer data available")
            else:
                print("Warning: Head IMU not active")

        if has_chassis_imu:
            if robot.sensors.chassis_imu.wait_for_active(timeout=5.0):
                if robot.sensors.chassis_imu.has_mag():
                    print("Chassis IMU has magnetometer data available")
            else:
                print("Warning: Chassis IMU not active")

        display_live_imu_data(robot, fps)


if __name__ == "__main__":
    tyro.cli(main)
