# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""This example demonstrates how to retrieve and visualize LIDAR data.

The script initializes the robot with LIDAR enabled, captures scan data,
and displays it in a simple Cartesian plot.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tyro
from dexcontrol.config.vega import get_vega_config
from loguru import logger

from dexcontrol.robot import Robot

try:
    # Try TkAgg backend first (more reliable for live display)
    matplotlib.use("TkAgg")
    print("Using TkAgg backend for display")
except ImportError:
    try:
        # Fallback to Qt5Agg
        matplotlib.use("Qt5Agg")
        print("Using Qt5Agg backend for display")
    except ImportError:
        # Last resort - use default
        print("Using default matplotlib backend")


def plot_lidar_scan(ranges: np.ndarray, angles: np.ndarray) -> None:
    """Plot LIDAR scan data in Cartesian coordinates.

    Args:
        ranges: Array of range measurements in meters.
        angles: Array of angle measurements in radians.
    """
    x = ranges * np.cos(angles)
    y = ranges * np.sin(angles)

    # Create plot
    plt.figure(figsize=(10, 8))
    plt.scatter(x, y, s=2, alpha=0.7, color="blue")
    plt.scatter(0, 0, s=50, color="red", marker="o", label="Robot")

    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.title("LIDAR Scan Data")
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.legend()
    plt.show()


def print_scan_info(
    ranges: np.ndarray, angles: np.ndarray, qualities: np.ndarray | None = None
) -> None:
    """Print basic information about the LIDAR scan.

    Args:
        ranges: Array of range measurements in meters.
        angles: Array of angle measurements in radians.
        qualities: Array of quality values, if available.
    """
    print("\nLIDAR Scan Information:")
    print(f"  Points: {len(ranges)}")
    print(f"  Range: {np.min(ranges):.2f} - {np.max(ranges):.2f} meters")
    print(f"  Angular span: {np.degrees(np.max(angles) - np.min(angles)):.1f}°")
    if qualities is not None:
        print(f"  Quality data: Available (avg: {np.mean(qualities):.1f})")
    else:
        print("  Quality data: Not available")


def main() -> None:
    """Initializes the robot, retrieves LIDAR data, and displays it."""
    configs = get_vega_config()
    configs.enable_sensor("lidar")
    robot = Robot(configs=configs)

    try:
        # Get LIDAR scan data
        scan_data = robot.sensors.lidar.get_obs()
        if not scan_data:
            logger.error("Failed to retrieve LIDAR data.")
            return

        # Extract scan data
        ranges = scan_data["ranges"]
        angles = scan_data["angles"]
        qualities = scan_data.get("qualities")

        logger.info("LIDAR data retrieved successfully!")

        # Print scan information
        print_scan_info(ranges, angles, qualities)

        # Plot the scan data
        plot_lidar_scan(ranges, angles)

    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        robot.shutdown()


if __name__ == "__main__":
    tyro.cli(main)
