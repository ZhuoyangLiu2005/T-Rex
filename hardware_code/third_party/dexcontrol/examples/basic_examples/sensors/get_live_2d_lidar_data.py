# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""This example demonstrates how to retrieve and visualize LIDAR data live.

The script initializes the robot with LIDAR enabled, continuously captures scan data,
and displays it in a live animated Cartesian plot.
"""

import time

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import tyro
from loguru import logger

from dexcontrol.core.config import get_robot_config
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


class LiveLidarPlotter:
    """Live LIDAR data plotter with animation."""

    def __init__(self, robot: Robot, update_interval: int = 100):
        """Initialize the live plotter.

        Args:
            robot: Robot instance with lidar sensor enabled.
            update_interval: Update interval in milliseconds for animation.
        """
        self.robot = robot
        self.update_interval = update_interval

        # Initialize plot
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.scatter = None
        self.robot_marker = None
        self.text_info = None

        # Statistics tracking
        self.scan_count = 0
        self.start_time = time.time()

        # Setup plot
        self._setup_plot()

    def _setup_plot(self):
        """Setup the initial plot configuration."""
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)
        self.ax.set_xlabel("X (meters)")
        self.ax.set_ylabel("Y (meters)")
        self.ax.set_title("Live LIDAR Scan Data")
        self.ax.grid(True, alpha=0.3)
        self.ax.set_aspect("equal")

        # Add robot marker at origin
        self.robot_marker = self.ax.scatter(
            0, 0, s=100, color="red", marker="o", label="Robot", zorder=5
        )

        # Add legend
        self.ax.legend(loc="upper right")

        # Add text for info display
        self.text_info = self.ax.text(
            0.02,
            0.98,
            "",
            transform=self.ax.transAxes,
            verticalalignment="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    def _get_info_text(
        self,
        ranges: np.ndarray,
        angles: np.ndarray,
        qualities: np.ndarray | None = None,
    ) -> str:
        """Generate information text for display."""
        runtime = time.time() - self.start_time
        scan_rate = self.scan_count / runtime if runtime > 0 else 0

        # Distance statistics
        close_points = np.sum(ranges < 1.0)
        medium_points = np.sum((ranges >= 1.0) & (ranges < 3.0))
        far_points = np.sum(ranges >= 3.0)

        # Obstacle detection
        obstacle_threshold = 0.5
        obstacles = ranges < obstacle_threshold
        obstacle_info = ""
        if np.any(obstacles):
            closest_distance = np.min(ranges[obstacles])
            closest_angle = angles[np.argmin(ranges)]
            obstacle_info = f"⚠️ Closest obstacle: {closest_distance:.2f}m at {np.degrees(closest_angle):.1f}°\n"

        info_text = (
            f"📊 Live LIDAR Monitor\n"
            f"⏱️ Runtime: {runtime:.1f}s | Scans: {self.scan_count} | Rate: {scan_rate:.1f} Hz\n"
            f"📍 Points: {len(ranges)} | Range: {np.min(ranges):.2f}-{np.max(ranges):.2f}m\n"
            f"🔄 Angular span: {np.degrees(np.max(angles) - np.min(angles)):.1f}°\n"
            f"📏 Close(<1m): {close_points} | Medium(1-3m): {medium_points} | Far(>3m): {far_points}\n"
            f"{obstacle_info}"
            f"{'✅ Quality data available' if qualities is not None else '❌ No quality data'}"
        )

        return info_text

    def _update_plot_limits(self, x: np.ndarray, y: np.ndarray):
        """Dynamically update plot limits based on data."""
        if len(x) > 0 and len(y) > 0:
            max_range = max(np.max(np.abs(x)), np.max(np.abs(y)))
            # Add some padding and ensure minimum range
            limit = max(max_range * 1.2, 2.0)
            self.ax.set_xlim(-limit, limit)
            self.ax.set_ylim(-limit, limit)

    def update_frame(self, frame):
        """Update function for animation."""
        try:
            # Get LIDAR scan data
            scan_data = self.robot.sensors.lidar.get_obs()
            if not scan_data:
                return self.scatter, self.text_info

            # Extract scan data
            ranges = np.asarray(scan_data.get("ranges", []), dtype=float).reshape(-1)
            angles = np.asarray(scan_data.get("angles", []), dtype=float).reshape(-1)
            qualities = scan_data.get("qualities")
            if qualities is not None:
                qualities = np.asarray(qualities).reshape(-1)

            # Guard: no data
            if ranges.size == 0 or angles.size == 0:
                return self.scatter, self.text_info

            # Align lengths if mismatched
            if ranges.size != angles.size:
                min_len = min(ranges.size, angles.size)
                ranges = ranges[:min_len]
                angles = angles[:min_len]
                if qualities is not None and qualities.size >= min_len:
                    qualities = qualities[:min_len]

            # Filter invalid and non-positive ranges
            valid_mask = np.isfinite(ranges) & np.isfinite(angles) & (ranges > 0)
            if not np.any(valid_mask):
                return self.scatter, self.text_info
            ranges = ranges[valid_mask]
            angles = angles[valid_mask]
            if qualities is not None and qualities.size == valid_mask.size:
                qualities = qualities[valid_mask]

            # Convert to Cartesian coordinates
            x = ranges * np.cos(angles)
            y = ranges * np.sin(angles)

            # Update scatter plot
            if self.scatter is not None:
                self.scatter.remove()

            # Color points based on distance for better visualization
            if ranges.size > 0:
                max_r = np.max(ranges)
                colors = plt.cm.viridis(ranges / max_r) if max_r > 0 else "blue"
            else:
                colors = "blue"
            self.scatter = self.ax.scatter(
                x, y, s=3, c=colors, alpha=0.7, cmap="viridis"
            )

            # Dynamically update plot limits based on current points
            self._update_plot_limits(x, y)

            # Update info text
            self.scan_count += 1
            info_text = self._get_info_text(ranges, angles, qualities)
            self.text_info.set_text(info_text)

            return self.scatter, self.text_info

        except Exception as e:
            logger.error(f"Error updating plot: {e}")
            return self.scatter, self.text_info

    def start_animation(self):
        """Start the live animation."""
        logger.info("Starting live LIDAR visualization...")

        # Wait for lidar to become active
        if not self.robot.sensors.lidar.wait_for_active(timeout=10.0):
            logger.error("LIDAR failed to become active within timeout")
            return

        logger.info("LIDAR is active! Starting live animation...")

        # Create and start animation
        ani = animation.FuncAnimation(
            self.fig,
            self.update_frame,
            interval=self.update_interval,
            blit=False,
            cache_frame_data=False,
        )

        plt.show()
        return ani


def main(update_rate: float = 10.0) -> None:
    """Initializes the robot and displays live animated LIDAR data.

    Args:
        update_rate: Update frequency in Hz for the animation.
    """
    configs = get_robot_config()
    configs.enable_sensor("lidar")
    robot = Robot(configs=configs)

    try:
        # Calculate update interval in milliseconds
        update_interval = int(1000 / update_rate)

        # Create and start live plotter
        plotter = LiveLidarPlotter(robot, update_interval)
        _ = plotter.start_animation()

    except KeyboardInterrupt:
        logger.info("Live visualization stopped by user")
    except Exception as e:
        logger.error(f"Error during live visualization: {e}")
    finally:
        logger.info("Shutting down robot...")
        robot.shutdown()
        logger.info("Robot shutdown complete")


if __name__ == "__main__":
    tyro.cli(main)
