# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Example script to show ZED X One wrist camera data live.

This script demonstrates how to retrieve wrist camera data (RGB images)
from the robot's ZED X One wrist cameras and display them live using matplotlib
animation. It showcases the simple API for getting camera data and provides
live visualization for both left and right wrist cameras.

Note: This example assumes ZED X One wrist cameras. The wrist cameras only
provide RGB streams (single image stream per camera), unlike the head camera
which provides RGB and depth streams.
"""

import os
from typing import Any

# Fix Qt plugin issues when running inside headless containers.
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = ""

import matplotlib

try:
    matplotlib.use("TkAgg")
    print("Using TkAgg backend for display")
except ImportError:
    try:
        matplotlib.use("Qt5Agg")
        print("Using Qt5Agg backend for display")
    except ImportError:
        print("Using default matplotlib backend")

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import tyro

from dexcontrol.core.config import get_robot_config
from dexcontrol.robot import Robot

# Sensor config key -> (stream key for data retrieval, display title)
CAMERA_STREAMS: list[tuple[str, str, str, list[str]]] = [
    ("left_rgb", "Left Wrist ZED X One RGB", "left_wrist_camera", ["rgb"]),
    ("right_rgb", "Right Wrist ZED X One RGB", "right_wrist_camera", ["rgb"]),
]


def get_camera_data(robot: Robot) -> dict[str, Any]:
    """Get ZED X One wrist camera data from robot sensors."""
    results = {}

    # Left wrist camera
    if robot.has_sensor("left_wrist_camera"):
        try:
            obs = robot.sensors.left_wrist_camera.get_obs(include_timestamp=True)
            results["left_rgb"] = obs
        except Exception as e:
            print(f"Failed to get left wrist camera: {e}")
            results["left_rgb"] = None

    # Right wrist camera
    if robot.has_sensor("right_wrist_camera"):
        try:
            obs = robot.sensors.right_wrist_camera.get_obs(include_timestamp=True)
            results["right_rgb"] = obs
        except Exception as e:
            print(f"Failed to get right wrist camera: {e}")
            results["right_rgb"] = None

    return results


def print_camera_info(camera_info, camera_name):
    """Nicely format and print any nested dictionary."""

    def print_dict(d, indent=0):
        """Recursively print dictionary with proper indentation."""
        for key, value in d.items():
            # Create indentation
            spaces = "  " * indent

            if isinstance(value, dict):
                print(f"{spaces}{key}:")
                print_dict(value, indent + 1)
            elif isinstance(value, (list, tuple)):
                print(f"{spaces}{key}: {value}")
            else:
                print(f"{spaces}{key}: {value}")

    if not camera_info:
        print(f"No {camera_name} camera information available")
        return

    print("\n" + "=" * 50)
    print(f"{camera_name.upper()} CAMERA INFORMATION")
    print("=" * 50)
    print_dict(camera_info)
    print("=" * 50)
    print()


def visualize_camera_data(robot: Robot, fps: float = 30.0) -> None:
    """Visualize ZED X One wrist camera data using matplotlib."""
    available_streams = [
        (stream_key, title)
        for stream_key, title, attr_name, _ in CAMERA_STREAMS
        if getattr(robot.sensors, attr_name, None) is not None
    ]

    if not available_streams:
        print("No wrist cameras available for visualization!")
        return

    fig, axes_raw = plt.subplots(
        1, len(available_streams), figsize=(7 * len(available_streams), 5)
    )
    axes = [axes_raw] if len(available_streams) == 1 else list(axes_raw)

    fig.suptitle("Live ZED X One Wrist Camera Feeds", fontsize=16)

    displays = [ax.imshow(np.zeros((480, 640, 3), dtype=np.uint8)) for ax in axes]
    for ax, (_stream_key, title) in zip(axes, available_streams):
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()

    def update(_frame: int) -> None:
        camera_data = get_camera_data(robot)

        for axis, (stream_key, title), display in zip(
            axes, available_streams, displays
        ):
            data = camera_data.get(stream_key)
            if data is None:
                continue

            # Extract image and timestamp
            if isinstance(data, dict):
                frame = data.get("data")
                timestamp_ns = data.get("timestamp")
                ts_ms = (timestamp_ns / 1e6) if timestamp_ns else None
            else:
                frame = data
                ts_ms = None

            if frame is None:
                continue

            # Update title with shape and timestamp
            new_title = f"{title}\n{frame.shape}"
            if ts_ms is not None:
                new_title += f"\nt={ts_ms:.1f}ms"

            axis.set_title(new_title)
            display.set_array(frame)

    # Start animation
    _ = animation.FuncAnimation(
        fig,
        update,
        interval=int(1000 / max(fps, 1e-3)),
        blit=False,
        cache_frame_data=False,
    )
    plt.show()


def _wait_for_camera(camera: Any, name: str, timeout: float = 5.0) -> None:
    """Wait for a camera sensor to become active before streaming."""
    if camera is None or not hasattr(camera, "wait_for_active"):
        return

    if camera.wait_for_active(timeout=timeout):
        print(f"{name} stream active")
    else:
        print(f"Warning: {name} stream not active within {timeout}s")


def main(fps: float = 30.0, left: bool = True, right: bool = True) -> None:
    """Main function to initialize robot and display ZED X One wrist camera feeds.

    Args:
        fps: Display refresh rate in Hz (default: 30.0)
        left: Enable left wrist camera (default: True)
        right: Enable right wrist camera (default: True)
    """
    if not (left or right):
        print("At least one wrist camera must be enabled!")
        return

    configs = get_robot_config()

    # Enable requested wrist cameras (skip if not available on this robot)
    if left and not configs.has_sensor("left_wrist_camera"):
        print("Left wrist camera is not available on this robot configuration")
        left = False
    if right and not configs.has_sensor("right_wrist_camera"):
        print("Right wrist camera is not available on this robot configuration")
        right = False

    if not (left or right):
        print("No wrist cameras available on this robot configuration!")
        return

    if left:
        configs.enable_sensor("left_wrist_camera")
    if right:
        configs.enable_sensor("right_wrist_camera")

    with Robot(configs=configs) as robot:
        # Get camera references
        left_camera = (
            getattr(robot.sensors, "left_wrist_camera", None) if left else None
        )
        right_camera = (
            getattr(robot.sensors, "right_wrist_camera", None) if right else None
        )

        if left_camera is None and left:
            print("Left wrist camera is not available on this robot configuration")
        if right_camera is None and right:
            print("Right wrist camera is not available on this robot configuration")

        # Wait for cameras to become active
        print("Waiting for camera streams to become active...")
        if left_camera is not None:
            _wait_for_camera(left_camera, "Left wrist")
            if hasattr(left_camera, "camera_info"):
                print_camera_info(left_camera.camera_info, "Left Wrist ZED X One")

        if right_camera is not None:
            _wait_for_camera(right_camera, "Right wrist")
            if hasattr(right_camera, "camera_info"):
                print_camera_info(right_camera.camera_info, "Right Wrist ZED X One")

        # Start live camera visualization
        visualize_camera_data(robot, fps)


if __name__ == "__main__":
    tyro.cli(main)
