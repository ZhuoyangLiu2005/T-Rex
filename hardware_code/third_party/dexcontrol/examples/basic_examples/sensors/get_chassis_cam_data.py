# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""This example demonstrates how to retrieve and display chassis camera data in real-time.

The script initializes the robot with all chassis cameras enabled, captures images
from each camera continuously, and displays them in a single tiled window that updates live.

Note: This example uses WebRTC (RTC) mode by default. To use standard Zenoh subscription,
use the --no-use-rtc flag.

Press Ctrl+C to exit the live feed.
"""

import math
import time

# Set matplotlib backend before importing pyplot
import matplotlib

try:
    # Try TkAgg backend first (more reliable for live display)
    matplotlib.use("TkAgg")
except ImportError:
    try:
        # Fallback to Qt5Agg
        matplotlib.use("Qt5Agg")
    except ImportError:
        pass  # Use default

import matplotlib.pyplot as plt
import numpy as np
import tyro
from dexbot_utils.configs.components.sensors.cameras import CameraConfig
from loguru import logger

from dexcontrol.core.config import get_robot_config
from dexcontrol.robot import Robot
from dexcontrol.utils.compat import supported_models


class LiveCameraDisplay:
    """Handles live display of multiple camera feeds in a tiled layout."""

    def __init__(self, camera_definitions: dict):
        """Initialize the live camera display.

        Args:
            camera_definitions: Dictionary mapping camera names to display names
        """
        self.camera_definitions = camera_definitions
        self.num_cameras = len(camera_definitions)

        # Calculate grid layout
        self.cols = math.ceil(math.sqrt(self.num_cameras))
        self.rows = math.ceil(self.num_cameras / self.cols)

        # Setup matplotlib figure and axes
        plt.ion()  # Turn on interactive mode
        self.fig, self.axes = plt.subplots(
            self.rows, self.cols, figsize=(15, 10), squeeze=False
        )
        self.fig.suptitle(
            "Live Chassis Camera Feeds (Press Ctrl+C to exit)", fontsize=16
        )

        # Initialize image plots
        self.image_plots = {}
        for i, (camera_name, display_name) in enumerate(camera_definitions.items()):
            ax = self.axes[i // self.cols, i % self.cols]
            ax.set_title(display_name)
            ax.axis("off")
            # Create empty image plot
            self.image_plots[camera_name] = ax.imshow(
                np.zeros((480, 640, 3), dtype=np.uint8)
            )

        # Hide unused subplots
        for i in range(self.num_cameras, self.rows * self.cols):
            self.axes[i // self.cols, i % self.cols].axis("off")

        plt.tight_layout()
        plt.show(block=False)

    def update_images(self, robot) -> None:
        """Update all camera images in the display.

        Args:
            robot: The robot instance to get camera data from
        """
        updated = False
        for camera_name, display_name in self.camera_definitions.items():
            try:
                camera_sensor = getattr(robot.sensors, camera_name, None)
                if camera_sensor and camera_sensor.is_active():
                    image = camera_sensor.get_obs()
                    if image is not None:
                        self.image_plots[camera_name].set_array(image)
                        updated = True
                    else:
                        logger.debug(f"No image data from '{display_name}'.")
                else:
                    logger.debug(
                        f"Camera sensor '{display_name}' is not available or not active."
                    )
            except Exception as e:
                logger.error(
                    f"Error getting data from '{display_name}': {e}",
                    exc_info=True,
                )

        if updated:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def close(self):
        """Close the display window."""
        plt.ioff()
        plt.close(self.fig)


@supported_models("vega_1", "vega_1p")
def main(use_rtc: bool = False, fps: float = 10.0) -> None:
    """Initializes the robot, retrieves chassis camera data, and displays it in real-time.

    Args:
        use_rtc: If True, use WebRTC for camera streams (default: False)
        fps: Display refresh rate in Hz (default: 10.0)
    """
    configs = get_robot_config()

    # Define chassis cameras to be used in this example.
    camera_definitions = {
        "base_left_camera": "Left Camera",
        "base_right_camera": "Right Camera",
        "base_front_camera": "Front Camera",
        "base_back_camera": "Back Camera",
    }

    # Add cameras to config if they don't exist
    for camera_name, display_name in camera_definitions.items():
        if camera_name not in configs.sensors:
            logger.info(f"Adding '{camera_name}' to configuration...")
            transport = "rtc" if use_rtc else "zenoh"
            configs.sensors[camera_name] = CameraConfig(
                name=camera_name,
                enabled=True,
                transport=transport,
                topic=f"sensors/{camera_name}/rgb",
                rtc_channel=f"sensors/{camera_name}/rgb_rtc",
            )
        else:
            # Enable existing camera
            configs.sensors[camera_name].enabled = True
            if use_rtc:
                configs.sensors[camera_name].transport = "rtc"
            logger.info(f"Enabled '{camera_name}' (RTC: {use_rtc}).")

    if use_rtc:
        logger.info("Using WebRTC mode for camera streams.")
    else:
        logger.info("Using standard Zenoh subscription for camera streams.")

    with Robot(configs=configs) as robot:
        # Wait for cameras to become active
        logger.info("Waiting for camera streams to become active...")
        active_cameras = []
        for camera_name in camera_definitions:
            camera_sensor = getattr(robot.sensors, camera_name, None)
            if camera_sensor:
                if camera_sensor.wait_for_active(timeout=5.0):
                    active_cameras.append(camera_name)
                    logger.success(f"✓ {camera_definitions[camera_name]} is active")
                    # Print camera info if available
                    if (
                        hasattr(camera_sensor, "camera_info")
                        and camera_sensor.camera_info
                    ):
                        info = camera_sensor.camera_info
                        logger.info(f"  Camera: {info.get('name', 'Unknown')}")
                        if "rtc" in info and use_rtc:
                            rtc_info = info["rtc"].get("streams", {}).get("rgb", {})
                            if "signaling_url" in rtc_info:
                                logger.info(f"  RTC URL: {rtc_info['signaling_url']}")
                else:
                    logger.warning(f"✗ {camera_definitions[camera_name]} is not active")

        if not active_cameras:
            logger.error("No cameras are active. Exiting...")
            return

        logger.info(f"Active cameras: {len(active_cameras)}/{len(camera_definitions)}")

        # Initialize live display
        display = LiveCameraDisplay(camera_definitions)

        logger.info(f"Starting live camera feed at {fps} FPS. Press Ctrl+C to exit...")
        frame_delay = 1.0 / fps

        try:
            while True:
                display.update_images(robot)
                time.sleep(frame_delay)

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received. Stopping live feed...")

        except Exception as e:
            logger.error(f"Unexpected error in live feed: {e}", exc_info=True)

        finally:
            logger.info("Shutting down...")
            display.close()
            logger.info("Shutdown complete.")


if __name__ == "__main__":
    tyro.cli(main)
