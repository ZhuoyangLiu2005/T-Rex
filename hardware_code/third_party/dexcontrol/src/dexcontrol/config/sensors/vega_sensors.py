# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Configuration dataclass for Vega robot sensors.

This module defines the VegaSensorsConfig dataclass which specifies the default
configurations for all sensors on the Vega robot, including cameras, IMUs,
LiDAR and ultrasonic sensors.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

from .cameras import RGBCameraConfig, ZedCameraConfig
from .imu import ChassisIMUConfig, ZedIMUConfig
from .lidar import RPLidarConfig
from .ultrasonic import UltraSonicConfig


def _make_rgb_camera(name: str) -> Callable[[], RGBCameraConfig]:
    """Helper function to create RGB camera config factory.

    Args:
        name: Camera instance name

    Returns:
        Factory function that creates an RGBCameraConfig
    """
    return (
        lambda: RGBCameraConfig(
            subscriber_config=dict(
                rgb=dict(
                    enable=True,
                    info_key=f"camera/base_{name}/info",  # Matches dexsensor: camera/base_{name}/info
                    topic=f"camera/base_{name}/rgb",
                )
            ),
            name=f"base_{name}_camera",
        )
    )


@dataclass
class VegaSensorsConfig:
    """Configuration for all sensors on the Vega robot.

    Contains default configurations for:
    - Head camera (Zed)
    - Base cameras (RGB) - left, right, front, back
    - IMUs - base and head (Zed)
    - LiDAR
    - Ultrasonic sensors
    """

    head_camera: ZedCameraConfig = field(default_factory=ZedCameraConfig)
    base_left_camera: RGBCameraConfig = field(default_factory=_make_rgb_camera("left"))
    base_right_camera: RGBCameraConfig = field(
        default_factory=_make_rgb_camera("right")
    )
    base_front_camera: RGBCameraConfig = field(
        default_factory=_make_rgb_camera("front")
    )
    base_back_camera: RGBCameraConfig = field(default_factory=_make_rgb_camera("back"))
    base_imu: ChassisIMUConfig = field(default_factory=ChassisIMUConfig)
    head_imu: ZedIMUConfig = field(default_factory=ZedIMUConfig)
    lidar: RPLidarConfig = field(default_factory=RPLidarConfig)
    ultrasonic: UltraSonicConfig = field(default_factory=UltraSonicConfig)

    def __post_init__(self) -> None:
        self.head_camera.enable = False
        self.base_left_camera.enable = False
        self.base_right_camera.enable = False
        self.base_front_camera.enable = False
        self.base_back_camera.enable = False
