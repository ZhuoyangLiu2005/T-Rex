# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

from dataclasses import dataclass, field


@dataclass
class ZedCameraConfig:
    """Configuration for the ZedCameraSensor.

    Attributes:
        _target_: The target class to instantiate.
        name: A unique name for the sensor instance.
        enable: Whether the sensor is enabled.
        use_rtc: If True, RGB streams use the RTCSubscriber.
                 If False, they use the standard RGBCameraSubscriber. Depth
                 always uses a standard Zenoh subscriber.
        enable_fps_tracking: If True, tracks and logs the FPS of all streams.
        fps_log_interval: The interval in seconds for logging FPS.
        subscriber_config: A dictionary containing the configurations for each
                           individual stream (left_rgb, right_rgb, depth).
    """

    _target_: str = "dexcontrol.sensors.camera.zed_camera.ZedCameraSensor"
    name: str = "zed_camera"
    enable: bool = False
    use_rtc: bool = False
    enable_fps_tracking: bool = False
    fps_log_interval: int = 30

    subscriber_config: dict = field(
        default_factory=lambda: {
            "left_rgb": {
                "enable": True,
                "info_key": "camera/head/info",  # Query the main camera info endpoint
                "topic": "camera/head/left_rgb",
            },
            "right_rgb": {
                "enable": True,
                "info_key": "camera/head/info",  # Query the main camera info endpoint
                "topic": "camera/head/right_rgb",
            },
            "depth": {
                "enable": True,
                "topic": "camera/head/depth",
            },
        }
    )
