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
class RGBCameraConfig:
    """Configuration for the RGBCameraSensor.

    Attributes:
        _target_: The target class to instantiate.
        name: A unique name for the sensor instance.
        enable: Whether the sensor is enabled.
        use_rtc: If True, uses the RTCSubscriber. If False,
                 uses the standard RGBCameraSubscriber with a Zenoh topic.
        enable_fps_tracking: If True, tracks and logs the FPS of the stream.
        fps_log_interval: The interval in seconds for logging FPS.
        subscriber_config: A dictionary containing the configuration for the
                           underlying subscriber (either RTC or standard).
    """

    _target_: str = "dexcontrol.sensors.camera.rgb_camera.RGBCameraSensor"
    name: str = "rgb_camera"
    enable: bool = False
    use_rtc: bool = True
    enable_fps_tracking: bool = False
    fps_log_interval: int = 30

    subscriber_config: dict = field(
        default_factory=lambda: {
            "rgb": {
                "enable": True,
                "info_key": "camera/rgb/info",
                "topic": "camera/rgb",
            }
        }
    )
