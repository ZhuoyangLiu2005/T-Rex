# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Production-ready camera sensors with Zenoh and RTC transport support.

This module provides high-performance camera sensor implementations that
receive data from dexsensor using either Zenoh (JPEG/PNG compressed, reliable)
or RTC (real-time video, hardware accelerated) transports.

Key Features:
    - Zenoh transport: Uses dexcomm Node API with automatic codec decoding
    - RTC transport: Uses dexcomm VideoSubscriber with FFmpeg hardware acceleration
    - Thread-safe operations with zero-copy data access where possible
    - Production-grade error handling and logging
    - Google-style docstrings with comprehensive examples

Available Classes:
    - USBCameraSensor: Single RGB stream camera
    - ZedCameraSensor: Multi-stream camera (left_rgb, right_rgb, depth)
    - BaseCameraSensor: Abstract base class for custom cameras
    - TransportType: Enum for transport types (ZENOH, RTC)
    - StreamType: Enum for stream data types (RGB, DEPTH)

Example:
    ```python
    from dexcontrol.sensors.camera import USBCameraSensor

    camera = USBCameraSensor(
        name="camera_front",
        stream_config={
            "enable": True,
            "transport": "rtc",
            "rtc_channel": "sensors/camera_front/rgb_rtc",
            "codec": "h264"
            # width/height optional - VideoSubscriber auto-queries from publisher
        }
    )

    # Get latest frame
    frame = camera.get_rgb()  # np.ndarray (1080, 1920, 3) uint8
    ```
"""

from dexcontrol.sensors.camera.base_camera import (
    BaseCameraSensor,
    StreamType,
    TransportType,
)
from dexcontrol.sensors.camera.usb_camera import USBCameraSensor
from dexcontrol.sensors.camera.zed_camera import ZedCameraSensor, ZedXOneCameraSensor

# Backward compatibility alias
RGBCameraSensor = USBCameraSensor

__all__ = [
    # Sensor classes
    "USBCameraSensor",
    "ZedCameraSensor",
    "ZedXOneCameraSensor",
    "RGBCameraSensor",  # Backward compatibility
    # Base classes and types
    "BaseCameraSensor",
    "TransportType",
    "StreamType",
]
