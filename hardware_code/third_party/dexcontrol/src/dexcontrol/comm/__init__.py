# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""DexControl communication module.

Clean, modular communication layer providing:
- DexComm Raw API integration for standard pub/sub
- WebRTC support for real-time video streaming
- Unified API across all subscriber types
"""

from dexcontrol.comm.rtc import (
    RTCSubscriber,
    create_rtc_camera_subscriber,
)
from dexcontrol.comm.subscribers import (
    create_buffered_subscriber,
    create_camera_subscriber,
    create_depth_subscriber,
    create_generic_subscriber,
    create_imu_subscriber,
    create_lidar_subscriber,
    create_subscriber,
)

__all__ = [
    # Core functions
    "create_subscriber",
    "create_buffered_subscriber",
    # Sensor-specific
    "create_camera_subscriber",
    "create_depth_subscriber",
    "create_imu_subscriber",
    "create_lidar_subscriber",
    "create_generic_subscriber",
    # WebRTC
    "RTCSubscriber",
    "create_rtc_camera_subscriber",
    # Utilities
]
