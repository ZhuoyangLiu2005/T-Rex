# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""IMU sensors package for dexcontrol.

This package provides sensor classes for various IMU (Inertial Measurement Unit)
sensors using Zenoh subscribers for data communication.

Available sensors:
    - NineAxisIMUSensor: Standard 9-axis IMU with accelerometer, gyroscope, and magnetometer
    - ZedIMUSensor: IMU specific to Zed hardware (6-axis: accelerometer + gyroscope)
"""

from .chassis_imu import ChassisIMUSensor
from .zed_imu import ZedIMUSensor

__all__ = [
    "ChassisIMUSensor",
    "ZedIMUSensor",
]
