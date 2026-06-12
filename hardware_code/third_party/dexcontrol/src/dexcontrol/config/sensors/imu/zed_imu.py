# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

from dataclasses import dataclass


@dataclass
class ZedIMUConfig:
    _target_: str = "dexcontrol.sensors.imu.zed_imu.ZedIMUSensor"
    topic: str = "/imu/head_camera"
    name: str = "head_camera_imu"
    enable_fps_tracking: bool = False
    fps_log_interval: int = 30
    enable: bool = False
