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
class UltraSonicConfig:
    _target_: str = "dexcontrol.sensors.ultrasonic.UltrasonicSensor"
    topic: str = "state/ultrasonic"
    name: str = "ultrasonic"
    enable_fps_tracking: bool = False
    fps_log_interval: int = 30
    enable: bool = False
