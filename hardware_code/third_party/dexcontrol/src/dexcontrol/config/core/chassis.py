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
class ChassisConfig:
    _target_: str = "dexcontrol.core.chassis.Chassis"
    steer_control_pub_topic: str = "control/chassis/steer"
    steer_state_sub_topic: str = "state/chassis/steer"
    drive_control_pub_topic: str = "control/chassis/drive"
    drive_state_sub_topic: str = "state/chassis/drive"
    dof: int = 2
    center_to_wheel_axis_dist: float = (
        0.219  # the distance between base center and wheel axis in m
    )
    wheels_dist: float = 0.45  # the distance between two wheels in m (0.41 for vega-rc2, 0.45 for vega-1)
    steer_joint_name: list[str] = field(
        default_factory=lambda: ["L_wheel_j1", "R_wheel_j1"]
    )
    drive_joint_name: list[str] = field(
        default_factory=lambda: ["L_wheel_j2", "R_wheel_j2"]
    )
    max_vel: float = 0.8
