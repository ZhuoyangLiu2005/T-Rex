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

import numpy as np


@dataclass
class ArmConfig:
    _target_: str = "dexcontrol.core.arm.Arm"
    state_sub_topic: str = "state/arm/right"
    wrench_sub_topic: str = "state/wrench/right"
    control_pub_topic: str = "control/arm/right"
    set_mode_query: str = "mode/arm/right"
    ee_pass_through_pub_topic: str = "control/ee_pass_through/right"
    dof: int = 7
    default_control_hz: int = 100
    joint_name: list[str] = field(
        default_factory=lambda: [f"R_arm_j{i + 1}" for i in range(7)]
    )
    joint_limit: list[list[float]] = field(
        default_factory=lambda: [[-np.pi, np.pi] for _ in range(7)]
    )
    joint_vel_limit: list[float] = field(
        default_factory=lambda: [2.4, 2.4, 2.7, 2.7, 2.7, 2.7, 2.7]
    )
    pose_pool: dict[str, list[float]] = field(
        default_factory=lambda: {
            "folded": [-1.57079, 0.0, 0, -3.1, 0, 0, 0.69813],
            "folded_closed_hand": [-1.57079, 0.0, 0, -3.1, 0, 0, 0.9],
            "L_shape": [-0.064, 0.3, 0.0, -1.556, -1.271, 0.0, 0.0],
            "lift_up": [-0.064, 0.3, 0.0, -2.756, -1.271, 0.0, 0.0],
            "zero": [1.57079, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
