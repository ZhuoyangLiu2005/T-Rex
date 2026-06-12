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
class TorsoConfig:
    _target_: str = "dexcontrol.core.torso.Torso"
    state_sub_topic: str = "state/torso"
    control_pub_topic: str = "control/torso"
    dof: int = 3
    joint_name: list[str] = field(
        default_factory=lambda: ["torso_j1", "torso_j2", "torso_j3"]
    )
    joint_limit: list[list[float]] = field(
        default_factory=lambda: [
            [0, 1.570],
            [0, 3.141],
            [-1.570, 1.570],
        ]
    )
    joint_vel_limit: list[float] = field(default_factory=lambda: [0.6, 0.6, 0.6])
    pose_pool: dict[str, list[float]] = field(
        default_factory=lambda: {
            "home": [0.0, 0.0, 0.0],
            "folded": [0.0, 0.0, -1.5708],
            "crouch20_low": [0.0, 0.0, -0.35],
            "crouch20_medium": [0.52, 1.05, 0.18],
            "crouch20_high": [0.78, 1.57, 0.44],
            "crouch45_low": [0.0, 0.0, -0.79],
            "crouch45_medium": [0.52, 1.05, -0.26],
            "crouch45_high": [0.78, 1.57, 0],
            "crouch90_low": [0.0, 0.0, -1.57],
            "crouch90_medium": [0.52, 1.05, -1.04],
            "crouch90_high": [0.78, 1.57, -0.78],
        }
    )
